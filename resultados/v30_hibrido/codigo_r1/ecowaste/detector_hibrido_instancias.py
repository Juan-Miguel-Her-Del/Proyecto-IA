"""Alternativa V30 propia por consultas; no carga modelos ni pesos de terceros.

No forma parte de la ablación A/B de tokenizador. Debe entrenarse y evaluarse
con un protocolo separado antes de ofrecer resultados o integrarla en la app.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ecowaste.detector_hibrido import DetectorHibrido
from ecowaste.backbone_cnn import mascara_a
from ecowaste.detector_consultas import (BloqueConsultas, auto_atencion_posicional,
    cxcywh_a_xyxy, perdida_consultas, postprocesar_consultas)


def cruzar_enmascarado(modulo, x, memoria, sesgo, validos, posicion_q, posicion_m):
    """Reutiliza proyecciones propias; excluye explícitamente claves padding."""
    b, n, d = x.shape; h = modulo.cabezas
    q = modulo.q(x+posicion_q).reshape(b, n, h, d//h).transpose(1, 2)
    k = F.linear(memoria+posicion_m, modulo.kv.weight[:d], modulo.kv.bias[:d]).reshape(b, -1, h, d//h).transpose(1, 2)
    v = F.linear(memoria, modulo.kv.weight[d:], modulo.kv.bias[d:]).reshape(b, -1, h, d//h).transpose(1, 2)
    mascara = sesgo[None, None].expand(b, 1, -1, -1).masked_fill(~validos[:, None, None], -torch.inf)
    z = F.scaled_dot_product_attention(q, k, v, attn_mask=mascara.to(q.dtype), dropout_p=0.)
    return modulo.salida(z.transpose(1, 2).reshape(b, n, d))


class DetectorHibridoInstancias(DetectorHibrido):
    def __init__(self, consultas=100, capas=2, dimension=96):
        if math.isqrt(consultas)**2 != consultas or capas < 1:
            raise ValueError('Consultas cuadradas y al menos una capa requeridas')
        super().__init__(cnn=True, dimension=dimension, auxiliar_clasificacion=True)
        self.cabezal = nn.Identity()
        self.configuracion_instancias = dict(consultas=consultas, capas=capas, dimension=dimension)
        self.consultas = nn.Parameter(torch.randn(consultas, dimension)*.02)
        self.decodificador = nn.ModuleList([BloqueConsultas(dimension) for _ in range(capas)])
        self.norma_memoria = nn.LayerNorm(dimension)
        self.norma_salida = nn.LayerNorm(dimension)
        self.material = nn.Linear(dimension, 7)  # índice 6 = ausencia de objeto
        self.caja = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, 4))
        lado = math.isqrt(consultas)
        y, x = torch.meshgrid((torch.arange(lado)+.5)/lado, (torch.arange(lado)+.5)/lado, indexing='ij')
        ref = torch.stack([x, y], -1).flatten(0, 1)
        yy, xx = torch.meshgrid((torch.arange(48)+.5)/48, (torch.arange(48)+.5)/48, indexing='ij')
        pos = torch.stack([xx, yy], -1).flatten(0, 1)
        self.register_buffer('referencia', ref)
        self.register_buffer('sesgo_espacial', -(ref[:, None]-pos[None]).square().sum(-1)/(2*.15**2))
        self.register_buffer('prior_wh', torch.tensor([.1, .1]))
        with torch.no_grad():
            self.material.bias.zero_(); self.material.bias[6] = math.log(54)
            self.caja[-1].weight.zero_(); self.caja[-1].bias.zero_()

    def forward(self, imagenes, validos=None):
        f = self.caracteristicas(imagenes, validos)
        memoria = self.norma_memoria(f.flatten(1, 2))
        mask = torch.ones(len(f), 48*48, dtype=torch.bool, device=f.device) if validos is None else mascara_a(validos, (48, 48)).flatten(1)
        pq = F.layer_norm(self.consultas, (self.consultas.shape[-1],))[None]
        pm = self.posicion_detalle*self.escala_posicion
        x = torch.zeros_like(pq).expand(len(f), -1, -1)
        for bloque in self.decodificador:
            x = x+auto_atencion_posicional(bloque.propia, bloque.normas[0](x), pq)
            x = x+cruzar_enmascarado(bloque.cruzada, bloque.normas[1](x), memoria, self.sesgo_espacial, mask, pq, pm)
            x = x+bloque.mlp(bloque.normas[2](x))
        x = self.norma_salida(x); raw = self.caja(x)
        centro = (raw[..., :2]+torch.logit(self.referencia)[None]).sigmoid()
        wh = (raw[..., 2:]+torch.logit(self.prior_wh)[None, None]).sigmoid()
        return {'logits': self.material(x), 'boxes': cxcywh_a_xyxy(torch.cat([centro, wh], -1))}


__all__ = ['DetectorHibridoInstancias', 'perdida_consultas', 'postprocesar_consultas']
