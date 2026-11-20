"""Detector propio con atención local/global, fusión y cabezal de cuadrícula."""
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from ecowaste.clasificador import posiciones_2d
from ecowaste.datasets import CLASES
from ecowaste.transformer import Bloque


def patches_rgb(imagenes, patch):
    b = imagenes.shape[0]
    return imagenes.unfold(2, patch, patch).unfold(3, patch, patch).permute(0, 2, 3, 1, 4, 5).reshape(b, -1, 3 * patch * patch)


def dividir_ventanas(x, lado_ventana):
    b, h, w, d = x.shape
    v = lado_ventana
    if h % v or w % v:
        raise ValueError("Las ventanas deben dividir la cuadrícula")
    return x.reshape(b, h // v, v, w // v, v, d).permute(0, 1, 3, 2, 4, 5).reshape(-1, v * v, d)


def unir_ventanas(ventanas, batch, lado, lado_ventana):
    v, d = lado_ventana, ventanas.shape[-1]
    return ventanas.reshape(batch, lado // v, lado // v, v, v, d).permute(0, 1, 3, 2, 4, 5).reshape(batch, lado, lado, d)


def atender_ventanas(x,bloque,ventana,desplazamiento=0):
    """Cambia la partición sin unir bordes opuestos; padding excluido de atención."""
    b,h,w,_=x.shape
    if h!=w or h%ventana or desplazamiento not in [0,ventana//2]:
        raise ValueError('Geometría de ventanas incompatible')
    s=desplazamiento
    if s:
        if ventana%2:raise ValueError('El desplazamiento requiere ventana par')
        extendido=F.pad(x,(0,0,s,s,s,s))
        validos=F.pad(torch.ones(b,h,w,1,dtype=torch.bool,device=x.device),(0,0,s,s,s,s))
        mascara=dividir_ventanas(validos,ventana).squeeze(-1)
    else:extendido=x;mascara=None
    ventanas=dividir_ventanas(extendido,ventana)
    salida=bloque(ventanas,validos=mascara)
    resultado=unir_ventanas(salida,b,extendido.shape[1],ventana)
    return resultado[:,s:s+h,s:s+w] if s else resultado


class DetectorResiduos(nn.Module):
    def __init__(self, imagen=384, patch_detalle=8, patch_contexto=16, dimension=96,
                 bloques_por_rama=2, cabezas=4, mlp=192, ventana=8, candidatos=2, clases=6, radio_centros=0, escala_posicion=1.0,ventanas_desplazadas=False):
        super().__init__()
        if imagen % patch_detalle or imagen % patch_contexto or (imagen // patch_detalle) % ventana or radio_centros not in [0, 1, 2]:
            raise ValueError("Patches y ventanas incompatibles con la imagen")
        if not math.isfinite(escala_posicion) or not 0 <= escala_posicion <= 10:
            raise ValueError("Escala posicional no válida")
        self.configuracion = dict(imagen=imagen, patch_detalle=patch_detalle, patch_contexto=patch_contexto,
            dimension=dimension, bloques_por_rama=bloques_por_rama, cabezas=cabezas, mlp=mlp,
            ventana=ventana, candidatos=candidatos, clases=clases, radio_centros=radio_centros, escala_posicion=escala_posicion,ventanas_desplazadas=ventanas_desplazadas)
        self.imagen, self.lado, self.ventana = imagen, imagen // patch_detalle, ventana
        self.candidatos, self.clases = candidatos, clases
        self.escala_posicion = escala_posicion
        self.tokenizador_detalle = nn.Linear(3 * patch_detalle ** 2, dimension)
        self.tokenizador_contexto = nn.Linear(3 * patch_contexto ** 2, dimension)
        self.register_buffer("posicion_detalle", posiciones_2d(self.lado, dimension))
        self.register_buffer("posicion_contexto", posiciones_2d(imagen // patch_contexto, dimension))
        self.detalle = nn.ModuleList([Bloque(dimension, cabezas, mlp) for _ in range(bloques_por_rama)])
        self.contexto = nn.ModuleList([Bloque(dimension, cabezas, mlp) for _ in range(bloques_por_rama)])
        self.fusion_valor = nn.Linear(dimension * 2, dimension)
        self.fusion_compuerta = nn.Linear(dimension * 2, dimension)
        self.norma_fusion = nn.LayerNorm(dimension)
        self.cabezal = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, candidatos * (clases + 5)))
        self.register_buffer("umbral_escala", torch.tensor(0.1))
        # Sesgo de presencia baja: no es un peso externo.
        with torch.no_grad():
            for candidato in range(candidatos):
                self.cabezal[-1].bias[candidato * (clases + 5)] = math.log(0.01 / 0.99)

    @torch.no_grad()
    def configurar_priores(self, priors_wh, umbral_escala):
        priors = torch.as_tensor(priors_wh, dtype=self.cabezal[-1].bias.dtype, device=self.cabezal[-1].bias.device)
        if priors.shape != (self.candidatos, 2) or not bool(((priors > 0) & (priors < 1)).all()):
            raise ValueError("Los tamaños iniciales deben estar normalizados y ser positivos")
        for i in range(self.candidatos):
            comienzo = i * (self.clases + 5) + self.clases + 3
            self.cabezal[-1].bias[comienzo:comienzo + 2] = torch.logit(priors[i])
        self.umbral_escala.fill_(float(umbral_escala))

    def caracteristicas(self, imagenes):
        if imagenes.ndim != 4 or tuple(imagenes.shape[1:]) != (3, self.imagen, self.imagen):
            raise ValueError(f"Entrada esperada [B,3,{self.imagen},{self.imagen}]")
        b, d = len(imagenes), self.configuracion["dimension"]
        x = (imagenes - 0.5) / 0.5
        detalle = self.tokenizador_detalle(patches_rgb(x, self.configuracion["patch_detalle"])) + self.posicion_detalle * self.escala_posicion
        detalle=detalle.reshape(b,self.lado,self.lado,d)
        for i,bloque in enumerate(self.detalle):
            desplazamiento=self.ventana//2 if self.configuracion['ventanas_desplazadas'] and i%2 else 0
            detalle=atender_ventanas(detalle,bloque,self.ventana,desplazamiento)
        contexto = self.tokenizador_contexto(patches_rgb(x, self.configuracion["patch_contexto"])) + self.posicion_contexto * self.escala_posicion
        for bloque in self.contexto:
            contexto = bloque(contexto)
        lado_contexto = self.imagen // self.configuracion["patch_contexto"]
        contexto = contexto.reshape(b, lado_contexto, lado_contexto, d).permute(0, 3, 1, 2)
        contexto = F.interpolate(contexto, size=(self.lado, self.lado), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        concatenado = torch.cat([detalle, contexto], dim=-1)
        fusion = self.norma_fusion(detalle + self.fusion_compuerta(concatenado).sigmoid() * self.fusion_valor(concatenado))
        return fusion

    def forward(self, imagenes):
        fusion = self.caracteristicas(imagenes)
        return self.cabezal(fusion).reshape(len(imagenes), self.lado, self.lado, self.candidatos, self.clases + 5)


def decodificar(salida, radio_centros=0):
    """Devuelve cajas xyxy normalizadas; se limita a la imagen solo al inferir."""
    raw = salida.float()
    h, w = raw.shape[1:3]
    y, x = torch.meshgrid(torch.arange(h, device=raw.device), torch.arange(w, device=raw.device), indexing="ij")
    rejilla = torch.stack([x, y], dim=-1)[None, :, :, None, :]
    # El rango ampliado permite regresar a la caja real desde celdas vecinas supervisadas.
    offsets = raw[..., -4:-2].sigmoid() * (1 + 2 * radio_centros) - radio_centros
    centros = (offsets + rejilla) / raw.new_tensor([w, h])
    tamanos = raw[..., -2:].sigmoid()
    cajas = torch.cat([centros - tamanos / 2, centros + tamanos / 2], dim=-1)
    return cajas, raw[..., 0], raw[..., 1:-4]


def cargar_detector(checkpoint, dispositivo="cpu"):
    datos = torch.load(Path(checkpoint), map_location=dispositivo, weights_only=True)
    if datos.get("tipo") != "ecowaste_detector_desde_cero_v1" or datos.get("clases") != CLASES:
        raise ValueError("Checkpoint o clases del detector no compatibles")
    modelo = DetectorResiduos(**datos["configuracion"]).to(dispositivo)
    modelo.load_state_dict(datos["pesos"])
    return modelo.eval(), datos
