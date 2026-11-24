"""Control Transformer y CNN+Transformer V30, con la misma cabeza de mapas."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from ecowaste.backbone_cnn import BackboneCNN, mascara_a
from ecowaste.detector import patches_rgb, dividir_ventanas, unir_ventanas
from ecowaste.detector_mapas import DetectorMapas
from ecowaste.datasets import CLASES

TIPO = 'ecowaste_hibrido_v30_desde_cero_1'


def mascara_letterbox(geometrias, device='cpu'):
    """Solo geometría de la fotografía, nunca etiquetas GT ni predicciones."""
    if not geometrias or any(g['lado'] != 384 for g in geometrias):
        raise ValueError('Geometrías 384 requeridas')
    yy, xx = torch.meshgrid(torch.arange(384, device=device), torch.arange(384, device=device), indexing='ij')
    ms = []
    for g in geometrias:
        x, y = g['relleno_x'], g['relleno_y']
        w, h = round(g['ancho_original'] * g['escala_x']), round(g['alto_original'] * g['escala_y'])
        ms.append((xx >= x) & (xx < x+w) & (yy >= y) & (yy < y+h))
    return torch.stack(ms)


def atender_con_mascara(x, bloque, validos, ventana=8):
    b, lado, _, _ = x.shape
    tokens = dividir_ventanas(x, ventana)
    mask = dividir_ventanas(validos[..., None], ventana).squeeze(-1)
    # Las ventanas completamente de padding no aportan claves ni salidas.
    salida = bloque(tokens, validos=mask).masked_fill(~mask[..., None], 0)
    return unir_ventanas(salida, b, lado, ventana)


class DetectorHibrido(DetectorMapas):
    def __init__(self, cnn=True, dimension=96, auxiliar_clasificacion=True):
        super().__init__(dimension=dimension, auxiliar_clasificacion=auxiliar_clasificacion)
        self.configuracion_hibrido = dict(cnn=cnn, dimension=dimension,
                                          auxiliar_clasificacion=auxiliar_clasificacion)
        self.backbone = BackboneCNN(dimension) if cnn else None
        if cnn:
            self.tokenizador_detalle = nn.Linear(dimension, dimension)
            self.tokenizador_contexto = nn.Linear(dimension, dimension)

    def caracteristicas(self, imagenes, validos=None):
        if imagenes.ndim != 4 or imagenes.shape[1:] != (3, 384, 384):
            raise ValueError('Entrada RGB 384x384 requerida')
        if validos is None:
            validos = torch.ones_like(imagenes[:, 0], dtype=torch.bool)
        if validos.shape != imagenes[:, 0].shape or validos.dtype != torch.bool:
            raise ValueError('Máscara de padding incompatible')
        b, d = len(imagenes), self.configuracion['dimension']
        x = ((imagenes-.5)/.5).masked_fill(~validos[:, None], 0)
        if self.backbone is not None:
            detalle, contexto = [z.flatten(2).transpose(1, 2) for z in self.backbone(x, validos)]
        else:
            detalle, contexto = patches_rgb(x, 8), patches_rgb(x, 16)
        md, mc = mascara_a(validos, (48, 48)), mascara_a(validos, (24, 24))
        detalle = self.tokenizador_detalle(detalle) + self.posicion_detalle*self.escala_posicion
        detalle = detalle.reshape(b, 48, 48, d).masked_fill(~md[..., None], 0)
        for bloque in self.detalle:
            detalle = atender_con_mascara(detalle, bloque, md, self.ventana)
        contexto = self.tokenizador_contexto(contexto) + self.posicion_contexto*self.escala_posicion
        for bloque in self.contexto:
            contexto = bloque(contexto, mc.flatten(1)).masked_fill(~mc.flatten(1)[..., None], 0)
        contexto = contexto.reshape(b, 24, 24, d).permute(0, 3, 1, 2)
        contexto = F.interpolate(contexto, (48, 48), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        concatenado = torch.cat([detalle, contexto], -1)
        fusion = self.norma_fusion(detalle + self.fusion_compuerta(concatenado).sigmoid()*self.fusion_valor(concatenado))
        return fusion.masked_fill(~md[..., None], 0)

    def forward(self, imagenes, validos=None, devolver_caracteristicas=False):
        fusion = self.caracteristicas(imagenes, validos)
        mapas = self.cabezal(fusion).permute(0, 3, 1, 2)
        return (mapas, fusion) if devolver_caracteristicas else mapas

    def clasificar(self, imagenes, validos=None):
        if self.clasificador_auxiliar is None:
            raise ValueError('Sin cabeza auxiliar')
        f = self.caracteristicas(imagenes, validos)
        if validos is None:
            pooled = f.mean((1, 2))
        else:
            m = mascara_a(validos, (48, 48))
            pooled = f.sum((1, 2))/m.sum((1, 2)).clamp(min=1)[:, None]
        return self.clasificador_auxiliar(pooled)


def crear_pareja(semilla=3001):
    """Pesos aleatorios propios; comparte inicialización de módulos compatibles."""
    torch.manual_seed(semilla)
    a = DetectorHibrido(cnn=False)
    torch.manual_seed(semilla)
    b = DetectorHibrido(cnn=True)
    propios = b.state_dict()
    for n, t in a.state_dict().items():
        if n in propios and propios[n].shape == t.shape:
            propios[n].copy_(t)
    b.load_state_dict(propios)
    return a, b


def cargar_hibrido(path, dispositivo='cpu'):
    ck = torch.load(Path(path), map_location='cpu', weights_only=True)
    if ck.get('tipo') != TIPO or ck.get('clases') != CLASES or ck.get('pesos_externos') is not False:
        raise ValueError('Checkpoint híbrido incompatible')
    m = DetectorHibrido(**ck['configuracion_hibrido']).to(dispositivo)
    m.load_state_dict(ck['pesos'], strict=True)
    return m.eval(), ck
