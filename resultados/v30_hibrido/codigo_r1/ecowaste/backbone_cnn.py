"""CNN propia V30, sin descarga ni inicialización de pesos externos."""
import torch
from torch import nn
from torch.nn import functional as F


def mascara_a(validos, forma):
    return F.interpolate(validos[:, None].float(), size=forma, mode='nearest-exact')[:, 0].bool()


class ConvNormAct(nn.Sequential):
    def __init__(self, entrada, salida, stride=1):
        super().__init__(nn.Conv2d(entrada, salida, 3, stride, 1, bias=False),
                         nn.GroupNorm(8, salida), nn.SiLU())


class ResidualCNN(nn.Module):
    def __init__(self, canales):
        super().__init__()
        self.rama = nn.Sequential(ConvNormAct(canales, canales),
                                  nn.Conv2d(canales, canales, 3, padding=1, bias=False),
                                  nn.GroupNorm(8, canales))
        self.activacion = nn.SiLU()

    def forward(self, x):
        return self.activacion(x + self.rama(x))


class BackboneCNN(nn.Module):
    """Stride 8 y 16; GroupNorm no depende del tamaño del microbatch.

    El padding se anula antes de entrar y después de cada escala. GroupNorm
    incluye esos ceros en sus estadísticas; no se afirma normalización enmascarada.
    """
    def __init__(self, dimension=96):
        super().__init__()
        if dimension != 96:
            raise ValueError('V30 fija 96 canales para aislar el experimento')
        self.etapas = nn.ModuleList([ConvNormAct(3, 24, 2), ConvNormAct(24, 48, 2),
            nn.Sequential(ConvNormAct(48, dimension, 2), ResidualCNN(dimension)),
            nn.Sequential(ConvNormAct(dimension, dimension, 2), ResidualCNN(dimension))])

    def forward(self, x, validos=None):
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[-2:] != (384, 384):
            raise ValueError('CNN requiere RGB 384x384')
        if validos is None:
            validos = torch.ones_like(x[:, 0], dtype=torch.bool)
        if validos.dtype != torch.bool or validos.shape != x[:, 0].shape:
            raise ValueError('Máscara de padding incompatible')
        x = x.masked_fill(~validos[:, None], 0)
        mapas = []
        for i, etapa in enumerate(self.etapas):
            x = etapa(x)
            x = x.masked_fill(~mascara_a(validos, x.shape[-2:])[:, None], 0)
            if i >= 2:
                mapas.append(x)
        return tuple(mapas)
