"""EcoWasteFormer-Cls: tokenización lineal y atención global desde cero."""
import math
from pathlib import Path

import torch
from torch import nn
from ecowaste.datasets import CLASES
from ecowaste.transformer import Bloque


def posiciones_2d(cuadricula, dimension):
    if dimension % 4:
        raise ValueError("La codificación 2D requiere embedding divisible entre cuatro")
    mitad = dimension // 2
    eje = torch.arange(cuadricula).float().unsqueeze(1)
    frecuencia = torch.exp(torch.arange(0, mitad, 2).float() * (-math.log(10000.0) / mitad))
    codificacion = torch.zeros(cuadricula, mitad)
    codificacion[:, 0::2] = (eje * frecuencia).sin()
    codificacion[:, 1::2] = (eje * frecuencia).cos()
    y = codificacion[:, None, :].expand(cuadricula, cuadricula, mitad)
    x = codificacion[None, :, :].expand(cuadricula, cuadricula, mitad)
    return torch.cat([y, x], dim=-1).reshape(1, cuadricula ** 2, dimension)


class ClasificadorResiduos(nn.Module):
    def __init__(self, imagen=224, patch=16, dimension=128, bloques=4, cabezas=4, mlp=256, clases=6):
        super().__init__()
        if imagen % patch or dimension % cabezas:
            raise ValueError("Dimensiones incompatibles con patches o cabezas")
        self.configuracion = dict(imagen=imagen, patch=patch, dimension=dimension, bloques=bloques, cabezas=cabezas, mlp=mlp, clases=clases)
        self.imagen, self.patch = imagen, patch
        self.tokenizador = nn.Linear(3 * patch * patch, dimension)
        self.register_buffer("posicion", posiciones_2d(imagen // patch, dimension))
        self.bloques = nn.ModuleList([Bloque(dimension, cabezas, mlp) for _ in range(bloques)])
        self.norma = nn.LayerNorm(dimension)
        self.cabezal = nn.Linear(dimension, clases)

    def extraer_patches(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, self.imagen, self.imagen):
            raise ValueError(f"Se esperaba [B,3,{self.imagen},{self.imagen}]")
        p = self.patch
        return x.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).reshape(x.shape[0], -1, 3 * p * p)

    def forward(self, imagenes):
        # Entrada RGB [0,1]; normalización fija propia, idéntica en entrenamiento/inferencia.
        x = self.tokenizador(self.extraer_patches((imagenes - 0.5) / 0.5)) + self.posicion
        for bloque in self.bloques:
            x = bloque(x)
        return self.cabezal(self.norma(x.mean(dim=1)))


def cargar_clasificador(checkpoint, dispositivo="cpu"):
    datos = torch.load(Path(checkpoint), map_location=dispositivo, weights_only=True)
    if datos.get("tipo") != "ecowaste_clasificador_desde_cero_v1" or datos.get("clases") != CLASES:
        raise ValueError("Checkpoint o taxonomía no compatibles")
    modelo = ClasificadorResiduos(**datos["configuracion"]).to(dispositivo)
    modelo.load_state_dict(datos["pesos"])
    return modelo.eval(), datos
