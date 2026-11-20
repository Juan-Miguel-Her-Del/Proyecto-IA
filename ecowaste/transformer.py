"""Bloques comunes propios: proyecciones Q/K/V, atención y MLP residual."""
from torch import nn
from torch.nn import functional as F


class Atencion(nn.Module):
    def __init__(self, dimension, cabezas):
        super().__init__()
        if dimension % cabezas:
            raise ValueError("El embedding debe ser divisible entre las cabezas")
        self.cabezas = cabezas
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.proyeccion = nn.Linear(dimension, dimension)

    def forward(self, x, validos=None):
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.cabezas, d // self.cabezas).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # En SDPA True permite atender; None permite todos los patches visuales.
        mascara = validos[:, None, None, :] if validos is not None else None
        salida = F.scaled_dot_product_attention(q, k, v, attn_mask=mascara, dropout_p=0.0)
        return self.proyeccion(salida.transpose(1, 2).reshape(b, t, d))


class Bloque(nn.Module):
    def __init__(self, dimension, cabezas, mlp):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.atencion = Atencion(dimension, cabezas)
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(nn.Linear(dimension, mlp), nn.GELU(), nn.Linear(mlp, dimension))

    def forward(self, x, validos=None):
        x = x + self.atencion(self.norm1(x), validos)
        return x + self.mlp(self.norm2(x))
