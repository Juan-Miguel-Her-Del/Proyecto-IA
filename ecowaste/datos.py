"""Validación geométrica y agrupación de imágenes similares."""
import math


def caja_valida(caja, ancho, alto):
    if len(caja) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in caja):
        return False
    x, y, w, h = caja
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= ancho + 1e-6 and y + h <= alto + 1e-6


def recortar_desborde_minimo(caja, ancho, alto):
    """Intersección geométrica solo para desbordes <=2px y pérdida de área <=2%."""
    if len(caja) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in caja):
        return None
    x, y, w, h = caja
    if w <= 0 or h <= 0:
        return None
    if max(-x, -y, x + w - ancho, y + h - alto) > 2:
        return None
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, ancho), min(y + h, alto)
    nueva = [x0, y0, x1 - x0, y1 - y0]
    if caja_valida(nueva, ancho, alto) and nueva[2] * nueva[3] / (w * h) >= 0.98:
        return nueva
    return None


class UnionFind:
    def __init__(self, cantidad):
        self.padre = list(range(cantidad))

    def find(self, i):
        while i != self.padre[i]:
            self.padre[i] = self.padre[self.padre[i]]
            i = self.padre[i]
        return i

    def unir(self, i, j):
        a, b = self.find(i), self.find(j)
        self.padre[max(a, b)] = min(a, b)


class ArbolHash:
    """BK-tree para búsqueda exacta en radio de Hamming (sin ANN)."""
    def __init__(self):
        self.raiz = None

    def agregar(self, valor, indice):
        if self.raiz is None:
            self.raiz = [valor, [indice], {}]
            return
        nodo = self.raiz
        while True:
            distancia = (valor ^ nodo[0]).bit_count()
            if distancia == 0:
                nodo[1].append(indice)
                return
            if distancia not in nodo[2]:
                nodo[2][distancia] = [valor, [indice], {}]
                return
            nodo = nodo[2][distancia]

    def buscar(self, valor, radio):
        pendientes = [self.raiz] if self.raiz else []
        while pendientes:
            nodo = pendientes.pop()
            distancia = (valor ^ nodo[0]).bit_count()
            if distancia <= radio:
                yield from nodo[1]
            pendientes.extend(hijo for d, hijo in nodo[2].items() if distancia - radio <= d <= distancia + radio)
