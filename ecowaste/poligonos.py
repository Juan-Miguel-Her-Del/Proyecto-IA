"""Supervisión semántica desde contornos originales TACO, con procedencia comprobada."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from ecowaste.datasets import CLASES, objetivo_deteccion
from ecowaste.detector_mapas import rasterizar

POLITICA = "poligonos_taco_384_promedio48_cobertura025_v1"


def nombre_politica(cobertura):
    if cobertura == .25:return POLITICA
    if cobertura == 1/64:return "poligonos_taco_384_promedio48_cualquier_pixel_v2"
    raise ValueError("Cobertura no registrada")


class ContornosTACO:
    def __init__(self, raiz, cobertura_minima=.25):
        self.cobertura_minima=cobertura_minima
        self.politica=nombre_politica(cobertura_minima)
        path = Path(raiz) / "datos/originales/taco/annotations.json"
        contenido = path.read_bytes()
        datos = json.loads(contenido)
        self.sha256 = hashlib.sha256(contenido).hexdigest()
        self.anotaciones = datos["annotations"]
        self.imagenes = {i["id"]: i for i in datos["images"]}
        self.categorias = {c["id"]: c["name"] for c in datos["categories"]}
        inventario = (Path(raiz) / "datos/reportes/inventario_imagenes.jsonl").read_bytes()
        self.sha256_inventario = hashlib.sha256(inventario).hexdigest()
        self.inventario = {r["id"]: r for r in (json.loads(l) for l in inventario.splitlines() if l)}

    def contornos(self, registro, caja):
        if not 1<=caja['id_interno']<=len(self.anotaciones):
            raise ValueError('Índice interno fuera de rango')
        a = self.anotaciones[caja["id_interno"] - 1]
        if (a["id"] != caja["id_original"] or a["bbox"] != caja["bbox_original"]
                or self.categorias[a["category_id"]] != caja["categoria_original"]):
            raise ValueError("El contorno no corresponde a la caja procesada")
        im = self.imagenes[a["image_id"]]
        if (im["width"], im["height"]) != (registro["ancho"], registro["alto"]):
            raise ValueError("Contorno e imagen tienen geometrías distintas")
        if a["image_id"] != registro["id_original"]:
            # La preparación fusionó una imagen exactamente duplicada. Exigir
            # que ese hecho conste en el manifiesto; no confiar solo en tamaños.
            fuente = self.inventario.get(f"taco:{a['image_id']}", {})
            if (not fuente.get("sha256_pixeles") or fuente["sha256_pixeles"] != registro["sha256_pixeles"]
                    or fuente.get("grupo") != registro["grupo"]):
                raise ValueError("Imagen de contorno distinta sin fusión documentada")
        segmentos = a.get("segmentation")
        if not isinstance(segmentos, list) or not segmentos:
            raise ValueError("No hay polígonos originales; no se inventa una máscara")
        for p in segmentos:
            if len(p) < 6 or len(p) % 2 or not np.isfinite(p).all():
                raise ValueError("Polígono original inválido")
        return segmentos

    def rasterizar(self, registro, lado=48):
        o = objetivo_deteccion(registro)
        segmentos = [(self.contornos(registro, a), CLASES.index(a["clase"]))
                     for a in registro["cajas"] if a["clase"]]
        return rasterizar_contornos(o, segmentos, lado, self.cobertura_minima)


def rasterizar_contornos(objetivo, segmentos, lado=48, cobertura_minima=.25, recortar_al_contenido=False):
    """Rellena a 384 px y promedia a la rejilla; fracciones pequeñas se ignoran.

    Los segmentos están en coordenadas originales y se agrupan por material.
    El bbox se conserva para evaluar instancias, no para fabricar positivos.
    """
    g = objetivo["geometria"]
    resolucion = g["lado"]
    if resolucion % lado:
        raise ValueError("La resolución debe ser múltiplo de la rejilla")
    lienzos = [Image.new("L", (resolucion, resolucion)) for _ in CLASES]
    pinceles = [ImageDraw.Draw(im) for im in lienzos]
    for poligonos, etiqueta in segmentos:
        for p in poligonos:
            puntos = [(x * g["escala_x"] + g["relleno_x"], y * g["escala_y"] + g["relleno_y"])
                      for x, y in zip(p[::2], p[1::2])]
            pinceles[etiqueta].polygon(puntos, fill=1)
    pixeles = torch.from_numpy(np.stack([np.array(im) for im in lienzos])).float()
    if recortar_al_contenido:
        # Un polígono que cruza el borde del recorte no debe pintar el padding.
        valido=torch.zeros(resolucion,resolucion,dtype=torch.bool)
        x,y=g['relleno_x'],g['relleno_y']
        w,h=round(g['ancho_original']*g['escala_x']),round(g['alto_original']*g['escala_y'])
        valido[y:y+h,x:x+w]=True
        pixeles*=valido
    cobertura = F.avg_pool2d(pixeles, resolucion // lado)
    # Hereda solo fondo, padding e ignoradas. Nunca positivos rectangulares.
    vacio = dict(objetivo, boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.long))
    resultado = rasterizar(vacio, lado)
    padding=resultado==255 if not len(objetivo['ignore_boxes']) else rasterizar(dict(vacio,ignore_boxes=torch.empty(0,4)),lado)==255
    resultado[cobertura.sum(0) > 0] = 255
    positivas = cobertura >= cobertura_minima
    unica = positivas.sum(0) == 1
    for c in range(6):
        resultado[positivas[c] & unica] = c + 1
    if recortar_al_contenido:resultado[padding]=255
    return resultado
