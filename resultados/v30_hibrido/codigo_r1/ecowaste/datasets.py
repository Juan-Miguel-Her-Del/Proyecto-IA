"""Lectores de los manifiestos: rutas relativas, letterbox y regiones a ignorar."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

CLASES = ["plastico", "papel", "carton", "vidrio", "metal", "organico"]


def geometria_letterbox(ancho, alto, lado):
    escala = lado / max(ancho, alto)
    nuevo_ancho, nuevo_alto = max(1, round(ancho * escala)), max(1, round(alto * escala))
    izquierda, arriba = (lado - nuevo_ancho) // 2, (lado - nuevo_alto) // 2
    return {"escala_x": nuevo_ancho / ancho, "escala_y": nuevo_alto / alto,
        "relleno_x": izquierda, "relleno_y": arriba, "lado": lado, "ancho_original": ancho, "alto_original": alto}


def letterbox(imagen, lado):
    ancho, alto = imagen.size
    geometria = geometria_letterbox(ancho, alto, lado)
    nuevo_ancho, nuevo_alto = round(ancho * geometria["escala_x"]), round(alto * geometria["escala_y"])
    lienzo = Image.new("RGB", (lado, lado), (114, 114, 114))
    lienzo.paste(imagen.resize((nuevo_ancho, nuevo_alto), Image.Resampling.BILINEAR), (geometria["relleno_x"], geometria["relleno_y"]))
    tensor = torch.from_numpy(np.array(lienzo, copy=True)).permute(2, 0, 1).float().div_(255)
    return tensor, geometria


def transformar_caja(bbox, geometria):
    x, y, w, h = bbox
    g = geometria
    x0, y0 = x * g["escala_x"] + g["relleno_x"], y * g["escala_y"] + g["relleno_y"]
    return [x0 / g["lado"], y0 / g["lado"], (x0 + w * g["escala_x"]) / g["lado"], (y0 + h * g["escala_y"]) / g["lado"]]


def objetivo_deteccion(registro, geometria=None):
    g = geometria or geometria_letterbox(registro["ancho"], registro["alto"], 384)
    objetos, ignoradas, etiquetas = [], [], []
    for a in registro["cajas"]:
        caja = transformar_caja(a["bbox"], g)
        if a["clase"]:
            objetos.append(caja)
            etiquetas.append(CLASES.index(a["clase"]))
        else:
            ignoradas.append(caja)
    return {"id": registro["id"], "boxes": torch.tensor(objetos, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(etiquetas, dtype=torch.int64),
        "ignore_boxes": torch.tensor(ignoradas, dtype=torch.float32).reshape(-1, 4),
        "geometria": g, "formato_boxes": "xyxy_normalizado"}


class ResiduosDataset(Dataset):
    def __init__(self, raiz, tarea, split):
        if tarea not in ["clasificacion", "deteccion"] or split not in ["train", "val", "test"]:
            raise ValueError("Tarea o partición no reconocida")
        self.raiz, self.tarea = Path(raiz), tarea
        path = self.raiz / "datos/procesados" / tarea / f"{split}.jsonl"
        self.registros = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, indice):
        r = self.registros[indice]
        with Image.open(self.raiz / r["ruta"]) as imagen:
            # Las rutas TACO con orientación EXIF ya apuntan al PNG normalizado.
            tensor, geometria = letterbox(imagen.convert("RGB"), 224 if self.tarea == "clasificacion" else 384)
        if self.tarea == "clasificacion":
            return tensor, CLASES.index(r["clase"])
        return tensor, objetivo_deteccion(r, geometria)


def collate_deteccion(lote):
    imagenes, objetivos = zip(*lote)
    return torch.stack(imagenes), list(objetivos)
