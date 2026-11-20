"""Comprueba todas las asignaciones train sin leer ni evaluar el conjunto test."""
from collections import Counter
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.datasets import ResiduosDataset, objetivo_deteccion
from ecowaste.perdida_detector import asignar, priores_train

ds = ResiduosDataset(RAIZ, "deteccion", "train")
objetivos = [objetivo_deteccion(r) for r in ds.registros]
_, corte = priores_train(objetivos)
reporte = {"imagenes_train": len(ds), "objetos_train": sum(len(o["boxes"]) for o in objetivos),
    "candidatos_totales_train": len(ds) * 48 * 48 * 2, "test_evaluado": False, "modos": {}}
for radio in [0, 1]:
    positivos, colisiones, extremos, clases = 0, 0, 0, Counter()
    for o in objetivos:
        a = asignar(o, umbral_escala=corte, radio_centros=radio)
        assert len(a["indices"].unique()) == len(a["indices"])
        boxes = o["boxes"][a["objetos"]]
        celdas = a["indices"] // 2
        xy = torch.stack([celdas % 48, celdas // 48], -1)
        offset = (((boxes[:, :2] + boxes[:, 2:]) / 2) * 48 - xy + radio) / (1 + 2 * radio)
        assert bool(((offset >= -1e-6) & (offset <= 1 + 1e-6)).all()), o["id"]
        extremos += int(((offset <= 1e-6) | (offset >= 1 - 1e-6)).any(-1).sum())
        positivos += len(a["indices"])
        colisiones += len(a["colisiones"])
        clases.update(o["labels"][a["objetos"]].tolist())
    reporte["modos"][str(radio)] = {"positivos": positivos, "colisiones_sin_representar": colisiones,
        "porcentaje_positivos_sobre_candidatos": 100 * positivos / reporte["candidatos_totales_train"],
        "offsets_en_limite_del_rango": extremos, "asignaciones_por_clase_id": dict(clases),
        "cajas_regresables_y_candidatos_sin_duplicados": True}
(RAIZ / "resultados/auditoria_supervision_detector.json").write_text(json.dumps(reporte, indent=2), encoding="utf-8")
print(json.dumps(reporte, indent=2))
