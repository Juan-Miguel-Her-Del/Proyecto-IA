"""Separa la localización del material usando predicciones val ya guardadas."""
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.datasets import objetivo_deteccion
from ecowaste.evaluacion_detector import evaluar_detecciones

filas = [json.loads(l) for l in (RAIZ / "datos/procesados/deteccion/val.jsonl").read_text(encoding="utf-8").splitlines()]
objetivos = [objetivo_deteccion(r) for r in filas]
for o in objetivos:
    o["labels"] = torch.zeros_like(o["labels"])
informe = {"metodo": "Todas las clases válidas se colapsan en residuo, sin alterar cajas ni puntuaciones; mismo evaluador propio.",
    "test_evaluado": False, "experimentos": {}}
for nombre in ["detector_piloto_v1", "detector_piloto_v2"]:
    datos = json.loads((RAIZ / f"resultados/{nombre}_predicciones_val.json").read_text(encoding="utf-8"))
    assert [d["id"] for d in datos] == [r["id"] for r in filas]
    pred = [{"boxes": torch.tensor(d["boxes"], dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(d["scores"], dtype=torch.float32),
        "labels": torch.zeros(len(d["scores"]), dtype=torch.long)} for d in datos]
    m = evaluar_detecciones(pred, objetivos)
    informe["experimentos"][nombre] = {"map50_sin_material": m["map50"], "recall50_sin_material": m["recall50"],
        "precision50_sin_material": m["precision50"], "objetos": m["objetos"]}
(RAIZ / "resultados/diagnostico_localizacion.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(informe, ensure_ascii=False, indent=2))
