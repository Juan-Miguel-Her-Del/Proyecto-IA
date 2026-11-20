"""Mide ajuste en un subconjunto train y en las celdas supervisadas; no es evaluación final."""
import argparse
import json
from pathlib import Path
import random
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from torch.utils.data import DataLoader, Subset
from ecowaste.datasets import collate_deteccion
from ecowaste.detector import cargar_detector, decodificar
from ecowaste.cajas import iou_pares
from ecowaste.ejecucion_detector import CacheDeteccion, evaluar_detector


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nombre", default="detector_piloto_v5")
    parser.add_argument("--estado", choices=["mejor", "ultimo"], default="ultimo")
    args = parser.parse_args()
    if Path(args.nombre).name != args.nombre:
        raise ValueError("Nombre no válido")
    torch.set_num_threads(4)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, meta = cargar_detector(RAIZ / f"checkpoints/{args.nombre}_{args.estado}.pt", dispositivo)
    radio = modelo.configuracion.get("radio_centros", 0)
    cache = CacheDeteccion(RAIZ, "train", meta["umbral_escala"], radio)
    rng = random.Random(43)
    indices = set()
    for clase in range(6):
        candidatos = [i for i, o in enumerate(cache.objetivos) if bool((o["labels"] == clase).any())]
        indices.update(rng.sample(candidatos, min(10, len(candidatos))))
    indices = sorted(indices)
    loader = DataLoader(Subset(cache, indices), batch_size=4, collate_fn=collate_deteccion)
    metrica, _ = evaluar_detector(modelo, loader, dispositivo, False, meta["config_entrenamiento"])
    ious, correctas, probabilidades = [], [], []
    for x, objetivos in loader:
        raw = modelo(x.to(dispositivo).float() / 255)
        cajas, presencia, clases = decodificar(raw, radio)
        for b, o in enumerate(objetivos):
            a = o["asignacion"]
            ix = a["indices"].to(dispositivo)
            gt = o["boxes"][a["objetos"]].to(dispositivo)
            iou = iou_pares(cajas[b].reshape(-1, 4)[ix], gt).diag()
            ious.extend(iou.tolist())
            correctas.extend((clases[b].reshape(-1, 6)[ix].argmax(-1).cpu() == o["labels"][a["objetos"]]).tolist())
            probabilidades.extend(presencia[b].flatten()[ix].sigmoid().tolist())
    informe = {"experimento": args.nombre, "checkpoint": args.estado, "epoca": meta["epoca"],
        "ids_solo_train": [cache.objetivos[i]["id"] for i in indices], "escenas": len(indices),
        "deteccion_en_subconjunto_train": metrica,
        "solo_celdas_supervisadas": {"iou_medio": sum(ious) / len(ious), "fraccion_iou_mayor_05": sum(v >= .5 for v in ious) / len(ious),
            "accuracy_material": sum(correctas) / len(correctas), "presencia_media": sum(probabilidades) / len(probabilidades)},
        "test_evaluado": False, "limite": "Escenas de entrenamiento elegidas por presencia de clases, no distribución representativa. Las métricas de celdas supervisadas usan índices GT conocidos y no se presentan como inferencia o evaluación de generalización."}
    (RAIZ / f"resultados/{args.nombre}_ajuste_train.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"escenas": len(indices), "map50_train_subconjunto": metrica["map50"], **informe["solo_celdas_supervisadas"]}, indent=2))


if __name__ == "__main__":
    main()
