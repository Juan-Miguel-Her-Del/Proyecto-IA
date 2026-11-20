"""AP interpolada a 101 recalls. Evaluador propio, no COCOeval oficial."""
import torch
from ecowaste.cajas import cubiertas_por_ignorar, iou_pares
from ecowaste.datasets import CLASES


def evaluar_detecciones(predicciones, objetivos, umbrales=None):
    if len(predicciones) != len(objetivos) or not objetivos:
        raise ValueError("Se requieren predicciones y objetivos por imagen")
    if umbrales is None:
        umbrales = [0.5 + i * 0.05 for i in range(10)]
    cache = []
    for p, o in zip(predicciones, objetivos):
        cache.append({"iou": iou_pares(p["boxes"].float().cpu(), o["boxes"].float().cpu()).tolist(),
            "ignorar": cubiertas_por_ignorar(p["boxes"].float().cpu(), o["ignore_boxes"].float().cpu()).tolist(),
            "etiquetas": o["labels"].tolist()})
    por_clase = {}
    for clase, nombre in enumerate(CLASES):
        total = sum(int((o["labels"] == clase).sum()) for o in objetivos)
        candidatos = [(float(p["scores"][j]), i, j) for i, p in enumerate(predicciones) for j in range(len(p["boxes"])) if int(p["labels"][j]) == clase]
        candidatos.sort(key=lambda r: -r[0])
        resultados = []
        for umbral in umbrales:
            usados = [set() for _ in objetivos]
            tp, fp, omitidas = [], [], 0
            for score, i, j in candidatos:
                coincidencias = cache[i]["iou"][j]
                validos = [k for k, etiqueta in enumerate(cache[i]["etiquetas"]) if etiqueta == clase]
                libres = [k for k in validos if k not in usados[i] and coincidencias[k] >= umbral]
                if libres:
                    k = max(libres, key=lambda k: coincidencias[k])
                    usados[i].add(k)
                    tp.append(1)
                    fp.append(0)
                else:
                    solapa_valido = any(v >= umbral for v in coincidencias)
                    ignorar = cache[i]["ignorar"][j]
                    # Una detección duplicada o con material equivocado sobre GT válida cuenta como error.
                    if ignorar and not solapa_valido:
                        omitidas += 1
                        continue
                    tp.append(0)
                    fp.append(1)
            tp_cum = torch.tensor(tp, dtype=torch.float32).cumsum(0)
            fp_cum = torch.tensor(fp, dtype=torch.float32).cumsum(0)
            if total:
                recalls = tp_cum / total
                precisiones = tp_cum / (tp_cum + fp_cum).clamp(min=1)
                ap = sum(float(precisiones[recalls >= r].max()) if bool((recalls >= r).any()) else 0 for r in torch.linspace(0, 1, 101)) / 101
            else:
                ap = None
            resultados.append({"iou": round(umbral, 2), "ap": ap, "tp": sum(tp), "fp": sum(fp), "predicciones_ignoradas": omitidas,
                "recall": sum(tp) / total if total else None,
                "precision": sum(tp) / (sum(tp) + sum(fp)) if tp else 0.0})
        por_clase[nombre] = {"objetos": total, "ap50": resultados[0]["ap"],
            "ap50_95": sum(r["ap"] for r in resultados) / len(resultados) if total else None,
            "por_umbral": resultados}
    presentes = [r for r in por_clase.values() if r["objetos"]]
    tp = sum(r["por_umbral"][0]["tp"] for r in por_clase.values())
    fp = sum(r["por_umbral"][0]["fp"] for r in por_clase.values())
    total = sum(r["objetos"] for r in por_clase.values())
    return {"map50": sum(r["ap50"] for r in presentes) / len(presentes) if presentes else None,
        "map50_95": sum(r["ap50_95"] for r in presentes) / len(presentes) if presentes else None,
        "recall50": tp / total if total else None, "precision50": tp / max(tp + fp, 1),
        "objetos": total, "clases_con_soporte": len(presentes), "por_clase": por_clase,
        "evaluador": "propio_101_puntos_con_ignore_regions_no_COCOeval_oficial"}
