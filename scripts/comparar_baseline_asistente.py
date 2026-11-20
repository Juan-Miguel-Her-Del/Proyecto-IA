"""Compara contra Naive Bayes de palabras sin cambiar el Transformer ni sus datos."""
from pathlib import Path
import json
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import numpy as np
from ecowaste.asistente import INTENCIONES, palabras
from scripts.entrenar_asistente import cargar, metricas


def main():
    train = cargar(RAIZ / "datos/asistente/train.jsonl")
    vocab = {w: i for i, w in enumerate(sorted({w for r in train for w in palabras(r["pregunta"])}))}
    cuentas = np.ones((len(INTENCIONES), len(vocab)), dtype=np.float64)
    for r in train:
        for w in palabras(r["pregunta"]):
            cuentas[INTENCIONES.index(r["intencion"]), vocab[w]] += 1
    logprobs = np.log(cuentas / cuentas.sum(1, keepdims=True))
    informe = {"modelo": "multinomial_naive_bayes", "smoothing": 1.0, "vocabulario_solo_train": len(vocab), "pesos_preentrenados": False}
    for split in ["val", "test"]:
        registros = cargar(RAIZ / "datos/asistente" / f"{split}.jsonl")
        predichos = []
        for r in registros:
            ids = [vocab[w] for w in palabras(r["pregunta"]) if w in vocab]
            puntajes = logprobs[:, ids].sum(1)
            predichos.append(int(puntajes.argmax()))
        informe[split] = metricas(predichos, [INTENCIONES.index(r["intencion"]) for r in registros])
    informe["nota"] = "Misma prueba inicial que el Transformer. La comparación no usa pesos externos ni modifica el corpus o el checkpoint; no representa conversaciones con usuarios independientes."
    destino = RAIZ / "resultados/baseline_asistente.json"
    destino.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({s: {k: informe[s][k] for k in ["accuracy", "f1_macro"]} for s in ["val", "test"]}, indent=2))


if __name__ == "__main__":
    main()
