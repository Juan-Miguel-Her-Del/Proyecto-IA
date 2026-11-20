"""Entrenamiento breve desde cero; selección/umbral usan val, test se evalúa al final."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ecowaste.asistente import INTENCIONES, Tokenizador, TransformerIntenciones, palabras


def cargar(path):
    return [json.loads(linea) for linea in path.read_text(encoding="utf-8").splitlines() if linea.strip()]


def metricas(predichos, reales):
    matriz = [[0] * len(INTENCIONES) for _ in INTENCIONES]
    for p, r in zip(predichos, reales):
        matriz[r][p] += 1
    por_clase = {}
    for i, clase in enumerate(INTENCIONES):
        tp = matriz[i][i]
        precision = tp / max(sum(f[i] for f in matriz), 1)
        recall = tp / max(sum(matriz[i]), 1)
        por_clase[clase] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}
    return {"accuracy": sum(p == r for p, r in zip(predichos, reales)) / max(len(reales), 1),
        "f1_macro": sum(v["f1"] for v in por_clase.values()) / len(INTENCIONES),
        "por_clase": por_clase, "matriz_confusion": matriz, "orden_clases": INTENCIONES}


@torch.inference_mode()
def evaluar(modelo, entradas, etiquetas):
    modelo.eval()
    probs = modelo(entradas).softmax(-1)
    return metricas(probs.argmax(-1).tolist(), etiquetas.tolist()), probs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epocas", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(4)
    random.seed(42)
    torch.manual_seed(42)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    carpeta = RAIZ / "datos" / "asistente"
    corpus = {s: cargar(carpeta / f"{s}.jsonl") for s in ["train", "val", "test"]}
    for s, registros in corpus.items():
        if set(Counter(r["intencion"] for r in registros)) != set(INTENCIONES):
            raise ValueError(f"Faltan intenciones en {s}")
        for otro in corpus:
            if otro == s:
                continue
            grupos = {r["grupo"] for r in registros} & {r["grupo"] for r in corpus[otro]}
            textos = {tuple(palabras(r["pregunta"])) for r in registros} & {tuple(palabras(r["pregunta"])) for r in corpus[otro]}
            if grupos or textos:
                raise ValueError(f"Fuga entre {s} y {otro}")
    tokenizador = Tokenizador.ajustar(corpus["train"])
    datos = {}
    for s, registros in corpus.items():
        x = torch.tensor([tokenizador.codificar(r["pregunta"], r["contexto_texto"]) for r in registros], device=dispositivo)
        y = torch.tensor([INTENCIONES.index(r["intencion"]) for r in registros], device=dispositivo)
        datos[s] = x, y
    modelo = TransformerIntenciones(len(tokenizador.vocabulario)).to(dispositivo)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=0.001, weight_decay=0.01)
    loader = DataLoader(TensorDataset(*datos["train"]), batch_size=32, shuffle=True)
    perdida_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    mejor_f1, mejor_peso, historial, paciencia = -1, None, [], 0
    inicio = time.perf_counter()
    for epoca in range(1, args.epocas + 1):
        modelo.train()
        acumulada = 0.0
        for x, y in loader:
            optimizador.zero_grad(set_to_none=True)
            perdida = perdida_fn(modelo(x), y)
            if not bool(torch.isfinite(perdida)):
                raise RuntimeError("Pérdida no finita")
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0, error_if_nonfinite=True)
            optimizador.step()
            acumulada += perdida.item() * len(x)
        resultado, _ = evaluar(modelo, *datos["val"])
        historial.append({"epoca": epoca, "loss_train": acumulada / len(datos["train"][0]), "f1_val": resultado["f1_macro"]})
        if resultado["f1_macro"] > mejor_f1 + 1e-6:
            mejor_f1 = resultado["f1_macro"]
            mejor_peso = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
            mejor_epoca, paciencia = epoca, 0
        else:
            paciencia += 1
        if epoca % 10 == 0:
            print(f"Época {epoca}: loss={historial[-1]['loss_train']:.4f}, F1 val={resultado['f1_macro']:.3f}", flush=True)
        if paciencia >= 25:
            break
    modelo.load_state_dict(mejor_peso)
    resultado_val, probs_val = evaluar(modelo, *datos["val"])
    confianza_val, indices_val = probs_val.max(-1)
    # Umbral elegido únicamente con val. Penaliza respuestas equivocadas aceptadas.
    candidatos = []
    for umbral in [0.0, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        aceptadas = confianza_val >= umbral
        aciertos = ((indices_val == datos["val"][1]) & aceptadas).sum().item()
        errores = ((indices_val != datos["val"][1]) & aceptadas).sum().item()
        utilidad = aciertos - 2 * errores - 0.1 * (~aceptadas).sum().item()
        candidatos.append((utilidad, umbral))
    umbral = max(candidatos)[1]
    resultado_test, probs_test = evaluar(modelo, *datos["test"])
    confianza, indices = probs_test.max(-1)
    conocidas = []
    for registro in corpus["test"]:
        toks = palabras(registro["pregunta"])
        conocidas.append(sum(t not in tokenizador.vocabulario for t in toks) / max(len(toks), 1) <= 0.5)
    aceptadas = (confianza >= umbral) & torch.tensor(conocidas, device=dispositivo)
    cantidad_aceptada = int(aceptadas.sum().item())
    precision_aceptada = ((indices == datos["test"][1]) & aceptadas).sum().item() / cantidad_aceptada if cantidad_aceptada else None
    resultado_test["cobertura_con_abstencion"] = cantidad_aceptada / len(aceptadas)
    resultado_test["accuracy_aceptadas"] = precision_aceptada
    checkpoint = RAIZ / "checkpoints" / "asistente_v1.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tipo": "ecowaste_intenciones_desde_cero_v1", "configuracion": modelo.configuracion,
        "vocabulario": tokenizador.vocabulario, "pesos": mejor_peso, "umbral": umbral,
        "semilla": 42, "epoca": mejor_epoca, "pesos_externos": False}, checkpoint)
    ejemplos = [{"pregunta": r["pregunta"], "real": r["intencion"], "predicha": INTENCIONES[p],
                 "aceptada": a} for r, p, a in zip(corpus["test"], indices.tolist(), aceptadas.tolist())]
    reporte = {"estado": "prototipo_entrenado_evaluado_en_corpus_inicial", "semilla": 42,
        "dispositivo": dispositivo, "torch": torch.__version__, "pesos_externos": False,
        "parametros": sum(p.numel() for p in modelo.parameters()), "vocabulario": len(tokenizador.vocabulario),
        "epocas_ejecutadas": len(historial), "mejor_epoca_por_val": mejor_epoca, "umbral_por_val": umbral,
        "duracion_segundos": time.perf_counter() - inicio,
        "sha256_corpus": {s: hashlib.sha256((carpeta / f"{s}.jsonl").read_bytes()).hexdigest() for s in corpus},
        "conteos": {s: len(r) for s, r in corpus.items()}, "validacion": resultado_val,
        "test": resultado_test, "ejemplos_test": ejemplos, "historial": historial,
        "limitacion": "Corpus inicial pequeño con asistencia IA, no validado con usuarios independientes. Estas métricas no miden detección visual ni garantizan generalización lingüística."}
    salida = RAIZ / "resultados" / "asistente_v1.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: reporte[k] for k in ["parametros", "mejor_epoca_por_val", "umbral_por_val", "test", "duracion_segundos"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
