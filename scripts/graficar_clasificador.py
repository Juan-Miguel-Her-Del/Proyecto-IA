"""Figuras exportables del piloto; requiere Matplotlib, sin cargar modelos ni imágenes."""
import json
import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--nombre', default='clasificador_piloto_v1')
args = parser.parse_args()
if Path(args.nombre).name != args.nombre: raise ValueError('Nombre inválido')
fuente = RAIZ / f'resultados/{args.nombre}.json'
reporte = json.loads(fuente.read_text(encoding="utf-8"))
historial = reporte["historial"]
epocas = [r["epoca"] for r in historial]
fig, ejes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
fig.suptitle(f"EcoWasteFormer-Cls · {args.nombre} · test reservado", fontsize=15)
ejes[0].plot(epocas, [r["loss_train_ponderada_con_aumentos"] for r in historial], color="#157f72", marker="o", markersize=3)
ejes[0].set(title="Objetivo de entrenamiento", xlabel="Época", ylabel="Entropía cruzada ponderada + smoothing")
ejes[0].grid(alpha=0.2)
ejes[1].plot(epocas, [r["val"]["accuracy"] for r in historial], label="Aciertos en validación", color="#2469a0")
ejes[1].plot(epocas, [r["val"]["f1_macro"] for r in historial], label="F1 macro en validación", color="#157f72")
ejes[1].axhline(reporte["baseline_mayoritaria_train_sobre_val"]["accuracy"], linestyle="--", color="#a14a22", label="Clase mayoritaria: aciertos")
ejes[1].axvline(reporte["mejor_epoca"], linestyle=":", color="gray", label="Checkpoint seleccionado")
ejes[1].set(title="Validación: 610 imágenes", xlabel="Época", ylim=(0, 1))
ejes[1].yaxis.set_major_formatter(PercentFormatter(1))
ejes[1].grid(alpha=0.2)
ejes[1].legend(fontsize=8, loc="lower right")
matriz = np.array(reporte["val"]["matriz_confusion"])
nombres = ["Plástico", "Papel", "Cartón", "Vidrio", "Metal", "Orgánico"]
ejes[2].imshow(matriz, cmap="Blues", vmin=0, vmax=matriz.max())
ejes[2].set(xticks=range(6), yticks=range(6), xticklabels=nombres, yticklabels=nombres,
    title=f"Matriz de confusión · época {reporte['mejor_epoca']}", xlabel="Predicción", ylabel="Clase real")
plt.setp(ejes[2].get_xticklabels(), rotation=40, ha="right")
for i in range(6):
    for j in range(6):
        ejes[2].text(j, i, str(matriz[i, j]), ha="center", va="center", color="white" if matriz[i, j] > matriz.max()/2 else "black", fontsize=9)
destino = RAIZ / "resultados/figuras"
destino.mkdir(parents=True, exist_ok=True)
for extension in ["png", "svg"]:
    fig.savefig(destino / f"{args.nombre}.{extension}", dpi=170)
plt.close(fig)
(destino / f"{args.nombre}_procedencia.json").write_text(json.dumps({"matplotlib": matplotlib.__version__,
    "fuente": fuente.relative_to(RAIZ).as_posix(), 'sha256_fuente': hashlib.sha256(fuente.read_bytes()).hexdigest(),
    "test_evaluado": False}, indent=2), encoding="utf-8")
print("Figuras PNG y SVG guardadas en resultados/figuras/")
