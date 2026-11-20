"""Figuras del detector a partir de resultados reales; requiere Matplotlib opcional."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--nombre", default="detector_piloto_v1")
args = parser.parse_args()
if Path(args.nombre).name != args.nombre:
    parser.error("El nombre debe ser un identificador de experimento")
nombre = args.nombre
d = json.loads((RAIZ / f"resultados/{nombre}.json").read_text(encoding="utf-8"))
salida = RAIZ / "resultados/figuras"
salida.mkdir(parents=True, exist_ok=True)
hist = d["historial"]
epocas = [r["epoca"] for r in hist]
fig, ejes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
fig.suptitle("EcoWasteFormer-Det · piloto desde cero · test reservado")
ejes[0].plot(epocas, [r["loss_train"] for r in hist], marker="o", color="#157f72")
ejes[0].set(title="Pérdida de entrenamiento", xlabel="Época", ylabel="Objetivo combinado")
ejes[1].plot(epocas, [r["map50_val"] for r in hist], label="mAP@50", color="#2469a0", marker="o")
ejes[1].plot(epocas, [r["map50_95_val"] for r in hist], label="mAP@50:95", color="#157f72", marker="o")
ejes[1].axvline(d["mejor_epoca"], color="gray", linestyle=":")
limite_ap=min(1,max(.02,max(r['map50_val'] for r in hist)*1.25))
ejes[1].set(title="Validación · AP de evaluador propio", xlabel="Época", ylim=(0, limite_ap))
ejes[1].yaxis.set_major_formatter(PercentFormatter(1))
ejes[1].legend()
clases = list(d["val"]["por_clase"])
valores = [d["val"]["por_clase"][c]["ap50"] or 0 for c in clases]
ejes[2].bar(clases, valores, color="#2469a0")
limite_clases=min(1,max(.03,max(valores)*1.3))
ejes[2].set(title="AP@50 por material", ylim=(0, limite_clases), ylabel="AP@50")
ejes[2].yaxis.set_major_formatter(PercentFormatter(1))
ejes[2].tick_params(axis="x", labelrotation=35)
for i, c in enumerate(clases):
    ejes[2].text(i, valores[i] + limite_clases*.03, f"n={d['val']['por_clase'][c]['objetos']}", ha="center", fontsize=8)
for ax in ejes[:2]:
    ax.grid(alpha=.2)
    if d.get('origen_continuacion'):
        ax.axvline(d['origen_continuacion']['epoca_inicio']-.5,color='#b36b00',linestyle='--',alpha=.7)
for extension in ["png", "svg"]:
    fig.savefig(salida / f"{nombre}.{extension}", dpi=160)
plt.close(fig)

if d.get('clasificacion_auxiliar'):
    fig,ejes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    coef=d['clasificacion_auxiliar']['coeficiente_perdida']
    ejes[0].plot(epocas,[r['loss_train'] for r in hist],label='Detección')
    ejes[0].plot(epocas,[coef*r['loss_aux_train'] for r in hist],label=f'{coef:g} × clasificación')
    ejes[0].plot(epocas,[r['loss_total_train'] for r in hist],label='Total',linestyle='--')
    ejes[0].set(title='Pérdidas de entrenamiento',xlabel='Época');ejes[0].legend()
    ejes[1].plot(epocas,[r['accuracy_aux_train'] for r in hist],marker='o')
    ejes[1].set(title='Clasificación auxiliar en train aumentado',xlabel='Época',ylim=(0,1))
    ejes[1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle('Supervisión de imagen completa · no es mAP ni evaluación independiente')
    for ax in ejes:ax.grid(alpha=.2)
    for ext in ['png','svg']:fig.savefig(salida/f'{nombre}_auxiliar.{ext}',dpi=160)
    plt.close(fig)

filas = [json.loads(l) for l in (RAIZ / "datos/procesados/deteccion/val.jsonl").read_text(encoding="utf-8").splitlines()]
pred = {r["id"]: r for r in json.loads((RAIZ / f"resultados/{nombre}_predicciones_val.json").read_text(encoding="utf-8"))}
umbral_visualizacion = d["configuracion"].get("umbral_visualizacion", .25)
grupos, elegidas = set(), []
for r in filas:
    if r["grupo"] not in grupos:
        grupos.add(r["grupo"])
        elegidas.append(r)
    if len(elegidas) == 3:
        break
fig, ejes = plt.subplots(len(elegidas), 2, figsize=(10, 4 * len(elegidas)), layout="constrained", squeeze=False)
for i, r in enumerate(elegidas):
    with Image.open(RAIZ / r["ruta"]) as origen:
        vista = origen.convert("RGB")
        vista.thumbnail((700, 700))
    sx_v, sy_v = vista.width / r["ancho"], vista.height / r["alto"]
    for ax in ejes[i]:
        ax.imshow(vista)
        ax.axis("off")
    ejes[i, 0].set_title(f"{r['id']} · anotaciones de la fuente")
    ejes[i, 1].set_title("Predicciones · puntuación ≥ 0,25")
    for a in r["cajas"]:
        x, y, w, h = a["bbox"]
        color = "#087f5b" if a["clase"] else "gray"
        ejes[i, 0].add_patch(Rectangle((x * sx_v, y * sy_v), w * sx_v, h * sy_v, fill=False, edgecolor=color, linewidth=2))
        ejes[i, 0].text(x * sx_v, y * sy_v, a["clase"] or "ignorar", color=color, fontsize=8, backgroundcolor="white")
    escala = 384 / max(r["ancho"], r["alto"])
    nw, nh = round(r["ancho"] * escala), round(r["alto"] * escala)
    px, py = (384 - nw) // 2, (384 - nh) // 2
    mostradas = 0
    for box, score, clase in zip(pred[r["id"]]["boxes"], pred[r["id"]]["scores"], pred[r["id"]]["labels"]):
        if score < umbral_visualizacion:
            continue
        x0, x1 = [(box[k] * 384 - px) / (nw / r["ancho"]) * sx_v for k in [0, 2]]
        y0, y1 = [(box[k] * 384 - py) / (nh / r["alto"]) * sy_v for k in [1, 3]]
        ejes[i, 1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#c92a2a", linewidth=1.2, alpha=.65))
        if mostradas < 5:
            ejes[i, 1].text(x0, y0, f"{clases[clase]} {score:.2f}", color="#c92a2a", fontsize=7, backgroundcolor="white")
        mostradas += 1
    ejes[i, 1].set_title(f"Predicciones ≥{umbral_visualizacion:.2f}: {mostradas} cajas; 5 etiquetas como máximo", fontsize=10)
fig.suptitle("Primeras escenas de grupos distintos de validación; sin selección por aciertos")
fig.savefig(salida / f"{nombre}_comparacion_val.png", dpi=150)
plt.close(fig)
(salida / f"{nombre}_graficas_procedencia.json").write_text(json.dumps({"matplotlib": matplotlib.__version__,
    "fuente": f"resultados/{nombre}.json", "ids_comparacion": [r["id"] for r in elegidas],
    "umbral_visualizacion": umbral_visualizacion, "test_evaluado": False}, indent=2), encoding="utf-8")
print("Figuras y comparación con anotaciones guardadas en resultados/figuras/")
