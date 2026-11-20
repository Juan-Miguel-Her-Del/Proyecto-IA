"""Sondeo de dependencia visual: escenas val frente a entrada gris uniforme."""
import argparse
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.datasets import ResiduosDataset, objetivo_deteccion
from ecowaste.detector import cargar_detector, decodificar
from ecowaste.inferencia_detector import postprocesar
from ecowaste.ejecucion_detector import CacheDeteccion


def correlacion(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.square().sum() * b.square().sum()).sqrt().clamp(min=1e-9))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nombres", nargs="+", default=["detector_piloto_v1"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    cache = CacheDeteccion(RAIZ, "val", .1)
    indices = torch.linspace(0, len(cache) - 1, 6).long().tolist()
    entradas = cache.imagenes[indices].to(dispositivo).float() / 255
    gris = torch.full_like(entradas[:1], 114/255)
    reporte = {"ids_sondeo_val": [cache.objetivos[i]["id"] for i in indices], "test_evaluado": False,
        "limite": "Sondeo de seis imágenes y una entrada sintética uniforme; no es una evaluación de escenas vacías reales ni una demostración causal de todos los fallos.", "experimentos": {}}
    for nombre in args.nombres:
        if Path(nombre).name != nombre:
            raise ValueError("Identificador no válido")
        modelo, _ = cargar_detector(RAIZ / f"checkpoints/{nombre}_mejor.pt", dispositivo)
        raw = modelo(torch.cat([entradas, gris]))
        _, obj, clases = decodificar(raw, modelo.configuracion.get("radio_centros", 0))
        celdas = (torch.arange(48, device=dispositivo) + .5) / 48
        yy, xx = torch.meshgrid(celdas, celdas, indexing="ij")
        mediciones = []
        for j, indice in enumerate(indices):
            g = cache.objetivos[indice]["geometria"]
            x0, y0 = g["relleno_x"] / 384, g["relleno_y"] / 384
            x1 = x0 + g["ancho_original"] * g["escala_x"] / 384
            y1 = y0 + g["alto_original"] * g["escala_y"] / 384
            mascara = ((xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1))[:, :, None].expand(48, 48, 2)
            mediciones.append({"id": cache.objetivos[indice]["id"],
                "correlacion_presencia_vs_gris": correlacion(obj[j][mascara], obj[-1][mascara]),
                "cambio_medio_probabilidad_presencia": float((obj[j].sigmoid()[mascara] - obj[-1].sigmoid()[mascara]).abs().mean()),
                "material_coincide_con_gris": float((clases[j].argmax(-1)[mascara] == clases[-1].argmax(-1)[mascara]).float().mean())})
        pred_gris = postprocesar(raw[-1:], umbral=.25, radio_centros=modelo.configuracion.get("radio_centros", 0))[0]
        reporte["experimentos"][nombre] = {"mediciones": mediciones,
            "correlacion_media_presencia_vs_gris": sum(m["correlacion_presencia_vs_gris"] for m in mediciones) / len(mediciones),
            "cambio_medio_probabilidad_presencia": sum(m["cambio_medio_probabilidad_presencia"] for m in mediciones) / len(mediciones),
            "material_coincide_con_gris_media": sum(m["material_coincide_con_gris"] for m in mediciones) / len(mediciones),
            "cajas_sobre_gris_con_puntuacion_025": len(pred_gris["boxes"])}
    (RAIZ / "resultados/sensibilidad_detector.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({n: {k:v for k,v in d.items() if k!='mediciones'} for n,d in reporte["experimentos"].items()}, indent=2))


if __name__ == "__main__":
    main()
