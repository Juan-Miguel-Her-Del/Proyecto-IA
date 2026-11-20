"""Memorización diagnóstica de seis escenas train; sin reutilizar pesos en el piloto."""
import argparse
import json
from pathlib import Path
import sys
import time

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.datasets import CLASES, ResiduosDataset, objetivo_deteccion
from ecowaste.detector import DetectorResiduos
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.inferencia_detector import postprocesar
from ecowaste.perdida_detector import PerdidaDetector, asignar, priores_train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio-centros", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--nombre", default="diagnostico_detector")
    parser.add_argument("--ignorar-interiores-gt", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--normalizacion-clases", choices=["por_imagen", "global"], default="por_imagen")
    parser.add_argument("--escala-posicion", type=float, default=1.0)
    args = parser.parse_args()
    if Path(args.nombre).name != args.nombre:
        raise ValueError("Nombre de informe no válido")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("La prueba requiere CUDA")
    ds = ResiduosDataset(RAIZ, "deteccion", "train")
    todos = [objetivo_deteccion(r) for r in ds.registros]
    priors, corte = priores_train(todos)
    indices = []
    for clase in range(6):
        opciones = []
        for i, o in enumerate(todos):
            if i in indices or len(o["boxes"]) > 4 or asignar(o, umbral_escala=corte)["colisiones"]:
                continue
            seleccion = o["labels"] == clase
            if bool(seleccion.any()):
                tam = o["boxes"][seleccion, 2:] - o["boxes"][seleccion, :2]
                lado_min = float(tam.min(-1).values.max()) * 384
                if 40 <= lado_min <= 270:
                    opciones.append((lado_min, i))
        if not opciones:
            raise RuntimeError(f"No hay escena adecuada para el diagnóstico de {CLASES[clase]}")
        # Escenas fáciles elegidas explícitamente para diagnosticar ajuste, no evaluación.
        indices.append(sorted(opciones, reverse=True)[0][1])
    lote = [ds[i] for i in indices]
    imagenes = torch.stack([x for x, _ in lote]).to("cuda")
    objetivos = [o for _, o in lote]
    for o in objetivos:
        o["asignacion"] = asignar(o, umbral_escala=corte, radio_centros=args.radio_centros)
    modelo = DetectorResiduos(radio_centros=args.radio_centros, escala_posicion=args.escala_posicion).to("cuda")
    modelo.configurar_priores(priors, corte)
    ignorar_interiores = args.radio_centros > 0 if args.ignorar_interiores_gt is None else args.ignorar_interiores_gt
    criterio = PerdidaDetector(corte, ignorar_interiores_gt=ignorar_interiores, radio_centros=args.radio_centros, normalizacion_clases=args.normalizacion_clases).to("cuda")
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=0.001, weight_decay=0.0)
    historial, aprobada = [], False
    torch.cuda.reset_peak_memory_stats()
    inicio = time.perf_counter()
    for paso in range(1, 401):
        modelo.train()
        optimizador.zero_grad(set_to_none=True)
        raw = modelo(imagenes)
        loss, detalle = criterio(raw, objetivos)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Pérdida no finita")
        loss.backward()
        if paso == 1:
            for nombre, p in modelo.named_parameters():
                if p.grad is None or not bool(torch.isfinite(p.grad).all()) or not bool((p.grad != 0).any()):
                    raise RuntimeError(f"Gradiente inválido en {nombre}")
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0, error_if_nonfinite=True)
        optimizador.step()
        if paso % 25 == 0:
            modelo.eval()
            with torch.no_grad():
                pred = postprocesar(modelo(imagenes), umbral=0.05, radio_centros=args.radio_centros)
            metrica = evaluar_detecciones(pred, objetivos)
            fila = {"paso": paso, "loss": float(loss.detach()), "detalle": detalle,
                "map50": metrica["map50"], "recall50": metrica["recall50"], "precision50": metrica["precision50"]}
            historial.append(fila)
            print(f"Paso {paso}: loss={fila['loss']:.3f}, mAP50={fila['map50']:.3f}, recall={fila['recall50']:.3f}, precision={fila['precision50']:.3f}", flush=True)
            if metrica["map50"] >= 0.8 and metrica["recall50"] >= 0.9 and metrica["precision50"] >= 0.7:
                aprobada = True
                break
    torch.cuda.synchronize()
    reporte = {"aprobada": aprobada, "radio_centros": args.radio_centros, "ignorar_interiores_gt": ignorar_interiores,
        "normalizacion_clases": args.normalizacion_clases,
        "escala_posicion": args.escala_posicion,
        "parametros": sum(p.numel() for p in modelo.parameters()),
        "semilla": 42, "precision": "FP32", "ids_solo_train": [o["id"] for o in objetivos],
        "objetos": sum(len(o["boxes"]) for o in objetivos), "priores_wh_solo_train": priors, "umbral_escala_solo_train": corte,
        "historial": historial, "metricas_sobre_escenas_memorizadas": metrica,
        "duracion_segundos": time.perf_counter() - inicio, "vram_pico_asignada_bytes": torch.cuda.max_memory_allocated(),
        "test_evaluado": False,
        "limite": "Escenas fáciles seleccionadas de entrenamiento. No miden generalización y sus pesos no se reutilizan en el piloto."}
    (RAIZ / f"resultados/{args.nombre}.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Diagnóstico {'aprobado' if aprobada else 'no aprobado'}", flush=True)
    return 0 if aprobada else 1


if __name__ == "__main__":
    raise SystemExit(main())
