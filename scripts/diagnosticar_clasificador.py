"""Prueba de aprendizaje sobre 18 imágenes train; no acredita generalización."""
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.clasificador import ClasificadorResiduos
from ecowaste.datasets import CLASES, ResiduosDataset


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng = random.Random(42)
    if not torch.cuda.is_available():
        raise RuntimeError("La prueba prevista requiere la GPU local")
    dataset = ResiduosDataset(RAIZ, "clasificacion", "train")
    indices = []
    for clase in CLASES:
        indices.extend(rng.sample([i for i, r in enumerate(dataset.registros) if r["clase"] == clase], 3))
    lote = [dataset[i] for i in indices]
    imagenes = torch.stack([x for x, _ in lote]).to("cuda")
    etiquetas = torch.tensor([y for _, y in lote], device="cuda")
    modelo = ClasificadorResiduos().to("cuda")
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=0.001, weight_decay=0.0)
    torch.cuda.reset_peak_memory_stats()
    historial, aprobada, consecutivas = [], False, 0
    inicio = time.perf_counter()
    with torch.no_grad():
        inicial = torch.nn.functional.cross_entropy(modelo(imagenes), etiquetas).item()
    for paso in range(1, 301):
        modelo.train()
        optimizador.zero_grad(set_to_none=True)
        perdida = torch.nn.functional.cross_entropy(modelo(imagenes), etiquetas)
        if not bool(torch.isfinite(perdida)):
            raise RuntimeError("Pérdida no finita")
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0, error_if_nonfinite=True)
        optimizador.step()
        if paso % 10 == 0:
            modelo.eval()
            with torch.no_grad():
                logits = modelo(imagenes)
                loss = torch.nn.functional.cross_entropy(logits, etiquetas).item()
                accuracy = (logits.argmax(-1) == etiquetas).float().mean().item()
            historial.append({"paso": paso, "loss": loss, "accuracy": accuracy})
            print(f"Paso {paso}: pérdida={loss:.4f}, aciertos={accuracy:.1%}", flush=True)
            consecutivas = consecutivas + 1 if accuracy == 1.0 and loss < 0.15 else 0
            if consecutivas >= 3:
                aprobada = True
                break
    torch.cuda.synchronize()
    reporte = {"fecha_utc": datetime.now(timezone.utc).isoformat(), "aprobada": aprobada,
        "semilla": 42, "gpu": torch.cuda.get_device_name(0), "parametros": sum(p.numel() for p in modelo.parameters()),
        "configuracion": modelo.configuracion, "imagenes": len(indices), "por_clase": 3,
        "ids_solo_train": [dataset.registros[i]["id"] for i in indices], "loss_inicial": inicial,
        "historial": historial, "duracion_segundos": time.perf_counter() - inicio,
        "vram_pico_asignada_bytes": torch.cuda.max_memory_allocated(),
        "precision_prueba": "FP32", "test_evaluado": False,
        "limite": "Memorizar estas imágenes prueba capacidad de ajuste, no calidad en imágenes nuevas. Los pesos de esta prueba no se reutilizan en el piloto."}
    salida = RAIZ / "resultados/diagnostico_clasificador.json"
    salida.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Diagnóstico {'aprobado' if aprobada else 'no aprobado'}; informe: {salida}")
    return 0 if aprobada else 1


if __name__ == "__main__":
    raise SystemExit(main())
