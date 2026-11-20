"""Piloto del detector desde cero, con selección por validación y test reservado."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from torch.utils.data import DataLoader
from ecowaste.datasets import CLASES, ResiduosDataset, collate_deteccion, objetivo_deteccion
from ecowaste.detector import DetectorResiduos
from ecowaste.ejecucion_detector import CacheDeteccion, aumentar_lote, evaluar_detector
from ecowaste.perdida_detector import PerdidaDetector, priores_train
from scripts.entrenar_clasificador import guardar_checkpoint, escribir_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=RAIZ / "configuracion/entrenamiento_detector.json")
    continuacion = parser.add_mutually_exclusive_group()
    continuacion.add_argument("--reanudar", type=Path)
    continuacion.add_argument("--continuar-desde", type=Path, help="Amplía solo el presupuesto en un experimento nuevo, conservando pesos propios, optimizador y RNG")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", cfg["nombre"]) or cfg["evaluar_test"]:
        raise ValueError("Nombre no válido o intento de evaluar test en el piloto")
    salida = RAIZ / "resultados" / (cfg["nombre"] + ".json")
    mejor_path = RAIZ / "checkpoints" / (cfg["nombre"] + "_mejor.pt")
    ultimo_path = RAIZ / "checkpoints" / (cfg["nombre"] + "_ultimo.pt")
    if not args.reanudar and any(p.exists() for p in [salida, mejor_path, ultimo_path]):
        raise FileExistsError("Experimento existente; reanudar o elegir un nombre nuevo")
    validacion = json.loads((RAIZ / "datos/reportes/validacion.json").read_text(encoding="utf-8"))
    diagnostico = json.loads((RAIZ / "resultados" / cfg.get("diagnostico", "diagnostico_detector.json")).read_text(encoding="utf-8"))
    if not validacion["validacion_estructural_correcta"] or not diagnostico["aprobada"]:
        raise ValueError("Se requieren datos validados y diagnóstico de ajuste aprobado")
    if diagnostico.get("radio_centros", 0) != cfg.get("radio_centros", 0):
        raise ValueError("El diagnóstico debe usar la misma supervisión espacial que el entrenamiento")
    if "ignorar_interiores_gt" in diagnostico and diagnostico["ignorar_interiores_gt"] != cfg.get("ignorar_interiores_gt", False):
        raise ValueError("El diagnóstico debe usar la misma política de negativos interiores")
    if diagnostico.get("normalizacion_clases", "por_imagen") != cfg.get("normalizacion_clases", "por_imagen"):
        raise ValueError("El diagnóstico debe usar la misma reducción de clases")
    if diagnostico.get("escala_posicion", 1.0) != cfg.get("escala_posicion", 1.0):
        raise ValueError("El diagnóstico debe usar la misma escala de posiciones")
    hashes = {}
    for split in ["train", "val"]:
        relativo = f"datos/procesados/deteccion/{split}.jsonl"
        hashes[split] = hashlib.sha256((RAIZ / relativo).read_bytes()).hexdigest()
        if hashes[split] != validacion["sha256_manifiestos"][relativo]:
            raise ValueError("Los manifiestos cambiaron desde su validación")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["semilla"])
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dispositivo == "cuda" and cfg["precision_mixta_cuda"]
    origen = ResiduosDataset(RAIZ, "deteccion", "train")
    metadatos_train = [objetivo_deteccion(r) for r in origen.registros]
    priors, corte = priores_train(metadatos_train)
    radio = cfg.get("radio_centros", 0)
    train = CacheDeteccion(RAIZ, "train", corte, radio)
    val = CacheDeteccion(RAIZ, "val", corte, radio)
    generador = torch.Generator().manual_seed(cfg["semilla"])
    comunes = dict(batch_size=cfg["batch_size"], collate_fn=collate_deteccion, pin_memory=dispositivo == "cuda")
    loader_train = DataLoader(train, shuffle=True, generator=generador, **comunes)
    loader_val = DataLoader(val, **comunes)
    modelo = DetectorResiduos(radio_centros=radio, escala_posicion=cfg.get("escala_posicion", 1.0)).to(dispositivo)
    modelo.configurar_priores(priors, corte)
    conteos = torch.bincount(torch.cat([o["labels"] for o in train.objetivos]), minlength=6).float()
    pesos = conteos.clamp(min=1).rsqrt()
    pesos = (pesos / pesos.mean()).clamp(max=3.0)
    criterio = PerdidaDetector(corte, pesos.to(dispositivo), cfg.get("ignorar_interiores_gt", False), radio, cfg.get("normalizacion_clases", "por_imagen")).to(dispositivo)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024)
    historial, mejor_map, mejor_epoca, paciencia, inicio_epoca, duracion_anterior = [], -1.0, 0, 0, 1, 0.0
    origen_continuacion = None
    fuente_estado = args.reanudar or args.continuar_desde
    if fuente_estado:
        previo = torch.load(fuente_estado, map_location=dispositivo, weights_only=True)
        if previo["hashes_manifiestos"] != hashes:
            raise ValueError("Los datos deben coincidir con la ejecución original")
        if args.continuar_desde:
            permitidas = {"nombre", "epocas", "paciencia", "min_epocas"}
            config_previa = {k:v for k,v in previo["config_entrenamiento"].items() if k not in permitidas}
            config_nueva = {k:v for k,v in cfg.items() if k not in permitidas}
            if config_previa != config_nueva or previo["config_entrenamiento"]["nombre"] == cfg["nombre"] or cfg["epocas"] <= previo["epoca"]:
                raise ValueError("La continuación debe usar otro nombre y ampliar solo presupuesto/parada, manteniendo el aprendizaje y los datos")
            nombre_previo = previo["config_entrenamiento"]["nombre"]
            mejor_previo_path = fuente_estado.parent / f"{nombre_previo}_mejor.pt"
            mejor_previo = torch.load(mejor_previo_path, map_location="cpu", weights_only=True)
            origen_continuacion = {"checkpoint": str(fuente_estado), "sha256": hashlib.sha256(fuente_estado.read_bytes()).hexdigest(),
                "experimento": nombre_previo, "epoca_inicial_continuacion": previo["epoca"] + 1,
                "nota": "Mismo entrenamiento propio desde cero; se preservan pesos, optimizador, scaler y estados aleatorios. No utiliza pesos externos."}
            mejor_previo["config_entrenamiento"] = cfg
            mejor_previo["origen_continuacion"] = origen_continuacion
            guardar_checkpoint(mejor_path, mejor_previo)
        elif previo["config_entrenamiento"] != cfg:
            raise ValueError("Configuración o manifiestos diferentes al experimento que se reanuda")
        modelo.load_state_dict(previo["pesos"])
        optimizador.load_state_dict(previo["optimizador"])
        scaler.load_state_dict(previo["scaler"])
        torch.set_rng_state(previo["rng_torch"].cpu())
        generador.set_state(previo["rng_loader"].cpu())
        if dispositivo == "cuda" and previo["rng_cuda"]:
            torch.cuda.set_rng_state_all([r.cpu() for r in previo["rng_cuda"]])
        historial, mejor_map, mejor_epoca = previo["historial"], previo["mejor_map"], previo["mejor_epoca"]
        paciencia, inicio_epoca = previo["paciencia"], previo["epoca"] + 1
        duracion_anterior = previo["duracion_segundos"]
        if args.reanudar:
            origen_continuacion = previo.get("origen_continuacion")
    if dispositivo == "cuda":
        torch.cuda.reset_peak_memory_stats()
    inicio = time.perf_counter()
    for epoca in range(inicio_epoca, cfg["epocas"] + 1):
        modelo.train()
        for grupo in optimizador.param_groups:
            grupo["lr"] = cfg["learning_rate"] * min(epoca / max(cfg["warmup_epocas"], 1), 1)
        total, componentes, cantidad = 0.0, {}, 0
        for imagenes, objetivos in loader_train:
            x = imagenes.to(dispositivo, non_blocking=True).float().div_(255)
            x, objetivos_aug = aumentar_lote(x, objetivos, cfg["aumentos_solo_train"], corte, radio)
            optimizador.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dispositivo, dtype=torch.float16, enabled=amp):
                raw = modelo(x)
            # Geometría y pérdidas en FP32 para objetos de área pequeña.
            loss, detalle = criterio(raw.float(), objetivos_aug)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Pérdida no finita; detener y diagnosticar")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizador)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            scaler.step(optimizador)
            scaler.update()
            total += loss.item() * len(x)
            cantidad += len(x)
            for k, v in detalle.items():
                componentes[k] = componentes.get(k, 0.0) + v
        metrica, _ = evaluar_detector(modelo, loader_val, dispositivo, amp, cfg)
        historial.append({"epoca": epoca, "loss_train": total / cantidad, "componentes_suma_lotes": componentes,
            "map50_val": metrica["map50"], "map50_95_val": metrica["map50_95"], "recall50_val": metrica["recall50"]})
        mejora = metrica["map50"] > mejor_map + 1e-8
        if mejora:
            mejor_map, mejor_epoca, paciencia = metrica["map50"], epoca, 0
        else:
            paciencia += 1
        checkpoint = {"tipo": "ecowaste_detector_desde_cero_v1", "configuracion": modelo.configuracion,
            "clases": CLASES, "pesos": modelo.state_dict(), "optimizador": optimizador.state_dict(),
            "scaler": scaler.state_dict(), "config_entrenamiento": cfg, "epoca": epoca,
            "mejor_map": mejor_map, "mejor_epoca": mejor_epoca, "paciencia": paciencia, "historial": historial,
            "hashes_manifiestos": hashes, "rng_torch": torch.get_rng_state(), "rng_loader": generador.get_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if dispositivo == "cuda" else [],
            "priores_wh": priors, "umbral_escala": corte, "pesos_externos": False,
            "origen_continuacion": origen_continuacion,
            "duracion_segundos": duracion_anterior + time.perf_counter() - inicio}
        guardar_checkpoint(ultimo_path, checkpoint)
        if mejora:
            guardar_checkpoint(mejor_path, checkpoint)
        escribir_json(salida, {"estado": "en_curso", "configuracion": cfg, "historial": historial, "test_evaluado": False})
        print(f"Época {epoca}/{cfg['epocas']}: loss={total/cantidad:.3f}, mAP50 val={metrica['map50']:.3f}, mAP50:95={metrica['map50_95']:.3f}, recall50={metrica['recall50']:.3f}", flush=True)
        if epoca >= cfg.get("min_epocas", 0) and paciencia >= cfg["paciencia"]:
            break
    mejor = torch.load(mejor_path, map_location=dispositivo, weights_only=True)
    modelo.load_state_dict(mejor["pesos"])
    val_final, predicciones = evaluar_detector(modelo, loader_val, dispositivo, amp, cfg)
    # Tensor lists se guardan en JSON para inspección sin ejecutar el modelo.
    escribir_json(RAIZ / "resultados" / (cfg["nombre"] + "_predicciones_val.json"), [
        {"id": o["id"], **{k: v.tolist() for k, v in p.items()}} for o, p in zip(val.objetivos, predicciones)])
    colisiones_train = sum(len(o["asignacion"]["colisiones"]) for o in train.objetivos)
    reporte = {"estado": "piloto_completado", "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "arquitectura": modelo.configuracion, "configuracion": cfg, "parametros": sum(p.numel() for p in modelo.parameters()),
        "gpu": torch.cuda.get_device_name(0) if dispositivo == "cuda" else "cpu", "torch": str(torch.__version__),
        "precision": "AMP FP16; geometria y perdidas FP32" if amp else "FP32", "pesos_externos": False,
        "mejor_epoca": mejor_epoca, "epocas_ejecutadas": len(historial), "historial": historial, "val": val_final,
        "priores_wh_solo_train": priors, "umbral_escala_solo_train": corte, "pesos_clase_train": pesos.tolist(),
        "objetos_train": int(conteos.sum()), "conteos_clase_train": dict(zip(CLASES, conteos.int().tolist())),
        "colisiones_train_sin_aumentos": colisiones_train, "hashes_manifiestos": hashes,
        "vram_pico_asignada_bytes": torch.cuda.max_memory_allocated() if dispositivo == "cuda" else None,
        "duracion_segundos": duracion_anterior + time.perf_counter() - inicio,
        "checkpoint_mejor": mejor_path.relative_to(RAIZ).as_posix(), "checkpoint_ultimo": ultimo_path.relative_to(RAIZ).as_posix(),
        "test_evaluado": False,
        "origen_continuacion": origen_continuacion,
        "limitaciones": ["Evaluador propio de AP de 101 puntos con regiones ignoradas; no es COCOeval oficial.",
            "Solo cinco objetos orgánicos de entrenamiento y uno en validación; no permite afirmar generalización de esa clase.",
            "Piloto con datos TACO; no acredita rendimiento sobre webcam o nuevas sesiones."]}
    escribir_json(salida, reporte)
    print(json.dumps({"mejor_epoca": mejor_epoca, "map50_val": val_final["map50"], "map50_95_val": val_final["map50_95"],
        "test_evaluado": False, "colisiones_train": colisiones_train}, indent=2), flush=True)


if __name__ == "__main__":
    main()
