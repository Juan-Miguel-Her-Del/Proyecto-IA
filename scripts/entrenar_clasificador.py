"""Piloto reproducible con train/val; test permanece reservado."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ecowaste.clasificador import ClasificadorResiduos
from ecowaste.datasets import CLASES, ResiduosDataset
from ecowaste.evaluacion import metricas_clasificacion


def escribir_json(path, valor):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporal = path.with_suffix(path.suffix + ".tmp")
    temporal.write_text(json.dumps(valor, ensure_ascii=False, indent=2), encoding="utf-8")
    temporal.replace(path)


def guardar_checkpoint(path, valor):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporal = path.with_suffix(".tmp")
    torch.save(valor, temporal)
    temporal.replace(path)


def cache_particion(split):
    ds = ResiduosDataset(RAIZ, "clasificacion", split)
    imagenes = torch.empty(len(ds), 3, 224, 224, dtype=torch.uint8)
    etiquetas = torch.empty(len(ds), dtype=torch.long)
    for i in range(len(ds)):
        x, y = ds[i]
        imagenes[i] = (x * 255).round().to(torch.uint8)
        etiquetas[i] = y
    print(f"Cache RAM {split}: {len(ds)} imágenes, {imagenes.numel() / 1024**2:.1f} MiB", flush=True)
    return TensorDataset(imagenes, etiquetas), ds


def aumentar(x, config):
    b = len(x)
    mascara = torch.rand(b, 1, 1, 1, device=x.device) < config["flip_horizontal"]
    x = torch.where(mascara, x.flip(-1), x)
    def factor(limites):
        return torch.empty(b, 1, 1, 1, device=x.device).uniform_(*limites)
    x = ((x - 0.5) * factor(config["contraste"]) + 0.5).mul(factor(config["brillo"])).clamp(0, 1)
    if config.get('afin'):
        a = config['afin']
        angulo = torch.empty(b, device=x.device).uniform_(-a['grados'], a['grados']) * math.pi / 180
        escala = torch.empty(b, device=x.device).uniform_(*a['escala'])
        theta = torch.zeros(b, 2, 3, device=x.device)
        theta[:, 0, 0] = theta[:, 1, 1] = angulo.cos() / escala
        theta[:, 0, 1] = -angulo.sin() / escala
        theta[:, 1, 0] = angulo.sin() / escala
        theta[:, :, 2] = torch.empty(b, 2, device=x.device).uniform_(-2*a['traslacion'], 2*a['traslacion'])
        grid = nn.functional.affine_grid(theta, x.shape, align_corners=False)
        relleno = 114 / 255
        x = nn.functional.grid_sample(x - relleno, grid, align_corners=False,
                                     padding_mode='zeros') + relleno
    return x.clamp(0, 1)


def mezclar_lote(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1.0
    concentracion = x.new_tensor(float(alpha))
    peso = float(torch.distributions.Beta(concentracion, concentracion).sample())
    permutacion = torch.randperm(len(x), device=x.device)
    return peso*x + (1-peso)*x[permutacion], y, y[permutacion], peso


def tasa_aprendizaje(epoca, cfg):
    warmup = max(cfg['warmup_epocas'], 1)
    if epoca <= warmup:
        return cfg['learning_rate'] * epoca / warmup
    if cfg.get('programacion_lr', 'constante') == 'constante':
        return cfg['learning_rate']
    if cfg['programacion_lr'] != 'coseno':
        raise ValueError('Programación de aprendizaje desconocida')
    proporcion = min((epoca - warmup) / max(cfg['epocas'] - warmup, 1), 1)
    suelo = cfg.get('fraccion_lr_final', .1)
    if not 0 <= suelo <= 1:
        raise ValueError('Fracción final de aprendizaje inválida')
    return cfg['learning_rate'] * (suelo + (1 - suelo) * (1 + math.cos(math.pi * proporcion)) / 2)


@torch.inference_mode()
def evaluar(modelo, loader, dispositivo, amp):
    modelo.eval()
    predichos, reales, loss = [], [], 0.0
    for x, y in loader:
        x = x.to(dispositivo, non_blocking=True).float().div_(255)
        y = y.to(dispositivo, non_blocking=True)
        with torch.autocast(device_type=dispositivo, dtype=torch.float16, enabled=amp):
            logits = modelo(x)
            perdida = nn.functional.cross_entropy(logits, y)
        loss += perdida.item() * len(x)
        predichos.extend(logits.argmax(-1).tolist())
        reales.extend(y.tolist())
    resultado = metricas_clasificacion(predichos, reales, CLASES)
    resultado["loss_ce_no_ponderada"] = loss / len(reales)
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=RAIZ / "configuracion/entrenamiento_clasificador.json")
    parser.add_argument("--reanudar", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", cfg["nombre"]):
        raise ValueError("Nombre de experimento no válido")
    if cfg["evaluar_test"]:
        raise ValueError("El piloto mantiene test reservado; usar una evaluación final separada")
    resultado_path = RAIZ / "resultados" / (cfg["nombre"] + ".json")
    mejor_path = RAIZ / "checkpoints" / (cfg["nombre"] + "_mejor.pt")
    ultimo_path = RAIZ / "checkpoints" / (cfg["nombre"] + "_ultimo.pt")
    if not args.reanudar and any(p.exists() for p in [resultado_path, mejor_path, ultimo_path]):
        raise FileExistsError("Experimento existente; reanudar o elegir otro nombre para conservar la evidencia")
    validacion = json.loads((RAIZ / "datos/reportes/validacion.json").read_text(encoding="utf-8"))
    diagnostico = json.loads((RAIZ / "resultados/diagnostico_clasificador.json").read_text(encoding="utf-8"))
    if not validacion["validacion_estructural_correcta"] or not diagnostico["aprobada"]:
        raise ValueError("Se requieren datos validados y diagnóstico de aprendizaje aprobado")
    hashes = {}
    for split in ["train", "val"]:
        relativo = f"datos/procesados/clasificacion/{split}.jsonl"
        hashes[split] = hashlib.sha256((RAIZ / relativo).read_bytes()).hexdigest()
        if hashes[split] != validacion["sha256_manifiestos"][relativo]:
            raise ValueError("Manifiesto modificado desde la validación")
    torch.set_num_threads(4)
    random.seed(cfg["semilla"])
    torch.manual_seed(cfg["semilla"])
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dispositivo == "cuda" and cfg["precision_mixta_cuda"]
    train, ds_train = cache_particion("train")
    val, _ = cache_particion("val")
    generador = torch.Generator().manual_seed(cfg["semilla"])
    loader_train = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, generator=generador, pin_memory=dispositivo == "cuda")
    loader_val = DataLoader(val, batch_size=cfg["batch_size"], pin_memory=dispositivo == "cuda")
    modelo = ClasificadorResiduos().to(dispositivo)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024)
    conteos = torch.bincount(train.tensors[1], minlength=6).float()
    pesos = conteos.rsqrt()
    pesos /= pesos.mean()
    criterio = nn.CrossEntropyLoss(weight=pesos.to(dispositivo), label_smoothing=cfg["label_smoothing"])
    historial, mejor_f1, mejor_epoca, paciencia, inicio_epoca = [], -1.0, 0, 0, 1
    duracion_anterior = 0.0
    if args.reanudar:
        previo = torch.load(args.reanudar, map_location=dispositivo, weights_only=True)
        if previo["config_entrenamiento"] != cfg or previo["hashes_manifiestos"] != hashes:
            raise ValueError("La configuración y los datos deben coincidir para reanudar")
        modelo.load_state_dict(previo["pesos"])
        optimizador.load_state_dict(previo["optimizador"])
        scaler.load_state_dict(previo["scaler"])
        torch.set_rng_state(previo["rng_torch"].cpu())
        generador.set_state(previo["rng_loader"].cpu())
        if dispositivo == "cuda" and previo["rng_cuda"]:
            torch.cuda.set_rng_state_all([r.cpu() for r in previo["rng_cuda"]])
        historial = previo["historial"]
        mejor_f1, mejor_epoca, paciencia = previo["mejor_f1"], previo["mejor_epoca"], previo["paciencia"]
        inicio_epoca = previo["epoca"] + 1
        duracion_anterior = previo["duracion_entrenamiento_segundos"]
    if dispositivo == "cuda":
        torch.cuda.reset_peak_memory_stats()
    inicio = time.perf_counter()
    print(f"Entrenando {cfg['nombre']}: {sum(p.numel() for p in modelo.parameters())} parámetros, {dispositivo}, AMP={amp}", flush=True)
    for epoca in range(inicio_epoca, cfg["epocas"] + 1):
        modelo.train()
        lr = tasa_aprendizaje(epoca, cfg)
        for grupo in optimizador.param_groups:
            grupo["lr"] = lr
        total_loss, aciertos, cantidad = 0.0, 0, 0
        for x, y in loader_train:
            x = x.to(dispositivo, non_blocking=True).float().div_(255)
            y = y.to(dispositivo, non_blocking=True)
            x = aumentar(x, cfg["aumentos_solo_train"])
            x, y_a, y_b, mezcla = mezclar_lote(x, y, cfg.get('mixup_alpha', 0))
            optimizador.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dispositivo, dtype=torch.float16, enabled=amp):
                logits = modelo(x)
                loss = mezcla*criterio(logits, y_a) + (1-mezcla)*criterio(logits, y_b)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Pérdida no finita; detener y diagnosticar")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizador)
            nn.utils.clip_grad_norm_(modelo.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            scaler.step(optimizador)
            scaler.update()
            total_loss += loss.item() * len(x)
            aciertos += mezcla*(logits.argmax(-1) == y_a).sum().item() + (1-mezcla)*(logits.argmax(-1) == y_b).sum().item()
            cantidad += len(x)
        metrica_val = evaluar(modelo, loader_val, dispositivo, amp)
        fila = {"epoca": epoca, "learning_rate": lr, "loss_train_ponderada_con_aumentos": total_loss / cantidad,
            "accuracy_train_durante_actualizaciones": aciertos / cantidad, "val": metrica_val}
        historial.append(fila)
        mejora = metrica_val["f1_macro"] > mejor_f1 + 1e-6
        if mejora:
            mejor_f1, mejor_epoca, paciencia = metrica_val["f1_macro"], epoca, 0
        else:
            paciencia += 1
        checkpoint = {"tipo": "ecowaste_clasificador_desde_cero_v1", "configuracion": modelo.configuracion,
            "clases": CLASES, "pesos": modelo.state_dict(), "optimizador": optimizador.state_dict(),
            "scaler": scaler.state_dict(), "config_entrenamiento": cfg, "epoca": epoca,
            "mejor_f1": mejor_f1, "mejor_epoca": mejor_epoca, "paciencia": paciencia, "historial": historial,
            "hashes_manifiestos": hashes, "rng_torch": torch.get_rng_state(), "rng_loader": generador.get_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if dispositivo == "cuda" else [], "pesos_externos": False,
            "duracion_entrenamiento_segundos": duracion_anterior + time.perf_counter() - inicio}
        guardar_checkpoint(ultimo_path, checkpoint)
        if mejora:
            guardar_checkpoint(mejor_path, checkpoint)
        escribir_json(resultado_path, {"estado": "en_curso", "configuracion": cfg, "historial": historial,
            "mejor_epoca": mejor_epoca, "test_evaluado": False})
        print(f"Época {epoca}/{cfg['epocas']}: loss={total_loss/cantidad:.4f}, val accuracy={metrica_val['accuracy']:.1%}, F1={metrica_val['f1_macro']:.3f}", flush=True)
        if epoca >= cfg.get('min_epocas', 1) and paciencia >= cfg["paciencia"]:
            break
    mejor = torch.load(mejor_path, map_location=dispositivo, weights_only=True)
    modelo.load_state_dict(mejor["pesos"])
    loader_train_eval = DataLoader(train, batch_size=cfg["batch_size"], pin_memory=dispositivo == "cuda")
    train_final = evaluar(modelo, loader_train_eval, dispositivo, amp)
    val_final = evaluar(modelo, loader_val, dispositivo, amp)
    mayoritaria = int(conteos.argmax())
    val_reales = val.tensors[1].tolist()
    baseline = metricas_clasificacion([mayoritaria] * len(val_reales), val_reales, CLASES)
    reporte = {"estado": "piloto_completado", "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "configuracion": cfg, "arquitectura": modelo.configuracion, "parametros": sum(p.numel() for p in modelo.parameters()),
        "gpu": torch.cuda.get_device_name(0) if dispositivo == "cuda" else "cpu", "torch": str(torch.__version__),
        "inicializacion": "aleatoria_de_PyTorch_sin_pesos_externos; biases_y_normas_con_inicializacion_estandar",
        "precision": "AMP FP16" if amp else "FP32", "mejor_epoca": mejor_epoca,
        "epocas_ejecutadas": len(historial), "historial": historial, "train": train_final, "val": val_final,
        "baseline_mayoritaria_train_sobre_val": baseline, "pesos_clases_train": pesos.tolist(),
        "conteos_train": dict(Counter(r["clase"] for r in ds_train.registros)), "hashes_manifiestos": hashes,
        "vram_pico_asignada_bytes": torch.cuda.max_memory_allocated() if dispositivo == "cuda" else None,
        "duracion_entrenamiento_segundos": duracion_anterior + time.perf_counter() - inicio,
        "checkpoint_mejor": mejor_path.relative_to(RAIZ).as_posix(), "checkpoint_ultimo": ultimo_path.relative_to(RAIZ).as_posix(),
        "test_evaluado": False, "limitacion": "Piloto sobre un solo dataset. Val se usa para seleccionar checkpoint; no equivale a prueba final ni demuestra detección o generalización en webcam."}
    escribir_json(resultado_path, reporte)
    print(json.dumps({"mejor_epoca": mejor_epoca, "accuracy_val": val_final["accuracy"], "f1_macro_val": val_final["f1_macro"], "test_evaluado": False, "vram_pico_MiB": (reporte["vram_pico_asignada_bytes"] or 0)/1024**2}, indent=2), flush=True)


if __name__ == "__main__":
    main()
