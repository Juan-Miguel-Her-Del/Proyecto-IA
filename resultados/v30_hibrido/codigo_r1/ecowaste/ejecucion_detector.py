"""Caché y aumentos geométricamente coherentes para detección."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset
from ecowaste.datasets import ResiduosDataset, collate_deteccion, objetivo_deteccion
from ecowaste.inferencia_detector import postprocesar
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.perdida_detector import asignar


class CacheDeteccion(Dataset):
    def __init__(self, raiz, split, umbral_escala, radio_centros=0):
        self.origen = ResiduosDataset(raiz, "deteccion", split)
        raiz = Path(raiz)
        manifiesto = raiz / f"datos/procesados/deteccion/{split}.jsonl"
        hash_manifest = hashlib.sha256(manifiesto.read_bytes()).hexdigest()
        cache_path = raiz / "datos/cache" / f"deteccion_384_{split}_{hash_manifest[:12]}.pt"
        if cache_path.is_file():
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cache["sha256_manifiesto"] != hash_manifest or cache["version"] != "letterbox_384_uint8_v1":
                raise ValueError("Caché incompatible con el manifiesto")
            self.imagenes = cache["imagenes"]
            if tuple(self.imagenes.shape) != (len(self.origen), 3, 384, 384) or self.imagenes.dtype != torch.uint8:
                raise ValueError("Forma o tipo de caché incompatible")
        else:
            self.imagenes = torch.empty(len(self.origen), 3, 384, 384, dtype=torch.uint8)
            def leer(i):
                x, _ = self.origen[i]
                return (x * 255).round().to(torch.uint8)
            with ThreadPoolExecutor(max_workers=4) as pool:
                for i, x in enumerate(pool.map(leer, range(len(self.origen)))):
                    self.imagenes[i] = x
                    if (i + 1) % 200 == 0:
                        print(f"Preparación de caché {split}: {i+1}/{len(self.origen)}", flush=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporal = cache_path.with_suffix(".tmp")
            torch.save({"imagenes": self.imagenes, "sha256_manifiesto": hash_manifest, "version": "letterbox_384_uint8_v1"}, temporal)
            temporal.replace(cache_path)
        self.objetivos = []
        for r in self.origen.registros:
            o = objetivo_deteccion(r)
            o["asignacion"] = asignar(o, umbral_escala=umbral_escala, radio_centros=radio_centros)
            self.objetivos.append(o)
        print(f"Caché {split}: {len(self)} escenas, {self.imagenes.numel()/1024**2:.1f} MiB", flush=True)

    def __len__(self):
        return len(self.origen)

    def __getitem__(self, indice):
        return self.imagenes[indice], self.objetivos[indice]


def aumentar_lote(imagenes, objetivos, config, corte, radio_centros=0):
    nuevos = []
    flip = (torch.rand(len(imagenes)) < config["flip_horizontal"]).tolist()
    for i, o in enumerate(objetivos):
        if not flip[i]:
            nuevos.append(o)
            continue
        nuevo = dict(o)
        imagenes[i] = imagenes[i].flip(-1)
        for key in ["boxes", "ignore_boxes"]:
            b = o[key].clone()
            b[:, 0], b[:, 2] = 1 - o[key][:, 2], 1 - o[key][:, 0]
            nuevo[key] = b
        g = dict(o["geometria"])
        g["relleno_x"] = g["lado"] - (g["relleno_x"] + round(g["ancho_original"] * g["escala_x"]))
        nuevo["geometria"] = g
        nuevo["asignacion"] = asignar(nuevo, umbral_escala=corte, radio_centros=radio_centros)
        nuevos.append(nuevo)
    def factor(limites):
        return torch.empty(len(imagenes), 1, 1, 1, device=imagenes.device).uniform_(*limites)
    imagenes = ((imagenes - 0.5) * factor(config["contraste"]) + 0.5).mul(factor(config["brillo"])).clamp(0, 1)
    return imagenes, nuevos


@torch.inference_mode()
def evaluar_detector(modelo, loader, dispositivo, amp, config):
    modelo.eval()
    predicciones, objetivos = [], []
    for imagenes, lote in loader:
        x = imagenes.to(dispositivo, non_blocking=True).float().div_(255)
        with torch.autocast(device_type=dispositivo, dtype=torch.float16, enabled=amp):
            raw = modelo(x)
        predicciones.extend(postprocesar(raw, config["umbral_evaluacion"], config["nms_iou"], config["max_detecciones"],
            geometrias=[o["geometria"] for o in lote], radio_centros=modelo.configuracion.get("radio_centros", 0)))
        objetivos.extend(lote)
    resultado = evaluar_detecciones(predicciones, objetivos)
    return resultado, predicciones
