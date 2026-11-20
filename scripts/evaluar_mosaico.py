"""Valida inferencia por zonas con pesos congelados, sin usar GT para proponer cajas."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from PIL import Image, ImageOps
import torch
from ecowaste.datasets import ResiduosDataset, objetivo_deteccion
from ecowaste.detector_mapas import cargar_detector_mapas
from ecowaste.detector_centros import cargar_detector_centros
from ecowaste.mosaico import predecir_mosaico, VERSION
from ecowaste.evaluacion_detector import evaluar_detecciones
from scripts.entrenar_clasificador import escribir_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--nombre', required=True)
    a = p.parse_args()
    if Path(a.nombre).name != a.nombre: raise ValueError('Nombre inválido')
    salida = RAIZ / f'resultados/{a.nombre}.json'
    if salida.exists(): raise FileExistsError('Resultado existente')
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    cargar = cargar_detector_centros if ck['tipo'] == 'ecowaste_centros_desde_cero_v1' else cargar_detector_mapas
    m, ck = cargar(a.checkpoint, device)
    sha = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    val = ResiduosDataset(RAIZ, 'deteccion', 'val')
    if hashlib.sha256((RAIZ/'datos/procesados/deteccion/val.jsonl').read_bytes()).hexdigest() != ck['hashes_manifiestos']['val']:
        raise ValueError('Validación modificada')
    umbral = ck['config_entrenamiento'].get('umbral_evaluacion', .05)
    predicciones, objetivos = [], []
    inicio = time.perf_counter()
    for i, r in enumerate(val.registros):
        with Image.open(RAIZ / r['ruta']) as im:
            imagen = ImageOps.exif_transpose(im).convert('RGB')
        predicciones.append(predecir_mosaico(m, imagen, umbral, amp=device == 'cuda'))
        objetivos.append(objetivo_deteccion(r))
        if (i+1) % 30 == 0: print(f"Inferencia por zonas: {i+1}/{len(val)}", flush=True)
    metricas = evaluar_detecciones(predicciones, objetivos)
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() != sha: raise ValueError('Checkpoint modificado')
    escribir_json(RAIZ / f'resultados/{a.nombre}_predicciones_val.json',
        [{'id': o['id'], **{k: v.tolist() for k, v in p.items()}} for o, p in zip(objetivos, predicciones)])
    escribir_json(salida, {'estado': 'reevaluacion_val_completada', 'fecha_utc': datetime.now(timezone.utc).isoformat(),
        'version': VERSION, 'checkpoint': a.checkpoint.as_posix(), 'sha256_checkpoint': sha,
        'epoca': ck['epoca'], 'hashes_manifiestos': ck['hashes_manifiestos'], 'val': metricas,
        'umbral_evaluacion': umbral, 'vistas_por_imagen': 5, 'test_evaluado': False, 'pesos_externos': False,
        'duracion_segundos': time.perf_counter()-inicio,
        'limite': 'Inferencia experimental de mayor coste, con la misma validación de desarrollo y sin nuevos pesos.'})
    print(json.dumps({'map50': metricas['map50'], 'map50_95': metricas['map50_95'], 'test_evaluado': False}))


if __name__ == '__main__': main()
