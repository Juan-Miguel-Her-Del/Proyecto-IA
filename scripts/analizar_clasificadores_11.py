"""Reproduce validación y registra errores corregidos/nuevos sobre las mismas fotos."""
import hashlib
import json
from pathlib import Path
import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from torch.utils.data import DataLoader
from ecowaste.clasificador import cargar_clasificador
from ecowaste.datasets import CLASES
from ecowaste.evaluacion import metricas_clasificacion
from scripts.entrenar_clasificador import cache_particion, escribir_json


def main():
    salida = RAIZ / 'resultados/analisis_clasificadores_11.json'
    if salida.exists(): raise FileExistsError('Análisis existente')
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    val, ds = cache_particion('val')
    reales = val.tensors[1].tolist()
    nombres = ['clasificador_piloto_v1', 'clasificador_v2_coseno80', 'clasificador_v3_regularizado']
    predicciones, resumen = {}, {}
    for nombre in nombres:
        reporte = json.loads((RAIZ / f'resultados/{nombre}.json').read_text(encoding='utf-8'))
        checkpoint = RAIZ / reporte['checkpoint_mejor']
        m, ck = cargar_clasificador(checkpoint, device)
        sha = hashlib.sha256((RAIZ / 'datos/procesados/clasificacion/val.jsonl').read_bytes()).hexdigest()
        if ck['hashes_manifiestos']['val'] != sha: raise ValueError('Validación modificada')
        labels, scores = [], []
        with torch.inference_mode():
            for x, _ in DataLoader(val, batch_size=32):
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
                    logits = m(x.to(device).float() / 255)
                p, y = logits.float().softmax(-1).max(-1)
                labels.extend(y.cpu().tolist())
                scores.extend(p.cpu().tolist())
        metricas = metricas_clasificacion(labels, reales, CLASES)
        if metricas['matriz_confusion'] != reporte['val']['matriz_confusion']:
            raise ValueError(f'No se reproduce la matriz: {nombre}')
        registros = [{'id': r['id'], 'grupo': r['grupo'], 'grupo_adquisicion': r['grupo_adquisicion'],
                      'real': CLASES[y], 'predicho': CLASES[p], 'puntuacion_no_calibrada': s,
                      'correcto': y == p} for r, y, p, s in zip(ds.registros, reales, labels, scores)]
        escribir_json(RAIZ / f'resultados/{nombre}_predicciones_val.json', registros)
        predicciones[nombre] = registros
        resumen[nombre] = {'metricas': metricas, 'sha256_checkpoint': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          'epoca': ck['epoca'], 'sha256_val': sha}
        del m, ck
    seleccionado = max(nombres, key=lambda n: resumen[n]['metricas']['f1_macro'])
    pares = list(zip(predicciones[nombres[0]], predicciones[seleccionado]))
    corregidos = [b['id'] for a, b in pares if not a['correcto'] and b['correcto']]
    nuevos = [b['id'] for a, b in pares if a['correcto'] and not b['correcto']]
    quedan = [b['id'] for a, b in pares if not a['correcto'] and not b['correcto']]
    informe = {'val_reproducida': True, 'imagenes_val': len(reales), 'modelos': resumen,
        'seleccionado_por_f1': seleccionado, 'errores_anteriores_corregidos': len(corregidos),
        'nuevos_errores': len(nuevos), 'errores_compartidos': len(quedan),
        'ids_corregidos': corregidos, 'ids_nuevos_errores': nuevos, 'ids_errores_compartidos': quedan,
        'test_evaluado': False, 'limite': 'Misma validación usada para seleccionar modelos; no evaluación independiente.'}
    escribir_json(salida, informe)
    print(json.dumps({k: informe[k] for k in ['val_reproducida', 'imagenes_val', 'seleccionado_por_f1',
        'errores_anteriores_corregidos', 'nuevos_errores', 'errores_compartidos', 'test_evaluado']}))


if __name__ == '__main__': main()
