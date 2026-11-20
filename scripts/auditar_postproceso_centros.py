"""Comprueba con GT train la conservación geométrica; no mide un modelo entrenado."""
import json
from pathlib import Path
import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.datasets import objetivo_deteccion
from ecowaste.detector_centros import objetivos_centros, postprocesar_centros
from ecowaste.perdida_detector import priores_train
from scripts.entrenar_clasificador import escribir_json


def main():
    torch.set_num_threads(4)
    registros = [json.loads(l) for l in (RAIZ / 'datos/procesados/deteccion/train.jsonl').read_text(encoding='utf-8').splitlines()]
    objetivos = [objetivo_deteccion(r) for r in registros]
    _, corte = priores_train(objetivos)
    totales = {str(iou): 0 for iou in [.5, .7, .9]}
    casos = []
    total = 0
    for o in objetivos:
        t = objetivos_centros(o, corte)
        raw = {'logits': torch.full((1, 48, 48, 6, 2), -20.),
               'boxes': torch.zeros(1, 48, 48, 6, 2, 4)}
        raw['logits'].view(-1)[t['indices']] = 20
        raw['boxes'].view(-1, 4)[t['indices']] = o['boxes'][t['objetos']]
        total += len(o['boxes'])
        fila = {'id': o['id'], 'objetos': len(o['boxes'])}
        for iou in [.5, .7, .9]:
            n = len(postprocesar_centros(raw, [o['geometria']], .35, iou)[0]['boxes'])
            totales[str(iou)] += n
            fila[f'cajas_conservadas_nms_{iou}'] = n
        if fila['cajas_conservadas_nms_0.5'] < len(o['boxes']): casos.append(fila)
    informe = {'fuente': 'solo_anotaciones_train_sin_modelo', 'objetos': total,
               'objetos_asignados': total, 'cajas_gt_conservadas_por_iou_nms': totales, 'casos': casos,
               'test_evaluado': False, 'limite': 'Se inyectan centros y cajas GT perfectos. No es AP, predicción autónoma ni techo absoluto de rendimiento.'}
    escribir_json(RAIZ / 'resultados/auditoria_postproceso_centros_train.json', informe)
    print(json.dumps({k: v for k, v in informe.items() if k != 'casos'}))


if __name__ == '__main__': main()
