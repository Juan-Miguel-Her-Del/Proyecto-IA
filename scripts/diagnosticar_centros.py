"""Memorización controlada y auditoría completa de asignación de centros train."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from torch.nn import functional as F
from ecowaste.datasets import ResiduosDataset, objetivo_deteccion
from ecowaste.auxiliar_clasificacion import DatosClasificacionAuxiliar
from ecowaste.perdida_detector import priores_train
from ecowaste.detector_centros import DetectorCentros, objetivos_centros, perdida_centros, postprocesar_centros, VERSION
from ecowaste.evaluacion_detector import evaluar_detecciones
from scripts.entrenar_clasificador import escribir_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nombre', default='diagnostico_centros')
    args = parser.parse_args()
    if Path(args.nombre).name != args.nombre: raise ValueError('Nombre inválido')
    salida = RAIZ / f'resultados/{args.nombre}.json'
    if salida.exists(): raise FileExistsError('Diagnóstico existente')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = ResiduosDataset(RAIZ, 'deteccion', 'train')
    todos = [objetivo_deteccion(r) for r in ds.registros]
    priors, corte = priores_train(todos)
    asignados = sum(len(objetivos_centros(o, corte)['indices']) for o in todos)
    total = sum(len(o['boxes']) for o in todos)
    if asignados != total: raise ValueError('Se han perdido objetos train')
    origen = json.loads((RAIZ / 'resultados/diagnostico_auxiliar_clasificacion.json').read_text(encoding='utf-8'))
    indices = [next(i for i, r in enumerate(ds.registros) if r['id'] == ident) for ident in origen['ids_taco_train']]
    lote = [ds[i] for i in indices]
    objetivos = [o for _, o in lote]
    x = torch.stack([im for im, _ in lote]).to(device)
    aux = DatosClasificacionAuxiliar(RAIZ)
    ia = [aux.ids.index(ident) for ident in origen['ids_realwaste_train']]
    xa = torch.stack([aux[i][0] for i in ia]).to(device).float() / 255
    ya = aux.etiquetas[ia].to(device)
    m = DetectorCentros().to(device)
    m.configurar_centros(priors, corte)
    opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=0)
    historial, aprobada = [], False
    inicio = time.perf_counter()
    for paso in range(1, 401):
        m.train()
        opt.zero_grad(set_to_none=True)
        loss, detalle = perdida_centros(m(x), objetivos, corte)
        loss_aux = F.cross_entropy(m.clasificar(xa), ya)
        (loss + .5 * loss_aux).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        if paso % 25 == 0:
            m.eval()
            with torch.no_grad():
                pred = postprocesar_centros(m(x), [o['geometria'] for o in objetivos], .35)
                acc = float((m.clasificar(xa).argmax(1) == ya).float().mean())
            met = evaluar_detecciones(pred, objetivos)
            historial.append({'paso': paso, 'loss': loss.item(), 'accuracy_aux': acc,
                              **{k: met[k] for k in ['map50', 'recall50', 'precision50']}, **detalle})
            print(f"Paso {paso}: AP50={met['map50']:.3f}, recall={met['recall50']:.3f}, precisión={met['precision50']:.3f}, aux={acc:.3f}", flush=True)
            if met['map50'] >= .8 and met['recall50'] >= .9 and met['precision50'] >= .7 and acc == 1:
                aprobada = True
                break
    escribir_json(salida, {'aprobada': aprobada, 'version': VERSION, 'arquitectura': m.configuracion_centros,
        'parametros': sum(p.numel() for p in m.parameters()), 'historial': historial,
        'sha256_train': hashlib.sha256((RAIZ / 'datos/procesados/deteccion/train.jsonl').read_bytes()).hexdigest(),
        'sha256_clasificacion_train': aux.metadata['sha256_manifiesto'], 'ids_taco_train': origen['ids_taco_train'],
        'ids_realwaste_train': origen['ids_realwaste_train'], 'objetos_train_auditados': total,
        'objetos_train_asignados': asignados, 'priores_train': priors, 'corte_train': corte,
        'duracion_segundos': time.perf_counter() - inicio, 'pesos_reutilizados': False, 'test_evaluado': False,
        'limite': 'Memorización de escenas fáciles train; no generalización. Los pesos se descartan.'})
    return 0 if aprobada else 1


if __name__ == '__main__': raise SystemExit(main())
