"""Reevalúa un checkpoint propio con filtrado de padding previo a componentes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.detector_mapas import cargar_detector_mapas
from scripts.entrenar_detector_mapas import DatosMapas,evaluar


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--nombre',required=True)
    a=p.parse_args()
    if Path(a.nombre).name!=a.nombre:raise ValueError('Nombre inválido')
    salida=RAIZ/f'resultados/{a.nombre}.json'
    if salida.exists():raise FileExistsError('Informe existente')
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    modelo,ck=cargar_detector_mapas(a.checkpoint,device)
    cfg=ck['config_entrenamiento']
    manifiesto=RAIZ/'datos/procesados/deteccion/val.jsonl'
    sha=hashlib.sha256(manifiesto.read_bytes()).hexdigest()
    if sha!=ck['hashes_manifiestos']['val']:raise ValueError('Validación distinta del checkpoint')
    datos=DatosMapas('val')
    dl=DataLoader(datos,batch_size=cfg['batch_size'],pin_memory=device=='cuda')
    metrica,pred=evaluar(modelo,dl,device,device=='cuda' and cfg['precision_mixta_cuda'],cfg)
    reporte={'estado':'reevaluacion_val_completada','postprocesamiento':'componentes_solo_contenido_real_v2',
        'val':metrica,'checkpoint':str(a.checkpoint),'sha256_checkpoint':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'sha256_val':sha,'test_evaluado':False,'epoca':ck['epoca'],'configuracion':cfg,
        'limite':'Sin entrenamiento ni ajuste de umbral; separación de padding corregida. Conserva todos los GT.'}
    salida.write_text(json.dumps(reporte,indent=2,ensure_ascii=False),encoding='utf-8')
    (RAIZ/f'resultados/{a.nombre}_predicciones_val.json').write_text(json.dumps([
        {'id':o['id'],**{k:v.tolist() for k,v in q.items()}} for o,q in zip(datos.base.objetivos,pred)]),encoding='utf-8')
    print(json.dumps({'map50':metrica['map50'],'map50_95':metrica['map50_95']}),flush=True)


if __name__=='__main__':main()
