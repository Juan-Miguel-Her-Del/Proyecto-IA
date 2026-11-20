"""Evalúa ranking de consultas sin descartar una caja por argmax de clase nula."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from torch.utils.data import DataLoader
from ecowaste.datasets import collate_deteccion
from ecowaste.ejecucion_detector import CacheDeteccion
from ecowaste.detector_consultas import cargar_detector_consultas,postprocesar_consultas
from ecowaste.evaluacion_detector import evaluar_detecciones
from scripts.entrenar_clasificador import escribir_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--nombre',required=True)
    a=p.parse_args()
    if Path(a.nombre).name!=a.nombre:raise ValueError('Nombre inválido')
    salida=RAIZ/f'resultados/{a.nombre}.json'
    if salida.exists():raise FileExistsError('Informe existente')
    contenido=a.checkpoint.read_bytes();sha=hashlib.sha256(contenido).hexdigest()
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    m,ck=cargar_detector_consultas(a.checkpoint,device)
    manifiesto=RAIZ/'datos/procesados/deteccion/val.jsonl'
    if hashlib.sha256(manifiesto.read_bytes()).hexdigest()!=ck['hashes_manifiestos']['val']:raise ValueError('Datos distintos')
    val=CacheDeteccion(RAIZ,'val',.1);dl=DataLoader(val,batch_size=8,collate_fn=collate_deteccion,pin_memory=device=='cuda')
    pred=[];objetivos=[]
    for x,lote in dl:
        with torch.inference_mode(),torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):raw=m(x.to(device).float()/255)
        pred.extend(postprocesar_consultas(raw,[o['geometria'] for o in lote],.05,descartar_nula=False));objetivos.extend(lote)
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=sha:raise ValueError('El checkpoint cambió durante la evaluación')
    met=evaluar_detecciones(pred,objetivos)
    escribir_json(RAIZ/f'resultados/{a.nombre}_predicciones_val.json',[{'id':o['id'],**{k:v.tolist() for k,v in q.items()}} for o,q in zip(objetivos,pred)])
    escribir_json(salida,{'estado':'reevaluacion_val_completada','variante_inferencia':'consultas_ranking_sin_corte_nula',
        'checkpoint':str(a.checkpoint),'sha256_checkpoint':sha,'epoca':ck['epoca'],'val':met,'hashes_manifiestos':ck['hashes_manifiestos'],
        'configuracion':{'umbral_evaluacion':.05,'descartar_nula':False},'pesos_externos':False,'test_evaluado':False,
        'limite':'Otra política de inferencia sobre val; pesos y entrenamiento no cambian. No es evaluación independiente.'})
    print(json.dumps({k:met[k] for k in ['map50','map50_95','recall50']}),flush=True)


if __name__=='__main__':main()
