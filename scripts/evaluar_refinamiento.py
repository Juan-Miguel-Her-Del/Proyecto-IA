"""Valida regiones con recortes y clasificador auxiliar; no usa GT al predecir."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from PIL import Image
import torch
from ecowaste.datasets import ResiduosDataset,letterbox,objetivo_deteccion
from ecowaste.detector_mapas import cargar_detector_mapas
from ecowaste.refinamiento import refinar_imagen,VERSION
from ecowaste.evaluacion_detector import evaluar_detecciones
from scripts.entrenar_clasificador import escribir_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--nombre',required=True)
    a=parser.parse_args()
    if Path(a.nombre).name!=a.nombre:raise ValueError('Nombre inválido')
    salida=RAIZ/f'resultados/{a.nombre}.json'
    if salida.exists():raise FileExistsError('Experimento existente')
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    m,ck=cargar_detector_mapas(a.checkpoint,device)
    if ck['configuracion'].get('formulacion','softmax7')!='softmax7':raise ValueError('El refinamiento espera softmax7')
    sha=hashlib.sha256((RAIZ/'datos/procesados/deteccion/val.jsonl').read_bytes()).hexdigest()
    if ck['hashes_manifiestos']['val']!=sha:raise ValueError('Datos distintos')
    ds=ResiduosDataset(RAIZ,'deteccion','val');objetivos=[];pred=[]
    for i,r in enumerate(ds.registros):
        with Image.open(RAIZ/r['ruta']) as im:imagen=im.convert('RGB')
        x,g=letterbox(imagen,384)
        with torch.inference_mode(),torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
            raw=m(x[None].to(device))[0]
        pred.append(refinar_imagen(m,raw,imagen,g,umbral=.05,amp=device=='cuda'))
        objetivos.append(objetivo_deteccion(r,g))
        if (i+1)%40==0:print(f'Refinamiento val {i+1}/{len(ds)}',flush=True)
    metrica=evaluar_detecciones(pred,objetivos)
    escribir_json(RAIZ/f'resultados/{a.nombre}_predicciones_val.json',[{'id':o['id'],**{k:v.tolist() for k,v in p.items()}} for o,p in zip(objetivos,pred)])
    escribir_json(salida,{'estado':'reevaluacion_val_completada','variante_inferencia':VERSION,
        'checkpoint':str(a.checkpoint),'sha256_checkpoint':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'epoca':ck['epoca'],'val':metrica,'hashes_manifiestos':ck['hashes_manifiestos'],
        'configuracion':{'umbral_evaluacion':.05,'umbral_presencia':.5,'contexto':1.2,'max_propuestas':40},
        'pesos_externos':False,'test_evaluado':False,
        'limite':'Nueva variante de inferencia sobre val, no prueba independiente. Cajas GT solo se consultan en evaluación.'})
    print(json.dumps({k:metrica[k] for k in ['map50','map50_95','recall50']}),flush=True)


if __name__=='__main__':main()
