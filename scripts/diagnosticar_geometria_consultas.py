"""Cobertura optimista de candidatos sobre train; no equivale a AP ni inferencia."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from torch.utils.data import DataLoader
from ecowaste.datasets import collate_deteccion,CLASES
from ecowaste.ejecucion_detector import CacheDeteccion
from ecowaste.detector_consultas import DetectorConsultas,VERSION_CONSULTAS,VERSION_PONDERADA
from ecowaste.cajas import iou_pares
from scripts.entrenar_clasificador import escribir_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--nombre',required=True)
    a=p.parse_args()
    if Path(a.nombre).name!=a.nombre:raise ValueError('Nombre inválido')
    salida=RAIZ/f'resultados/{a.nombre}.json'
    if salida.exists():raise FileExistsError('Informe existente')
    contenido=a.checkpoint.read_bytes();ck=torch.load(io.BytesIO(contenido),map_location='cpu',weights_only=True)
    if ck['tipo']!='ecowaste_consultas_desde_cero_v1' or ck['version'] not in [VERSION_CONSULTAS,VERSION_PONDERADA]:raise ValueError('Modelo incompatible')
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    ds=CacheDeteccion(RAIZ,'train',.1)
    if hashlib.sha256((RAIZ/'datos/procesados/deteccion/train.jsonl').read_bytes()).hexdigest()!=ck['hashes_manifiestos']['train']:
        raise ValueError('Datos distintos')
    m=DetectorConsultas(**ck['configuracion']).to(device);m.load_state_dict(ck['pesos']);m.eval()
    ious=[];correctos=[];rechazados=[];scores=[]
    for x,objetivos in DataLoader(ds,batch_size=8,collate_fn=collate_deteccion,pin_memory=device=='cuda'):
        with torch.inference_mode(),torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):raw=m(x.to(device).float()/255)
        b=raw['boxes'].float().cpu();prob=raw['logits'].float().softmax(-1).cpu()
        for i,o in enumerate(objetivos):
            valor,indice=iou_pares(o['boxes'],b[i]).max(1)
            material=prob[i,indice,:6].argmax(-1)
            ious.extend(valor.tolist());correctos.extend((material==o['labels']).tolist())
            rechazados.extend((prob[i,indice].argmax(-1)==6).tolist())
            scores.extend(prob[i,indice].gather(1,o['labels'][:,None]).squeeze(1).tolist())
    t=torch.tensor(ious);bien=t>=.5;ac=torch.tensor(correctos);re=torch.tensor(rechazados)
    informe={'epoca_checkpoint':ck['epoca'],'sha256_checkpoint':hashlib.sha256(contenido).hexdigest(),
        'objetos_train':len(t),'iou_medio_mejor_propuesta':float(t.mean()),'cobertura_optimista_iou050':float(bien.float().mean()),
        'material_correcto_mejor_propuesta':float(ac.float().mean()),'nula_argmax_mejor_propuesta':float(re.float().mean()),
        'material_correcto_condicionado_iou050':float(ac[bien].float().mean()) if bool(bien.any()) else None,
        'nula_argmax_condicionado_iou050':float(re[bien].float().mean()) if bool(bien.any()) else None,
        'score_medio_material_gt':sum(scores)/len(scores),'test_evaluado':False,
        'limite':'Se elige con GT la propuesta de mayor IoU sin exigir emparejamiento único. Cobertura optimista de cajas en train, no AP ni inferencia autónoma.'}
    escribir_json(salida,informe);print(json.dumps(informe),flush=True)


if __name__=='__main__':main()
