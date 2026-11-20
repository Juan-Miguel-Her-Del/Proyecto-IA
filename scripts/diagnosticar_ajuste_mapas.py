"""Separa ajuste semántico en train de la reconstrucción de cajas; test reservado."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from torch.utils.data import DataLoader,Subset
from ecowaste.datasets import CLASES
from ecowaste.poligonos import ContornosTACO
from ecowaste.recortes import CacheRecortes
from ecowaste.detector_mapas import cargar_detector_mapas,etiquetas_mapas
from scripts.entrenar_detector_mapas import DatosMapas
from scripts.entrenar_clasificador import escribir_json


@torch.inference_mode()
def medir(modelo,dl):
    matriz=torch.zeros(7,7,dtype=torch.long,device='cuda')
    material_correcto=total_fg=0
    for x,y,_ in dl:
        x=x.cuda().float()/255;y=y.cuda()
        with torch.autocast('cuda',dtype=torch.float16):raw=modelo(x)
        p=etiquetas_mapas(raw,modelo.configuracion_mapas.get('formulacion','softmax7'))
        valida=y!=255
        fg=valida&(y>0)
        material_correcto+=int(((raw[:,1:].argmax(1)+1==y)&fg).sum())
        total_fg+=int(fg.sum())
        matriz+=torch.bincount(y[valida]*7+p[valida],minlength=49).reshape(7,7)
    matriz=matriz.cpu();tp=matriz.diag();union=matriz.sum(0)+matriz.sum(1)-tp
    por_clase={}
    for i,c in enumerate(['fondo']+CLASES):
        por_clase[c]={'celdas_gt':int(matriz[i].sum()),'celdas_predichas':int(matriz[:,i].sum()),
            'recall':float(tp[i]/matriz[i].sum().clamp(min=1)),
            'iou':float(tp[i]/union[i].clamp(min=1))}
    return {'matriz_confusion_filas_gt':matriz.tolist(),'por_clase':por_clase,
        'accuracy_solo_foreground_gt':float(tp[1:].sum()/matriz[1:].sum().clamp(min=1)),
        'accuracy_material_condicionada_a_celdas_GT':material_correcto/max(total_fg,1),
        'foreground_gt_predicho_fondo':float(matriz[1:,0].sum()/matriz[1:].sum().clamp(min=1)),
        'iou_macro_seis_materiales':sum(por_clase[c]['iou'] for c in CLASES)/6}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nombre',default='ajuste_semantico_mapas_train')
    parser.add_argument('--experimentos',nargs='+',default=['detector_mapas_poligonos_v2','detector_mapas_recortes_v3'])
    args=parser.parse_args()
    if any(Path(n).name!=n for n in [args.nombre]+args.experimentos):raise ValueError('Nombre inválido')
    salida=RAIZ/f'resultados/{args.nombre}.json'
    if salida.exists():raise FileExistsError('Diagnóstico existente')
    torch.set_num_threads(4)
    c=ContornosTACO(RAIZ,1/64);completas=DatosMapas('train',c)
    hashes={'train':hashlib.sha256((RAIZ/'datos/procesados/deteccion/train.jsonl').read_bytes()).hexdigest(),
        'anotaciones_poligonos':c.sha256,'inventario_procedencia':c.sha256_inventario}
    recortes=CacheRecortes(RAIZ,'datos/cache/recortes_multiescala_v1',hashes)
    rng=random.Random(42);indices=[]
    for clase in CLASES:
        disponibles=[i for i,r in enumerate(recortes.plan) if r['clase_ancla']==clase]
        indices.extend(rng.sample(disponibles,min(10,len(disponibles))))
    datasets={'completas_train':completas,'recortes_train_estratificados':Subset(recortes,indices)}
    resultado={'test_evaluado':False,'seleccion_recortes':[recortes.plan[i]['id'] for i in indices],
        'limites':['Las 964 escenas completas son train, no generalización.','Los 60 recortes se seleccionan por clase de ancla; no estiman la distribución natural.','IoU semántico no equivale a AP de cajas.'],
        'experimentos':{}}
    for nombre in args.experimentos:
        ck=RAIZ/f'checkpoints/{nombre}_mejor.pt'
        modelo,meta=cargar_detector_mapas(ck,'cuda')
        if meta['hashes_manifiestos']['train']!=hashes['train']:raise ValueError('Datos distintos')
        mediciones={}
        for tipo,ds in datasets.items():
            mediciones[tipo]=medir(modelo,DataLoader(ds,batch_size=8,pin_memory=True))
            print(nombre,tipo,'accuracy_fg',round(mediciones[tipo]['accuracy_solo_foreground_gt'],3),flush=True)
        resultado['experimentos'][nombre]={'sha256_checkpoint':hashlib.sha256(ck.read_bytes()).hexdigest(),'mediciones':mediciones}
    escribir_json(salida,resultado)


if __name__=='__main__':main()
