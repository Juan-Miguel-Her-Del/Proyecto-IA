"""Recortes multiescala exclusivamente train, desde imágenes y contornos originales."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ecowaste.datasets import CLASES, objetivo_deteccion
from ecowaste.poligonos import rasterizar_contornos

VERSION='recortes_taco_originales_contexto2_35_min64_v1'


def ventana_recorte(registro,caja,factor):
    if not math.isfinite(factor) or factor<1:raise ValueError('Factor debe ser al menos uno')
    W,H=registro['ancho'],registro['alto']
    x,y,w,h=caja['bbox']
    lado=max(64,math.ceil(max(w,h)*factor))
    cw,ch=min(W,lado),min(H,lado)
    x0=min(max(0,round(x+w/2-cw/2)),W-cw)
    y0=min(max(0,round(y+h/2-ch/2)),H-ch)
    return [x0,y0,x0+cw,y0+ch]


def planificar(registros,factores=(2.,3.5)):
    plan=[]
    for i,r in enumerate(registros):
        if r['split']!='train':raise ValueError('Solo se permiten recortes de train')
        for a in r['cajas']:
            if not a['clase']:continue
            for factor in factores:
                v=ventana_recorte(r,a,factor)
                w,h=a['bbox'][2:]
                plan.append({'id':f"{r['id']}/ancla{a['id_interno']}/escala{factor:g}",
                    'id_fuente':r['id'],'indice_registro':i,'ancla_id_interno':a['id_interno'],
                    'clase_ancla':a['clase'],'ventana_xyxy':v,'factor_contexto':factor,
                    'grupo':r['grupo'],'split':'train',
                    'lado_menor_ancla_384_antes':min(w,h)*384/max(r['ancho'],r['alto']),
                    'lado_menor_ancla_384_despues':min(w,h)*384/max(v[2]-v[0],v[3]-v[1])})
    return plan


def objetivo_recorte(registro,ventana,contornos):
    x0,y0,x1,y1=ventana
    if not (0<=x0<x1<=registro['ancho'] and 0<=y0<y1<=registro['alto']):
        raise ValueError('Ventana fuera de la imagen')
    cajas=[];segmentos=[];ids=[]
    for a in registro['cajas']:
        x,y,w,h=a['bbox']
        ix0,iy0,ix1,iy1=max(x,x0),max(y,y0),min(x+w,x1),min(y+h,y1)
        if ix1<=ix0 or iy1<=iy0:continue
        cajas.append(dict(a,bbox=[ix0-x0,iy0-y0,ix1-ix0,iy1-iy0]))
        ids.append(a['id_interno'])
        if a['clase']:
            desplazados=[[v-(x0 if j%2==0 else y0) for j,v in enumerate(p)]
                          for p in contornos.contornos(registro,a)]
            segmentos.append((desplazados,CLASES.index(a['clase'])))
    local=dict(registro,ancho=x1-x0,alto=y1-y0,cajas=cajas)
    objetivo=objetivo_deteccion(local)
    objetivo['ids_anotaciones_visibles']=ids
    mascara=rasterizar_contornos(objetivo,segmentos,cobertura_minima=contornos.cobertura_minima,
                                recortar_al_contenido=True)
    return objetivo,mascara


class CacheRecortes(Dataset):
    def __init__(self,raiz,carpeta,hashes):
        path=Path(raiz)/carpeta
        meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        if meta['version']!=VERSION or not meta['aprobada']:raise ValueError('Recortes no validados')
        for k,v in meta['hashes_origen'].items():
            if hashes.get(k)!=v:raise ValueError('Recortes de otros datos')
        self.plan=[json.loads(l) for l in (path/'plan.jsonl').read_text(encoding='utf-8').splitlines()]
        if any(r['split']!='train' for r in self.plan):raise ValueError('Recorte ajeno a train')
        for nombre,sha in meta['sha256_archivos'].items():
            h=hashlib.sha256()
            with (path/nombre).open('rb') as f:
                for bloque in iter(lambda:f.read(8*1024**2),b''):h.update(bloque)
            if h.hexdigest()!=sha:raise ValueError(f'Archivo alterado: {nombre}')
        self.imagenes=np.load(path/'imagenes.npy',mmap_mode='r')
        self.mascaras=np.load(path/'mascaras.npy',mmap_mode='r')
        n=len(self.plan)
        if (self.imagenes.shape!=(n,3,384,384) or self.mascaras.shape!=(n,48,48)
                or self.imagenes.dtype!=np.uint8 or self.mascaras.dtype!=np.uint8):
            raise ValueError('Caché de recortes incompatible')
        self.metadata=meta

    def __len__(self):return len(self.plan)

    def __getitem__(self,i):
        return torch.from_numpy(self.imagenes[i].copy()),torch.from_numpy(self.mascaras[i].astype(np.int64)),i


class MezclaRecortes(Dataset):
    def __init__(self,completas,recortes,probabilidad=.5,balancear_anclas=False):
        if not 0<probabilidad<1:raise ValueError('La mezcla debe conservar ambas fuentes')
        self.completas,self.recortes=completas,recortes
        pesos_recortes=torch.full((len(recortes),),probabilidad/len(recortes),dtype=torch.double)
        if balancear_anclas:
            conteos=Counter(r['clase_ancla'] for r in recortes.plan)
            if set(conteos)!=set(CLASES):raise ValueError('Faltan materiales para balancear anclas')
            pesos_recortes=torch.tensor([probabilidad/(6*conteos[r['clase_ancla']]) for r in recortes.plan],dtype=torch.double)
        self.pesos_muestreo=torch.cat([
            torch.full((len(completas),),(1-probabilidad)/len(completas),dtype=torch.double),
            pesos_recortes])

    def __len__(self):return len(self.completas)+len(self.recortes)

    def __getitem__(self,i):
        x,y,_=self.completas[i] if i<len(self.completas) else self.recortes[i-len(self.completas)]
        return x,y,i
