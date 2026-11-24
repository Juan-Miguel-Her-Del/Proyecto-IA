"""Datos experimentales V28.2: solo train; ningún cambio en lectores históricos."""
from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from ecowaste.incorporacion import leer,sha,CLASES
from ecowaste.datasets import letterbox,objetivo_deteccion
from ecowaste.detector_mapas import rasterizar
from ecowaste.validacion_anotaciones import mask_poligonos,celdas

VERSION='auxiliar_exploratorio_v28_2_1'


def objetivo_aux(im,anns,cats):
    cajas=[{'bbox':a['bbox'],'clase':cats[a['category_id']] if cats[a['category_id']] in CLASES else None} for a in anns]
    return objetivo_deteccion({'id':im['eco']['imagen_id_auxiliar'],'ancho':im['width'],'alto':im['height'],'cajas':cajas})


def mascara_aux(im,anns,cats):
    """Rasterización histórica por cobertura, conservando huecos nuevos y ignore.

    La pérdida no se modifica. Sin huecos, equivale a rasterizar_contornos.
    Ignore de bbox conservador y precedencia de foreground conservan V6.
    """
    o=objetivo_aux(im,anns,cats);g=o['geometria'];lienzos=np.zeros((6,384,384),dtype=np.uint8)
    for a in anns:
        c=cats[a['category_id']]
        if c not in CLASES:continue
        rings={'poligonos':a['eco']['poligonos_usuario'],'huecos':a['eco']['huecos_usuario']}
        lienzos[CLASES.index(c)]|=mask_poligonos(rings,im['width'],im['height'],geometria=g)
    cover=np.stack([celdas(m) for m in lienzos]);vacio=dict(o,boxes=torch.empty(0,4),labels=torch.empty(0,dtype=torch.long))
    target=rasterizar(vacio,48);target[torch.from_numpy(cover.sum(0)>0)]=255
    positivas=cover>=1/64;unica=positivas.sum(0)==1
    for c in range(6):target[torch.from_numpy(positivas[c]&unica)]=c+1
    return target


class AuxiliarExperimental:
    def __init__(self,raiz,carpeta):
        self.raiz,self.carpeta=Path(raiz),Path(carpeta)
        meta=leer(self.carpeta/'MANIFIESTO.json')
        if meta['version']!=VERSION or meta['estado']!='EXPLORATORIA_NO_VALIDADA_PARA_ADOPCION' or not meta['autorizada_solo_diagnostico']:
            raise ValueError('No es la colección experimental autorizada')
        for p,h in meta['archivos'].items():
            if sha(self.carpeta/p)!=h:raise ValueError('Colección experimental alterada: '+p)
        d=leer(self.carpeta/'annotations.json');self.meta=meta;self.coco=d;cats={c['id']:c['name'] for c in d['categories']}
        self.registros=[];self.objetivos=[];xs=[];ys=[]
        for im in d['images']:
            anns=[a for a in d['annotations'] if a['image_id']==im['id']]
            p=self.carpeta/im['file_name']
            if sha(p)!=im['eco']['sha256_archivo']:raise ValueError('Foto auxiliar alterada')
            with Image.open(p) as pil:x,_=letterbox(pil.convert('RGB'),384)
            xs.append((x*255).round().to(torch.uint8));ys.append(mascara_aux(im,anns,cats));self.objetivos.append(objetivo_aux(im,anns,cats))
            self.registros.append({'id':im['eco']['imagen_id_auxiliar'],'ruta':p.relative_to(self.raiz).as_posix(),
                'ancho':im['width'],'alto':im['height'],'sha256_archivo':im['eco']['sha256_archivo'],'fuente':im['eco']['fuente']})
        self.imagenes=torch.stack(xs);self.mascaras=torch.stack(ys)
        if set(self.mascaras.unique().tolist())-set([0,1,2,3,4,5,6,255]):raise ValueError('Etiquetas inválidas')

    def __len__(self):return len(self.registros)


def historico_train(raiz):
    # DatosMapas solo abre la caché train existente. Rechazar creación implícita.
    from scripts.entrenar_detector_mapas import DatosMapas
    from ecowaste.poligonos import ContornosTACO
    raiz=Path(raiz);h=sha(raiz/'datos/procesados/deteccion/train.jsonl')
    if not (raiz/f'datos/cache/deteccion_384_train_{h[:12]}.pt').exists():raise FileNotFoundError('Falta caché histórica train; no crear/escribir históricos')
    ds=DatosMapas('train',ContornosTACO(raiz,1/64))
    return SimpleNamespace(imagenes=ds.base.imagenes,mascaras=ds.mascaras,
        objetivos=ds.base.objetivos,registros=ds.base.origen.registros)


def auxiliar_global_train(raiz):
    """Leer caché train validada sin instanciar el lector que inspecciona test."""
    raiz=Path(raiz);path=raiz/'datos/procesados/clasificacion/train.jsonl';h=sha(path)
    folder=raiz/f'datos/cache/clasificacion_auxiliar_{h[:12]}';meta=leer(folder/'metadata.json')
    if not meta['aprobada'] or meta['sha256_manifiesto']!=h or sha(folder/'imagenes.npy')!=meta['sha256_imagenes']:
        raise ValueError('Caché train auxiliar inválida')
    rows=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    return SimpleNamespace(imagenes=np.load(folder/'imagenes.npy',mmap_mode='r'),
        etiquetas=torch.tensor([CLASES.index(r['clase']) for r in rows]),ids=[r['id'] for r in rows])
