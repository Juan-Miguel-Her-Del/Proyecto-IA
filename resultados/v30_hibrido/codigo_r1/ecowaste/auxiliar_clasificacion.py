"""Clasificación auxiliar RealWaste: etiquetas de imagen, nunca cajas inventadas."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from ecowaste.datasets import CLASES,ResiduosDataset,letterbox

VERSION='clasificacion_auxiliar_384_uint8_train_v1'


def sha_archivo(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


class DatosClasificacionAuxiliar(Dataset):
    def __init__(self,raiz):
        raiz=Path(raiz);origen=ResiduosDataset(raiz,'clasificacion','train')
        sha=sha_archivo(raiz/'datos/procesados/clasificacion/train.jsonl')
        valid=json.loads((raiz/'datos/reportes/validacion.json').read_text(encoding='utf-8'))
        if sha!=valid['sha256_manifiestos']['datos/procesados/clasificacion/train.jsonl']:
            raise ValueError('Manifiesto auxiliar distinto del validado')
        grupos={r['grupo'] for r in origen.registros}
        for tarea in ['clasificacion','deteccion']:
            for split in ['val','test']:
                retenidos=ResiduosDataset(raiz,tarea,split).registros
                if grupos&{r['grupo'] for r in retenidos}:raise ValueError('Fuga en la tarea auxiliar')
        path=raiz/'datos/cache'/f'clasificacion_auxiliar_{sha[:12]}'
        meta_path=path/'metadata.json';archivo=path/'imagenes.npy'
        if path.exists():
            meta=json.loads(meta_path.read_text(encoding='utf-8'))
            if (not meta['aprobada'] or meta['version']!=VERSION or meta['sha256_manifiesto']!=sha
                    or meta['sha256_imagenes']!=sha_archivo(archivo)):raise ValueError('Caché auxiliar no válida')
        else:
            if shutil.disk_usage(raiz).free<len(origen)*3*384*384+256*1024**2:raise OSError('Espacio insuficiente')
            path.mkdir(parents=True)
            meta_path.write_text(json.dumps({'aprobada':False}),encoding='utf-8')
            array=np.lib.format.open_memmap(archivo,mode='w+',dtype='uint8',shape=(len(origen),3,384,384))
            def leer(i):
                r=origen.registros[i];contenido=(raiz/r['ruta']).read_bytes()
                if hashlib.sha256(contenido).hexdigest()!=r['sha256_archivo']:raise ValueError('Imagen auxiliar alterada')
                with Image.open(io.BytesIO(contenido)) as im:x,_=letterbox(im.convert('RGB'),384)
                return (x*255).round().to(torch.uint8).numpy()
            with ThreadPoolExecutor(max_workers=4) as pool:
                for i,x in enumerate(pool.map(leer,range(len(origen)))):
                    array[i]=x
                    if (i+1)%500==0:print(f'Caché clasificación auxiliar {i+1}/{len(origen)}',flush=True)
            array.flush();del array
            meta={'aprobada':True,'version':VERSION,'sha256_manifiesto':sha,
                'sha256_imagenes':sha_archivo(archivo),'imagenes':len(origen),
                'grupos_disjuntos_val_test':True,'cajas_o_mascaras_generadas':False,'test_evaluado':False}
            meta_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        self.imagenes=np.load(archivo,mmap_mode='r')
        if self.imagenes.shape!=(len(origen),3,384,384) or self.imagenes.dtype!=np.uint8:raise ValueError('Forma auxiliar inválida')
        self.etiquetas=torch.tensor([CLASES.index(r['clase']) for r in origen.registros])
        self.ids=[r['id'] for r in origen.registros]
        self.metadata=meta;self.metadata_path=meta_path

    def __len__(self):return len(self.etiquetas)

    def __getitem__(self,i):return torch.from_numpy(self.imagenes[i].copy()),self.etiquetas[i]
