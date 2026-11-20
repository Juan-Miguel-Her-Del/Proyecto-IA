"""Construye y audita una caché multiescala exclusivamente de entrenamiento."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import numpy as np
from PIL import Image,ImageDraw
import torch
from ecowaste.datasets import CLASES,letterbox,ResiduosDataset
from ecowaste.poligonos import ContornosTACO
from ecowaste.recortes import VERSION,planificar,objetivo_recorte
from scripts.entrenar_clasificador import escribir_json


def sha_archivo(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--carpeta',default='datos/cache/recortes_multiescala_v1')
    a=p.parse_args()
    destino=(RAIZ/a.carpeta).resolve()
    if not destino.is_relative_to((RAIZ/'datos/cache').resolve()):raise ValueError('Destino fuera de caché')
    if destino.exists():raise FileExistsError('Caché existente; usar otro nombre')
    torch.set_num_threads(4)
    inicio=time.perf_counter()
    ds=ResiduosDataset(RAIZ,'deteccion','train')
    c=ContornosTACO(RAIZ,1/64)
    plan=planificar(ds.registros)
    hashes={'train':sha_archivo(RAIZ/'datos/procesados/deteccion/train.jsonl'),
        'anotaciones_poligonos':c.sha256,'inventario_procedencia':c.sha256_inventario}
    validacion=json.loads((RAIZ/'datos/reportes/validacion.json').read_text(encoding='utf-8'))
    if not validacion['validacion_estructural_correcta'] or hashes['train']!=validacion['sha256_manifiestos']['datos/procesados/deteccion/train.jsonl']:
        raise ValueError('Los datos train no coinciden con la auditoría')
    otros={}
    for split in ['val','test']:
        # Solo metadatos para comprobar separación; no se cargan imágenes/objetivos.
        registros=ResiduosDataset(RAIZ,'deteccion',split).registros
        otros[split]={r['grupo'] for r in registros}
        if {r['grupo'] for r in ds.registros}&otros[split]:raise ValueError('Fuga entre grupos')
    necesario=len(plan)*(3*384*384+48*48)
    if shutil.disk_usage(RAIZ).free<necesario+512*1024**2:raise OSError('Espacio insuficiente para caché')
    destino.mkdir(parents=True)
    escribir_json(destino/'metadata.json',{'version':VERSION,'aprobada':False,'hashes_origen':hashes})
    imagenes=np.lib.format.open_memmap(destino/'imagenes.npy',mode='w+',dtype='uint8',shape=(len(plan),3,384,384))
    mascaras=np.lib.format.open_memmap(destino/'mascaras.npy',mode='w+',dtype='uint8',shape=(len(plan),48,48))
    conteos=torch.zeros(7,dtype=torch.long);sin_ancla=[];visibles=0
    imagen=None;indice_previo=None;muestras=[]
    for j,t in enumerate(plan):
        i=t['indice_registro'];r=ds.registros[i]
        if i!=indice_previo:
            contenido=(RAIZ/r['ruta']).read_bytes()
            if hashlib.sha256(contenido).hexdigest()!=r['sha256_archivo']:raise ValueError(f"Imagen alterada: {r['id']}")
            if imagen is not None:imagen.close()
            with Image.open(io.BytesIO(contenido)) as im:imagen=im.convert('RGB')
            if imagen.size!=(r['ancho'],r['alto']):raise ValueError('Dimensiones incompatibles')
            indice_previo=i
        o,y=objetivo_recorte(r,t['ventana_xyxy'],c)
        x,g=letterbox(imagen.crop(t['ventana_xyxy']),384)
        if o['geometria']!=g:raise ValueError('Imagen y máscaras con geometría distinta')
        if not bool(torch.isfinite(o['boxes']).all()) or not bool(((o['boxes']>=0)&(o['boxes']<=1)).all()):
            raise ValueError('Cajas recortadas inválidas')
        if t['ancla_id_interno'] not in o['ids_anotaciones_visibles']:raise ValueError('Se perdió el ancla')
        celdas_ancla=int((y==CLASES.index(t['clase_ancla'])+1).sum())
        if not celdas_ancla:sin_ancla.append(t['id'])
        t['celdas_material_ancla']=celdas_ancla
        t['anotaciones_visibles']=o['ids_anotaciones_visibles']
        imagenes[j]=(x*255).round().to(torch.uint8).numpy()
        mascaras[j]=y.numpy().astype('uint8')
        conteos+=torch.bincount(y[y!=255],minlength=7)
        visibles+=len(o['ids_anotaciones_visibles'])
        if len(muestras)<6 and (not muestras or t['clase_ancla'] not in [s[0] for s in muestras]):
            muestras.append((t['clase_ancla'],Image.fromarray(imagenes[j].transpose(1,2,0)),y.clone(),t['id']))
        if (j+1)%250==0:print(f'Recortes {j+1}/{len(plan)}',flush=True)
    if imagen is not None:imagen.close()
    imagenes.flush();mascaras.flush()
    (destino/'plan.jsonl').write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in plan),encoding='utf-8')
    # Leer el resultado completo en bloques para verificar rangos y formas.
    for j in range(0,len(plan),128):
        valores=np.unique(mascaras[j:j+128])
        if not np.isin(valores,[0,1,2,3,4,5,6,255]).all():raise ValueError('Etiquetas inválidas en disco')
    resumen={'version':VERSION,'aprobada':True,'hashes_origen':hashes,'cobertura_minima':1/64,
        'sha256_archivos':{n:sha_archivo(destino/n) for n in ['plan.jsonl','imagenes.npy','mascaras.npy']},
        'recortes':len(plan),'imagenes_fuente':len(ds),'anclas_unicas':len({t['ancla_id_interno'] for t in plan}),
        'factores_contexto':[2.,3.5],'anclas_por_material':dict(Counter(t['clase_ancla'] for t in plan)),
        'ventanas_unicas':len({(t['id_fuente'],tuple(t['ventana_xyxy'])) for t in plan}),
        'celdas_fondo_y_clases':conteos.tolist(),'anotaciones_visibles_acumuladas':visibles,
        'recortes_sin_celdas_del_material_ancla':sin_ancla,'grupos_val_test_disjuntos':True,
        'tamano_menor_ancla_mediana_antes_px':float(np.median([t['lado_menor_ancla_384_antes'] for t in plan])),
        'tamano_menor_ancla_mediana_despues_px':float(np.median([t['lado_menor_ancla_384_despues'] for t in plan])),
        'bytes_imagenes_y_mascaras':necesario,'duracion_segundos':time.perf_counter()-inicio,'test_evaluado':False,
        'limites':['Son vistas de las mismas 2.505 anotaciones, no objetos nuevos ni mayor diversidad orgánica.',
            'Anclas ocluidas por solapes pueden perder su material semántico; se registran sin borrar el recorte.',
            'No se usan métricas de recortes para declarar calidad; val se evalúa completo.']}
    escribir_json(destino/'metadata.json',resumen)
    escribir_json(RAIZ/'datos/reportes/recortes_multiescala.json',dict(resumen,carpeta=destino.relative_to(RAIZ).as_posix()))
    panel=Image.new('RGB',(768,240*len(muestras)),'white')
    colores=np.array([[120,120,120],[255,80,80],[80,170,255],[200,150,60],[80,230,150],[210,120,255],[255,200,60]],dtype=np.uint8)
    for i,(clase,im,y,identificador) in enumerate(muestras):
        a=np.zeros((48,48,3),dtype=np.uint8);valid=y.numpy()!=255;a[valid]=colores[y.numpy()[valid]]
        capa=Image.fromarray(a).resize((384,384),Image.Resampling.NEAREST)
        panel.paste(im.resize((224,224)),(0,i*240+16))
        panel.paste(Image.blend(im,capa,.45).resize((224,224)),(224,i*240+16))
        ImageDraw.Draw(panel).text((0,i*240),identificador+' '+clase,fill='black')
    panel.crop((0,0,448,panel.height)).save(RAIZ/'datos/reportes/recortes_multiescala.png')
    print(json.dumps({k:resumen[k] for k in ['recortes','anclas_unicas','tamano_menor_ancla_mediana_antes_px','tamano_menor_ancla_mediana_despues_px']}),flush=True)


if __name__=='__main__':main()
