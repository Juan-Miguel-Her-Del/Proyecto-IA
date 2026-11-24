"""Validación prefijada y evaluación final de V31; jamás test."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,sys,time
from pathlib import Path
from types import SimpleNamespace
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
import torch
from PIL import Image
from ecowaste.experimento_v31 import OUT,contexto,contrato,comprobar
from ecowaste.detector_instancias_v31 import cargar
from ecowaste.detector_consultas import postprocesar_consultas
from ecowaste.detector_hibrido import mascara_letterbox
from ecowaste.datasets import letterbox,geometria_letterbox
from ecowaste.mosaico import ventanas,combinar_vistas
from ecowaste.backend_numerico import aplicar_backend,politica,registro_efectivo
from ecowaste.ejecucion_detector import CacheDeteccion
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.evaluacion_v28_2 import geometria_posthoc
from ecowaste.datos_v29 import cargar_coleccion
from ecowaste.ablacion_fondo import huella,resumir_deteccion
from ecowaste.reanudacion_regional import publicar
from ecowaste.incorporacion import leer,sha


def checkpoint(rama,k,c):
    ev=leer(OUT/f'entrenamiento/{rama}/confirmaciones/{k:04d}.json');p=RAIZ/ev['checkpoint']
    if sha(p)!=ev['sha256']:raise ValueError('Checkpoint alterado')
    m,ck=cargar(p,'cuda');comprobar(ck,c,rama)
    if ck['paso']!=k:raise ValueError('Paso distinto')
    return m,ck


@torch.inference_mode()
def capturar(m,imagenes,meta,folder,vistas):
    if any(set(r)!={'id','ruta','ancho','alto','sha256'} for r in meta):raise ValueError('Inferencia solo recibe metadatos, no GT')
    key={'pesos':huella(m.state_dict()),'imagenes':huella(imagenes),'meta':huella(meta),'vistas':vistas,'backend':registro_efectivo()}
    if (folder/'captura.json').exists():
        cert=leer(folder/'captura.json')
        if cert['clave']!=key:raise ValueError('Captura de otro protocolo')
        for p,h in cert['archivos'].items():
            if sha(folder/p)!=h:raise ValueError('Captura alterada')
        return cert
    archivos={};costes={}
    for v in vistas:
        preds=[];inicio=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for k in range(0,len(meta),8 if v==1 else 1):
            ms=meta[k:k+8] if v==1 else [meta[k]];p=folder/f'raw_{v}v/{k:04d}.pt'
            if p.exists():
                r=torch.load(p,weights_only=True)
                if r['clave']!=key or r['meta']!=ms:raise ValueError('Parcial ajeno')
                s={n:t.cuda() for n,t in r['salida'].items()};gs=r['geometrias'];zs=r['zonas']
            else:
                if v==1:
                    x=imagenes[k:k+8].cuda().float()/255;gs=[geometria_letterbox(r['ancho'],r['alto'],384) for r in ms];zs=None
                else:
                    r=ms[0]
                    if sha(RAIZ/r['ruta'])!=r['sha256']:raise ValueError('Fotografía modificada')
                    with Image.open(RAIZ/r['ruta']) as im:im=im.convert('RGB')
                    zs=ventanas(im.width,im.height);xgs=[letterbox(im.crop(z),384) for z in zs]
                    x=torch.stack([x for x,g in xgs]).cuda();gs=[g for x,g in xgs]
                with torch.autocast('cuda',dtype=torch.float16):s=m(x,mascara_letterbox(gs,'cuda'))
                publicar(p,{'clave':key,'meta':ms,'salida':{n:t.cpu() for n,t in s.items()},'geometrias':gs,'zonas':zs},tensor=True)
            archivos[p.relative_to(folder).as_posix()]=sha(p)
            ps=postprocesar_consultas(s,gs,.05,True)
            if v==1:preds.extend(ps)
            else:preds.append(combinar_vistas(ps,gs,zs,ms[0]['ancho'],ms[0]['alto'],384,.05))
            if k+len(ms)==len(meta) or (k+len(ms))%100==0:print(folder.parent.name,folder.name,v,k+len(ms),'/',len(meta),flush=True)
        serial=[{'id':r['id'],**{n:t.tolist() for n,t in p.items()}} for r,p in zip(meta,preds)];p=folder/f'predicciones_{v}v.json'
        if p.exists():
            if leer(p)!=serial:raise ValueError('Predicciones no reproducidas')
        else:publicar(p,serial)
        archivos[p.name]=sha(p);torch.cuda.synchronize()
        costes[v]={'segundos_incluye_E_S_y_postproceso':time.perf_counter()-inicio,'pico_gpu_bytes':torch.cuda.max_memory_allocated()}
    cert={'clave':key,'archivos':archivos,'costes':costes,'gt_utilizado':False};publicar(folder/'captura.json',cert);return cert


def analizar(folder,ds,vistas):
    res={}
    for v in vistas:
        rows=leer(folder/f'predicciones_{v}v.json')
        if [p['id'] for p in rows]!=[r['id'] for r in ds.registros]:raise ValueError('Correspondencia de imágenes incorrecta')
        ps=[{'boxes':torch.tensor(r['boxes'],dtype=torch.float32).reshape(-1,4),'scores':torch.tensor(r['scores']),
             'labels':torch.tensor(r['labels'],dtype=torch.long)} for r in rows]
        detalle=evaluar_detecciones(ps,ds.objetivos)
        res[v]={'metricas':resumir_deteccion(detalle),'detalle':detalle,'geometria':geometria_posthoc(ps,ds.objetivos)}
    return res


def meta(ds):
    return [{**{k:r[k] for k in ['id','ruta','ancho','alto']},'sha256':r.get('sha256_archivo') or sha(RAIZ/r['ruta'])} for r in ds.registros]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paso',type=int,choices=[800,1600,2400],required=True);p.add_argument('--rama',choices=['A','B'],required=True);a=p.parse_args()
    cfg,ht,nu,gl,pm=contexto();del gl,pm
    plan=torch.load(OUT/'plan.pt',weights_only=True);c=contrato(cfg,plan)
    if c!=leer(OUT/'protocolo.json'):raise ValueError('Código/protocolo distintos')
    va0=CacheDeteccion(RAIZ,'val',.1);va=SimpleNamespace(imagenes=va0.imagenes,objetivos=va0.objetivos,registros=va0.origen.registros)
    if a.paso==2400:
        ret=cargar_coleccion(RAIZ,RAIZ/cfg['coleccion'],'val')
        dominios={'historico_train':(ht,[1]),'nuevo_train':(nu,[1]),'desarrollo':(va,[1,5]),'externo_retenido':(ret,[1,5])}
    else:dominios={'desarrollo':(va,[1])}
    res={}
    with aplicar_backend(politica('completo')):
        m,ck=checkpoint(a.rama,a.paso,c)
        for dom,(ds,vistas) in dominios.items():
            folder=OUT/f'evaluacion/{a.rama}/{a.paso:04d}/{dom}';cert=capturar(m,ds.imagenes,meta(ds),folder,vistas)
            q=folder/'analisis.json'
            if q.exists():an=leer(q)
            else:an=analizar(folder,ds,vistas);publicar(q,an)
            res[dom]={'analisis':q.relative_to(RAIZ).as_posix(),'sha256':sha(q),'metricas':{str(v):an[str(v) if str(v) in an else v]['metricas'] for v in vistas},'costes':cert['costes']}
            print(a.rama,a.paso,dom,res[dom]['metricas'],flush=True)
        publicar(OUT/f'evaluacion/{a.rama}/{a.paso:04d}/resumen.json',{'rama':a.rama,'paso':a.paso,'resultados':res,'backend':registro_efectivo(),'test':False,'adopcion':False})


if __name__=='__main__':main()
