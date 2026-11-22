"""V25: inferencia congelada train, sin GT; val se reutiliza, nunca test."""
import argparse
import json
from pathlib import Path
import sys
import time
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from PIL import Image
import torch
from ecowaste.datasets import letterbox,geometria_letterbox
from ecowaste.detector_mapas import cargar_detector_mapas,cajas_desde_mapas
from ecowaste.mosaico import ventanas
from ecowaste.traza_propuestas import trazar_logits
from ecowaste.backend_numerico import registro_efectivo
from ecowaste.reanudacion_regional import publicar,sha,huella
from scripts.auditar_propuestas_internas import serializar_traza

SALIDA=RAIZ/'resultados/v25_datos_representacion'
CHECKPOINT='checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt'
SHA='eb8fd00ee536ead21e3a2a0ecd5b06baad04acb2e437b22e0eac3686a7bf36b9'


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def metadatos_sin_gt(registros):
    return [{k:r[k] for k in ['id','ruta','ancho','alto','sha256_archivo']} for r in registros]


def verificar_metadatos(ms):
    if any(set(m)!={'id','ruta','ancho','alto','sha256_archivo'} for m in ms):
        raise ValueError('La captura solo admite procedencia y geometria de imagen, nunca GT')


def preparar():
    if SALIDA.exists():raise FileExistsError('Conservar auditoria anterior; no sobrescribir')
    SALIDA.mkdir(parents=True)
    archivos=[]
    for carpeta,patron in [('ecowaste','*.py'),('scripts','*.py'),('pruebas','*.py'),('configuracion','*.json'),
        ('documentacion','*.md'),('resultados','*.json'),('datos/procesados','*.jsonl')]:
        archivos.extend(p for p in (RAIZ/carpeta).rglob(patron) if not p.is_relative_to(SALIDA))
    archivos.extend([RAIZ/'README.md',RAIZ/CHECKPOINT,RAIZ/'datos/originales/taco/annotations.json'])
    publicar(SALIDA/'protegidos.json',{p.relative_to(RAIZ).as_posix():sha(p) for p in archivos})
    publicar(SALIDA/'checkpoints_estadisticas.json',{p.relative_to(RAIZ).as_posix():[p.stat().st_size,p.stat().st_mtime_ns]
        for p in (RAIZ/'checkpoints').rglob('*.pt')})
    publicar(SALIDA/'cierre_v20.json',{'decision':'cerrado_no_adoptado','informe':'documentacion/24_resultados_v20_backend_determinista.md',
        'motivo':'AP50 carton y vidrio cero; mAP50 inferior a la referencia original',
        'checkpoint_predeterminado':CHECKPOINT,'sha256':SHA,'nuevos_entrenamientos_autorizados':False,
        'test_evaluado':False,'configuracion_original_sha256':sha(RAIZ/'configuracion/arquitectura.json')})


@torch.inference_mode()
def capturar(ms,imagenes):
    verificar_metadatos(ms)
    if sha(RAIZ/CHECKPOINT)!=SHA:raise ValueError('Checkpoint original alterado')
    entorno=registro_efectivo()
    historico=leer(RAIZ/'resultados/v22_reproducibilidad_v20/historico/A/C1/entorno_inicio.json')
    for k in ['torch','cuda','cudnn','gpu','CUBLAS_WORKSPACE_CONFIG','deterministic_algorithms','cudnn_deterministic',
        'cudnn_benchmark','tf32_matmul','tf32_cudnn','float32_matmul_precision','sdpa','threads','threads_interop']:
        if entorno[k]!=historico[k]:raise ValueError('Backend historico diferente: '+k)
    destino=SALIDA/'captura_train'
    if destino.exists():raise FileExistsError('Captura anterior no se reemplaza')
    destino.mkdir()
    m,ck=cargar_detector_mapas(RAIZ/CHECKPOINT,'cuda');m.requires_grad_(False)
    if ck['epoca']!=19:raise ValueError('Epoca inesperada')
    del ck
    estado=huella(m.state_dict());costes={};archivos=[];preds={}
    for nv in [1,5]:
        torch.cuda.reset_peak_memory_stats();inicio=time.perf_counter();ps=[]
        if nv==1:
            for k in range(0,len(ms),8):
                lote=ms[k:k+8];gs=[geometria_letterbox(x['ancho'],x['alto'],384) for x in lote]
                with torch.autocast('cuda',dtype=torch.float16):raw=m(imagenes[k:k+8].cuda().float()/255)
                ps.extend(cajas_desde_mapas(raw,.05,2,gs,politica='argmax7'))
                p=destino/'logits_1v'/f'{k:03d}.pt'
                publicar(p,{'metadatos':lote,'logits':raw.cpu(),'geometrias':gs},tensor=True);archivos.append(p)
        else:
            for i,meta in enumerate(ms):
                if sha(RAIZ/meta['ruta'])!=meta['sha256_archivo']:raise ValueError('Imagen modificada')
                with Image.open(RAIZ/meta['ruta']) as im:im=im.convert('RGB')
                zs=ventanas(im.width,im.height);xs=[letterbox(im.crop(z),384) for z in zs];gs=[g for _,g in xs]
                with torch.autocast('cuda',dtype=torch.float16):raw=m(torch.stack([x for x,_ in xs]).cuda())
                t=trazar_logits(raw,gs,zs,im.width,im.height,meta['id']);ps.append(t['final'])
                for sub,obj in [('logits_5v',{'metadatos':meta,'logits':raw.cpu(),'geometrias':gs,'ventanas':zs}),
                    ('trazas_5v',serializar_traza(t))]:
                    p=destino/sub/f'{i:03d}.pt';publicar(p,obj,tensor=True);archivos.append(p)
                if (i+1)%40==0 or i+1==len(ms):print(f'Train sin GT: {i+1}/{len(ms)} escenas, cinco vistas',flush=True)
        torch.cuda.synchronize();costes[str(nv)]={'segundos_incluye_observador_y_archivos':time.perf_counter()-inicio,
            'memoria_gpu_allocated_max':torch.cuda.max_memory_allocated(),'memoria_gpu_reserved_max':torch.cuda.max_memory_reserved()}
        pp=[{'id':meta['id'],**{k:v.tolist() for k,v in p.items()}} for meta,p in zip(ms,ps)]
        p=destino/f'predicciones_{nv}v_train.json';publicar(p,pp);archivos.append(p);preds[str(nv)]=len(pp)
    if huella(m.state_dict())!=estado or any(p.grad is not None for p in m.parameters()):raise AssertionError('Modelo alterado')
    publicar(destino/'captura.json',{'checkpoint_sha256':SHA,'epoca':19,'imagenes':len(ms),'vistas':[1,5],
        'backend':entorno,'costes':costes,'regresion_observador_exacta':len(ms),'gt_utilizado':False,
        'modelo_antes_despues_identico':True,'sin_gradientes':True,'entrenamiento':False,'test_evaluado':False,
        'sha256_manifiesto_train':sha(RAIZ/'datos/procesados/deteccion/train.jsonl'),
        'archivos':{p.relative_to(SALIDA).as_posix():sha(p) for p in archivos},
        'codigo':{p.relative_to(RAIZ).as_posix():sha(p) for p in [Path(__file__),*list((RAIZ/'ecowaste').glob('*.py')),
            RAIZ/'scripts/auditar_propuestas_internas.py']}})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preparar',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    if a.preparar:preparar()
    registros=[json.loads(l) for l in (RAIZ/'datos/procesados/deteccion/train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    if len(registros)!=964 or any(r['split']!='train' or r['fuente']!='taco' for r in registros):raise ValueError('Train inesperado')
    ms=metadatos_sin_gt(registros);del registros
    hm=sha(RAIZ/'datos/procesados/deteccion/train.jsonl')
    cache=torch.load(RAIZ/f'datos/cache/deteccion_384_train_{hm[:12]}.pt',map_location='cpu',weights_only=True)
    if cache['sha256_manifiesto']!=hm:raise ValueError('Cache de imagenes incompatible')
    capturar(ms,cache['imagenes'])


if __name__=='__main__':main()
