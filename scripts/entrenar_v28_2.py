"""V28.2: piloto real con aprobación numérica previa; nunca consulta val ni test."""
import os
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in [None,':4096:8']:raise RuntimeError('cuBLAS incompatible')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import sys
import tempfile
import time
import uuid
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.experimento_v28_2 import (SALIDA,CONFIG,CERTIFICADOS,contexto,codigo,crear_plan,restaurar,paso,instantanea,Diario)
from ecowaste.incorporacion import leer,sha,digest
from ecowaste.reanudacion_regional import publicar,bloqueo,huellas_estado
from ecowaste.backend_numerico import aplicar_backend,registro_efectivo,verificar_registro
from ecowaste.ablacion_fondo import huella
from ecowaste.reproducibilidad_v20 import comparar_arbol


def contrato(cfg,plan):
    return {'config':cfg,'codigo':codigo(),'plan_sha256':huella(plan),
        'train_sha256':sha(RAIZ/'datos/procesados/deteccion/train.jsonl'),
        'global_train_sha256':sha(RAIZ/'datos/procesados/clasificacion/train.jsonl'),
        'qa_previa_sha256':sha(SALIDA/'verificacion_previa_datos.json'),
        'seleccion_visual_sha256':sha(SALIDA/'seleccion_visual_prefijada.json')}


def controles():
    cfg,origen,ht,nu,gl,pm=contexto();plan=crear_plan(origen,len(ht.registros),len(nu),gl.etiquetas,cfg)
    qa=leer(SALIDA/'verificacion_previa_datos.json')
    if not qa['aprobada']:raise ValueError('QA de datos no aprobada')
    c=contrato(cfg,plan);publicar(CERTIFICADOS/'plan.pt',plan,tensor=True);publicar(CERTIFICADOS/'protocolo_congelado.json',c)
    ramas={}
    with aplicar_backend(cfg['backend_numerico']):
        for rama in ['A','B']:
            m,opt,sc,rg,ra,rest=restaurar(origen,cfg);continuos=[]
            # Mismo motor real que producción, diagnóstico con copias completas por etapa.
            with tempfile.TemporaryDirectory(prefix='descartar_',dir=SALIDA) as tmp:
                tmp=Path(tmp)
                for k in range(2):
                    tr,detalle=paso(m,opt,sc,ht,nu,gl,pm,plan,rama,k,rg,ra,True);continuos.append((tr,detalle))
                    if k==0:publicar(tmp/'interrumpido.pt',instantanea(origen,m,opt,sc,rg,ra,rama,1,[],c,None),tensor=True)
                del m,opt,sc;torch.cuda.empty_cache()
                base=torch.load(tmp/'interrumpido.pt',map_location='cpu',weights_only=True)
                m,opt,sc,rg,ra,rest_r=restaurar(base,cfg)
                tr,detalle=paso(m,opt,sc,ht,nu,gl,pm,plan,rama,1,rg,ra,True)
                comp=comparar_arbol(continuos[1][1],detalle)
                r={'restauracion_origen_exacta':rest==huellas_estado(origen),
                    'restauracion_intermedia_exacta':rest_r==huellas_estado(base),
                    'comparacion_paso2':comp,'pasos_continuos':[t for t,d in continuos],
                    'paso_reanudado':tr,'pesos_piloto_descartados':True,'backend':registro_efectivo(opt)}
                publicar(CERTIFICADOS/f'control_{rama}.json',r);ramas[rama]=r
                del m,opt,sc;torch.cuda.empty_cache()
    aprobada=all(r['restauracion_origen_exacta'] and r['restauracion_intermedia_exacta'] and
        r['comparacion_paso2']['dentro_tolerancia'] for r in ramas.values())
    publicar(CERTIFICADOS/'controles.json',{'aprobada':aprobada,'contrato':c,'ramas':{r:sha(CERTIFICADOS/f'control_{r}.json') for r in ramas},
        'criterio':'restauración exacta; todas las etapas rtol=1e-4 atol=1e-5; no ampliar tolerancia',
        'rtol':1e-4,'atol':1e-5,'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name()})
    if not aprobada:raise RuntimeError('Controles fallidos: no entrenar')
    print('Controles A/B aprobados; pesos diagnósticos descartados',flush=True)


def entrenar(rama,hasta=300):
    if hasta not in [100,200,300]:raise ValueError('Etapa fuera del protocolo')
    cfg,origen,ht,nu,gl,pm=contexto();plan=torch.load(CERTIFICADOS/'plan.pt',map_location='cpu',weights_only=True)
    c=contrato(cfg,plan);d=leer(CERTIFICADOS/'controles.json')
    if not d['aprobada'] or d['contrato']!=c or leer(CERTIFICADOS/'protocolo_congelado.json')!=c:raise ValueError('Controles incompatibles o código cambiado')
    for r,h in d['ramas'].items():
        if sha(CERTIFICADOS/f'control_{r}.json')!=h:raise ValueError('Control alterado')
    diario=Diario(rama,c)
    with bloqueo(diario.base/'ejecucion.lock'),aplicar_backend(cfg['backend_numerico']):
        ck,previo=diario.leer();m,opt,sc,rg,ra,rest=restaurar(ck or origen,cfg)
        if ck:
            verificar_registro(cfg['backend_numerico'],registro_efectivo(opt),ck['v28_2']['backend_efectivo'])
            historial=ck['v28_2']['historial'];k0=ck['v28_2']['paso']
        else:
            historial=[];k0=0
            previo=diario.confirmar(instantanea(origen,m,opt,sc,rg,ra,rama,0,historial,c,previo),previo)
        if k0>=hasta:raise ValueError('Etapa ya completada; conservar resultados')
        intento=uuid.uuid4().hex;inicio=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        publicar(diario.base/'intentos'/f'{intento}_inicio.json',{'paso_inicial':k0,'restauracion_exacta':rest,
            'backend':registro_efectivo(opt),'contrato_sha256':digest(c)})
        for k in range(k0,hasta):
            r,_=paso(m,opt,sc,ht,nu,gl,pm,plan,rama,k,rg,ra)
            tr=diario.base/'trazas'/intento/f'paso_{k+1:04d}.json';publicar(tr,r)
            historial.append({**r,'archivo_traza':tr.relative_to(RAIZ).as_posix(),'sha256_traza':sha(tr)})
            if (k+1)%100==0:
                nuevo=instantanea(origen,m,opt,sc,rg,ra,rama,k+1,historial,c,previo)
                previo=diario.confirmar(nuevo,previo)
            if (k+1)%10==0:print(f'{rama}: {k+1}/300 loss={r["loss_total"]:.6f} AdamW={r["paso_adamw"]}',flush=True)
        publicar(diario.base/'intentos'/f'{intento}_fin.json',{'paso_final':hasta,'actualizaciones_este_intento':hasta-k0,
            'segundos':time.perf_counter()-inicio,'memoria_gpu_max_allocated_bytes':torch.cuda.max_memory_allocated(),
            'memoria_gpu_max_reserved_bytes':torch.cuda.max_memory_reserved(),'ultimo_commit_sha256':previo})
        # Releer toda la cadena y verificar estados, trazas y presupuesto, sin cambiar aliases.
        final,_=diario.leer()
        if final['v28_2']['paso']!=hasta:raise ValueError('Etapa incompleta')
    print('Rama '+rama+' etapa '+str(hasta)+' completa; checkpoint original conservado',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--controles',action='store_true');g.add_argument('--rama',choices=['A','B'])
    p.add_argument('--hasta',type=int,choices=[100,200,300],default=300);a=p.parse_args()
    try:
        if a.controles:controles()
        else:entrenar(a.rama,a.hasta)
    except Exception:
        import traceback
        publicar(SALIDA/'incidencias'/f'{uuid.uuid4().hex}.json',{'error':traceback.format_exc(),'entrenamiento_detenido':True})
        raise


if __name__=='__main__':main()
