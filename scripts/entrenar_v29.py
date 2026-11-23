"""Entrenamiento V29 autorizado: controles reales antes del presupuesto fijo A/B."""
import os
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in [None,':4096:8']:raise RuntimeError('cuBLAS incompatible')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,sys,tempfile,time,uuid
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.experimento_v29 import (SALIDA,CONFIG,codigo,contexto,crear_plan,restaurar,rng_nuevo_inicial,
    contrato,paso,instantanea,Diario)
from ecowaste.incorporacion import leer,sha,digest
from ecowaste.reanudacion_regional import publicar,bloqueo,huellas_estado
from ecowaste.backend_numerico import aplicar_backend,registro_efectivo,verificar_registro
from ecowaste.reproducibilidad_v20 import comparar_arbol
CERT=SALIDA/'controles'


def controles():
    pruebas=leer(sorted((SALIDA/'pruebas').glob('*/pruebas.json'))[-1])
    if not pruebas['aprobada'] or pruebas['codigo']!=codigo():raise ValueError('Suite/codec pendientes o código cambió')
    cfg,origen,ht,nu,gl,pm=contexto()
    qa=leer(RAIZ/cfg['coleccion']/'MANIFIESTO.json')
    if not qa['qa_tecnica_aprobada']:raise ValueError('QA técnica pendiente')
    plan=crear_plan(origen,len(ht.registros),[r['fuente'] for r in nu.registros],gl.etiquetas,cfg)
    c=contrato(cfg,plan);publicar(CERT/'plan.pt',plan,tensor=True);publicar(CERT/'protocolo_congelado.json',c)
    ramas={};comunes={}
    with aplicar_backend(cfg['backend_numerico']):
        for rama in ['A','B']:
            referencias=[];restauraciones=[];comparaciones=[];trazas={}
            with tempfile.TemporaryDirectory(prefix='pesos_diagnostico_descartar_',dir=SALIDA) as td:
                punto=Path(td)/'paso2.pt'
                for trayectoria in ['C1','C2','R']:
                    m,opt,sc,rg,ra,rest=restaurar(origen,cfg);restauraciones.append(rest==huellas_estado(origen))
                    rn=rng_nuevo_inicial(cfg);trs=[]
                    for k in range(4):
                        tr,detalle=paso(m,opt,sc,ht,nu,gl,pm,plan,rama,k,rg,ra,rn,True);trs.append(tr)
                        if trayectoria=='C1':referencias.append(detalle)
                        else:
                            comp=comparar_arbol(referencias[k],detalle)
                            p=CERT/rama/f'{trayectoria}_vs_C1_paso{k+1}.json';publicar(p,comp)
                            comparaciones.append({'trayectoria':trayectoria,'paso':k+1,'exacto':comp['exacto'],
                                'dentro_tolerancia':comp['dentro_tolerancia'],'archivo':p.relative_to(RAIZ).as_posix(),'sha256':sha(p)})
                        if trayectoria=='R' and k==1:
                            ck=instantanea(origen,m,opt,sc,rg,ra,rn,rama,2,trs,c,None);publicar(punto,ck,tensor=True)
                            del m,opt,sc;torch.cuda.empty_cache()
                            ck=torch.load(punto,map_location='cpu',weights_only=True);m,opt,sc,rg,ra,rest=restaurar(ck,cfg)
                            rn=copy.deepcopy(ck['v29']['rng_nuevo']);restauraciones.append(rest==huellas_estado(ck))
                    trazas[trayectoria]=trs
                    del m,opt,sc;torch.cuda.empty_cache()
            referencias.clear();comunes[rama]=trazas['C1']
            r={'rama':rama,'restauraciones_exactas':restauraciones,'comparaciones':comparaciones,
                'trazas':trazas,'rtol':1e-4,'atol':1e-5,'pesos_diagnosticos_descartados':True,
                'aprobada':all(restauraciones) and all(x['dentro_tolerancia'] for x in comparaciones)}
            publicar(CERT/f'control_{rama}.json',r);ramas[rama]=r
    igualdad=all(a['huella_entradas_comunes']==b['huella_entradas_comunes'] and
        a['rng_comun_antes']==b['rng_comun_antes'] and a['rng_comun_despues']==b['rng_comun_despues'] for a,b in zip(comunes['A'],comunes['B']))
    salida={'aprobada':all(r['aprobada'] for r in ramas.values()) and igualdad,'contrato':c,
        'ramas':{r:sha(CERT/f'control_{r}.json') for r in ramas},'entradas_y_rng_historicos_identicos_A_B':igualdad,
        'criterio':'Restauración exacta; C1/C2 y C1/R en cada paso y todas las etapas rtol=1e-4 atol=1e-5; mismos aumentos históricos A/B',
        'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name()}
    publicar(CERT/'controles.json',salida)
    if not salida['aprobada']:raise RuntimeError('Controles fallidos: entrenamiento bloqueado')
    print('Controles reales C1/C2/R A/B aprobados; pesos descartados',flush=True)


def entrenar(rama,hasta):
    cfg,origen,ht,nu,gl,pm=contexto()
    if hasta not in cfg['checkpoints'][1:]:raise ValueError('Presupuesto no autorizado')
    plan=torch.load(CERT/'plan.pt',map_location='cpu',weights_only=True);c=contrato(cfg,plan);cert=leer(CERT/'controles.json')
    if not cert['aprobada'] or cert['contrato']!=c or leer(CERT/'protocolo_congelado.json')!=c:raise ValueError('Certificado incompatible o código cambiado')
    for r,h in cert['ramas'].items():
        if sha(CERT/f'control_{r}.json')!=h:raise ValueError('Certificado alterado')
        for comp in leer(CERT/f'control_{r}.json')['comparaciones']:
            if sha(RAIZ/comp['archivo'])!=comp['sha256']:raise ValueError('Comparación diagnóstica alterada')
    diario=Diario(rama,c)
    with bloqueo(diario.base/'ejecucion.lock'),aplicar_backend(cfg['backend_numerico']):
        ck,previo=diario.leer();m,opt,sc,rg,ra,rest=restaurar(ck or origen,cfg)
        if ck:
            verificar_registro(cfg['backend_numerico'],registro_efectivo(opt),ck['v29']['backend_efectivo'])
            hist=ck['v29']['historial'];k0=ck['v29']['paso'];rn=copy.deepcopy(ck['v29']['rng_nuevo'])
        else:
            hist=[];k0=0;rn=rng_nuevo_inicial(cfg)
            previo=diario.confirmar(instantanea(origen,m,opt,sc,rg,ra,rn,rama,0,hist,c,previo),previo)
        if k0>=hasta:raise ValueError('Etapa ya completada; no sobrescribir')
        intento=uuid.uuid4().hex;inicio=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        publicar(diario.base/'intentos'/f'{intento}_inicio.json',{'paso_inicial':k0,'restauracion_exacta':rest,
            'backend':registro_efectivo(opt),'contrato_sha256':digest(c)})
        for k in range(k0,hasta):
            tr,_=paso(m,opt,sc,ht,nu,gl,pm,plan,rama,k,rg,ra,rn)
            p=diario.base/'trazas'/intento/f'paso_{k+1:04d}.json';publicar(p,tr)
            hist.append({**tr,'archivo_traza':p.relative_to(RAIZ).as_posix(),'sha256_traza':sha(p)})
            if k+1 in cfg['checkpoints']:previo=diario.confirmar(instantanea(origen,m,opt,sc,rg,ra,rn,rama,k+1,hist,c,previo),previo)
            if (k+1)%25==0:print(f'{rama} {k+1}/{cfg["actualizaciones"]} loss={tr["loss_total"]:.6f} AdamW={tr["paso_adamw"]}',flush=True)
        publicar(diario.base/'intentos'/f'{intento}_fin.json',{'paso_final':hasta,'actualizaciones':hasta-k0,
            'segundos':time.perf_counter()-inicio,'memoria_gpu_max_allocated_bytes':torch.cuda.max_memory_allocated(),
            'memoria_gpu_max_reserved_bytes':torch.cuda.max_memory_reserved(),'ultimo_commit_sha256':previo})
        final,_=diario.leer()
        if final['v29']['paso']!=hasta:raise ValueError('Entrenamiento incompleto')
    print('Rama',rama,'completa hasta',hasta,'sin modificar el detector predeterminado',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--controles',action='store_true');g.add_argument('--rama',choices=['A','B'])
    p.add_argument('--hasta',type=int,choices=[100,300,600],default=600);a=p.parse_args()
    try:
        if a.controles:controles()
        else:entrenar(a.rama,a.hasta)
    except Exception:
        import traceback
        publicar(SALIDA/'incidencias'/f'{uuid.uuid4().hex}.json',{'error':traceback.format_exc(),'entrenamiento_detenido':True});raise


if __name__=='__main__':main()
