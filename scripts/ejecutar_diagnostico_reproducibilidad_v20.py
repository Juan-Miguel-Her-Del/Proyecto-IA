"""Orquesta procesos cortos C1/C2/R y compara todos los estados; train-only."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.reanudacion_regional import publicar,sha
from ecowaste.reproducibilidad_v20 import comparar_arbol,comparar_envolvente,cpu
from scripts.probar_entrenamiento_v20 import CONFIG,validar_diagnostico
from scripts.preparar_v20 import huellas_codigo

ETAPAS=['pre.pesos','pre.adamw','pre.rng_inicio','pre.entradas','pre.logits_taco','pre.losses',
    'pre.gradientes_taco_escalados','pre.logits_auxiliar','pre.gradientes_total_escalados',
    'pre.gradientes_sin_escala','pre.norma_antes_clip','pre.gradientes_clip','post.pesos','post.adamw','post.scaler','post.rng_fin']


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def resumen_comparacion(c):
    r={k:v for k,v in c.items() if k!='filas'};r['etapas']={}
    for e in ETAPAS:
        ff=[f for f in c['filas'] if f['nombre']==e or f['nombre'].startswith(e+'.') or f['nombre'].startswith(e+'[')]
        r['etapas'][e]={'exacta':all(f['exacto'] for f in ff),'fuera_tolerancia':sum(f['fuera_tolerancia'] for f in ff),
            'max_delta':max((f.get('max_diferencia_absoluta') or 0. for f in ff),default=0.),
            'max_error_normalizado':max((f.get('max_error_normalizado') or 0. for f in ff),default=0.)}
    r['primera_etapa_no_exacta']=next((e for e in ETAPAS if not r['etapas'][e]['exacta']),None)
    return r


def cargar_paso(carpeta,k):
    return {e:torch.load(carpeta/f'paso_{k:02d}_{e}.pt',map_location='cpu',weights_only=True) for e in ['pre','post']}


def replay_adam(pre,serial=False):
    """Mismos pesos, gradientes y momentos: elimina forward/backward del ensayo."""
    nombres=[n for g in pre['adamw']['grupos'] for n in g['params']]
    ps={n:torch.nn.Parameter(pre['pesos'][n].cuda().clone()) for n in nombres}
    grupos=[];indices={n:i for i,n in enumerate(nombres)}
    for g in pre['adamw']['grupos']:
        op={k:copy.deepcopy(v) for k,v in g.items() if k!='params'}
        if serial:op.update(foreach=False,fused=False)
        grupos.append({**op,'params':[ps[n] for n in g['params']]})
    opt=torch.optim.AdamW(grupos)
    state={'state':{indices[n]:cpu(v) for n,v in pre['adamw']['estado'].items()},
        'param_groups':[{**{k:copy.deepcopy(v) for k,v in g.items() if k!='params'},
            'params':[indices[n] for n in g['params']]} for g in pre['adamw']['grupos']]}
    opt.load_state_dict(state)
    if serial:
        for g in opt.param_groups:g.update(foreach=False,fused=False)
    for n,p in ps.items():p.grad=pre['gradientes_clip'][n].cuda().clone() if pre['gradientes_clip'][n] is not None else None
    opt.step()
    resultado={'pesos':{n:cpu(p) for n,p in ps.items()},'momentos':{n:cpu(opt.state[p]) for n,p in ps.items()}}
    del ps,opt;torch.cuda.empty_cache();return resultado


def ejecutar():
    cfg,p,d=validar_diagnostico();salida=RAIZ/p['salida']
    if (salida/'protocolo_ejecutado.json').exists():raise FileExistsError('Diagnóstico existente; no repetir ni sobrescribir')
    publicar(salida/'protocolo_ejecutado.json',{'protocolo':p,'hashes_codigo':huellas_codigo(),
        'sha256_configuracion_diagnostica':sha(CONFIG),'sha256_diagnostico':sha(RAIZ/p['diagnostico_regenerado']),
        'AB_habilitado':False,'fijado_antes_de_las_trayectorias':True})
    inicio=time.perf_counter();procesos=[]
    def correr(args,nombre,modo='historico'):
        env=os.environ.copy()
        if modo=='determinista':env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        log=salida/'logs'/f'{nombre}.txt';log.parent.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter()
        with log.open('x',encoding='utf-8') as f:
            proc=subprocess.run([sys.executable,str(RAIZ/'scripts/probar_entrenamiento_v20.py'),*args],cwd=RAIZ,env=env,stdout=f,stderr=subprocess.STDOUT)
        procesos.append({'nombre':nombre,'argumentos':args,'codigo_salida':proc.returncode,'segundos':time.perf_counter()-start,
            'log':log.relative_to(RAIZ).as_posix(),'sha256_log':sha(log)})
        publicar(salida/'procesos'/f'{nombre}.json',procesos[-1]);print(nombre,'salida',proc.returncode,flush=True)
        return proc.returncode==0
    if not correr(['--crear-plan'],'plan'):raise RuntimeError('No se pudo preparar plan')
    for modo in p['modos']:
        for rama in p['ramas']:
            for t in p['trayectorias']:
                ok=correr(['--modo',modo,'--rama',rama,'--trayectoria',t],f'{modo}_{rama}_{t}_inicio',modo)
                if t=='R' and ok:correr(['--modo',modo,'--rama',rama,'--trayectoria',t,'--fase','reanudar'],f'{modo}_{rama}_{t}_reanudar',modo)
    publicar(salida/'procesos.json',procesos)
    resumen={}
    for modo in p['modos']:
        resumen[modo]={}
        for rama in p['ramas']:
            out=salida/'comparaciones'/modo/rama;filas=[]
            for k in range(1,5):
                try:ss={t:cargar_paso(salida/modo/rama/t,k) for t in p['trayectorias']}
                except FileNotFoundError:
                    filas.append({'paso':k,'incompleto':True});continue
                cc=comparar_arbol(ss['C1'],ss['C2']);cr=comparar_arbol(ss['C1'],ss['R'])
                publicar(out/f'paso_{k:02d}_C1_C2.json',cc);publicar(out/f'paso_{k:02d}_C1_R.json',cr)
                filas.append({'paso':k,'C1_C2':resumen_comparacion(cc),'C1_R':resumen_comparacion(cr),
                    'envolvente':comparar_envolvente(cc,cr)})
            restore=salida/modo/rama/'R/restauracion.json'
            restauracion=leer(restore)['exacta'] if restore.exists() else False
            entradas_exactas=all(not f.get('incompleto') and all(f[c]['etapas'][et]['exacta'] for c in ['C1_C2','C1_R'] for et in ['pre.entradas','pre.rng_inicio','post.rng_fin']) for f in filas)
            resumen[modo][rama]={'pasos':filas,'restauracion_exacta':restauracion,'entradas_y_RNG_exactos':entradas_exactas,
                'sin_divergencia_adicional_segun_envolvente':restauracion and entradas_exactas and all(f.get('envolvente',{}).get('cumple',False) for f in filas),
                'todos_dentro_tolerancia_original':restauracion and entradas_exactas and all(not f.get('incompleto') and f['C1_C2']['dentro_tolerancia'] and f['C1_R']['dentro_tolerancia'] for f in filas)}
            publicar(out/'resumen.json',resumen[modo][rama])
            print('Comparación',modo,rama,'restauración',restauracion,flush=True)
    replays={}
    for rama in p['ramas']:
        s=cargar_paso(salida/'historico'/rama/'C1',1);pre=s['pre'];r1=replay_adam(pre);r2=replay_adam(pre);rs=replay_adam(pre,serial=True)
        esperado={'pesos':{n:s['post']['pesos'][n] for n in r1['pesos']},'momentos':s['post']['adamw']['estado']}
        replays[rama]={}
        for nombre,x,y in [('dos_replays_default',r1,r2),('replay_frente_actual',r1,esperado),('serial_frente_default',rs,r1)]:
            c=comparar_arbol(x,y);publicar(salida/'replay_adam'/f'{rama}_{nombre}.json',c);replays[rama][nombre]=resumen_comparacion(c)
    publicar(salida/'resumen.json',{'estado':'diagnostico_numerico_completo','modos':resumen,'replay_adam':replays,
        'procesos_con_error':[x for x in procesos if x['codigo_salida']!=0],
        'actualizaciones_trayectorias':sum(1 for f in salida.glob('*/*/*/paso_*_post.pt')),
        'actualizaciones_replay_adam':6,'segundos':time.perf_counter()-inicio,'hashes_codigo':huellas_codigo(),
        'prueba_corta_original_aprobada':False,'AB_habilitado':False,'validacion_evaluada':False,'test_evaluado':False})


if __name__=='__main__':ejecutar()
