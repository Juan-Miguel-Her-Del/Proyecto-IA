"""V23: mínimo C1/C2/R, candidato explícito y reanudación real tras 121 lotes."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.backend_numerico import politica,aplicar_backend,configurar_optimizador,registro_efectivo,verificar_registro,VERSION
from ecowaste.reanudacion_regional import publicar,sha,huellas_estado
from ecowaste.reanudacion_v20 import DiarioV20
from ecowaste.reproducibilidad_v20 import cpu,comparar_arbol
from ecowaste.observacion_backend import ObservadorBackend
from ecowaste.ablacion_fondo import restaurar_aleatorio,estado_aleatorio
from scripts.preparar_v20 import leer_cfg,protocolo,huellas_codigo,contexto_train
from scripts.supervision_regional_materiales import restaurar,loaders
from scripts.entrenar_v20 import ejecutar_epoca_train
from scripts.ejecutar_diagnostico_reproducibilidad_v20 import cargar_paso,resumen_comparacion

CONFIG=RAIZ/'configuracion/validacion_backend_v23.json'
ROOT=RAIZ/'resultados/v23_backend_v20'


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def diagnostico():
    d=leer(ROOT/'preparacion_codigo/diagnostico.json');d['sha256_archivo']=sha(ROOT/'preparacion_codigo/diagnostico.json')
    if not d['aprobada'] or d['hashes_codigo']!=huellas_codigo():raise ValueError('Regenerar diagnóstico de código')
    if leer_cfg()['entrenamiento_completo_habilitado']:raise ValueError('A/B debe seguir deshabilitado')
    return d


def ejecutar_proceso(args,nombre,modo='minimo',script='validar_backend_v23.py'):
    env=os.environ.copy()
    if modo=='completo':env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    log=ROOT/'logs'/f'{nombre}.txt';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf-8') as f:
        r=subprocess.run([sys.executable,str(RAIZ/'scripts'/script),*args],cwd=RAIZ,env=env,stdout=f,stderr=subprocess.STDOUT)
    publicar(ROOT/'procesos'/f'{nombre}.json',{'argumentos':args,'codigo_salida':r.returncode,'log':log.relative_to(RAIZ).as_posix(),'sha256':sha(log)})
    print(nombre,'salida',r.returncode,flush=True)
    if r.returncode:raise RuntimeError('Proceso fallido; conservar evidencia: '+nombre)


def corto(modo):
    p=leer(CONFIG);nombre='sdpa_math_cudnn' if modo=='minimo' else 'determinista';res={}
    for rama in p['ramas']:
        for t in p['trayectorias']:
            args=['--config',str(CONFIG),'--modo',nombre,'--rama',rama,'--trayectoria',t]
            ejecutar_proceso(args,f'corto_{modo}_{rama}_{t}',modo,'probar_entrenamiento_v20.py')
            if t=='R':ejecutar_proceso([*args,'--fase','reanudar'],f'corto_{modo}_{rama}_{t}_reanudar',modo,'probar_entrenamiento_v20.py')
        pasos=[]
        for k in range(1,5):
            ss={t:cargar_paso(ROOT/'corto'/nombre/rama/t,k) for t in ['C1','C2','R']};cc={}
            for otro in ['C2','R']:
                c=comparar_arbol(ss['C1'],ss[otro]);publicar(ROOT/'corto_comparaciones'/modo/rama/f'{k:02d}_C1_{otro}.json',c)
                cc[otro]=resumen_comparacion(c)
            pasos.append(cc)
        recarga=leer(ROOT/'corto'/nombre/rama/'R/restauracion.json')['exacta']
        entradas=all(s[t]['etapas'][e]['exacta'] for s in pasos for t in ['C2','R'] for e in ['pre.entradas','pre.rng_inicio','post.rng_fin'])
        res[rama]={'pasos':pasos,'restauracion_exacta':recarga,'entradas_y_RNG_exactos':entradas,
            'aprobada':recarga and entradas and all(s[t]['dentro_tolerancia'] for s in pasos for t in ['C2','R'])}
    publicar(ROOT/f'corto_{modo}.json',{'modo':modo,'aprobada':all(x['aprobada'] for x in res.values()),'ramas':res,'hashes_codigo':huellas_codigo()})
    return all(x['aprobada'] for x in res.values())


def cfg_control(modo,t):
    cfg=leer_cfg();cfg.update(nombre=f'v23_control_{modo}_{t}',version_protocolo=VERSION,
        backend_numerico=politica(modo),diagnostico_reanudacion_larga=True,
        diagnostico='resultados/v23_backend_v20/preparacion_codigo/diagnostico.json')
    return cfg


def worker_largo(modo,rama,t,continuar=False):
    d=diagnostico();cfg=cfg_control(modo,t);ck,train,aux,regiones,pm,est=contexto_train(cfg)
    diario=DiarioV20(ROOT/'largo'/modo,cfg,rama,d,huellas_codigo())
    anterior,regs,previo=diario.leer();origen=anterior if continuar else ck
    if continuar and (anterior is None or t!='R'):raise ValueError('Reanudación diagnóstica inválida')
    if not continuar and anterior is not None:raise FileExistsError('Diario ya existente')
    m,opt,sc,rng,ra,real=restaurar(origen);dl,da=loaders(train,aux,rng,ra);restaurar_aleatorio(origen,rng,ra)
    p=politica(modo);salida=ROOT/'largo'/modo/'trazas'/rama/t
    with aplicar_backend(p):
        configurar_optimizador(p,opt);ef=registro_efectivo(opt);verificar_registro(p,ef,origen.get('backend_efectivo') if continuar else None)
        publicar(salida/('restauracion.json' if continuar else 'inicio.json'),{'estado_restaurado':real,
            'exacta':real==huellas_estado(origen),'backend_efectivo':ef,'pid':os.getpid()})
        obs=ObservadorBackend(m,opt,sc,rng,ra,salida,ROOT/'largo'/modo/'trazas'/rama/'C1',121 if continuar else 0)
        if not continuar:
            fila=ejecutar_epoca_train(m,opt,sc,dl,da,train,regiones,pm,cfg['ramas'][rama],est,20,0,p,obs)
            if fila['huella_orden_y_aumentos']!=d['trazas_preparadas'][rama][0]['huella_orden_y_aumentos']:raise ValueError('Datos distintos al plan')
            estado={**ck,'pesos':m.state_dict(),'optimizador':opt.state_dict(),'scaler':sc.state_dict(),**estado_aleatorio(rng,ra),
                'epoca':20,'historial':[fila],'actualizaciones_desde_origen':121,'mejor_epoca':None,'mejor_map':None,
                'protocolo_v20':protocolo(cfg),'sha256_origen':cfg['sha256_checkpoint_origen'],'hashes_codigo_v20':huellas_codigo(),
                'huella_regiones':d['huella_regiones'],'backend_numerico':p,'backend_efectivo':ef,
                'config_entrenamiento':{**ck['config_entrenamiento'],'v20':{'coeficiente':cfg['ramas'][rama],
                    'margen':1.,'fraccion':.25,'delta_huber':1.,'pesos':[1.,1.,2.,2.,1.,1.]}}}
            estado['reanudacion']={'tipo':'v20_margen_huber','rama':rama,'commit_previo_sha256':previo,'epoca_completa':True,
                'estado_inicial_origen':d['estado_inicial'],'sha256_diagnostico':d['sha256_archivo'],'huellas_estado':huellas_estado(estado)}
            previo=diario.confirmar(estado,previo)
            if t=='R':return
        fila=ejecutar_epoca_train(m,opt,sc,dl,da,train,regiones,pm,cfg['ramas'][rama],est,21,121,p,obs,limite=4)
        fin=cpu({'pesos':m.state_dict(),'optimizador':opt.state_dict(),'scaler':sc.state_dict(),**estado_aleatorio(rng,ra)})
        publicar(salida/'final_125.pt',fin,tensor=True)
        publicar(salida/'fin.json',{'actualizaciones':125,'paso_adamw':fila['paso_adamw'],'huellas_estado':huellas_estado(fin),
            'backend_efectivo':ef,'entrenamiento_AB':False,'epoca_21_parcial_sin_commit':True})


def largo(modo):
    for rama in ['A','B']:
        for t in ['C1','C2','R']:
            args=['--worker-largo','--modo',modo,'--rama',rama,'--trayectoria',t]
            ejecutar_proceso(args,f'largo_{modo}_{rama}_{t}',modo)
            if t=='R':ejecutar_proceso([*args,'--reanudar'],f'largo_{modo}_{rama}_{t}_reanudar',modo)
    res={}
    for rama in ['A','B']:
        root=ROOT/'largo'/modo/'trazas'/rama;pasos=[]
        for k in range(1,126):
            jj={t:leer(root/t/f'paso_{k:04d}.json') for t in ['C1','C2','R']};pares={}
            for t in ['C2','R']:
                exacta=jj['C1']['hashes_etapas']==jj[t]['hashes_etapas'];comp=None
                if not exacta:
                    a=torch.load(root/'C1'/f'paso_{k:04d}.pt',weights_only=True);b=torch.load(root/t/f'paso_{k:04d}.pt',weights_only=True)
                    comp=comparar_arbol(a,b);publicar(ROOT/'largo'/modo/'comparaciones'/rama/f'{k:04d}_C1_{t}.json',comp)
                pares[t]={'huellas_exactas':exacta,'dentro_tolerancia':exacta or comp['dentro_tolerancia']}
                if jj[t]['paso_adamw']!=2299+k:raise ValueError('Presupuesto inconsistente')
            pasos.append({'paso':k,'pares':pares})
        aa=torch.load(root/'C1/final_125.pt',weights_only=True)
        finals={}
        for t in ['C2','R']:
            c=comparar_arbol(aa,torch.load(root/t/'final_125.pt',weights_only=True));publicar(ROOT/'largo'/modo/f'final_{rama}_C1_{t}.json',c)
            finals[t]={k:v for k,v in c.items() if k!='filas'}
        recarga=leer(root/'R/restauracion.json')['exacta']
        res[rama]={'pasos':pasos,'finales':finals,'restauracion_exacta':recarga,
            'aprobada':recarga and all(s['pares'][t]['dentro_tolerancia'] for s in pasos for t in ['C2','R']) and all(c['dentro_tolerancia'] for c in finals.values())}
    publicar(ROOT/f'largo_{modo}.json',{'modo':modo,'ramas':res,'aprobada':all(x['aprobada'] for x in res.values()),'hashes_codigo':huellas_codigo()})
    return all(x['aprobada'] for x in res.values())


def ejecutar():
    d=diagnostico()
    publicar(ROOT/'protocolo_ejecutado.json',{'protocolo':leer(CONFIG),'hashes_codigo':huellas_codigo(),'sha256_config':sha(CONFIG),'AB_habilitado':False})
    ejecutar_proceso(['--config',str(CONFIG),'--crear-plan'],'plan',script='probar_entrenamiento_v20.py')
    minimo=corto('minimo');modo='minimo' if minimo else 'completo'
    if modo=='completo' and not corto(modo):raise RuntimeError('Candidato completo falla el control corto')
    publicar(ROOT/'seleccion_inicial.json',{'minimo_aprobado':minimo,'candidato':modo})
    aprobado=largo(modo)
    if not aprobado and modo=='minimo':
        modo='completo'
        if not corto(modo):raise RuntimeError('Completo corto fallido')
        aprobado=largo(modo)
    publicar(ROOT/'decision.json',{'aprobada':aprobado,'candidato':modo,'backend_numerico':politica(modo),
        'minimo_corto_aprobado':minimo,'hashes_codigo':huellas_codigo(),'AB_habilitado':False,
        'actualizaciones_AB':{'A':0,'B':0},'validacion_evaluada':False,'test_evaluado':False})


def main():
    torch.set_num_threads(4);p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker-largo',action='store_true')
    p.add_argument('--modo',choices=['minimo','completo']);p.add_argument('--rama',choices=['A','B']);p.add_argument('--trayectoria',choices=['C1','C2','R']);p.add_argument('--reanudar',action='store_true');a=p.parse_args()
    if a.worker_largo:worker_largo(a.modo,a.rama,a.trayectoria,a.reanudar)
    else:ejecutar()


if __name__=='__main__':main()
