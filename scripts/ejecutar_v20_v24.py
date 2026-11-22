"""Ejecución autorizada por etapas: época 20 A/B, control, 21..31 y evaluación."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.reanudacion_regional import publicar,sha
from scripts.preparar_v20 import huellas_codigo,protocolo

ROOT=RAIZ/'resultados/v24_v20_evaluacion'
CONFIG=RAIZ/'configuracion/v20_backend_v23_1_completo_eval24_ep19.json'


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def correr(script,args,nombre,historico=False):
    env=os.environ.copy()
    env['PYTHONIOENCODING']='utf-8'
    if historico:env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    else:env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    log=ROOT/'logs'/f'{nombre}.txt';log.parent.mkdir(parents=True,exist_ok=True)
    if log.exists():
        original=nombre;i=2
        while log.exists():nombre=f'{original}_intento{i}';log=ROOT/'logs'/f'{nombre}.txt';i+=1
    with log.open('x',encoding='utf-8') as f:
        proc=subprocess.Popen([sys.executable,str(RAIZ/'scripts'/script),'--config',str(CONFIG),*args],cwd=RAIZ,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in proc.stdout:f.write(line);f.flush();print(line,end='',flush=True)
        codigo=proc.wait()
    publicar(ROOT/'procesos'/f'{nombre}.json',{'script':script,'argumentos':args,'codigo_salida':codigo,
        'CUBLAS_WORKSPACE_CONFIG':env.get('CUBLAS_WORKSPACE_CONFIG'),'sha256_log':sha(log)})
    if codigo:raise RuntimeError('Proceso fallido; detener sin cambiar hipótesis: '+nombre)


def habilitar():
    cfg=leer(CONFIG);h=huellas_codigo();d=leer(RAIZ/cfg['diagnostico'])
    if cfg['entrenamiento_completo_habilitado']:raise ValueError('Habilitación previa: inspeccionar registros')
    if not d['aprobada'] or d['hashes_codigo']!=h or d['protocolo']!=protocolo(cfg):raise ValueError('Diagnóstico desactualizado')
    for p in [cfg['prueba_corta'],cfg['control_backend']]:
        c=leer(RAIZ/p)
        if not c['aprobada'] or c['hashes_codigo']!=h or c['backend_numerico']!=cfg['backend_numerico']:raise ValueError('Certificados incompatibles')
    hist=leer(RAIZ/cfg['referencia_historica'])
    if not hist['aprobada'] or hist['hashes_codigo']!=h or any(hist['regresion'][v]['imagenes_exactas']!=181 for v in ['1','5']):raise ValueError('Regresión histórica pendiente')
    publicar(ROOT/'configuracion_antes_de_habilitar.json',cfg)
    publicar(ROOT/'autorizacion_por_etapas.json',{'protocolo':protocolo(cfg),'hashes_codigo':h,
        'origen':'Instrucción del usuario: época 20 A/B, comprobar diarios/estados/trazas/validación y, si pasan, reanudar hasta 31 y evaluar.',
        'limite_primera_etapa':121,'presupuesto_final_por_rama':1452,'test_reservado':True,'adopcion_automatica':False})
    cfg['entrenamiento_completo_habilitado']=True
    fd,tmp=tempfile.mkstemp(prefix='.habilitacion_v24_',dir=CONFIG.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(cfg,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,CONFIG)
    finally:Path(tmp).unlink(missing_ok=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fase',choices=['historico','entrenar','evaluar'],required=True);a=p.parse_args()
    if a.fase=='historico':correr('evaluar_v20.py',['--referencia-historica'],'referencia_historica',True);return
    if a.fase=='evaluar':
        args=[];compat=ROOT/'compatibilidad_evaluacion_cpu.json'
        if compat.exists():
            cfg=leer(CONFIG);nombre=cfg['nombre']
            args=['--compatibilidad-evaluacion',str(compat),'--directorio-capturas',str(RAIZ/f'resultados/{nombre}_evaluacion_revision_cpu'),
                '--reusar-origen',str(RAIZ/f'resultados/{nombre}_evaluacion/origen')]
        correr('evaluar_v20.py',args,'evaluacion_final');return
    habilitar()
    for rama in ['A','B']:correr('entrenar_v20.py',['--rama',rama,'--hasta-epoca','20'],f'{rama}_epoca20')
    correr('evaluar_v20.py',['--verificar-etapa-20'],'verificacion_epoca20')
    cfg=leer(CONFIG);c=leer(RAIZ/cfg['control_etapa_20'])
    if not c['aprobada'] or c['hashes_codigo']!=huellas_codigo():raise ValueError('No continuar sin control de ambas ramas')
    publicar(ROOT/'continuacion_autorizada.json',{'sha256_control_etapa20':sha(RAIZ/cfg['control_etapa_20']),
        'desde_epoca':21,'hasta_epoca':31,'presupuesto_por_rama':1452,'hipotesis_sin_cambios':True})
    for rama in ['A','B']:correr('entrenar_v20.py',['--rama',rama,'--reanudar','--hasta-epoca','31'],f'{rama}_epocas21_31')


if __name__=='__main__':main()
