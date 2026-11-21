"""Publica protocolo V20 numérico separado, siempre deshabilitado, tras los controles."""
import argparse
import copy
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.backend_numerico import VERSION,verificar_registro
from ecowaste.reanudacion_regional import publicar,sha
from ecowaste.reanudacion_v20 import validar_estado_v20
from scripts.preparar_v20 import leer_cfg,protocolo,validar,huellas_codigo,exigir_autorizacion
from scripts.validar_backend_v23 import cfg_control,diagnostico,ROOT


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def controles():
    r=leer(ROOT/'decision.json');modo=r['candidato'];h=huellas_codigo()
    for p in [ROOT/'decision.json',ROOT/f'corto_{modo}.json',ROOT/f'largo_{modo}.json']:
        c=leer(p)
        if not c['aprobada'] or c['hashes_codigo']!=h:raise ValueError('Controles no aprobados o código distinto')
    return r


def ruta_config(r):return RAIZ/f"configuracion/v20_backend_v23_1_{r['candidato']}_ep19.json"


def configurar():
    r=controles();cfg=leer_cfg();cfg.update(nombre=f"v20_backend_v23_1_{r['candidato']}_ep19",version_protocolo=VERSION,
        backend_numerico=r['backend_numerico'],diagnostico='resultados/v23_backend_v20/preparacion_protocolo/diagnostico.json',
        prueba_corta='resultados/v23_backend_v20/controles/corto_versionado.json',control_backend='resultados/v23_backend_v20/controles/largo_versionado.json',
        entrenamiento_completo_habilitado=False)
    validar(cfg);publicar(ruta_config(r),cfg)
    print(ruta_config(r))


def certificar():
    r=controles();cfg=leer(ruta_config(r));d=leer(RAIZ/cfg['diagnostico']);h=huellas_codigo();modo=r['candidato']
    if not d['aprobada'] or d['protocolo']!=protocolo(cfg) or d['hashes_codigo']!=h:raise ValueError('Regenerar diagnóstico del protocolo')
    cp=ROOT/'largo'/modo/'checkpoints'/f'v23_control_{modo}_R_A_epoca_020.pt'
    ck=torch.load(cp,map_location='cpu',weights_only=True);dd=diagnostico();cc=cfg_control(modo,'R')
    validar_estado_v20(ck,cc,'A',dd,h,None)
    ataques={}
    for nombre,mutar in {
        'otro_backend':lambda c:c.update(backend_numerico={**c['backend_numerico'],'modo':'completo' if modo=='minimo' else 'minimo'}),
        'sdpa_incompatible':lambda c:c['backend_efectivo']['sdpa'].update(efficient=True),
        'cudnn_incompatible':lambda c:c['backend_efectivo'].update(cudnn_deterministic=False)
    }.items():
        malo=copy.deepcopy(ck);mutar(malo)
        try:validar_estado_v20(malo,cc,'A',dd,h,None)
        except ValueError as e:ataques[nombre]={'rechazado':True,'motivo':str(e)}
        else:raise ValueError('Se aceptó una reanudación incompatible')
    ef=ck['backend_efectivo'];otro=copy.deepcopy(ef);otro['torch']='version_diferente'
    try:verificar_registro(cfg['backend_numerico'],ef,otro)
    except ValueError as e:ataques['entorno_diferente']={'rechazado':True,'motivo':str(e)}
    else:raise ValueError('Entorno distinto aceptado')
    if cfg['entrenamiento_completo_habilitado']:raise ValueError('Protocolo debe estar deshabilitado')
    try:exigir_autorizacion(cfg)
    except ValueError as e:
        if 'deshabilitado' not in str(e):raise
        ataques['entrenamiento_deshabilitado']={'rechazado':True,'motivo':str(e)}
    else:raise ValueError('Guardia de entrenamiento abierta')
    publicar(ROOT/'rechazos_reanudacion.json',ataques)
    base={'aprobada':True,'hashes_codigo':h,'sha256_diagnostico':sha(RAIZ/cfg['diagnostico']),
        'backend_numerico':cfg['backend_numerico'],'protocolo':protocolo(cfg),
        'prueba_corta_historica_no_reclasificada':True,'entrenamiento_AB_habilitado':False}
    publicar(RAIZ/cfg['prueba_corta'],{**base,'fuente':f'resultados/v23_backend_v20/corto_{modo}.json','sha256_fuente':sha(ROOT/f'corto_{modo}.json')})
    publicar(RAIZ/cfg['control_backend'],{**base,'fuente':f'resultados/v23_backend_v20/largo_{modo}.json','sha256_fuente':sha(ROOT/f'largo_{modo}.json'),
        'actualizaciones_diagnosticas_por_trayectoria':125,'corte_en_epoca':20,'backend_efectivo':ef})
    publicar(ROOT/'preparacion_completa.json',{'aprobada':True,'configuracion':ruta_config(r).relative_to(RAIZ).as_posix(),
        'sha256_configuracion':sha(ruta_config(r)),'hashes_codigo':h,'entrenamiento_AB_habilitado':False,'actualizaciones_AB':{'A':0,'B':0}})
    print('Protocolo versionado preparado; A/B permanece deshabilitado')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--certificar',action='store_true');a=p.parse_args()
    certificar() if a.certificar else configurar()
