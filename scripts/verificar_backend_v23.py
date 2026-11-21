"""Integridad histórica, controles V23 y bloqueo del A/B. Solo lectura salvo --crear."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.reanudacion_regional import publicar,sha
from scripts.preparar_v20 import huellas_codigo,leer_cfg,protocolo

ROOT=RAIZ/'resultados/v23_backend_v20'


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def comprobar():
    antes=leer(ROOT/'estado_previo.json');permitidos=antes['cambios_codigo_autorizados'];copias=leer(ROOT/'codigo_previo.json');n=0
    for p,h in antes['archivos'].items():
        if p in permitidos:
            if hashlib.sha256(base64.b64decode(copias[p]['bytes_base64'])).hexdigest()!=h:raise ValueError('Copia no exacta: '+p)
        elif sha(RAIZ/p)!=h:raise ValueError('Histórico modificado: '+p)
        else:n+=1
    r=leer(ROOT/'decision.json');prep=leer(ROOT/'preparacion_completa.json');cfg=leer(RAIZ/prep['configuracion']);h=huellas_codigo()
    if not r['aprobada'] or not prep['aprobada'] or r['hashes_codigo']!=h or prep['hashes_codigo']!=h:raise ValueError('Controles desactualizados')
    if leer_cfg()['entrenamiento_completo_habilitado'] or cfg['entrenamiento_completo_habilitado']:raise ValueError('A/B habilitado')
    if sha(RAIZ/prep['configuracion'])!=prep['sha256_configuracion']:raise ValueError('Configuración alterada')
    d=leer(RAIZ/cfg['diagnostico'])
    if not d['aprobada'] or d['hashes_codigo']!=h or d['protocolo']!=protocolo(cfg):raise ValueError('Regenerar diagnóstico')
    for campo in ['prueba_corta','control_backend']:
        c=leer(RAIZ/cfg[campo])
        if not c['aprobada'] or c['hashes_codigo']!=h or c['backend_numerico']!=cfg['backend_numerico'] or c['sha256_diagnostico']!=sha(RAIZ/cfg['diagnostico']):raise ValueError('Certificado incompatible')
        if sha(RAIZ/c['fuente'])!=c['sha256_fuente']:raise ValueError('Control alterado')
    protocolo_c=leer(ROOT/'protocolo_ejecutado.json')
    if protocolo_c['hashes_codigo']!=h or protocolo_c['sha256_config']!=sha(RAIZ/'configuracion/validacion_backend_v23.json'):raise ValueError('Protocolo de comparación cambiado')
    largo=leer(ROOT/f"largo_{r['candidato']}.json")
    if not largo['aprobada'] or not all(x['restauracion_exacta'] and x['aprobada'] and len(x['pasos'])==125 for x in largo['ramas'].values()):raise ValueError('Control largo incompleto')
    rechazos=leer(ROOT/'rechazos_reanudacion.json')
    if not all(x['rechazado'] for x in rechazos.values()):raise ValueError('Reanudación incompatible aceptada')
    if list((RAIZ/'checkpoints').glob(cfg['nombre']+'*.pt')):raise ValueError('Se inició el A/B completo')
    return {'historicos_intactos':n,'scripts_con_copia_exacta':len(permitidos),'pruebas_aprobadas':d['pruebas']['ejecutadas'],
        'candidato':r['candidato'],'backend_numerico':cfg['backend_numerico'],'configuracion':prep['configuracion'],
        'actualizaciones_AB':{'A':0,'B':0},'entrenamiento_AB_habilitado':False,'validacion_evaluada':False,'test_evaluado':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--crear',action='store_true');a=p.parse_args();c=comprobar();destino=ROOT/'manifiesto_integridad.json'
    if a.crear:
        if destino.exists():raise FileExistsError('Manifiesto existente')
        archivos={p.relative_to(RAIZ).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and p!=destino}
        otros=[*huellas_codigo(),c['configuracion'],'configuracion/validacion_backend_v23.json',
            'scripts/preparar_protocolo_v23.py','scripts/informar_backend_v23.py','scripts/verificar_backend_v23.py',
            'documentacion/23_validacion_backend_determinista_v20.md']
        archivos.update({p:sha(RAIZ/p) for p in otros})
        publicar(destino,{'fecha_utc':datetime.now(timezone.utc).isoformat(),'controles':c,'archivos':archivos,'hashes_codigo':huellas_codigo(),'sin_firma_externa':True})
    m=leer(destino)
    for p,h in m['archivos'].items():
        if sha(RAIZ/p)!=h:raise ValueError('Artefacto alterado: '+p)
    print(json.dumps({**c,'archivos_verificados':len(m['archivos'])}))


if __name__=='__main__':main()
