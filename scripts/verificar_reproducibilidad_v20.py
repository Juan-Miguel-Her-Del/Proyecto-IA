"""Integridad V22 y conservación histórica; no entrena, no evalúa ni habilita A/B."""
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
from scripts.preparar_v20 import huellas_codigo,leer_cfg

ROOT=RAIZ/'resultados/v22_reproducibilidad_v20'


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def comprobar():
    previo=leer(ROOT/'estado_previo.json');permitidos=set(previo['cambios_codigo_autorizados']);conservados=0
    anterior=leer(ROOT/'codigo_anterior_bytes.json')
    for p,h in previo['archivos'].items():
        if p in permitidos:
            if hashlib.sha256(base64.b64decode(anterior[p]['bytes_base64'])).hexdigest()!=h:raise ValueError('Copia histórica no exacta: '+p)
        elif sha(RAIZ/p)!=h:raise ValueError('Histórico alterado: '+p)
        else:conservados+=1
    if leer_cfg()['entrenamiento_completo_habilitado'] or (RAIZ/leer_cfg()['prueba_corta']).exists():raise ValueError('Habilitación no autorizada')
    if list((RAIZ/'checkpoints').glob('v20*.pt')):raise ValueError('Checkpoint A/B inesperado')
    protocolo=leer(ROOT/'protocolo_ejecutado.json');d=leer(ROOT/'preparacion/diagnostico.json');r=leer(ROOT/'resumen.json')
    if sha(RAIZ/'configuracion/diagnostico_reproducibilidad_v20.json')!=protocolo['sha256_configuracion_diagnostica']:raise ValueError('Criterio cambiado después de ejecutar')
    if protocolo['hashes_codigo']!=huellas_codigo() or d['hashes_codigo']!=huellas_codigo():raise ValueError('Código posterior al diagnóstico')
    if sha(ROOT/'preparacion/diagnostico.json')!=protocolo['sha256_diagnostico'] or not d['aprobada']:raise ValueError('Diagnóstico distinto')
    if r['procesos_con_error'] or r['actualizaciones_trayectorias']!=144 or r['actualizaciones_replay_adam']!=6:raise ValueError('Diagnóstico incompleto')
    for modo in r['modos'].values():
        for rama in modo.values():
            if not rama['restauracion_exacta'] or not rama['entradas_y_RNG_exactos']:raise ValueError('Estado o inputs distintos')
    if len(list(ROOT.glob('*/*/*/checkpoint_paso_02.pt')))!=36:raise ValueError('Faltan checkpoints de paso 2')
    return {'historicos_intactos':conservados,'fuentes_modificadas_con_copia_exacta':len(permitidos),
        'actualizaciones_AB':{'A':0,'B':0},'actualizaciones_diagnosticas_red':144,'replays_adam':6,
        'diagnostico_regenerado':True,'pruebas_aprobadas':d['pruebas']['ejecutadas'],'AB_habilitado':False,
        'validacion_evaluada':False,'test_evaluado':False,'hipotesis_y_referencia_intactas':True}


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--crear',action='store_true');args=a.parse_args()
    c=comprobar();path=ROOT/'manifiesto_integridad.json'
    if args.crear:
        if path.exists():raise FileExistsError('Manifiesto existente')
        archivos={p.relative_to(RAIZ).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and p!=path}
        otros=[*huellas_codigo(),'configuracion/diagnostico_reproducibilidad_v20.json','configuracion/aprendizaje_discriminativo_mapas.json',
            'scripts/analizar_reproducibilidad_v20.py','scripts/informar_reproducibilidad_v20.py',
            'scripts/verificar_reproducibilidad_v20.py','documentacion/22_diagnostico_reproducibilidad_v20.md']
        archivos.update({p:sha(RAIZ/p) for p in otros})
        publicar(path,{'fecha_utc':datetime.now(timezone.utc).isoformat(),'controles':c,'archivos':archivos,
            'hashes_codigo_diagnostico':huellas_codigo(),'sin_firma_externa':True})
    m=leer(path)
    for p,h in m['archivos'].items():
        if sha(RAIZ/p)!=h:raise ValueError('Artefacto alterado: '+p)
    print(json.dumps({**c,'archivos_verificados':len(m['archivos'])}))


if __name__=='__main__':main()
