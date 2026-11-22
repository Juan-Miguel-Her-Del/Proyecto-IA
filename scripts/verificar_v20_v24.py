"""Comprueba entrega A/B completa y archivos históricos sin repetir inferencia."""
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
from ecowaste.reanudacion_v20 import DiarioV20
from ecowaste.backend_numerico import verificar_registro
from scripts.preparar_v20 import huellas_codigo,protocolo
from scripts.entrenar_v20 import cargar_diagnostico

ROOT=RAIZ/'resultados/v24_v20_evaluacion'
CONFIG=RAIZ/'configuracion/v20_backend_v23_1_completo_eval24_ep19.json'


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def comprobar():
    antes=leer(ROOT/'estado_previo.json');copias=leer(ROOT/'codigo_previo.json');permitidos=set(antes['cambios_codigo_autorizados']);n=0
    for p,h in antes['archivos'].items():
        if p in permitidos:
            if hashlib.sha256(base64.b64decode(copias[p]['bytes_base64'])).hexdigest()!=h:raise ValueError('Copia histórica no exacta')
        elif sha(RAIZ/p)!=h:raise ValueError('Histórico modificado: '+p)
        else:n+=1
    cfg=leer(CONFIG);h=huellas_codigo();diag=cargar_diagnostico(cfg);nombre=cfg['nombre']
    train=leer(RAIZ/f'resultados/{nombre}.json');e=leer(RAIZ/f'resultados/{nombre}_evaluacion.json')
    entrenadas=train['hashes_codigo'];compat=leer(ROOT/'compatibilidad_evaluacion_cpu.json')
    if e['hashes_codigo']!=h or e['hashes_codigo_entrenamiento']!=entrenadas or compat['hashes_entrenamiento']!=entrenadas or compat['hashes_evaluacion']!=h or not compat['aprobada']:
        raise ValueError('Código de entrenamiento/evaluación no vinculado')
    if train['protocolo']!=protocolo(cfg) or e['protocolo']!=protocolo(cfg):raise ValueError('Protocolo cambiado')
    historico=leer(RAIZ/cfg['referencia_historica']);etapa=leer(RAIZ/cfg['control_etapa_20'])
    if not historico['aprobada'] or not etapa['aprobada'] or any(historico['regresion'][v]['imagenes_exactas']!=181 for v in ['1','5']):raise ValueError('Controles incompletos')
    finales={};registros=[]
    for rama in ['A','B']:
        d=DiarioV20(RAIZ,cfg,rama,diag,entrenadas);ck,rs,_=d.leer()
        if ck['epoca']!=31 or ck['actualizaciones_desde_origen']!=1452 or len(rs)!=12:raise ValueError('Presupuesto incompleto')
        if ck['historial']!=train['ramas'][rama]['historial']:raise ValueError('Historial diferente')
        finales[rama]=ck;registros+=rs
    if finales['A']['reanudacion']['estado_inicial_origen']!=finales['B']['reanudacion']['estado_inicial_origen']:raise ValueError('Origen distinto')
    if [x['huella_orden_y_aumentos'] for x in finales['A']['historial']]!=[x['huella_orden_y_aumentos'] for x in finales['B']['historial']]:raise ValueError('Entradas distintas')
    capturas={}
    for x in e['resultados'].values():
        verificar_registro(cfg['backend_numerico'],x['backend_efectivo'])
        cp=x['captura']
        if cp in capturas:continue
        c=leer(RAIZ/cp)
        if c['hashes_codigo'] not in [h,entrenadas] or c['imagenes']!=181 or c['gt_utilizado']:raise ValueError('Captura incompatible')
        for f in c['archivos']:
            if sha(RAIZ/f['archivo'])!=f['sha256']:raise ValueError('Logits o traza modificados')
        capturas[cp]=c
    if e['test_evaluado'] or e['cambio_predeterminado'] or e['adopcion_automatica']:raise ValueError('Ámbito modificado')
    if sha(RAIZ/cfg['checkpoint_origen'])!=cfg['sha256_checkpoint_origen']:raise ValueError('Referencia alterada')
    revision=leer(ROOT/'revision_cpu_diagnostico/diagnostico.json')
    if not revision['aprobada'] or revision['hashes_codigo']!=h:raise ValueError('Diagnóstico de evaluación incompatible')
    if not all(x['exactas'] for x in leer(ROOT/'comparacion_epoca20_vs_control125.json').values()):raise ValueError('Cambio respecto al control V23')
    return {'historicos_intactos':n,'scripts_con_copia_exacta':len(permitidos),'pruebas_entrenamiento':diag['pruebas'],'pruebas_evaluacion':revision['pruebas'],
        'actualizaciones_por_rama':{'A':1452,'B':1452},'checkpoints_epoca':len(registros),'capturas_numericas_unicas':len(capturas),
        'referencia_historica_exacta':{'1v':181,'5v':181},'test_evaluado':False,'adopcion_automatica':False,
        'referencia_original_intacta':True,'evidencia_125_pasos_intacta':True},cfg,registros


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--crear',action='store_true');a=p.parse_args();c,cfg,rs=comprobar();destino=ROOT/'manifiesto_integridad.json'
    if a.crear:
        if destino.exists():raise FileExistsError('Manifiesto existente')
        paths={p for p in ROOT.rglob('*') if p.is_file() and p!=destino}
        paths.update(p for p in (RAIZ/f"resultados/{cfg['nombre']}_evaluacion").rglob('*') if p.is_file())
        paths.update(p for p in (RAIZ/f"resultados/{cfg['nombre']}_evaluacion_revision_cpu").rglob('*') if p.is_file())
        paths.update(RAIZ/r['checkpoint'] for r in rs)
        paths.update(p for p in (RAIZ/'resultados'/cfg['nombre']).rglob('*') if p.is_file())
        paths.update((RAIZ/'checkpoints').glob(cfg['nombre']+'*.pt'))
        paths.update([CONFIG,RAIZ/f"resultados/{cfg['nombre']}.json",RAIZ/f"resultados/{cfg['nombre']}_evaluacion.json",
            RAIZ/'documentacion/24_resultados_v20_backend_determinista.md'])
        paths.update(RAIZ/p for p in huellas_codigo())
        paths.update(RAIZ/'scripts'/p for p in ['preparar_ejecucion_v24.py','ejecutar_v20_v24.py','informar_v20_v24.py','verificar_v20_v24.py','certificar_evaluacion_cpu_v24.py'])
        paths.add(RAIZ/'pruebas/test_agregacion_evaluacion_cpu.py')
        publicar(destino,{'fecha_utc':datetime.now(timezone.utc).isoformat(),'controles':c,'hashes_codigo':huellas_codigo(),
            'archivos':{p.relative_to(RAIZ).as_posix():sha(p) for p in sorted(paths)},'sin_firma_externa':True})
    m=leer(destino)
    for p,h in m['archivos'].items():
        if sha(RAIZ/p)!=h:raise ValueError('Artefacto modificado: '+p)
    print(json.dumps({**c,'archivos_verificados':len(m['archivos'])}))


if __name__=='__main__':main()
