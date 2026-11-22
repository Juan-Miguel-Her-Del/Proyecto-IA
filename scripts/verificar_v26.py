"""Certifica preparación V26 sin modelos, GPU, entrenamiento ni evaluación de test."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import VERSION,POLITICA,sha,leer,digest,politica,publicar
from ecowaste.coleccion_nueva import PROTOCOLO,consolidar,inventario
from scripts.preparar_v26 import SALIDA,NUEVOS


def comprobar(raiz,hashes):
    for path,esperado in hashes.items():
        archivo=raiz/path
        if not archivo.is_file() or sha(archivo)!=esperado:raise ValueError('Archivo modificado o ausente: '+path)


def controles():
    historicos=leer(SALIDA/'historicos_protegidos.json');comprobar(RAIZ,historicos)
    ck=leer(SALIDA/'checkpoints_previos.json')
    actual={p.relative_to(RAIZ).as_posix():[p.stat().st_size,p.stat().st_mtime_ns] for p in (RAIZ/'checkpoints').rglob('*.pt')}
    if actual!=ck:raise ValueError('Checkpoints históricos alterados')
    referencia='checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt'
    if sha(RAIZ/referencia)!='eb8fd00ee536ead21e3a2a0ecd5b06baad04acb2e437b22e0eac3686a7bf36b9':raise ValueError('Origen cambiado')
    p=politica();pre=leer(SALIDA/'preparacion.json')
    if pre['sha256_politica']!=sha(POLITICA) or pre['sha256_protocolo']!=sha(PROTOCOLO):raise ValueError('Preparación de otro protocolo')
    protocolo=leer(PROTOCOLO)
    if protocolo['evaluacion_autorizada'] or protocolo['entrenamiento_autorizado']:raise ValueError('Se desbloqueó el experimento')
    if leer(NUEVOS/'generalizacion_bloqueada/PROTOCOLO_PREREGISTRADO.json')!=protocolo:raise ValueError('Protocolo copiado incompatible')
    if leer(NUEVOS/'generalizacion_bloqueada/BLOQUEO.json')['sha256_protocolo']!=sha(PROTOCOLO):raise ValueError('Bloqueo de otro protocolo')
    esperado=inventario(consolidar([]))
    if leer(SALIDA/'inventario_inicial.json')!=esperado or pre['adquisiciones_reales_recibidas'] or pre['GT_real_aceptado']:
        raise ValueError('El inventario inicial acredita datos no recibidos')
    from collections import Counter
    for archivo,cuotas in [('plan_train_1550.csv',p['cuotas_train']),('plan_generalizacion_600.csv',{c:100 for c in p['clases']})]:
        with (NUEVOS/'adquisicion'/archivo).open(encoding='utf-8-sig',newline='') as f:filas=list(csv.DictReader(f))
        if dict(Counter(r['material_objetivo'] for r in filas))!=cuotas:raise ValueError('Cuotas incorrectas')
        if any(r['imagen_id'] or r['objeto_fisico_id'] or r['escena_id'] or r['estado']!='pendiente_adquisicion' for r in filas):
            raise ValueError('Plan inicial confundido con adquisiciones reales')
    plantilla=leer(NUEVOS/'plantillas/annotations.json')
    if plantilla['images'] or plantilla['annotations'] or plantilla['physical_objects'] or leer(NUEVOS/'plantillas/revisores.json')['revisores']:
        raise ValueError('Se inventaron imágenes, etiquetas o revisores en las plantillas')
    ensayo=leer(SALIDA/'ensayo_integracion_revision02.json')
    comprobar(RAIZ,ensayo['hashes_codigo'])
    if not all(ensayo[k] for k in ['aprobada','rechazo_congelacion_real','bloqueo_generalizacion','directorio_fixture_eliminado']):raise ValueError('Ensayo no aprobado')
    if ensayo['datos_reales_contabilizados'] or ensayo['entrenamiento'] or ensayo['inferencia'] or ensayo['test_evaluado']:
        raise ValueError('Ensayo fuera de alcance')
    pruebas=leer(SALIDA/'pruebas_finales.json');comprobar(RAIZ,pruebas['hashes_codigo_bajo_prueba'])
    if not pruebas['aprobadas'] or pruebas['ejecutadas']!=246 or pruebas['existentes_conservadas']!=201 or pruebas['nuevas']!=45 or any(pruebas[k] for k in ['fallos','errores','omitidas']):
        raise ValueError('Pruebas no aprobadas')
    # Los contratos se publican como JSON válido; la validación geométrica es Python.
    for f in (RAIZ/'esquemas').glob('*v26*json'):
        schema=leer(f)
        if not schema.get('$schema') or not schema.get('required'):raise ValueError('Contrato de datos incompleto')
    # Los comandos disponibles no pueden arrancar entrenamiento/evaluación.
    from scripts.datos_v26 import argumentos
    if argumentos(['inventario','--salida','pendiente.json']).orden!='inventario':raise ValueError('CLI incorrecto')
    for doc in [RAIZ/'documentacion/26_pipeline_datos_reales.md',NUEVOS/'README.md']:
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if not target.startswith('http') and not (doc.parent/target).exists():raise ValueError('Enlace roto: '+target)
    return {'aprobada':True,'historicos_sha256_intactos':len(historicos),'checkpoints_estadisticas_intactas':len(ck),
        'checkpoint_original_sha256_intacto':True,'pruebas':246,'pruebas_previas_conservadas':201,'pruebas_nuevas':45,
        'ensayo_descartable_aprobado':True,'nuevas_imagenes_reales_en_preparacion':0,'nuevos_GT_reales_aceptados':0,
        'instancias_train_pendientes':1550,'instancias_generalizacion_pendientes':600,'organicos_utilizables_pendientes':300,
        'entrenamiento':False,'evaluacion':False,'test_original_evaluado':False}


def archivos_preparacion():
    return [NUEVOS/'README.md',NUEVOS/'.gitignore',*list((NUEVOS/'plantillas').glob('*.json')),
        *list((NUEVOS/'adquisicion').glob('plan_*.csv')),*list((NUEVOS/'generalizacion_bloqueada').glob('*.json'))]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--publicar',action='store_true');a=ap.parse_args()
    resultado=controles();mp=SALIDA/'manifiesto_integridad.json';vp=SALIDA/'verificacion.json'
    if a.publicar:
        if mp.exists() or vp.exists():raise FileExistsError('Certificación existente; verificar sin --publicar')
        # La verificación V25 es de lectura y reutiliza inferencia histórica; no entrena.
        previo=subprocess.run([sys.executable,'scripts/verificar_presentacion_v25.py'],cwd=RAIZ,text=True,encoding='utf-8',capture_output=True)
        publicar(SALIDA/'verificacion_v25_conservada.json',{'aprobada':previo.returncode==0,'salida':previo.stdout,'errores':previo.stderr,
            'comando':'scripts/verificar_presentacion_v25.py','entrenamiento':False,'test_evaluado':False})
        if previo.returncode:raise ValueError('V25 no conserva integridad: '+previo.stderr)
        codigo=[RAIZ/p for p in ['ecowaste/incorporacion.py','ecowaste/revision_datos.py','ecowaste/coleccion_nueva.py',
            'pruebas/test_pipeline_v26.py','configuracion/incorporacion_v26.json','configuracion/protocolo_generalizacion_v26.json',
            'documentacion/26_pipeline_datos_reales.md']]
        codigo.extend((RAIZ/'scripts').glob('*v26.py'));codigo.extend((RAIZ/'esquemas').glob('*v26*json'))
        artefactos=[f for f in SALIDA.rglob('*') if f.is_file() and f not in [mp,vp]]
        publicar(mp,{'version':VERSION,'fase':'preparacion_sin_datos_reales',
            'codigo_y_documentacion':{f.relative_to(RAIZ).as_posix():sha(f) for f in codigo},
            'plantillas_y_planes':{f.relative_to(RAIZ).as_posix():sha(f) for f in archivos_preparacion()},
            'resultados':{f.relative_to(SALIDA).as_posix():sha(f) for f in artefactos},
            'protocolo_bloqueado_sha256':sha(PROTOCOLO),'entrenamiento_autorizado':False,'evaluacion_autorizada':False})
        resultado['sha256_manifiesto']=sha(mp);publicar(vp,resultado)
    else:
        m=leer(mp);comprobar(RAIZ,m['codigo_y_documentacion']);comprobar(RAIZ,m['plantillas_y_planes']);comprobar(SALIDA,m['resultados'])
        if leer(vp)['sha256_manifiesto']!=sha(mp):raise ValueError('Manifiesto cambiado')
        if not leer(SALIDA/'verificacion_v25_conservada.json')['aprobada']:raise ValueError('Verificación histórica no aprobada')
        resultado['sha256_manifiesto']=sha(mp)
    print(json.dumps(resultado,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
