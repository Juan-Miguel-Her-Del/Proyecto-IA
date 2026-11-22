"""Crea plantillas e inventarios pendientes V26; no inventa adquisiciones ni anotadores."""
import csv
import io
from pathlib import Path
import subprocess
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import VERSION,politica,sha,digest,publicar,publicar_bytes,leer
from ecowaste.coleccion_nueva import consolidar,inventario,PROTOCOLO

SALIDA=RAIZ/'resultados/v26_pipeline_datos'
NUEVOS=RAIZ/'datos/nuevos/v26'


def csv_pendiente(path,cuotas,pool):
    f=io.StringIO(newline='');campos=['slot_plan','destino','material_objetivo','estado','imagen_id','objeto_fisico_id','escena_id','sitio_id','sesion_id','secuencia_id']
    w=csv.DictWriter(f,fieldnames=campos);w.writeheader();i=0
    for c,n in cuotas.items():
        for _ in range(n):
            i+=1;w.writerow({'slot_plan':f'{pool}_{i:04d}','destino':pool,'material_objetivo':c,'estado':'pendiente_adquisicion',
                **{k:'' for k in campos if k.endswith('_id')}})
    publicar_bytes(path,f.getvalue().encode('utf-8-sig'))


def main():
    if SALIDA.exists() or NUEVOS.exists():raise FileExistsError('Preparación existente; conservarla')
    SALIDA.mkdir(parents=True)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=RAIZ).decode('utf-8').split('\0')
    archivos=[RAIZ/f for f in tracked if f]
    archivos.extend([*list((RAIZ/'datos/procesados').rglob('*.jsonl')),RAIZ/'checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt',
        RAIZ/'datos/originales/taco/annotations.json',RAIZ/'datos/reportes/inventario_imagenes.jsonl'])
    publicar(SALIDA/'historicos_protegidos.json',{p.relative_to(RAIZ).as_posix():sha(p) for p in archivos if p.is_file()})
    publicar(SALIDA/'checkpoints_previos.json',{p.relative_to(RAIZ).as_posix():[p.stat().st_size,p.stat().st_mtime_ns] for p in (RAIZ/'checkpoints').rglob('*.pt')})
    p=politica();NUEVOS.mkdir(parents=True)
    publicar(NUEVOS/'plantillas/revisores.json',{'tipo_datos':'real','revisores':[],
        'instruccion':'Registrar personas reales, persona_id estable, humano=true, tipo=real y roles anotador/adjudicador. No crear aprobaciones ficticias.'})
    publicar(NUEVOS/'plantillas/annotations.json',{'eco':{'version':VERSION,'coleccion_id':'REEMPLAZAR_POR_ID_REAL','tipo_datos':'real'},
        'images':[],'annotations':[],'physical_objects':[],
        'categories':[{'id':i+1,'name':c,'supercategory':'material'} for i,c in enumerate([*p['clases'],'ambiguo','compuesto','fuera_alcance'])]})
    publicar(NUEVOS/'plantillas/ficha_captura.json',{'id':None,'file_name':None,'width':None,'height':None,
        'eco':{'sha256_archivo':None,'escena_id':None,'sitio_id':None,'sesion_id':None,'grupo_captura_id':None,
            'secuencia_id':None,'fotograma':None,'dispositivo_id':None,'capturado_en':None,'destino_previsto':'sin_asignar','objetos_fisicos_ids':[],
            'procedencia':{'tipo':'captura_real','autor':None,'referencia':None,'licencia':None,'permiso_ref':None,'permiso_confirmado':False}}})
    publicar(NUEVOS/'plantillas/ficha_instancia.json',{'id':None,'image_id':None,'category_id':None,'segmentation':[],
        'eco':{'objeto_fisico_id':None,'origen':'humana','visible_only':True,'oclusion':None,'truncado':None,
            'transparente':None,'solapa_con':[],'motivo_ambiguo':None}})
    csv_pendiente(NUEVOS/'adquisicion/plan_train_1550.csv',p['cuotas_train'],'train_nuevo')
    csv_pendiente(NUEVOS/'adquisicion/plan_generalizacion_600.csv',{c:p['generalizacion']['instancias_por_clase'] for c in p['clases']},'generalizacion_bloqueada')
    publicar(NUEVOS/'generalizacion_bloqueada/PROTOCOLO_PREREGISTRADO.json',leer(PROTOCOLO))
    publicar(NUEVOS/'generalizacion_bloqueada/BLOQUEO.json',{'version':VERSION,'estado':'sin_imagenes_bloqueada',
        'sha256_protocolo':sha(PROTOCOLO),'imagenes_recibidas':0,'GT_aceptado':0,
        'evaluacion_autorizada':False,'entrenamiento_autorizado':False,'test_original_evaluado':False})
    publicar(SALIDA/'inventario_inicial.json',inventario(consolidar([])))
    publicar(SALIDA/'preparacion.json',{'version':VERSION,'sha256_propuesta_v25':sha(RAIZ/p['referencia_v25']),
        'sha256_politica':sha(RAIZ/'configuracion/incorporacion_v26.json'),'sha256_protocolo':sha(PROTOCOLO),
        'adquisiciones_reales_recibidas':0,'GT_real_aceptado':0,'slots_train_pendientes':1550,'slots_generalizacion_pendientes':600,
        'regiones_organicas_utilizables_recibidas':0,'entrenamiento':False,'evaluacion':False,'test_original_evaluado':False})
    print('V26 preparado: 1550 + 600 objetivos pendientes; cero datos reales nuevos recibidos.',flush=True)


if __name__=='__main__':main()
