"""Colección nueva autorizada solo para piloto V28.2; preserva NO_ENTRENAR de V28."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import subprocess
import hashlib
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.incorporacion import leer,publicar,publicar_bytes,sha,digest,CLASES
from ecowaste.auxiliares_publicos import Taller
from ecowaste.validacion_anotaciones import analizar_imagen,revision_humana_final
from ecowaste.datos_v28_2 import VERSION,AuxiliarExperimental
from ecowaste.backend_numerico import politica

SALIDA=RAIZ/'resultados/v28_2_exploratorio'
DATOS=RAIZ/'datos/nuevos/v28_2/coleccion'
CONFIG=RAIZ/'configuracion/entrenamiento_exploratorio_v28_2.json'


def main():
    if SALIDA.exists() or DATOS.exists():raise FileExistsError('No sobrescribir preparación V28.2')
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=RAIZ).decode().split('\0')
    paths=[RAIZ/p for p in tracked if p];paths.extend((RAIZ/'datos/procesados').rglob('*.jsonl'))
    paths.append(RAIZ/'checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt')
    publicar(SALIDA/'historicos_protegidos.json',{p.relative_to(RAIZ).as_posix():sha(p) for p in paths if p.is_file()})
    publicar(SALIDA/'checkpoints_previos.json',{p.relative_to(RAIZ).as_posix():[p.stat().st_size,p.stat().st_mtime_ns] for p in (RAIZ/'checkpoints').rglob('*.pt')})
    b=leer(RAIZ/'resultados/v28_validacion_anotaciones/respaldo_verificado.json');backup=RAIZ/b['destino']
    for p,h in b['archivos'].items():
        if sha(backup/p)!=h:raise ValueError('Respaldo de 731 archivos alterado')
    t=Taller(backup,RAIZ)
    if sum(len(t.diario(i)) for i in t.imagenes)!=134:raise ValueError('Revisiones humanas distintas')
    ids=leer(RAIZ/'resultados/v28_validacion_anotaciones/suficiencia_exploratoria.json')['imagenes_candidatas_sin_incidencias_identificadas']
    observaciones=leer(RAIZ/'resultados/v28_validacion_anotaciones/observaciones_visuales_y_taxonomia.json')
    exp=leer(RAIZ/'resultados/v28_validacion_anotaciones/exportacion_auxiliar_auditoria.json');prev=RAIZ/exp['destino'];c=leer(prev/'annotations.json')
    porid={i['eco']['imagen_id_auxiliar']:i for i in c['images']};cats={i['id']:i['name'] for i in c['categories']}
    nuevo={'info':{'version':VERSION,'estado':'EXPLORATORIA_NO_VALIDADA_PARA_ADOPCION',
                  'autorizada_solo_diagnostico':True,'aceptadas_V26':0},'images':[],'annotations':[],'categories':c['categories'],'physical_objects':[]}
    qa=[]
    for iid in ids:
        if any(o['imagen']==iid and o['requiere_decision'] for o in observaciones):raise ValueError('Incidencia conocida en candidata: '+iid)
        im=t.imagen(iid);ev=revision_humana_final(t.diario(iid))
        if not ev or not ev['cobertura_completa'] or ev['persona_id']!='juan_miguel':raise ValueError('Falta revisión completa: '+iid)
        q=analizar_imagen(im,ev['anotaciones'],True)
        if q['errores'] or q['avisos_visuales']:raise ValueError('Candidata no pasa QA técnico: '+iid)
        orig=porid[iid];row=deepcopy(orig);row['id']=len(nuevo['images'])+1
        source=prev/orig['file_name'];publicar_bytes(DATOS/row['file_name'],source.read_bytes())
        if sha(DATOS/row['file_name'])!=im['sha256_archivo']:raise ValueError('Imagen alterada')
        row['eco'].update(estado_coleccion='EXPLORATORIA_NO_VALIDADA_PARA_ADOPCION',ausencia_incidencias_no_prueba_exhaustividad=True)
        nuevo['images'].append(row)
        humans={a['id']:a for a in ev['anotaciones']}
        for a in c['annotations']:
            if a['image_id']!=orig['id']:continue
            h=humans[a['eco']['id_auxiliar']]
            if h['poligonos']!=a['eco']['poligonos_usuario'] or h['material']!=cats[a['category_id']]:raise ValueError('Exportación no corresponde al usuario')
            a=deepcopy(a);a['id']=len(nuevo['annotations'])+1;a['image_id']=row['id'];nuevo['annotations'].append(a)
        qa.append({'imagen':iid,'revision':ev['huella'],'persona':ev['persona_id'],'QA_tecnico_sin_incidencias':True,
                   'exhaustividad_no_certificada':True,'grupo_procedencia':'fuente:'+im['fuente']})
    publicar(DATOS/'annotations.json',nuevo);publicar(DATOS/'QA_ALCANCE.json',qa)
    lic=leer(prev/'MANIFIESTO_COMPLEMENTO_LICENCIAS.json')
    for p,h in lic['archivos'].items():
        if sha(prev/p)!=h:raise ValueError('Licencia previa alterada')
        publicar_bytes(DATOS/p,(prev/p).read_bytes())
    # Conservar los eventos humanos exactos dentro de la colección nueva.
    for iid in ids:
        folder=digest(iid)[:24]
        for p in (backup/'revisiones'/folder).glob('*.json'):publicar_bytes(DATOS/'revisiones'/folder/p.name,p.read_bytes())
    publicar(DATOS/'MANIFIESTO.json',{'version':VERSION,'estado':'EXPLORATORIA_NO_VALIDADA_PARA_ADOPCION',
        'autorizada_solo_diagnostico':True,'aceptadas_V26':0,'no_requiere_cuotas_40_y_8_por_autorizacion_V28_2':True,
        'archivos':{p.relative_to(DATOS).as_posix():sha(p) for p in DATOS.rglob('*') if p.is_file()}})
    datos=AuxiliarExperimental(RAIZ,DATOS)
    publicar(SALIDA/'dataset.json',{'imagenes':ids,'cantidad_imagenes':len(ids),'objetos_seis_materiales':dict(Counter(cats[a['category_id']] for a in nuevo['annotations'] if not a['ignore'])),
        'ignoradas':sum(a['ignore'] for a in nuevo['annotations']),'sha256_coleccion':sha(DATOS/'MANIFIESTO.json'),
        'respaldo_731_verificado':True,'eventos_134_conservados':True,'sin_modificar_exportacion_NO_ENTRENAR':True,
        'proposito':'diagnóstico train reducido autorizado; no adopción ni aceptación V26','deficits':{'imagenes_hasta40':12,'metal_hasta8':2,'HPI_alimentarios':3},
        'agrupacion':'fuente completa solo train; sin holdout ni identidades físicas inventadas','mascaras_sha256':hashlib.sha256(datos.mascaras.numpy().tobytes()).hexdigest()})
    origen=RAIZ/'checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt'
    cfg={'version':'v28_2_exploratorio_1','entrenamiento_autorizado':True,'proposito_autorizado':'piloto diagnóstico 300 pasos A/B; decisión expresa del usuario',
         'checkpoint_origen':origen.relative_to(RAIZ).as_posix(),'sha256_origen':sha(origen),
         'coleccion':DATOS.relative_to(RAIZ).as_posix(),'sha256_coleccion':sha(DATOS/'MANIFIESTO.json'),
         'actualizaciones':300,'batch_size':8,'historicas_comunes':6,'ranuras_sustituidas':2,'semilla_auxiliar':2801,
         'learning_rate':.0005,'weight_decay':.01,'coeficiente_fondo':3.,'coeficiente_auxiliar_global':.5,'auxiliar_batch':16,
         'clipping':1.,'nueva_perdida':False,'backend_numerico':politica('completo'),'warmup_nuevo':False,'parada_temprana':False,
         'vistas':[1,5],'umbral':.05,'visual':.35,'area_minima':2,'nms':.5,'top':100,'politica':'argmax7',
         'seleccion_mejor_checkpoint':False,'pasos_checkpoint':[0,100,200,300],'evaluar_val':False,'evaluar_test':False,
         'generalizacion':False,'adopcion_automatica':False,'presupuesto_no_ampliable':True}
    publicar(CONFIG,cfg)
    # Regla visual fija, anterior a entrenamiento e inspección de predicciones.
    hrows=[__import__('json').loads(l) for l in (RAIZ/'datos/procesados/deteccion/train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    visual=[]
    for dominio,rs in [('historico',hrows),('auxiliar',datos.registros)]:
        usados=set()
        for material in ['carton','vidrio','organico']:
            candidatos=[]
            for r in rs:
                if dominio=='historico':presente=any(a.get('clase')==material for a in r['cajas'])
                else:
                    im=next(i for i in nuevo['images'] if i['eco']['imagen_id_auxiliar']==r['id'])
                    presente=any(a['image_id']==im['id'] and cats[a['category_id']]==material for a in nuevo['annotations'])
                if presente and r['id'] not in usados:candidatos.append(r)
            candidatos.sort(key=lambda r:(0 if material=='organico' and r.get('fuente')=='hpi' else 1,hashlib.sha256(('v28.2_visual_1'+r['id']).encode()).hexdigest()))
            for r in candidatos[:2]:visual.append({'dominio':dominio,'material_prioritario':material,'imagen':r['id']});usados.add(r['id'])
    publicar(SALIDA/'seleccion_visual_prefijada.json',{'regla':'2 por dominio/material; SHA256 v28.2_visual_1+id; prioridad HPI para orgánico; sin ver predicciones','imagenes':visual})
    print('Colección y protocolo V28.2 preparados:',len(datos),'imágenes',flush=True)


if __name__=='__main__':main()
