"""Audita exclusivamente el respaldo humano verificado, nunca las propuestas como GT."""
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs,urlsplit

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,publicar,publicar_bytes,sha,digest,CLASES,candidatos_duplicados
from ecowaste.auxiliares_publicos import Taller,VERSION
from ecowaste.validacion_anotaciones import iniciales,analizar_imagen,comparar
from scripts.respaldar_anotaciones_v28 import SALIDA,ORIGEN,inventariar
from scripts.preparar_piloto_v27 import historicos_train_val


def csv_nuevo(path,rows,fields):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r.get(k),(list,dict)) else r.get(k) for k in fields})


def main():
    if (SALIDA/'auditoria.json').exists():raise FileExistsError('Auditoría existente: usar otra versión, no sobrescribir')
    rec=leer(SALIDA/'respaldo_verificado.json');backup=RAIZ/rec['destino']
    if inventariar(backup)!=rec['archivos']:raise ValueError('Respaldo alterado')
    t=Taller(backup,RAIZ);filas=[];objetos=[];pendientes=[];identidades=Counter();n_eventos=0;dif=Counter();cadenas=[]
    dirs={digest(i)[:24] for i in t.imagenes}
    orfanos=[p.relative_to(backup).as_posix() for p in (backup/'revisiones').rglob('*.json') if p.parent.name not in dirs]
    for iid,im in t.imagenes.items():
        eventos=t.diario(iid);n_eventos+=len(eventos);identidades.update(e['persona_id'] for e in eventos)
        for e in eventos:
            if e['version']!=VERSION or e['aceptada_V26'] or e['entrenamiento_autorizado'] or e['cuotas_identidad_acreditadas']:
                raise ValueError('Evento fuera del contrato')
        t.archivo_imagen(iid)
        asistidas=[e for e in eventos if e['modo']=='asistida'];last=asistidas[-1] if asistidas else None
        estado='sin_revision_guardada' if last is None else 'completa_usuario_unico' if last['cobertura_completa'] else 'borrador_usuario'
        base=iniciales(t,iid);final=deepcopy(last['anotaciones']) if last else []
        delta=comparar(base,final,im) if last else None
        if delta:
            for k,v in delta.items():dif[k]+=len(v)
        qa=analizar_imagen(im,final,bool(last and last['cobertura_completa']))
        n_act=sum(not a.get('descartada') for a in final)
        if not last or not last['cobertura_completa']:
            pendientes.append({'imagen':iid,'objeto':None,'indice_editor':None,'nivel':'guardado',
                'motivos':[estado],'archivo_original':im['archivo']})
        if last and not n_act:
            pendientes.append({'imagen':iid,'objeto':None,'indice_editor':None,'nivel':'visual',
                'motivos':['imagen_guardada_sin_objetos_activos_verificar_negativa'],'archivo_original':im['archivo']})
        for n,a in enumerate(qa['objetos'],1):
            a.update(imagen=iid,fuente=im['fuente'],indice_editor=n,anadido_por_usuario=bool(last and a['id'] not in {b['id'] for b in base}))
            objetos.append(a)
            if a['errores']:pendientes.append({'imagen':iid,'objeto':a['id'],'indice_editor':n,'nivel':'error_bloqueante','motivos':a['errores'],'archivo_original':im['archivo']})
        for v in qa['avisos_visuales']:
            aids=v.get('objetos',[v.get('objeto')])
            pendientes.append({'imagen':iid,'objeto':aids,'indice_editor':[next((n+1 for n,a in enumerate(final) if a['id']==aid),None) for aid in aids],
                'nivel':'visual','motivos':v['motivos'],'datos':v.get('datos'),'archivo_original':im['archivo']})
        if delta and delta['originales_ausentes_sin_registro']:
            pendientes.append({'imagen':iid,'objeto':delta['originales_ausentes_sin_registro'],'nivel':'error_bloqueante',
                'motivos':['propuestas_originales_borradas_sin_descarte_trazable'],'archivo_original':im['archivo']})
        row={'imagen':iid,'fuente':im['fuente'],'archivo_original':im['archivo'],'estado':estado,
             'eventos':len(eventos),'eventos_independientes':sum(e['modo']=='independiente' for e in eventos),
             'ultima_huella':last['huella'] if last else None,'persona_ultima':last['persona_id'] if last else None,
             'propuestas_iniciales':len(base),'registros_finales_guardados':len(final),'objetos_activos_guardados':n_act,
             'descartados':sum(bool(a.get('descartada')) for a in final),'comparacion':delta,
             'errores':qa['errores'],'avisos_visuales':qa['avisos_visuales'],'solapes':qa['solapes'],
             'fusiones_representacion_48':qa['fusiones_48'],'organicos_utilizables_48':qa['organicos_utilizables_48'],
             'qa_objetos':qa['objetos']}
        filas.append(row)
        print(len(filas),'/72',iid,estado,'objetos',n_act,'errores',len(qa['errores']),'avisos',len(qa['avisos_visuales']),flush=True)
    distrib=[]
    for c in [*CLASES,'ambiguo','compuesto','fuera_alcance','sin_asignar']:
        rs=[a for a in objetos if a['material']==c and not a['descartada']]
        completas=[a for a in rs if a['imagen_cobertura_completa']]
        distrib.append({'material':c,'objetos_guardados_activos':len(rs),'objetos_revisados_imagen_completa':len(completas),
            'poligonos_validos_imagen_completa':sum(a['valido_geometria'] for a in completas),
            'imagenes_distintas_completas':len({a['imagen'] for a in completas}),
            'objetos_pendientes_formales':sum(not a['imagen_cobertura_completa'] or bool(a['errores']) for a in rs),
            'objetos_con_avisos_visuales':sum(bool(a['avisos']) for a in rs)})
    logs=leer(SALIDA/'logs_respaldados.json');codigos=Counter();contextos=[]
    import re
    for nombre,recibo in logs.items():
        last_iid=None
        for n,linea in enumerate((RAIZ/recibo['archivo']).read_text(encoding='utf-8',errors='replace').splitlines(),1):
            m=re.search(r'GET (/api/tarea\?[^ ]+)',linea)
            if m:last_iid=parse_qs(urlsplit(m[1]).query).get('id',[None])[0]
            m=re.search(r'POST /api/guardar HTTP/[^\"]+\" (\d+)',linea)
            if m:
                codigos[m[1]]+=1
                if m[1]!='200':contextos.append({'log':nombre,'linea':n,'codigo':m[1],'ultima_imagen_GET_cercana_NO_prueba_payload':last_iid})
    publicar(SALIDA/'incidencias_guardado_logs.json',{'codigos_POST_guardar_observados':dict(codigos),'incidencias':contextos,
        'limite':'Los logs no conservan cuerpo del POST ni motivo de rechazo. Contexto GET no prueba qué corrección se intentó guardar.'})
    exportaciones=[];estado_busqueda='completada'
    try:
        descargas=Path.home()/'Downloads'
        candidatos=list(descargas.glob('ecowaste_auxiliar_COCO_pendiente*.json'))
        for p in candidatos:
            destino=backup.parent/'exportaciones_encontradas'/p.name;publicar_bytes(destino,p.read_bytes())
            exportaciones.append({'archivo':str(p),'respaldo':destino.relative_to(RAIZ).as_posix(),'sha256':sha(destino)})
    except OSError as e:estado_busqueda=str(e)
    publicar(SALIDA/'busqueda_recuperacion.json',{'estado_descargas':estado_busqueda,'exportaciones_encontradas':exportaciones,
        'otros_proyectos':list(str(p.relative_to(RAIZ)) for p in ORIGEN.parent.rglob('PROYECTO.json') if p.parent!=ORIGEN),
        'fusion_automatica_de_exportaciones':False,'eventos_huerfanos':orfanos})
    refs=[{**im,'id':iid} for iid,im in t.imagenes.items()]
    duplicados=candidatos_duplicados(refs,historicos_train_val())
    publicar(SALIDA/'duplicados_train_val_y_piloto.json',duplicados)
    publicar(SALIDA/'imagenes.json',filas);publicar(SALIDA/'objetos.json',objetos);publicar(SALIDA/'distribucion.json',distrib)
    publicar(SALIDA/'pendientes.json',pendientes)
    csv_nuevo(SALIDA/'distribucion.csv',distrib,list(distrib[0]))
    csv_nuevo(SALIDA/'imagenes.csv',filas,['imagen','fuente','estado','eventos','persona_ultima','propuestas_iniciales','registros_finales_guardados','objetos_activos_guardados','descartados','organicos_utilizables_48'])
    csv_nuevo(SALIDA/'pendientes.csv',pendientes,['imagen','objeto','indice_editor','nivel','motivos','archivo_original'])
    resumen={'imagenes_base':len(t.imagenes),'imagenes_con_guardado':sum(f['estado']!='sin_revision_guardada' for f in filas),
        'imagenes_completas':sum(f['estado']=='completa_usuario_unico' for f in filas),'imagenes_borrador':sum(f['estado']=='borrador_usuario' for f in filas),
        'imagenes_sin_guardado':sum(f['estado']=='sin_revision_guardada' for f in filas),'eventos':n_eventos,'personas':dict(identidades),
        'cadenas_validas':True,'eventos_huerfanos':orfanos,'anotaciones_guardadas':len(objetos),
        'objetos_activos_guardados':sum(not a['descartada'] for a in objetos),
        'objetos_activos_imagen_completa':sum(not a['descartada'] and a['imagen_cobertura_completa'] for a in objetos),
        'descartados':sum(a['descartada'] for a in objetos),'comparacion_propuestas':dict(dif),
        'organicos_utilizables_48':sum(f['organicos_utilizables_48'] for f in filas),
        'organicos_HPI_utilizables_48':sum(f['organicos_utilizables_48'] for f in filas if f['fuente']=='hpi'),
        'imagenes_con_errores':sum(bool(f['errores']) for f in filas),
        'imagenes_con_avisos_visuales':sum(bool(f['avisos_visuales']) for f in filas),
        'imagenes_fusion_representacion_48':sum(bool(f['fusiones_representacion_48']) for f in filas),
        'aceptadas_V26':0,'entrenamiento_autorizado':False,'test_evaluado':False,'respaldo':rec['destino']}
    publicar(SALIDA/'auditoria.json',resumen);print(resumen,flush=True)


if __name__=='__main__':main()
