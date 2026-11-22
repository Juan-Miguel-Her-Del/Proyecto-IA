"""Ensayo descartable con dos escenas TACO train reutilizadas, nunca adquisición nueva."""
import copy
import argparse
import json
from pathlib import Path
import sys
import tempfile
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.incorporacion import VERSION,CLASES,importar,leer,sha,digest,publicar
from ecowaste.revision_datos import registrar_revision,estado_revision,exportar_tareas
from ecowaste.coleccion_nueva import consolidar,particionar,inventario,congelar,ColeccionCongelada
from ecowaste.poligonos import ContornosTACO


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--salida',type=Path,default=RAIZ/'resultados/v26_pipeline_datos/ensayo_integracion.json');args=ap.parse_args()
    torch.set_num_threads(4);salida=args.salida
    if salida.exists():raise FileExistsError('Ensayo ya registrado')
    rr=[json.loads(l) for l in (RAIZ/'datos/procesados/deteccion/train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()][:2]
    cont=ContornosTACO(RAIZ,1/64);origenes=[]
    for r in rr:
        for a in r['cajas']:cont.contornos(r,a)  # Correspondencia original, sin fabricar geometría.
        origenes.append({'id':r['id'],'sha256':sha(RAIZ/r['ruta'])})
    with tempfile.TemporaryDirectory(prefix='ecowaste_v26_descartable_') as td:
        root=Path(td);fuente=root/'fuente';fuente.mkdir();images=[];anns=[];objs=[]
        for i,r in enumerate(rr):
            p=fuente/(str(i)+Path(r['ruta']).suffix);p.write_bytes((RAIZ/r['ruta']).read_bytes())
            aa=[a for a in r['cajas'] if a['clase']];ids=[f'fixture_obj_{a["id_interno"]}' for a in aa]
            images.append({'id':i,'file_name':p.name,'width':r['ancho'],'height':r['alto'],'eco':{
                'sha256_archivo':sha(p),'escena_id':f'fixture_escena_{i}','sitio_id':f'fixture_sitio_{i}',
                'sesion_id':f'fixture_sesion_{i}','grupo_captura_id':f'fixture_grupo_{i}','secuencia_id':None,
                'dispositivo_id':'fixture_no_dispositivo_real_registrado','capturado_en':'2026-11-21T12:00:00-05:00',
                'destino_previsto':'train_nuevo' if i==0 else 'generalizacion_bloqueada','objetos_fisicos_ids':ids,
                'procedencia':{'tipo':'reuso_historico','autor':'autores originales TACO, fixture local',
                    'referencia':r['id'],'licencia':r['licencia'],'permiso_ref':'ensayo_local_no_adquisicion_ni_republicacion',
                    'permiso_confirmado':True}}})
            for oid,a in zip(ids,aa):
                objs.append({'id':oid,'material_declarado':a['clase'],'evidencia_identidad':'identificador de fixture enlazado al GT histórico, no identidad física nueva'})
                anns.append({'id':a['id_interno'],'image_id':i,'category_id':CLASES.index(a['clase'])+1,
                    'segmentation':cont.contornos(r,a),'bbox':a['bbox'],'eco':{'objeto_fisico_id':oid,'origen':'humana',
                        'visible_only':True,'oclusion':'ninguna','truncado':False,'transparente':False,'solapa_con':[]}})
        data={'eco':{'version':VERSION,'coleccion_id':'fixture_taco_train_descartable','tipo_datos':'fixture'},
            'images':images,'annotations':anns,'physical_objects':objs,'categories':[{'id':i+1,'name':c} for i,c in enumerate(CLASES)]}
        entrada=fuente/'annotations.json';entrada.write_text(json.dumps(data),encoding='utf-8')
        revisores=[{'id':x,'persona_id':x,'tipo':'fixture','humano':True,'roles':['anotador','adjudicador']} for x in ['fixture_a','fixture_b','fixture_c']]
        paquete=root/'importado';c=importar(entrada,fuente,paquete,revisores,ensayo=True)
        tareas=exportar_tareas(paquete,root/'tareas_independientes')
        if any(t['instancias'] for t in tareas):raise AssertionError('Fuga de propuestas en tarea independiente')
        # Simulación explícita de revisión para probar la máquina de estados, nunca humanos reales.
        for im in c['imagenes']:
            for fase,actor in [('primera','fixture_a'),('segunda','fixture_b'),('adjudicacion','fixture_c')]:
                registrar_revision(paquete,{'version':VERSION,'imagen_id':im['id'],'sha256_coleccion':digest(c),'sha256_imagen':im['sha256_archivo'],
                    'fase':fase,'revisor_id':actor,'cobertura_completa':True,'motivo':'simulación descartable de integración; no adjudicación real',
                    'declaracion':'humana_adjudicada' if fase=='adjudicacion' else 'humana_independiente',
                    'instancias':copy.deepcopy(im['instancias']),'solapes_revisados':True})
        cohorte=consolidar([paquete]);plan=particionar(cohorte);inv=inventario(cohorte,plan)
        rechazo=False
        try:congelar(cohorte,plan,root/'produccion_prohibida')
        except ValueError:rechazo=True
        if not rechazo:raise AssertionError('Una fixture llegó a producción')
        export= root/'exportado';congelar(cohorte,plan,export,ensayo=True);ds=ColeccionCongelada(export,ensayo=True)
        x,gt=ds[0];r=ds.registros[0];mask=ds.rasterizar(r)
        bloqueo=False
        try:ColeccionCongelada(export,ensayo=True,split='generalizacion_bloqueada')
        except PermissionError:bloqueo=True
        prueba={'aprobada':True,'tipo':'fixture_reuso_de_train_no_nuevos_datos','imagenes':len(images),'instancias':len(anns),
            'origenes':origenes,'fases_simuladas':6,'tareas_independientes_sin_propuestas':True,
            'rechazo_congelacion_real':rechazo,'bloqueo_generalizacion':bloqueo,'forma_imagen':list(x.shape),
            'forma_mascara':list(mask.shape),'cajas_train_adaptador':len(gt['boxes']),
            'sha256_entrada_fixture':sha(entrada),'sha256_plan_fixture':digest(plan),
            'hashes_codigo':{f:sha(RAIZ/f) for f in ['ecowaste/incorporacion.py','ecowaste/revision_datos.py','ecowaste/coleccion_nueva.py','scripts/ensayar_v26.py']},
            'datos_reales_contabilizados':sum(q['instancias_computables_cuota'] for s in inv['splits'].values() for q in s['por_material'].values()),
            'entrenamiento':False,'inferencia':False,'test_evaluado':False}
    prueba['directorio_fixture_eliminado']=not root.exists()
    if not prueba['directorio_fixture_eliminado'] or prueba['datos_reales_contabilizados']:raise AssertionError('Fixture no descartada o contada como real')
    publicar(salida,prueba);print(json.dumps(prueba,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
