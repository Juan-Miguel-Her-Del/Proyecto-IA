"""Fixtures descartables: ninguna imagen/etiqueta de estas pruebas acredita datos reales."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

from ecowaste.incorporacion import (VERSION,CLASES,RAIZ,importar,sha,digest,cargar,poligonos_validos,
    caja_derivada,normalizar_paquete,qa_geometria,ruta_segura,destino_nuevo,publicar)
from ecowaste.revision_datos import registrar_revision,estado_revision,exportar_tareas,leer_diario,doble_obligatoria
from ecowaste.coleccion_nueva import consolidar,particionar,inventario,verificar_plan,congelar,ColeccionCongelada


def revisores(tipo='fixture'):
    return [{'id':x,'persona_id':'persona_'+x,'humano':True,'tipo':tipo,'roles':['anotador','adjudicador']} for x in ['a','b','c']]


def muestra(root,clases=('carton','organico'),tipo='fixture',cid='fixture_lote'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);ims=[];anns=[];fis=[]
    for i,c in enumerate(clases):
        imagen=root/f'{i}.png';Image.fromarray(np.random.default_rng(i+9).integers(0,255,(64,96,3),dtype=np.uint8)).save(imagen)
        eco={'sha256_archivo':sha(imagen),'escena_id':f'escena_{i}','sitio_id':f'sitio_{i}',
            'sesion_id':f'sesion_{i}','grupo_captura_id':f'captura_{i}','secuencia_id':None,
            'dispositivo_id':'camara_fixture','capturado_en':'2026-11-21T10:00:00-05:00','destino_previsto':'train_nuevo','objetos_fisicos_ids':[f'obj_{i}'],
            'procedencia':{'tipo':'fixture' if tipo=='fixture' else 'captura_real','autor':'fixture_no_real',
                'referencia':'fixture_descartable','licencia':'solo_fixture','permiso_ref':'test','permiso_confirmado':True}}
        ims.append({'id':str(i),'file_name':imagen.name,'width':96,'height':64,'eco':eco})
        fis.append({'id':f'obj_{i}','material_declarado':'por_verificar','evidencia_identidad':'registro de fixture, no objeto real'})
        anns.append({'id':str(i),'image_id':str(i),'category_id':CLASES.index(c)+1,
            'segmentation':[[12,12,42,12,42,37,12,37]],'bbox':[12,12,30,25],
            'eco':{'objeto_fisico_id':f'obj_{i}','origen':'propuesta_automatica','visible_only':True,
                'oclusion':'ninguna','truncado':False,'transparente':False,'solapa_con':[]}})
    d={'eco':{'version':VERSION,'coleccion_id':cid,'tipo_datos':tipo},'images':ims,'annotations':anns,
        'physical_objects':fis,'categories':[{'id':i+1,'name':c} for i,c in enumerate(CLASES)]}
    archivo=root/'annotations.json';archivo.write_text(json.dumps(d),encoding='utf-8')
    return d,archivo


def evento(paquete,im,fase,actor,instancias=None):
    c=cargar(paquete)
    return {'version':VERSION,'imagen_id':im['id'],'sha256_coleccion':digest(c),'sha256_imagen':im['sha256_archivo'],
        'fase':fase,'revisor_id':actor,'cobertura_completa':True,'motivo':'fixture de revisión, sin personas ni datos reales',
        'declaracion':'humana_adjudicada' if fase=='adjudicacion' else 'humana_independiente',
        'instancias':copy.deepcopy(im['instancias'] if instancias is None else instancias),'solapes_revisados':True}


def aceptar(paquete):
    c=cargar(paquete)
    for im in c['imagenes']:
        for fase,actor in [('primera','a'),('segunda','b'),('adjudicacion','c')]:registrar_revision(paquete,evento(paquete,im,fase,actor))


class PipelineV26(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.data,self.entrada=muestra(self.root/'fuente');self.paquete=self.root/'importado'

    def escribir(self):self.entrada.write_text(json.dumps(self.data),encoding='utf-8')

    def importar(self,**kw):return importar(self.entrada,self.entrada.parent,self.paquete,revisores(),ensayo=True,historicos=[],**kw)

    def test_importa_en_cuarentena_y_no_cuenta_propuestas(self):
        c=self.importar();self.assertEqual(len(c['imagenes']),2)
        est=estado_revision(self.paquete)[1];self.assertTrue(all(x['estado']=='pendiente_primera' for x in est))
        inv=inventario(consolidar([self.paquete]));self.assertEqual(inv['splits']['train_nuevo']['por_material']['organico']['instancias_computables_cuota'],0)
        self.assertFalse(inv['coleccion_suficiente_y_revisada'])

    def test_modo_real_no_admite_fixture(self):
        with self.assertRaises(ValueError):importar(self.entrada,self.entrada.parent,self.paquete,revisores('real'),historicos=[])
        self.assertFalse(self.paquete.exists())

    def test_no_sobrescribe_importacion(self):
        self.importar()
        with self.assertRaises(FileExistsError):self.importar()

    def test_rutas_relativas_y_escape_windows(self):
        for path in ['../afuera.png','C:/imagen.png','C:imagen.png','\\servidor\\imagen.png','/absoluta.png']:
            with self.subTest(path=path),self.assertRaises(ValueError):ruta_segura(self.root,path)

    def test_no_escribe_en_historicos_ni_por_api(self):
        for path in ['datos/procesados/nuevo_v26','datos/originales/nuevo_v26','resultados/v25_datos_representacion/otra_cosa','checkpoints/nuevo_v26']:
            with self.subTest(path=path),self.assertRaises(ValueError):destino_nuevo(RAIZ/path)

    def test_revision_invalida_no_crea_lock_historico(self):
        falso=self.root/'no_paquete';falso.mkdir()
        with self.assertRaises(FileNotFoundError):registrar_revision(falso,{})
        self.assertFalse((falso/'revision.lock').exists())

    def test_poligonos_cruces_repetidos_fuera_nan_y_sin_area(self):
        for p in [[0,0,10,10,0,10,10,0],[0,0,10,0,10,0],[0,0,1,float('nan'),2,0],[-1,0,10,0,10,10],[0,0,2,2,4,4]]:
            with self.subTest(p=p),self.assertRaises(ValueError):poligonos_validos([p],20,20)

    def test_poligono_cerrado_se_normaliza_sin_cambiar_bbox(self):
        ps=poligonos_validos([[0,0,10,0,10,10,0,10,0,0]],20,20)
        self.assertEqual(caja_derivada(ps),[0.,0.,10.,10.]);self.assertEqual(len(ps[0]),8)

    def test_no_global_labels_bbox_only_ni_huecos(self):
        for fallo in ['sin_poligono','holes','visible']:
            d=copy.deepcopy(self.data)
            if fallo=='sin_poligono':d['annotations'][0].pop('segmentation')
            elif fallo=='holes':d['annotations'][0]['eco']['holes']=[[1,1,2,1,2,2]]
            else:d['annotations'][0]['eco']['visible_only']=False
            with self.subTest(fallo=fallo),self.assertRaises(ValueError):normalizar_paquete(d,self.entrada.parent,revisores(),ensayo=True)

    def test_rechaza_bbox_incoherente_y_area_no_finita(self):
        self.data['annotations'][0]['bbox'][2]=50
        with self.assertRaises(ValueError):normalizar_paquete(self.data,self.entrada.parent,revisores(),ensayo=True)
        self.data['annotations'][0]['bbox'][2]=30;self.data['annotations'][0]['area']=float('inf')
        with self.assertRaises(ValueError):normalizar_paquete(self.data,self.entrada.parent,revisores(),ensayo=True)

    def test_rechaza_material_o_identidad_desconocidos(self):
        self.data['categories'][0]['name']='material_inventado'
        with self.assertRaises(ValueError):normalizar_paquete(self.data,self.entrada.parent,revisores(),ensayo=True)
        self.data['categories'][0]['name']='plastico';self.data['annotations'][0]['eco']['objeto_fisico_id']='no_registrado'
        with self.assertRaises(ValueError):normalizar_paquete(self.data,self.entrada.parent,revisores(),ensayo=True)

    def test_rechaza_dimension_sha_fecha_y_permiso(self):
        for campo,valor in [('width',99),('sha256_archivo','0'*64),('capturado_en','2026-11-21'),('permiso_confirmado',False)]:
            d=copy.deepcopy(self.data)
            if campo=='width':d['images'][0][campo]=valor
            elif campo=='permiso_confirmado':d['images'][0]['eco']['procedencia'][campo]=valor
            else:d['images'][0]['eco'][campo]=valor
            with self.subTest(campo=campo),self.assertRaises(ValueError):normalizar_paquete(d,self.entrada.parent,revisores(),ensayo=True)

    def test_duplicados_pixel_aunque_archivo_distinto(self):
        im=Image.open(self.entrada.parent/'0.png');im.save(self.entrada.parent/'1.png',compress_level=0)
        self.data['images'][1]['eco']['sha256_archivo']=sha(self.entrada.parent/'1.png');self.escribir()
        with self.assertRaises(ValueError):self.importar()

    def test_imagen_historica_no_cuenta_como_adquisicion_real(self):
        self.data['eco']['tipo_datos']='real'
        for im in self.data['images']:im['eco']['procedencia']['tipo']='captura_real'
        self.escribir();h={'id':'historico','sha256_archivo':self.data['images'][0]['eco']['sha256_archivo']}
        with self.assertRaises(ValueError):importar(self.entrada,self.entrada.parent,self.paquete,revisores('real'),historicos=[h])
        self.assertFalse(self.paquete.exists())

    def test_importacion_incompleta_y_tamper_rechazados(self):
        self.paquete.mkdir()
        with self.assertRaises(FileNotFoundError):cargar(self.paquete)
        self.paquete.rmdir();self.importar();(self.paquete/'coleccion.json').write_text('{}')
        with self.assertRaises(ValueError):cargar(self.paquete)

    def test_exportaciones_independientes_no_revelan_propuestas(self):
        self.importar();ts=exportar_tareas(self.paquete,self.root/'tareas')
        self.assertEqual(len(ts),4);self.assertTrue(all(x['instancias']==[] and 'anotaciones_previas' not in x for x in ts))

    def test_no_solo_aprobacion_automatica_o_version_antigua(self):
        c=self.importar();im=c['imagenes'][0]
        for campo,valor in [('declaracion','automatica'),('sha256_coleccion','0'*64),('sha256_imagen','0'*64),('cobertura_completa',False)]:
            ev=evento(self.paquete,im,'primera','a');ev[campo]=valor
            with self.subTest(campo=campo),self.assertRaises(ValueError):registrar_revision(self.paquete,ev)

    def test_prioritarias_requieren_dos_anotaciones_y_tercera_persona(self):
        c=self.importar();im=c['imagenes'][0];registrar_revision(self.paquete,evento(self.paquete,im,'primera','a'))
        with self.assertRaises(ValueError):registrar_revision(self.paquete,evento(self.paquete,im,'adjudicacion','c'))
        with self.assertRaises(ValueError):registrar_revision(self.paquete,evento(self.paquete,im,'segunda','a'))
        registrar_revision(self.paquete,evento(self.paquete,im,'segunda','b'))
        with self.assertRaises(ValueError):registrar_revision(self.paquete,evento(self.paquete,im,'adjudicacion','b'))
        registrar_revision(self.paquete,evento(self.paquete,im,'adjudicacion','c'))
        self.assertEqual(estado_revision(self.paquete)[1][0]['estado'],'aceptada')

    def test_alias_de_misma_persona_no_simula_doble_anotacion(self):
        rs=revisores();rs[1]['persona_id']=rs[0]['persona_id'];importar(self.entrada,self.entrada.parent,self.paquete,rs,ensayo=True,historicos=[])
        im=cargar(self.paquete)['imagenes'][0];registrar_revision(self.paquete,evento(self.paquete,im,'primera','a'))
        with self.assertRaises(ValueError):registrar_revision(self.paquete,evento(self.paquete,im,'segunda','b'))

    def test_discrepancia_material_y_omision_no_se_promedian(self):
        c=self.importar();im=c['imagenes'][0];registrar_revision(self.paquete,evento(self.paquete,im,'primera','a'))
        segunda=copy.deepcopy(im['instancias']);segunda[0]['material']='vidrio'
        registrar_revision(self.paquete,evento(self.paquete,im,'segunda','b',segunda))
        estado=estado_revision(self.paquete)[1][0];self.assertEqual(estado['estado'],'pendiente_adjudicacion')
        self.assertEqual(estado['discrepancias'][0]['tipo'],'material_o_geometria')
        registrar_revision(self.paquete,evento(self.paquete,im,'adjudicacion','c'))
        self.assertEqual(estado_revision(self.paquete)[1][0]['instancias_finales'][0]['material'],'carton')

    def test_adjudicador_detecta_omision(self):
        c=self.importar();im=c['imagenes'][0];registrar_revision(self.paquete,evento(self.paquete,im,'primera','a',[]))
        registrar_revision(self.paquete,evento(self.paquete,im,'segunda','b'))
        self.assertEqual(estado_revision(self.paquete)[1][0]['discrepancias'][0]['tipo'],'omision_o_instancia_adicional')

    def test_diario_inmutable_y_corrupcion(self):
        c=self.importar();im=c['imagenes'][0];ev=evento(self.paquete,im,'primera','a');registrar_revision(self.paquete,ev)
        with self.assertRaises(ValueError):registrar_revision(self.paquete,ev)
        path=self.paquete/'revisiones/000001.json';r=json.loads(path.read_text());r['motivo']='alterado';path.write_text(json.dumps(r))
        with self.assertRaises(ValueError):leer_diario(self.paquete)

    def test_qa_organico_utilizable_y_pequeno_no_descartado(self):
        c=self.importar();im=c['imagenes'][1];qa=qa_geometria(im)
        self.assertTrue(qa['instancias'][0]['organico_utilizable_48'])
        a=im['instancias'][0];a['poligonos']=[[12,12,12.5,12,12.5,12.5,12,12.5]];a['bbox']=[12,12,.5,.5];a['area_mask_px']=1
        tiny=qa_geometria(im)['instancias'][0];self.assertTrue(tiny['pequeno_menos_8px']);self.assertFalse(tiny['organico_utilizable_48'])

    def test_solape_material_no_es_organico_utilizable(self):
        c=self.importar();im=copy.deepcopy(c['imagenes'][1]);a=copy.deepcopy(im['instancias'][0]);a['id']='otro';a['material']='plastico';a['objeto_fisico_id']='otro'
        im['instancias'].append(a);q=qa_geometria(im)
        self.assertTrue(q['solapes'][0]['materiales_distintos']);self.assertFalse(q['instancias'][0]['organico_utilizable_48'])

    def test_una_sola_componente_no_se_divide_para_cumplir_cuotas(self):
        self.importar();co=consolidar([self.paquete])
        for im in co['imagenes']:im['sitio_id']='mismo_sitio';im['destino_previsto']='sin_asignar'
        p=particionar(co);self.assertEqual(len(p['grupos']),1);self.assertEqual(set(p['asignacion'].values()),{'train_nuevo'})

    def test_union_transitiva_fisica_sesion_sitio_secuencia(self):
        self.importar();co=consolidar([self.paquete]);base=copy.deepcopy(co['imagenes'][0]);ims=[]
        for i in range(5):
            x=copy.deepcopy(base);x.update(id=str(i),sitio_id='s'+str(i),sesion_id='ses'+str(i),escena_id='e'+str(i),grupo_captura_id='g'+str(i),secuencia_id=None,
                objetos_para_grupos=['o'+str(i)],destino_previsto='sin_asignar',phash=f'{(1<<(i*8))-1:016x}');ims.append(x)
        ims[1]['objetos_para_grupos']=ims[0]['objetos_para_grupos'];ims[2]['sesion_id']=ims[1]['sesion_id'];ims[3]['sitio_id']=ims[2]['sitio_id'];ims[3]['secuencia_id']='seq';ims[4]['secuencia_id']='seq'
        co['imagenes']=ims;p=particionar(co);self.assertEqual(len(p['grupos']),1)

    def test_destinos_previos_opuestos_rechazan_fuga(self):
        self.importar();co=consolidar([self.paquete]);co['imagenes'][1]['sitio_id']=co['imagenes'][0]['sitio_id'];co['imagenes'][1]['destino_previsto']='generalizacion_bloqueada'
        with self.assertRaises(ValueError):particionar(co)

    def test_grupos_reproducibles_sin_depender_del_orden(self):
        self.importar();co=consolidar([self.paquete]);p=particionar(co);co['imagenes'].reverse();q=particionar(co)
        self.assertEqual(p['asignacion'],q['asignacion']);self.assertEqual(p['grupos'],q['grupos'])

    def test_plan_alterado_y_cohorte_anticuada(self):
        self.importar();co=consolidar([self.paquete]);p=particionar(co);p['asignacion'][co['imagenes'][0]['id']]='generalizacion_bloqueada'
        with self.assertRaises(ValueError):verificar_plan(co,p)

    def test_fixture_revisada_nunca_satisface_cuota_real_ni_congela_prod(self):
        self.importar();aceptar(self.paquete);co=consolidar([self.paquete]);p=particionar(co);inv=inventario(co,p)
        self.assertEqual(inv['splits']['train_nuevo']['por_material']['organico']['organicos_utilizables_48'],0)
        with self.assertRaises(ValueError):congelar(co,p,self.root/'produccion')
        self.assertFalse((self.root/'produccion').exists())

    def test_exportacion_taco_ensayo_y_adaptador_sin_detector(self):
        self.importar();aceptar(self.paquete);co=consolidar([self.paquete]);p=particionar(co);out=self.root/'ensayo';congelar(co,p,out,ensayo=True)
        self.assertFalse((out/'CONGELADA.json').exists());ds=ColeccionCongelada(out,ensayo=True)
        x,obj=ds[0];self.assertEqual(tuple(x.shape),(3,384,384));self.assertEqual(len(obj['boxes']),1)
        r=ds.registros[0];self.assertTrue(ds.contornos(r,r['cajas'][0]));self.assertEqual(tuple(ds.rasterizar(r).shape),(48,48))
        with self.assertRaises(PermissionError):ColeccionCongelada(out,ensayo=True,split='generalizacion_bloqueada')
        with self.assertRaises(FileNotFoundError):ColeccionCongelada(out)
        falsa=copy.deepcopy(r['cajas'][0]);falsa['clase']='metal'
        with self.assertRaises(ValueError):ds.contornos(r,falsa)

    def test_congelacion_rechaza_estados_inventados(self):
        self.importar();co=consolidar([self.paquete]);co['imagenes'][0]['estado']='aceptada'
        with self.assertRaises(ValueError):congelar(co,particionar(co),self.root/'ensayo',ensayo=True)

    def test_no_computar_repeticion_de_frames_ni_cuotas_sinteticas(self):
        self.importar();aceptar(self.paquete);co=consolidar([self.paquete]);im=copy.deepcopy(co['imagenes'][1]);ims=[]
        for i in range(10):x=copy.deepcopy(im);x['id']=str(i);ims.append(x)
        co['imagenes']=ims;co['tipo_datos']='real';inv=inventario(co)
        q=inv['splits']['train_nuevo']['por_material']['organico'];self.assertEqual(q['instancias_adjudicadas'],10)
        self.assertEqual(q['instancias_computables_cuota'],1);self.assertEqual(q['organicos_utilizables_48'],1)

    def test_inventario_vacio_reporta_pendientes_no_imagenes_inventadas(self):
        inv=inventario(consolidar([]));self.assertFalse(inv['coleccion_suficiente_y_revisada'])
        self.assertEqual(sum(x['pendientes'] for x in inv['splits']['train_nuevo']['por_material'].values()),1550)
        self.assertEqual(inv['splits']['train_nuevo']['imagenes_recibidas'],0)

    def test_objeto_registrado_en_captura_aunque_falte_anotacion(self):
        self.data['annotations']=[];self.escribir();self.importar();co=consolidar([self.paquete])
        self.assertEqual(co['imagenes'][0]['objetos_para_grupos'],['obj_0'])

    def test_no_adjudicar_objeto_visible_omitido_sin_justificacion(self):
        c=self.importar();im=c['imagenes'][0]
        registrar_revision(self.paquete,evento(self.paquete,im,'primera','a'))
        registrar_revision(self.paquete,evento(self.paquete,im,'segunda','b'))
        fin=evento(self.paquete,im,'adjudicacion','c',[])
        with self.assertRaises(ValueError):registrar_revision(self.paquete,fin)
        fin['no_visibles']={'obj_0':{'motivo':'no_es_residuo','detalle':'control negativo simulado en fixture'}}
        registrar_revision(self.paquete,fin);self.assertEqual(estado_revision(self.paquete)[1][0]['instancias_finales'],[])

    def test_duplicados_perceptuales_unen_grupos_sin_borrar_imagenes(self):
        self.importar();co=consolidar([self.paquete])
        for im in co['imagenes']:im['phash']='0'*16;im['destino_previsto']='sin_asignar'
        p=particionar(co);self.assertEqual(len(p['grupos']),1);self.assertEqual(len(p['grupos'][0]['imagenes']),2)

    def test_no_alterar_imagen_despues_de_congelar(self):
        self.importar();aceptar(self.paquete);co=consolidar([self.paquete]);out=self.root/'ensayo'
        congelar(co,particionar(co),out,ensayo=True);ds=ColeccionCongelada(out,ensayo=True)
        (out/ds.registros[0]['ruta']).write_bytes(b'alterada')
        with self.assertRaises(ValueError):ds[0]
        with self.assertRaises(ValueError):ColeccionCongelada(out,ensayo=True)

    def test_limite_organicos_utiles_por_objeto_fisico(self):
        self.importar();aceptar(self.paquete);co=consolidar([self.paquete]);im=copy.deepcopy(co['imagenes'][1]);ims=[]
        for i in range(4):
            x=copy.deepcopy(im);x.update(id=str(i),escena_id='e'+str(i),sesion_id='s'+str(i),grupo_captura_id='g'+str(i));ims.append(x)
        co['imagenes']=ims;co['tipo_datos']='real';inv=inventario(co)
        q=inv['splits']['train_nuevo']['por_material']['organico']
        self.assertEqual(q['instancias_computables_cuota'],4);self.assertEqual(q['organicos_utilizables_48'],3)

    def test_material_ambiguo_se_exporta_como_ignore(self):
        from ecowaste.coleccion_nueva import exportar_taco
        c=self.importar();im=copy.deepcopy(c['imagenes'][0]);im['instancias'][0]['material']='compuesto'
        im['instancias'][0]['motivo_ambiguo']='multicapa no separable';taco,registros=exportar_taco([im])
        self.assertIsNone(registros[0]['cajas'][0]['clase']);self.assertEqual(taco['annotations'][0]['segmentation'],im['instancias'][0]['poligonos'])

    def test_conflicto_material_de_mismo_objeto_se_detecta(self):
        self.importar();aceptar(self.paquete);co=consolidar([self.paquete])
        # Cambiar los estados finales en memoria no permite pasar la puerta de congelación.
        co['imagenes'][1]['instancias'][0]['objeto_fisico_id']='obj_0'
        with self.assertRaises(ValueError):congelar(co,particionar(co),self.root/'falso',ensayo=True)

    def test_rechaza_dimensiones_float_y_objeto_no_presente(self):
        d=copy.deepcopy(self.data);d['images'][0]['width']=96.0
        with self.assertRaises(ValueError):normalizar_paquete(d,self.entrada.parent,revisores(),ensayo=True)
        d=copy.deepcopy(self.data);d['images'][0]['eco']['objetos_fisicos_ids']=[]
        with self.assertRaises(ValueError):normalizar_paquete(d,self.entrada.parent,revisores(),ensayo=True)

    def test_anotacion_duplicada_no_se_disfraza_con_otro_id_fisico(self):
        d=copy.deepcopy(self.data);d['physical_objects'].append({'id':'otro','material_declarado':'por_verificar','evidencia_identidad':'fixture'})
        d['images'][0]['eco']['objetos_fisicos_ids'].append('otro');a=copy.deepcopy(d['annotations'][0]);a['id']='duplicado'
        a['eco']['objeto_fisico_id']='otro';a['segmentation'][0]=[42,37,42,12,12,12,12,37];d['annotations'].append(a)
        with self.assertRaises(ValueError):normalizar_paquete(d,self.entrada.parent,revisores(),ensayo=True)

    def test_hash_perceptual_mismo_metodo_que_inventario_historico(self):
        import cv2
        c=self.importar();im=c['imagenes'][0]
        with Image.open(self.entrada.parent/'0.png') as image:g=np.asarray(image.convert('RGB').resize((32,32),Image.Resampling.LANCZOS).convert('L'),dtype=np.float32)
        dct=cv2.dct(g)[:8,:8].flatten();b=dct>np.median(dct[1:]);b[0]=False
        self.assertEqual(im['phash'],f'{sum(int(v)<<i for i,v in enumerate(b)):016x}')

    def test_manifiesto_generalizacion_no_se_etiqueta_train(self):
        from ecowaste.coleccion_nueva import exportar_taco
        c=self.importar();_,rr=exportar_taco(c['imagenes'],split='generalizacion_bloqueada')
        self.assertTrue(all(r['split']=='generalizacion_bloqueada' for r in rr))


if __name__=='__main__':unittest.main()
