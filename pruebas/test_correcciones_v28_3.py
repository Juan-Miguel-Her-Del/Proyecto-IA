"""Pruebas sobre figuras descartables; nunca registrar revisiones en fotos reales."""
from copy import deepcopy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.parse import quote
from pruebas.fixtures_anotacion_v27_2 import fixture
from ecowaste.correcciones_v28_3 import GestionCorrecciones,auditorias_verificadas,prioridad_vigente,PAPEL
from ecowaste.incorporacion import leer,sha
from scripts.anotar_v28_3 import servidor


def incidencia():
    return {'id':'inc_fixture','imagen':'fixture:1','objetos':['objeto_1'],'motivo':'comprobar material y cobertura',
        'tipo':'decision_visual','origen':'fixture_descartable','prioridad':1,'bloqueante':True,'posible_omision':False}


class CorreccionesV283(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.r=Path(self.tmp.name);self.t=fixture(self.r)
        self.g=GestionCorrecciones(self.t,self.r/'gestion',[incidencia()]);self.iid='fixture:1'
    def tearDown(self):self.tmp.cleanup()
    def datos(self,material='carton'):
        t=self.t.tarea(self.iid);anns=deepcopy(t['anotaciones']);anns[0]['material']=material
        return {'imagen_id':self.iid,'proyecto_sha256':self.t.base['huella'],'revision':t['revision'],'persona_id':'actor_fixture_1',
            'cobertura_completa':True,'anotaciones':anns,'id_solicitud':'sol_'+str(t['revision'])}
    def guardar(self,d=None):
        d=d or self.datos();return self.g.guardar(d,{'id':self.iid,'modo':'asistida','base':None,'revision':d['revision']})
    def resolver(self,estado='corregida',h=None):
        return self.g.resolver({'incidencia_id':'inc_fixture','estado':estado,'persona_id':'actor_fixture_1',
            'revision_huella':h or self.t.diario(self.iid)[-1]['huella'],'nota':'contorno y material comprobados en fixture'})

    def test_no_cerrar_sin_guardado_humano(self):
        with self.assertRaises(ValueError):self.g.resolver({'incidencia_id':'inc_fixture','estado':'corregida','persona_id':'actor_fixture_1','revision_huella':None,'nota':'no existe guardado'})
        self.assertEqual(self.t.diario(self.iid),[])
    def test_todas_las_omisiones_papel_son_prioridad_cuatro(self):
        self.assertEqual(len(PAPEL),12)
        for iid in PAPEL:self.assertEqual(prioridad_vigente({'imagen':iid,'prioridad':6}),4)
        self.assertEqual(prioridad_vigente({'imagen':PAPEL[0],'prioridad':7}),7)
    def test_cierre_no_crea_otra_revision_y_sigue_sin_V26(self):
        self.guardar();antes=sha(next((self.t.path/'revisiones').rglob('*.json')));e=self.resolver()
        self.assertFalse(e['aceptada_V26']);self.assertFalse(e['revision_independiente']);self.assertFalse(e['entrenamiento_autorizado'])
        self.assertEqual(self.g.listar()[0]['estado'],'corregida');self.assertEqual(len(self.t.diario(self.iid)),1)
        self.assertEqual(antes,sha(next((self.t.path/'revisiones').rglob('*.json'))))
    def test_guardado_nuevo_reabre_resolucion(self):
        self.guardar();self.resolver();self.guardar(self.datos('plastico'))
        r=self.g.listar()[0];self.assertEqual(r['estado'],'pendiente');self.assertTrue(r['resolucion_anterior_desactualizada'])
    def test_no_revisar_objeto_resuelto_si_solo_cambia_otro(self):
        self.guardar();self.resolver();d=self.datos();otro=deepcopy(d['anotaciones'][0]);otro.update(id='objeto_2',material='papel',poligonos=[[140,30,180,30,180,80,140,80]])
        d['anotaciones'].append(otro);self.guardar(d);self.assertEqual(self.g.listar()[0]['estado'],'corregida')
    def test_observacion_imagen_completa_se_reabre_si_cambia_cobertura(self):
        inc={**incidencia(),'objetos':[]};g=GestionCorrecciones(self.t,self.r/'gestion_global',[inc]);self.guardar()
        g.resolver({'incidencia_id':'inc_fixture','estado':'corregida','persona_id':'actor_fixture_1','revision_huella':self.t.diario(self.iid)[-1]['huella'],'nota':'imagen completa comprobada en fixture'})
        d=self.datos('papel');self.guardar(d);self.assertEqual(g.listar()[0]['estado'],'pendiente')
    def test_no_cerrar_usando_revision_antigua(self):
        self.guardar();h=self.t.diario(self.iid)[-1]['huella'];self.guardar(self.datos('plastico'))
        with self.assertRaises(ValueError):self.resolver(h=h)
    def test_estado_requiere_decision_no_libera_datos(self):
        self.guardar();self.resolver('requiere_decision');_,s=self.g.auditar(self.r/'auditorias')
        self.assertEqual(s['imagenes_candidatas'],0);self.assertFalse(s['entrenamiento_autorizado'])
    def test_cierre_no_elude_error_paridad(self):
        d=self.datos();d['anotaciones'][0]['poligonos'].append([40,40,70,40,70,65,40,65]);self.guardar(d)
        with self.assertRaisesRegex(ValueError,'errores técnicos'):self.resolver()
    def test_borrador_invalido_se_conserva_sin_ser_revision(self):
        d=self.datos();d['anotaciones'][0]['poligonos']=[[1,2]];r=self.g.borrador(d)
        self.assertEqual(leer(self.g.path/r['archivo'])['contenido'],d);self.assertEqual(len(self.t.diario(self.iid)),0)
    def test_borrador_de_otro_proyecto_rechazado(self):
        d=self.datos();d['proyecto_sha256']='ajeno'
        with self.assertRaises(ValueError):self.g.borrador(d)
    def test_recibo_evitar_duplicado_tras_respuesta_perdida(self):
        d=self.datos();a=self.guardar(d);b=self.g.guardar(d,None)
        self.assertTrue(b['reintento_idempotente']);self.assertEqual(a['revision_huella'],b['revision_huella']);self.assertEqual(len(self.t.diario(self.iid)),1)
    def test_recibo_reanudable_sin_vistas_en_memoria(self):
        d=self.datos();self.guardar(d);otra=GestionCorrecciones(self.t,self.g.path)
        self.assertTrue(otra.guardar(d,None)['reintento_idempotente']);self.assertEqual(len(self.t.diario(self.iid)),1)
    def test_id_solicitud_no_reutilizable_con_otro_contenido(self):
        d=self.datos();self.guardar(d);d['anotaciones'][0]['material']='plastico'
        with self.assertRaisesRegex(ValueError,'otro contenido'):self.g.guardar(d,None)
    def test_dos_vistas_no_sobrescriben_revision(self):
        a=self.datos();b=deepcopy(a);b['id_solicitud']='otra_solicitud';self.guardar(a)
        with self.assertRaisesRegex(ValueError,'Conflicto'):self.guardar(b)
        self.assertEqual(len(self.t.diario(self.iid)),1)
    def test_error_preciso_identifica_objeto(self):
        d=self.datos();d['anotaciones'][0]['material']='sin_asignar'
        with self.assertRaisesRegex(ValueError,'Objeto 1 .*objeto_1.*sin_asignar'):self.g.validar_guardado(d)
    def test_cache_reusa_y_recalcula_solo_guardado_nuevo(self):
        self.guardar();self.resolver();p,s=self.g.auditar(self.r/'auditorias');p2,s2=self.g.auditar(self.r/'auditorias')
        self.assertEqual(s['cache_recalculadas'],1);self.assertEqual(s2['cache_reusadas'],1);self.assertEqual(s2['snapshot_previo_sha256'],sha(p))
        self.guardar(self.datos('papel'));_,s3=self.g.auditar(self.r/'auditorias');self.assertEqual(s3['cache_recalculadas'],1)
        self.assertEqual(s3['imagenes_candidatas'],0);self.assertEqual(len(auditorias_verificadas(self.r/'auditorias')),3)
    def test_cache_alterada_no_autoriza_candidatura(self):
        self.guardar();self.g.auditar(self.r/'auditorias');p=next((self.g.path/'cache_qa').glob('*.json'));r=leer(p);r['qa']['errores']=['alterado'];p.write_text(json.dumps(r),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Caché QA alterada'):self.g.auditar(self.r/'auditorias')
    def test_snapshot_anterior_alterado_rechazado(self):
        self.guardar();p,_=self.g.auditar(self.r/'auditorias');(p.parent/'imagenes.json').write_text('[]',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'auditoría alterado'):self.g.auditar(self.r/'auditorias')
    def test_incidente_nuevo_aparece_navegable(self):
        d=self.datos();d['anotaciones'][0]['poligonos'].append([130,110,170,110,170,150,130,150]);self.guardar(d);self.g.auditar(self.r/'auditorias')
        self.assertTrue(any('fragmentacion' in r['motivo'] and r['estado']=='pendiente' for r in self.g.listar()))
    def test_aviso_tecnico_arreglado_deja_de_aparecer_sin_fingir_resolucion(self):
        d=self.datos();d['anotaciones'][0]['poligonos'].append([130,110,170,110,170,150,130,150]);self.guardar(d)
        self.g.auditar(self.r/'auditorias');inc=next(r for r in self.g.listar() if 'fragmentacion' in r['motivo'])
        d=self.datos();d['anotaciones'][0]['poligonos']=d['anotaciones'][0]['poligonos'][:1];self.guardar(d)
        self.g.auditar(self.r/'auditorias');r=next(r for r in self.g.listar() if r['id']==inc['id'])
        self.assertEqual(r['estado'],'no_vigente');self.assertIsNone(r['resolucion']);self.assertFalse(self.g.eventos())
        self.assertEqual(len(self.t.diario(self.iid)),2)
    def test_omision_visual_no_se_cierra_solo_por_guardar(self):
        self.guardar();self.g.auditar(self.r/'auditorias');self.assertEqual(self.g.listar()[0]['estado'],'pendiente')
    def test_sin_qa_actual_no_ocultar_aviso_tecnico(self):
        c={**incidencia(),'tipo':'visual','motivo':'fragmentacion_o_objetos_fusionados_requiere_inspeccion'}
        g=GestionCorrecciones(self.t,self.r/'sin_cache',[c]);self.guardar();self.assertEqual(g.listar()[0]['estado'],'pendiente')


class MotorFixture:
    def liberar(self):pass


class HTTPV283(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.r=Path(self.tmp.name);self.t=fixture(self.r)
        self.g=GestionCorrecciones(self.t,self.r/'gestion',[incidencia()]);self.iniciar()
    def iniciar(self):
        self.app=servidor(self.t,MotorFixture(),0,self.g);self.h=threading.Thread(target=self.app.serve_forever,daemon=True);self.h.start()
        self.puerto=self.app.server_port;self.token=self.req('GET','/api/config')[1]['token']
    def parar(self):self.app.shutdown();self.app.server_close();self.h.join()
    def tearDown(self):self.parar();self.tmp.cleanup()
    def req(self,method,path,d=None,**headers):
        c=http.client.HTTPConnection('127.0.0.1',self.puerto,timeout=10)
        if d is not None:headers={'Content-Type':'application/json','X-Eco-Token':self.token,**headers}
        c.request(method,path,json.dumps(d) if d is not None else None,headers);r=c.getresponse();blob=r.read();status=r.status
        data=json.loads(blob) if r.getheader('Content-Type','').startswith('application/json') else blob;c.close();return status,data
    def datos(self):
        t=self.req('GET','/api/tarea?id=fixture%3A1')[1]
        return {'imagen_id':'fixture:1','proyecto_sha256':self.t.base['huella'],'token_vista':t['token_vista'],
            'revision':t['revision'],'persona_id':'actor_fixture_1','cobertura_completa':True,'anotaciones':t['anotaciones'],'id_solicitud':'sol_http_fixture'}
    def test_config_CPU_e_interfaz_prioritaria(self):
        self.assertFalse(self.req('GET','/api/config')[1]['GPU_utilizada']);self.assertIn('Correcciones prioritarias',self.req('GET','/')[1].decode())
        self.assertEqual(self.req('GET','/api/prioridades')[1]['incidencias'][0]['estado'],'pendiente')
    def test_auditoria_vigente_no_depende_del_orden_de_la_lista(self):
        self.g.catalogo['incidencias'].append({**incidencia(),'id':'inc_fixture_otro'})
        p,s=self.g.auditar(self.g.path/'auditorias');i=p.parent/'incidencias.json'
        i.write_text(json.dumps(list(reversed(leer(i)))),encoding='utf-8')
        s['sha256_archivos']['incidencias.json']=sha(i);p.write_text(json.dumps(s),encoding='utf-8')
        self.assertTrue(self.req('GET','/api/auditoria')[1]['vigente']);self.assertFalse(self.t.diario('fixture:1'))
    def test_400_con_causa_objeto_y_copia_sin_evento(self):
        d=self.datos();d['anotaciones'][0]['poligonos']=[];s,r=self.req('POST','/api/guardar',d)
        self.assertEqual(s,400);self.assertIn('Objeto 1',r['error']);self.assertTrue(r['id_error'].startswith('err_'))
        p=self.g.path/r['borrador_emergencia_servidor']['archivo'];self.assertEqual(leer(p)['contenido'],d);self.assertEqual(len(self.t.diario('fixture:1')),0)
    def test_guardado_conserva_cobertura_y_contexto_nuevo(self):
        d=self.datos();s,r=self.req('POST','/api/guardar',d);self.assertEqual(s,200)
        self.assertTrue(r['tarea']['cobertura_completa']);self.assertEqual(r['tarea']['revision'],1);self.assertNotEqual(r['tarea']['token_vista'],d['token_vista'])
    def test_doble_clic_no_duplica_evento(self):
        d=self.datos();self.req('POST','/api/guardar',d);s,r=self.req('POST','/api/guardar',d)
        self.assertEqual(s,200);self.assertTrue(r['reintento_idempotente']);self.assertEqual(len(self.t.diario('fixture:1')),1)
    def test_recibo_sobrevive_reinicio(self):
        d=self.datos();self.req('POST','/api/guardar',d);self.parar();self.iniciar();s,r=self.req('POST','/api/guardar',d)
        self.assertEqual(s,200);self.assertTrue(r['reintento_idempotente']);self.assertEqual(len(self.t.diario('fixture:1')),1)
    def test_error_token_y_origen_no_escriben_revision(self):
        d=self.datos();self.assertEqual(self.req('POST','/api/guardar',d,**{'X-Eco-Token':'caducado'})[0],400)
        self.assertEqual(self.req('POST','/api/guardar',d,Origin='https://ajeno.example')[0],400);self.assertFalse(self.t.diario('fixture:1'))
    def test_no_recuperar_archivo_arbitrario(self):
        self.assertEqual(self.req('GET','/api/borradores?id=fixture%3A1&archivo=../../PROYECTO.json')[0],400)
    def test_conflicto_no_pierde_entrada(self):
        a,b=self.datos(),self.datos();b['id_solicitud']='sol_tab2';self.req('POST','/api/guardar',a);s,r=self.req('POST','/api/guardar',b)
        self.assertEqual(s,400);self.assertIn('Conflicto',r['error']);self.assertTrue((self.g.path/r['borrador_emergencia_servidor']['archivo']).is_file());self.assertEqual(len(self.t.diario('fixture:1')),1)
    def test_material_ambiguo_y_poligono_cruzado_dan_causa_exacta(self):
        for caso in ['ambiguo','cruzado']:
            d=self.datos()
            if caso=='ambiguo':d['anotaciones'][0].update(material='compuesto',motivo='')
            else:d['anotaciones'][0]['poligonos']=[[20,25,110,90,20,90,110,25]]
            s,r=self.req('POST','/api/guardar',d);self.assertEqual(s,400);self.assertIn('Objeto 1',r['error']);self.assertIsNotNone(r['borrador_emergencia_servidor'])
        self.assertFalse(self.t.diario('fixture:1'))


if __name__=='__main__':unittest.main()
