"""API local: aislamiento, versiones y estados, usando solo un fixture descartable."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.parse import quote

from pruebas.fixtures_anotacion_v27_2 import fixture
from scripts.anotar_v27_2 import servidor


class MotorFixture:
    def liberar(self):pass


class EditorHTTPV272(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='ecowaste_http_fixture_')
        self.t=fixture(Path(self.tmp.name));self.app=servidor(self.t,MotorFixture(),0)
        self.hilo=threading.Thread(target=self.app.serve_forever,daemon=True);self.hilo.start()
        self.puerto=self.app.server_port
        self.token=self.req('GET','/api/config')[1]['token']

    def tearDown(self):
        self.app.shutdown();self.app.server_close();self.hilo.join();self.tmp.cleanup()

    def req(self,method,path,body=None,**headers):
        c=http.client.HTTPConnection('127.0.0.1',self.puerto,timeout=5)
        if body is not None:
            headers={'Content-Type':'application/json','X-Eco-Token':self.token,**headers}
        c.request(method,path,json.dumps(body) if body is not None else None,headers)
        r=c.getresponse();raw=r.read();status=r.status
        data=json.loads(raw) if r.getheader('Content-Type','').startswith('application/json') else raw
        c.close();return status,data

    def tarea(self,independiente=False):
        return self.req('GET','/api/tarea?id='+quote('fixture:1')+'&modo='+('independiente' if independiente else 'asistida'))[1]

    def entrada(self,t):
        return {'imagen_id':'fixture:1','token_vista':t['token_vista'],'revision':t['revision'],
                'persona_id':'actor_fixture_1','cobertura_completa':True,'anotaciones':t['anotaciones']}

    def test_interfaz_y_config_sin_GPU(self):
        self.assertEqual(self.req('GET','/')[0],200)
        cfg=self.req('GET','/api/config')[1];self.assertEqual(cfg['dispositivo'],'cpu');self.assertFalse(cfg['GPU_utilizada'])
        self.assertIn(b'canvas',self.req('GET','/')[1])

    def test_post_sin_token_no_escribe_diario(self):
        status,_=self.req('POST','/api/guardar',self.entrada(self.tarea()),**{'X-Eco-Token':'incorrecto'})
        self.assertEqual(status,400);self.assertFalse(self.t.diario('fixture:1'))

    def test_origen_externo_no_puede_escribir(self):
        status,_=self.req('POST','/api/guardar',self.entrada(self.tarea()),Origin='https://ajeno.example')
        self.assertEqual(status,400);self.assertFalse(self.t.diario('fixture:1'))

    def test_host_ajeno_y_rutas_no_listadas_no_exponen_archivos(self):
        self.assertEqual(self.req('GET','/api/config',Host='ajeno.example')[0],400)
        self.assertEqual(self.req('GET','/../../README.md')[0],404)
        self.assertEqual(self.req('GET','/api/imagen?id=../../README.md')[0],400)

    def test_vista_no_admite_falsa_revision_independiente(self):
        d=self.entrada(self.tarea());d['modo']='independiente'
        self.assertEqual(self.req('POST','/api/guardar',d)[0],200)
        self.assertEqual(self.t.diario('fixture:1')[0]['modo'],'asistida')
        self.assertEqual(self.t.tarea('fixture:1')['estado'],'corregida_usuario_unico')

    def test_dos_pestanas_no_sobrescriben_correcciones(self):
        a,b=self.tarea(),self.tarea()
        self.assertEqual(self.req('POST','/api/guardar',self.entrada(a))[0],200)
        self.assertEqual(self.req('POST','/api/guardar',self.entrada(b))[0],400)
        self.assertEqual(len(self.t.diario('fixture:1')),1)

    def test_independiente_sin_propuestas_rechaza_mismo_actor(self):
        self.req('POST','/api/guardar',self.entrada(self.tarea()))
        t=self.tarea(True);self.assertEqual(t['anotaciones'],[])
        self.assertEqual(self.req('POST','/api/guardar',self.entrada(t))[0],400)

    def test_exportar_preserva_estado_pendiente_sin_crear_revision(self):
        self.assertEqual(self.req('GET','/api/exportar')[0],404)
        status,c=self.req('POST','/api/exportar',{})
        self.assertEqual(status,200);self.assertFalse(c['info']['entrenamiento_autorizado'])
        self.assertFalse(self.t.diario('fixture:1'));self.assertFalse(c['annotations'][0]['eco']['aceptada_V26'])


if __name__=='__main__':unittest.main()
