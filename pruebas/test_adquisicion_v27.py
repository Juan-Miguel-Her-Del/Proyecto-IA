"""Propuestas externas y descargas acotadas; ninguna revisión humana se simula en datos reales."""
import email.message
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from ecowaste.propuestas_publicas import mapear_material,propuesta,caja_yolo,diagnostico_regional,imagen_verificada
from scripts.adquirir_v27 import ZipPorRangos,solo_train,seleccionar


class Respuesta:
    def __init__(self,data,start,n,*,status=206,incorrecto=False):
        self.data=data;self.start=start;self.n=n;self.status=status;self.leido=False
        self.headers=email.message.Message();self.headers['content-range']=f'bytes {start+int(incorrecto)}-{start+n-1}/{len(data)}'
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,n=-1):self.leido=True;return self.data[self.start:self.start+self.n]


def remoto(data,**kwargs):
    def abrir(req,timeout):
        m=re.fullmatch(r'bytes=(\d+)-(\d+)',req.headers['Range']);a,b=map(int,m.groups())
        return Respuesta(data,a,b-a+1,**kwargs)
    return abrir


class AdquisicionV27(unittest.TestCase):
    def test_train_solamente_y_sin_escape(self):
        self.assertTrue(solo_train('Warp-D/train/images/x.jpg'))
        for s in ['test/x.jpg','a/test/train/x.jpg','a/val/train/x.jpg','../train/x.jpg','C:/train/x.jpg','/train/x.jpg','a/validation/data/x.jpg']:
            with self.subTest(s=s):self.assertFalse(solo_train(s))

    def test_rangos_con_cabecera_minuscula_y_cache(self):
        data=b'abcdefghijklmnopqrstuvwxyz'
        with patch('urllib.request.urlopen',side_effect=remoto(data)) as mock:
            r=ZipPorRangos('https://example.org/a',size=len(data));r.seek(3);self.assertEqual(r.read(4),b'defg')
            llamadas=mock.call_count;r.seek(0);self.assertEqual(r.read(2),b'ab');self.assertEqual(mock.call_count,llamadas)
            r.seek(-2,2);self.assertEqual(r.read(),b'yz')

    def test_no_lee_respuesta_servidor_que_ignora_range(self):
        respuesta=Respuesta(b'x'*100,0,1,status=200)
        with patch('urllib.request.urlopen',return_value=respuesta),self.assertRaises(ValueError):ZipPorRangos('https://example.org/a')
        self.assertFalse(respuesta.leido)

    def test_no_supera_presupuesto_ni_acepta_rango_incorrecto(self):
        with patch('urllib.request.urlopen',side_effect=remoto(b'x'*100)):
            r=ZipPorRangos('https://example.org/a',presupuesto=3)
            with self.assertRaises(ValueError):r.read(4)
        with patch('urllib.request.urlopen',side_effect=remoto(b'x'*100,incorrecto=True)),self.assertRaises(ValueError):ZipPorRangos('https://example.org/a')

    def test_tamano_origen_cambiado_se_rechaza(self):
        with patch('urllib.request.urlopen',side_effect=remoto(b'abcdef')),self.assertRaises(ValueError):ZipPorRangos('https://example.org/a',size=7)

    def test_mapeo_conservador_sin_carton_multicapa(self):
        self.assertEqual(mapear_material('warp','glass-transp'),'vidrio')
        self.assertEqual(mapear_material('warp','bottle-blue'),'plastico')
        for c in ['milk-cardboard','juice-cardboard','detergent-box','cans','canister']:
            with self.subTest(c=c):self.assertIsNone(mapear_material('warp',c))
        self.assertIsNone(mapear_material('hpi','vegetables'));self.assertEqual(mapear_material('zerowaste','cardboard'),'carton')

    def test_bbox_no_se_convierte_en_poligono(self):
        p=propuesta('warp','i','a','glass-green',(100,80),bbox=[1,2,30,40])
        self.assertIsNone(p['poligonos_originales']);self.assertIsNone(p['poligonos_normalizados_v26'])
        self.assertTrue(p['requiere_poligono_humano']);self.assertFalse(p['aceptada'])

    def test_no_se_inventan_identidades_tiempos_o_revision(self):
        p=propuesta('zerowaste','i','a','metal',(100,80),poligonos=[[1,1,20,1,20,20,1,20]],bbox=[1,1,19,19])
        for k in ['objeto_fisico_id','sesion_id','sitio_id','dispositivo_id','capturado_en','evidencia_identidad_fisica']:self.assertIsNone(p[k])
        self.assertFalse(p['visible_only_confirmado']);self.assertFalse(p['aceptada'])

    def test_caja_derivada_conserva_original_y_poligono(self):
        original=[0,0,21,21];ps=[[1,1,20,1,20,20,1,20,1,1]]
        p=propuesta('zerowaste','i','a','cardboard',(100,80),poligonos=ps,bbox=original)
        self.assertEqual(p['bbox_original'],original);self.assertEqual(p['bbox_v26_derivada'],[1,1,19,19])
        self.assertEqual(len(ps[0]),10);self.assertFalse(p['bbox_original_coincide_v26'])

    def test_geometria_invalida_solo_se_senala(self):
        ps=[[0,0,20,20,0,20,20,0]];p=propuesta('zerowaste','i','a','cardboard',(100,80),poligonos=ps,bbox=[0,0,20,20])
        self.assertTrue(p['errores_geometricos']);self.assertIsNone(p['poligonos_normalizados_v26']);self.assertEqual(p['poligonos_originales'],ps)

    def test_yolo_coordenadas_se_transforman_sin_inventar_mascara(self):
        self.assertEqual(caja_yolo([.5,.5,.2,.5],100,80),[40,20,20,40])
        for v in [[.01,.5,1,.5],[float('nan'),.5,.1,.1],[.5,.5,0,.3]]:
            with self.subTest(v=v),self.assertRaises(ValueError):caja_yolo(v,100,80)

    def test_no_infiere_instancias_organicas_desde_etiquetas_globales(self):
        r=diagnostico_regional({'id':'i','width':100,'height':80},[])
        self.assertFalse(r['calculado']);self.assertIsNone(r['organicos_utilizables'])

    def test_no_declara_regiones_utiles_con_mascaras_incompletas(self):
        p=propuesta('warp','i','a','glass-green',(100,80),bbox=[1,2,30,40])
        r=diagnostico_regional({'id':'i','width':100,'height':80},[p]);self.assertFalse(r['calculado']);self.assertIsNone(r['organicos_utilizables'])

    def test_seleccion_reproducible_sin_repetir_imagen(self):
        cand=[{'id':i,'nombre':str(i),'grupo_nombre':str(i%2)} for i in range(8)]
        cs={i:{'a','b'} for i in range(8)};x=seleccionar(cand,cs,{'a':2,'b':2});y=seleccionar(list(reversed(cand)),cs,{'a':2,'b':2})
        self.assertEqual(x,y);self.assertEqual(len({a['id'] for a in x}),4)

    def test_verifica_fichero_real_sin_generar_anotaciones(self):
        # Imagen mínima exclusivamente de unit test, borrada; no pertenece al piloto real.
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'fixture.png';Image.new('RGB',(31,17),'gray').save(p);m=imagen_verificada(p)
            self.assertEqual((m['width'],m['height']),(31,17));self.assertEqual(len(m['sha256_archivo']),64)
            self.assertNotIn('annotations',m)


if __name__=='__main__':unittest.main()
