import unittest
from copy import deepcopy
import numpy as np
from ecowaste.auxiliares_publicos import rle_coco
from ecowaste.ajustes_ia_v28_3 import (decodificar,guardar_mask,caja,canonizar,
    ambiguo,quitar_solape,nueva_propuesta,elegibilidad48,componentes_dudosos_a_ignore)


class AjustesIA(unittest.TestCase):
    def obj(self,n,m,material='plastico'):
        return guardar_mask({'id':n,'material':material,'poligonos':[],'huecos':[],'descartada':False},m)
    def test_rle_con_huecos_y_componentes_es_exacto(self):
        m=np.ones((20,30),bool);m[2:12,4:18]=False;m[15,:]=False
        np.testing.assert_array_equal(m,decodificar(rle_coco(m)))
    def test_rle_rechaza_longitud_incorrecta(self):
        with self.assertRaises(ValueError):decodificar({'size':[3,4],'counts':[1,2]})
    def test_rle_rechaza_numeros_negativos(self):
        with self.assertRaises(ValueError):decodificar({'size':[2,2],'counts':[-1,5]})
    def test_bbox_incluye_pixel_de_borde(self):
        m=np.zeros((8,9),bool);m[2:5,4:8]=True
        self.assertEqual(caja(m),[4,2,4,3])
    def test_vacia_no_se_convierte_en_caja(self):
        with self.assertRaises(ValueError):caja(np.zeros((5,5),bool))
    def test_paridad_preservada_no_union_silenciosa(self):
        a={'id':'a','material':'organico','poligonos':[[1,1,8,1,8,8,1,8],[3,3,5,3,5,5,3,5]],'huecos':[]}
        antes=deepcopy(a);b=canonizar(a,10,10);m=decodificar(b['segmentacion_canonica_RLE'])
        self.assertEqual(a,antes);self.assertFalse(m[4,4]);self.assertTrue(m[2,2])
        self.assertFalse(b['revision_humana_del_ajuste']);self.assertFalse(b['aceptada_V26'])
    def test_ignorar_no_borra_region_ni_origen(self):
        m=np.ones((5,6),bool);a=self.obj('a',m,'papel');ambiguo(a,'Composicion no demostrable')
        np.testing.assert_array_equal(m,decodificar(a['segmentacion_canonica_RLE']))
        self.assertEqual(a['material'],'ambiguo');self.assertFalse(a['revision_humana_del_ajuste'])
    def test_ignore_sin_mascara_rechazado(self):
        with self.assertRaises(ValueError):ambiguo({'material':'papel'},'Sin region')
    def test_solape_a_ignore_conserva_union_exacta(self):
        a=np.zeros((10,12),bool);a[1:7,1:7]=True;b=np.zeros_like(a);b[4:9,5:11]=True
        objs=[self.obj('a',a),self.obj('b',b)];s=quitar_solape(objs,['a','b'],'Identidad ambigua')
        masks=[decodificar(o['segmentacion_canonica_RLE']) for o in objs]
        np.testing.assert_array_equal(a|b,np.logical_or.reduce(masks));self.assertFalse((masks[0]&masks[1]).any())
        self.assertTrue(s['no_acredita_nueva_instancia']);self.assertEqual(s['material'],'ambiguo')
    def test_occlusion_preserva_objeto_frontal(self):
        a=np.zeros((8,8),bool);a[1:6,1:6]=True;b=np.zeros_like(a);b[4:7,4:7]=True
        objs=[self.obj('a',a),self.obj('b',b)];quitar_solape(objs,['a','b'],'Frontal visible',frente='b')
        np.testing.assert_array_equal(b,decodificar(objs[1]['segmentacion_canonica_RLE']))
        self.assertEqual(len(objs),2)
    def test_no_elimina_instancia_entera_por_solape(self):
        m=np.ones((5,5),bool);objs=[self.obj('a',m),self.obj('b',m)]
        with self.assertRaises(ValueError):quitar_solape(objs,['a','b'],'Todo superpuesto')
    def test_propuesta_no_se_convierte_en_revision_humana(self):
        s={'id':'asistente_123','resultado':{'rle_ia_original':rle_coco(np.ones((5,5),bool))}}
        a=nueva_propuesta(s,'papel','Hoja visible')
        self.assertFalse(a['revision_humana_del_ajuste']);self.assertFalse(a['aceptada_V26'])
        self.assertEqual(a['poligonos'],[]);self.assertIn('NO_humano',a['origen'])
    def test_rechaza_material_inventado(self):
        s={'id':'sam','resultado':{'rle_ia_original':rle_coco(np.ones((5,5),bool))}}
        with self.assertRaises(ValueError):nueva_propuesta(s,'nuevo_material','Sin evidencia')
    def test_rejilla_respeta_letterbox(self):
        m=np.ones((80,160),bool)
        self.assertEqual(elegibilidad48(m,160,80),48*24)
    def test_fragmentos_ignore_conservan_todos_los_pixeles(self):
        m=np.zeros((15,15),bool);m[1:7,1:7]=True;m[10:12,10:12]=True
        a=self.obj('a',m,'organico');b=componentes_dudosos_a_ignore(a,'Fragmento visual ambiguo')
        ma=decodificar(a['segmentacion_canonica_RLE']);mb=decodificar(b['segmentacion_canonica_RLE'])
        np.testing.assert_array_equal(m,ma|mb);self.assertFalse((ma&mb).any())
        self.assertEqual(b['material'],'ambiguo');self.assertFalse(b['aceptada_V26'])
    def test_fragmentos_sin_componente_dominante_no_se_asignan(self):
        m=np.zeros((15,15),bool)
        for x,y in [(1,1),(6,6),(11,11)]:m[y:y+2,x:x+2]=True
        with self.assertRaises(ValueError):componentes_dudosos_a_ignore(self.obj('a',m),'Identidad no demostrable')


if __name__=='__main__':unittest.main()
