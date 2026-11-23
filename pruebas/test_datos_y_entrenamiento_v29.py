"""Integridad de COCO y exposición histórica del nuevo experimento de datos."""
import copy,importlib.util,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from ecowaste.datos_v29 import (bbox_mascara,bbox_iou,decodificar_coco,rasterizar_instancias,
    conflicto_duplicado,cargar_coleccion,ajustar_mascara)
from ecowaste.experimento_v29 import crear_plan,indices,flujo_auxiliar,Diario,comprobar
from scripts.descargar_cohortes_v29 import seguro,es_test
from ecowaste.reanudacion_regional import publicar
from scripts.preparar_v29 import grupo,Grupos


class DatosYEntrenamientoV29(unittest.TestCase):
    def setUp(self):
        self.im={'id':'fixture','ancho':384,'alto':192}
        self.m=np.zeros((192,384),dtype=np.uint8);self.m[30:100,20:100]=1
        self.a={'id':1,'material':'carton'}
        self.cfg={'actualizaciones':600,'semilla_nuevos':2901,'version':'v29_datos_preserva_historico_1','checkpoints':[0,100,300,600]}
        self.ck={'rng_loader':torch.Generator().manual_seed(9).get_state(),'rng_auxiliar':torch.Generator().manual_seed(11).get_state()}

    def test_bbox_se_deriva_del_contorno(self):
        self.assertEqual(bbox_mascara(self.m),[20,30,80,70]);self.assertEqual(bbox_iou([20,30,80,70],bbox_mascara(self.m)),1.)
        self.assertEqual(bbox_iou([0,0,2,2],bbox_mascara(self.m)),0.)
        with self.assertRaises(ValueError):bbox_mascara(self.m*0)

    def test_orden_Fortran_y_C_preservan_geometria_exacta(self):
        from ecowaste.datasets import geometria_letterbox
        for w,h in [(384,192),(283,537),(513,189)]:
            rng=np.random.default_rng(29);m=rng.integers(0,2,(h,w),dtype=np.uint8);g=geometria_letterbox(w,h,384)
            self.assertEqual(bbox_mascara(m),bbox_mascara(np.asfortranarray(m)))
            self.assertTrue(np.array_equal(ajustar_mascara(m,g),ajustar_mascara(np.asfortranarray(m),g)))

    def test_padding_excluido_y_material_correcto(self):
        y,r=rasterizar_instancias(self.im,[self.a],[self.m]);self.assertTrue(bool((y==3).any()))
        self.assertTrue(bool((y[:12]==255).all()));self.assertTrue(bool((y[36:]==255).all()))
        self.assertEqual(r[0][0],2);self.assertEqual(r[0][2],1)
        self.assertTrue(np.array_equal(r[0][1],y.numpy()==3))

    def test_ignore_no_es_fondo_ni_se_sobrescribe(self):
        y,_=rasterizar_instancias(self.im,[self.a,{'id':2,'material':None}],[self.m,self.m])
        self.assertFalse(bool((y==3).any()));self.assertEqual(int(y[17,4]),255)

    def test_mezcla_materiales_ignorada(self):
        y,_=rasterizar_instancias(self.im,[self.a,{'id':2,'material':'papel'}],[self.m,self.m])
        self.assertFalse(bool(((y>0)&(y!=255)).any()))

    def test_huecos_en_mascara_se_conservan(self):
        m=self.m.copy();m[48:72,40:64]=0;y,_=rasterizar_instancias(self.im,[self.a],[m])
        self.assertEqual(int(y[19,6]),0);self.assertEqual(int(y[17,4]),3)

    def test_imagen_sin_instancias_no_fabrica_positivos(self):
        y,r=rasterizar_instancias(self.im,[],[]);self.assertEqual(r,[])
        self.assertFalse(bool(((y>0)&(y!=255)).any()))

    def test_duplicados_exactos_y_candidatos_no_son_identidad_fisica(self):
        a={'sha256_archivo':'a','sha256_pixeles':'b','dhash':'0000000000000000'}
        self.assertEqual(conflicto_duplicado(a,{**a,'sha256_archivo':'otro'}),'exacto')
        self.assertEqual(conflicto_duplicado(a,{'dhash':'0000000000000001'}),'perceptual_candidato')
        self.assertIsNone(conflicto_duplicado(a,{'dhash':'ffffffffffffffff'}))

    def test_rutas_inseguras_y_test_prohibidos(self):
        for s in ['../a','/a','C:/a','a\\b']:
            with self.assertRaises(ValueError):seguro(s)
        for s in ['images/test/a.jpg','multi class/annotations/instances_test.json','test.json']:
            self.assertTrue(es_test(s))
        self.assertFalse(es_test('data/train/a.jpg'))

    def test_secuencias_y_bloques_no_se_reparten_por_fotograma(self):
        for seq,sp in [(1,'train'),(2,'train'),(6,'val')]:
            a=grupo('packwise',f'Vollstrom{seq}_frame00012_Probe00006.jpg')
            b=grupo('packwise',f'Vollstrom{seq}_frame00013_Probe00006_00007.jpg')
            self.assertEqual(a,b);self.assertEqual(a[1],sp)
        for k in range(1,21):self.assertEqual(grupo('click',f'trash ({k}).jpg'),grupo('click','trash (1).jpg'))
        with self.assertRaises(ValueError):grupo('click','sin_procedencia.jpg')

    def test_relaciones_perceptuales_transitivas_unen_grupos(self):
        g=Grupos(['escena1','escena2','escena3','otra']);g.unir('escena2','escena3');g.unir('escena1','escena2')
        self.assertEqual(g.raiz('escena3'),g.raiz('escena1'));self.assertNotEqual(g.raiz('otra'),g.raiz('escena1'))

    def test_plan_mantiene_ocho_historicas_y_tarea_global(self):
        p=crear_plan(self.ck,964,['packwise']*7+['click']*9,torch.arange(60)%6,self.cfg)
        for k in range(600):
            a,b=indices(p,'A',k),indices(p,'B',k)
            self.assertTrue(torch.equal(a['historico'],b['historico']));self.assertEqual(len(a['historico']),8)
            self.assertTrue(torch.equal(a['global'],b['global']));self.assertEqual(len(b['nuevo']),4)
            self.assertTrue(bool((b['nuevo'][:2]<7).all()));self.assertTrue(bool((b['nuevo'][2:]>=7).all()))
        with self.assertRaises(ValueError):indices(p,'B',600)

    def test_plan_reproducible_y_reanudacion_mismas_entradas(self):
        antes=torch.get_rng_state().clone();args=(self.ck,964,['packwise']*3+['click']*4,torch.arange(60)%6,self.cfg)
        a,b=crear_plan(*args),crear_plan(*args)
        self.assertTrue(torch.equal(antes,torch.get_rng_state()))
        for s in ['historico','nuevo','global']:self.assertTrue(torch.equal(a[s],b[s]))
        self.assertTrue(torch.equal(torch.cat([a['nuevo'][:100],b['nuevo'][100:]]),a['nuevo']))

    def test_ausencia_fuente_o_clase_global_bloquea(self):
        with self.assertRaises(ValueError):crear_plan(self.ck,5,['click']*4,torch.arange(60)%6,self.cfg)
        with self.assertRaises(ValueError):crear_plan(self.ck,5,['click','packwise'],torch.zeros(10,dtype=torch.long),self.cfg)

    def test_rng_auxiliar_no_desplaza_flujo_historico_incluso_con_error(self):
        estado=torch.get_rng_state().clone();privado={'cpu':torch.Generator().manual_seed(2902).get_state(),'cuda':[]}
        esperado=torch.rand(5);torch.set_rng_state(estado)
        with patch('torch.cuda.get_rng_state_all',return_value=[]),patch('torch.cuda.set_rng_state_all'):
            with self.assertRaises(RuntimeError):
                with flujo_auxiliar(privado):torch.rand(100);raise RuntimeError('interrupción')
        self.assertTrue(torch.equal(esperado,torch.rand(5)))

    def test_checkpoint_otra_rama_protocolo_rechazado(self):
        with self.assertRaises(ValueError):comprobar({'v29':{'contrato':{},'rama':'A'}},{},'B')
        with self.assertRaises(ValueError):comprobar({'v29':{'contrato':{},'rama':'B'}},{'distinto':1},'B')

    def test_diario_incompleto_y_no_sobrescritura(self):
        with tempfile.TemporaryDirectory() as td:
            d=Diario('A',{'config':self.cfg},td);publicar(d.base/'confirmaciones/paso_0100.json',{})
            with self.assertRaises(ValueError):d.leer()
            with self.assertRaises(FileExistsError):publicar(d.base/'confirmaciones/paso_0100.json',{'sobrescribir':True})

    def test_lector_no_admite_test(self):
        with self.assertRaises(ValueError):cargar_coleccion('.','.','test')

    @unittest.skipUnless(importlib.util.find_spec('pycocotools'),'Codec solo requerido por preparación CPU')
    def test_codec_coco_poligono_RLE_huecos_y_errores(self):
        from pycocotools import mask
        m=np.ones((20,20),dtype=np.uint8);m[5:15,5:15]=0
        r=mask.encode(np.asfortranarray(m));r['counts']=r['counts'].decode('ascii')
        self.assertTrue(np.array_equal(m,decodificar_coco(r,20,20)))
        p=decodificar_coco([[1,1,8,1,8,8,1,8]],20,20);self.assertEqual(int(p.sum()),49)
        for s in [{'size':[20,20],'counts':[400]}, {'size':[20,20],'counts':[1,-1,400]},[[0,0,1,float('nan'),2,2]],r]:
            with self.assertRaises(ValueError):decodificar_coco(s,21,20)


if __name__=='__main__':unittest.main()
