from copy import deepcopy
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ecowaste.recortes import planificar,ventana_recorte,objetivo_recorte,MezclaRecortes


class ContornosFicticios:
    cobertura_minima=1/64
    def contornos(self,r,a):return a['poligono_prueba']


class RecortesTests(unittest.TestCase):
    def registro(self):
        return {'id':'taco:prueba','ancho':200,'alto':100,'split':'train','grupo':'g1',
            'cajas':[{'id_interno':1,'clase':'plastico','bbox':[20,20,20,20],
                       'poligono_prueba':[[20,20,40,20,40,40,20,40]]},
                      {'id_interno':2,'clase':'metal','bbox':[90,10,30,30],
                       'poligono_prueba':[[90,10,120,10,120,40,90,40]]},
                      {'id_interno':3,'clase':None,'bbox':[50,60,20,20]}]}

    def test_ventana_entera_dentro_con_ancla_en_borde(self):
        r=self.registro();a=dict(r['cajas'][0],bbox=[0,0,10,20])
        v=ventana_recorte(r,a,2)
        self.assertEqual(v,[0,0,64,64])
        self.assertEqual(ventana_recorte(r,a,100),[0,0,200,100])

    def test_rechaza_particiones_y_ventanas_invalidas(self):
        r=self.registro();r['split']='val'
        with self.assertRaises(ValueError):planificar([r])
        with self.assertRaises(ValueError):objetivo_recorte(r,[-1,0,100,100],ContornosFicticios())

    def test_corta_y_desplaza_todos_los_objetos_visibles(self):
        r=self.registro();original=deepcopy(r)
        o,y=objetivo_recorte(r,[30,0,100,100],ContornosFicticios())
        self.assertEqual(o['ids_anotaciones_visibles'],[1,2,3])
        self.assertEqual(o['labels'].tolist(),[0,4])
        self.assertEqual(tuple(o['ignore_boxes'].shape),(1,4))
        self.assertTrue(bool(((o['boxes']>=0)&(o['boxes']<=1)).all()))
        self.assertTrue(bool((y==1).any()));self.assertTrue(bool((y==5).any()))
        g=o['geometria'];b=o['boxes'][0]
        self.assertAlmostEqual(float(b[0]),g['relleno_x']/384,places=6)
        self.assertAlmostEqual(float(b[2]),(g['relleno_x']+10*g['escala_x'])/384,places=6)
        self.assertEqual(r,original)

    def test_poligono_cortado_no_pinta_padding(self):
        o,y=objetivo_recorte(self.registro(),[30,0,100,100],ContornosFicticios())
        self.assertTrue(bool((y[:,:7]==255).all()))
        self.assertTrue(bool((y[:,-7:]==255).all()))
        self.assertEqual(int(y[2,8]),0)

    def test_plan_dos_escalas_solo_anclas_positivas(self):
        r=self.registro();plan=planificar([r])
        self.assertEqual(len(plan),4)
        self.assertEqual({p['ancla_id_interno'] for p in plan},{1,2})
        self.assertTrue(all(p['grupo']=='g1' and p['split']=='train' for p in plan))
        self.assertTrue(all(p['lado_menor_ancla_384_despues']>=p['lado_menor_ancla_384_antes'] for p in plan))

    def test_mezcla_preserva_probabilidad_no_depende_del_numero_de_recortes(self):
        mezcla=MezclaRecortes([('escena1',0,0),('escena2',1,1)],[(f'recorte{i}',0,i) for i in range(100)],.5)
        self.assertAlmostEqual(float(mezcla.pesos_muestreo[:2].sum()),.5)
        self.assertAlmostEqual(float(mezcla.pesos_muestreo[2:].sum()),.5)
        self.assertEqual(mezcla[0],('escena1',0,0))
        self.assertEqual(mezcla[2],('recorte0',0,2))

    def test_balanceo_igual_probabilidad_por_material_y_mitad_escenas(self):
        from ecowaste.datasets import CLASES
        class RecortesDePrueba(list):pass
        r=RecortesDePrueba([(0,0,i) for i in range(8)])
        r.plan=[{'clase_ancla':c} for c in [CLASES[0]]*3+CLASES[1:]]
        mezcla=MezclaRecortes([(0,0,0)]*2,r,.5,True)
        self.assertAlmostEqual(float(mezcla.pesos_muestreo[:2].sum()),.5)
        for c in CLASES:
            indices=[i+2 for i,p in enumerate(r.plan) if p['clase_ancla']==c]
            self.assertAlmostEqual(float(mezcla.pesos_muestreo[indices].sum()),1/12)


if __name__=='__main__':unittest.main()
