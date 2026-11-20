import unittest
import torch
from ecowaste.detector_mapas import cajas_desde_mapas
from ecowaste.datasets import geometria_letterbox


class TestPropuestasMaterial(unittest.TestCase):
    def region(self, ganador=0):
        raw=torch.full((1,7,8,8),-10.)
        raw[:,0]=10.
        raw[:,0,2:5,1:4]=2. if ganador==0 else -10.
        raw[:,1,2:5,1:4]=1.
        if ganador!=0:raw[:,ganador,2:5,1:4]=2.
        return raw

    def test_propuesta_con_fondo_ganador(self):
        raw=self.region()
        self.assertEqual(len(cajas_desde_mapas(raw,umbral=.05)[0]['boxes']),0)
        p=cajas_desde_mapas(raw,umbral=.05,politica='por_material')[0]
        self.assertEqual(p['labels'].tolist(),[0])
        torch.testing.assert_close(p['boxes'],torch.tensor([[1/8,2/8,4/8,5/8]]))
        self.assertAlmostEqual(float(p['scores'][0]),float(raw.softmax(1)[0,1,2,1]),places=6)

    def test_dos_materiales_pueden_compartir_celdas(self):
        raw=self.region(ganador=4)
        base=cajas_desde_mapas(raw,umbral=.05)[0]
        nuevo=cajas_desde_mapas(raw,umbral=.05,politica='por_material')[0]
        self.assertEqual(base['labels'].tolist(),[3])
        self.assertEqual(set(nuevo['labels'].tolist()),{0,3})
        self.assertEqual(len(nuevo['boxes']),2)
        torch.testing.assert_close(nuevo['boxes'][0],nuevo['boxes'][1])

    def test_mapa_uniforme_puede_crear_falsas_propuestas(self):
        raw=torch.zeros(1,7,8,8)
        self.assertEqual(len(cajas_desde_mapas(raw,umbral=.05)[0]['boxes']),0)
        nuevo=cajas_desde_mapas(raw,umbral=.05,politica='por_material')[0]
        self.assertEqual(set(nuevo['labels'].tolist()),set(range(6)))
        torch.testing.assert_close(nuevo['scores'],torch.full((6,),1/7))
        torch.testing.assert_close(nuevo['boxes'],torch.tensor([[0.,0.,1.,1.]]).expand(6,-1))

    def test_padding_area_y_umbral_se_conservan(self):
        g=geometria_letterbox(16,32,32)
        raw=torch.full((1,7,8,8),-10.);raw[:,0]=10
        raw[:,0,:,:2]=2;raw[:,1,:,:2]=1
        self.assertEqual(len(cajas_desde_mapas(raw,umbral=.05,geometrias=[g],politica='por_material')[0]['boxes']),0)
        raw=self.region()
        self.assertEqual(len(cajas_desde_mapas(raw,umbral=.5,politica='por_material')[0]['boxes']),0)
        self.assertEqual(len(cajas_desde_mapas(raw,umbral=.05,area_minima=10,politica='por_material')[0]['boxes']),0)

    def test_politica_invalida_no_es_silenciosa(self):
        raw=self.region()
        with self.assertRaises(ValueError):cajas_desde_mapas(raw,politica='otra')
        with self.assertRaises(ValueError):cajas_desde_mapas(raw,politica='por_material',formulacion='presencia_y_material')


if __name__=='__main__':unittest.main()
