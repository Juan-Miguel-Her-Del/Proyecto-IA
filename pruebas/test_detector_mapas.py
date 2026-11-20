from pathlib import Path
import sys
import unittest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.detector_mapas import DetectorMapas, rasterizar, perdida_mapas, cajas_desde_mapas


class MapasTests(unittest.TestCase):
    def test_cobertura_de_cajas_y_solapes_se_ignoran(self):
        o = {"boxes": torch.tensor([[0.,0.,.5,.5],[.25,.25,.75,.75]]),
            "labels": torch.tensor([0,1]), "ignore_boxes": torch.tensor([[.75,.75,1.,1.]])}
        mapa = rasterizar(o, 4)
        self.assertEqual(int(mapa[0,0]), 1)
        self.assertEqual(int(mapa[1,1]), 255)
        self.assertEqual(int(mapa[2,2]), 2)
        self.assertEqual(int(mapa[3,3]), 255)
        self.assertEqual(int(mapa[0,3]), 0)

    def test_componente_tiene_caja_y_material_conocidos(self):
        logits = torch.full((1,7,8,8), -10.)
        logits[:,0] = 10
        logits[:,0,2:5,1:4] = -10
        logits[:,3,2:5,1:4] = 10
        p = cajas_desde_mapas(logits)[0]
        self.assertEqual(p['labels'].tolist(), [2])
        torch.testing.assert_close(p['boxes'],torch.tensor([[1/8,2/8,4/8,5/8]]))
        self.assertEqual(len(cajas_desde_mapas(torch.zeros(1,7,8,8))[0]['boxes']),0)

    def test_gradientes_y_todo_ignorado(self):
        x = torch.randn(2,7,8,8,requires_grad=True)
        target = torch.zeros(2,8,8,dtype=torch.long)
        target[:,2:5,2:5] = 1
        loss = perdida_mapas(x,target)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        x2 = torch.randn(1,7,8,8,requires_grad=True)
        loss2 = perdida_mapas(x2,torch.full((1,8,8),255,dtype=torch.long))
        self.assertEqual(float(loss2.detach()),0)
        loss2.backward()
        self.assertEqual(float(x2.grad.abs().sum()),0)

    def test_padding_no_crea_ni_conecta_componentes(self):
        from ecowaste.datasets import geometria_letterbox
        g=geometria_letterbox(16,32,32)
        raw=torch.full((1,7,8,8),-10.)
        raw[:,0]=10
        raw[:,0,:,:2]=-10;raw[:,1,:,:2]=10
        self.assertEqual(len(cajas_desde_mapas(raw,geometrias=[g])[0]['boxes']),0)
        raw[:,0,3:5,2:4]=-10;raw[:,1,3:5,2:4]=10
        p=cajas_desde_mapas(raw,geometrias=[g])[0]
        torch.testing.assert_close(p['boxes'],torch.tensor([[.25,.375,.5,.625]]))

    def test_forma_modelo_no_incluye_convoluciones(self):
        modelo = DetectorMapas()
        self.assertEqual(sum(p.numel() for p in modelo.parameters()),438727)
        self.assertFalse(any(isinstance(m,torch.nn.Conv2d) for m in modelo.modules()))
        with torch.no_grad():
            self.assertEqual(tuple(modelo(torch.rand(1,3,384,384)).shape),(1,7,48,48))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
