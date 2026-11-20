from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ecowaste.datasets import CLASES
from ecowaste.detector_mapas import DetectorMapas,cargar_detector_mapas,perdida_mapas,cajas_desde_mapas,etiquetas_mapas


class PresenciaMaterialTests(unittest.TestCase):
    def test_material_no_recibe_gradiente_de_celdas_fondo(self):
        raw=torch.zeros(1,7,4,4,requires_grad=True)
        y=torch.zeros(1,4,4,dtype=torch.long);y[:,1:3,1:3]=3
        perdida_mapas(raw,y,formulacion='presencia_y_material').backward()
        fg=y>0
        self.assertEqual(float(raw.grad[:,1:].permute(0,2,3,1)[~fg].abs().sum()),0)
        self.assertGreater(float(raw.grad[:,1:].permute(0,2,3,1)[fg].abs().sum()),0)
        self.assertTrue(bool((raw.grad[:,0][fg]<0).all()))
        self.assertTrue(bool((raw.grad[:,0][~fg]>0).all()))

    def test_todo_ignorado_y_escena_sin_material(self):
        for etiqueta in [0,255]:
            raw=torch.randn(1,7,4,4,requires_grad=True)
            loss=perdida_mapas(raw,torch.full((1,4,4),etiqueta),formulacion='presencia_y_material')
            loss.backward()
            self.assertTrue(bool(torch.isfinite(raw.grad).all()))
            self.assertEqual(float(raw.grad[:,1:].abs().sum()),0)
            if etiqueta==255:self.assertEqual(float(loss.detach()),0)

    def test_presencia_y_material_deciden_por_separado(self):
        raw=torch.zeros(1,7,4,4)
        raw[:,0]=-20
        raw[:,0,1:3,1:3]=torch.logit(torch.tensor(.8))
        p=cajas_desde_mapas(raw,.05,formulacion='presencia_y_material')[0]
        self.assertEqual(p['labels'].tolist(),[0])
        torch.testing.assert_close(p['boxes'],torch.tensor([[.25,.25,.75,.75]]))
        self.assertAlmostEqual(float(p['scores'][0]),.8/6,places=5)
        self.assertEqual(int((etiquetas_mapas(raw,'presencia_y_material')>0).sum()),4)

    def test_checkpoint_legacy_y_nueva_formulacion(self):
        with tempfile.TemporaryDirectory() as d:
            for formula in ['softmax7','presencia_y_material']:
                m=DetectorMapas(formulacion=formula)
                self.assertEqual(sum(p.numel() for p in m.parameters()),438727)
                cfg=dict(m.configuracion_mapas)
                if formula=='softmax7':cfg.pop('formulacion');cfg.pop('auxiliar_clasificacion')
                path=Path(d)/f'{formula}.pt'
                torch.save({'tipo':'ecowaste_mapas_desde_cero_v1','clases':CLASES,'configuracion':cfg,'pesos':m.state_dict()},path)
                nuevo,_=cargar_detector_mapas(path)
                self.assertEqual(nuevo.configuracion_mapas['formulacion'],formula)


if __name__=='__main__':unittest.main()
