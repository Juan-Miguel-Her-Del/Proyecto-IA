from pathlib import Path
import sys
import tempfile
import unittest
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ecowaste.detector_mapas import DetectorMapas,cargar_detector_mapas
from ecowaste.datasets import CLASES


class AuxiliarTests(unittest.TestCase):
    def test_etiqueta_global_entrena_extractor_sin_supervisar_cabezal_pixel(self):
        torch.manual_seed(42)
        m=DetectorMapas(auxiliar_clasificacion=True)
        loss=F.cross_entropy(m.clasificar(torch.rand(2,3,384,384)),torch.tensor([0,5]))
        loss.backward()
        for p in [m.tokenizador_detalle.weight,m.tokenizador_contexto.weight,m.clasificador_auxiliar.weight]:
            self.assertIsNotNone(p.grad)
            self.assertTrue(bool(torch.isfinite(p.grad).all()))
            self.assertGreater(float(p.grad.abs().sum()),0)
        self.assertIsNone(m.cabezal[-1].weight.grad)
        self.assertFalse(any(isinstance(a,torch.nn.Conv2d) for a in m.modules()))

    def test_guardado_carga_auxiliar_y_compatibilidad(self):
        m=DetectorMapas(auxiliar_clasificacion=True)
        self.assertEqual(sum(p.numel() for p in m.parameters()),439309)
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'aux.pt'
            torch.save({'tipo':'ecowaste_mapas_desde_cero_v1','clases':CLASES,'configuracion':m.configuracion_mapas,'pesos':m.state_dict()},path)
            copia,_=cargar_detector_mapas(path)
            self.assertIsNotNone(copia.clasificador_auxiliar)
        with self.assertRaises(ValueError):DetectorMapas().clasificar(torch.zeros(1,3,384,384))


if __name__=='__main__':unittest.main()
