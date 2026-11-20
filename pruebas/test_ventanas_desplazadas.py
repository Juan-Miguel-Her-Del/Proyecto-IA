from pathlib import Path
import sys
import tempfile
import unittest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ecowaste.detector import atender_ventanas
from ecowaste.detector_mapas import DetectorMapas,cargar_detector_mapas
from ecowaste.datasets import CLASES


class MediaDeVentana(nn.Module):
    def forward(self,x,validos=None):
        pesos=torch.ones_like(x[...,:1]) if validos is None else validos[...,None].float()
        media=(x*pesos).sum(1,keepdim=True)/pesos.sum(1,keepdim=True)
        return media.expand_as(x)


class VentanasDesplazadasTests(unittest.TestCase):
    def test_intercambia_informacion_entre_ventanas_vecinas(self):
        x=torch.zeros(1,16,16,1);x[:,7,7]=1
        inicial=atender_ventanas(x,MediaDeVentana(),8)
        fijo=atender_ventanas(inicial,MediaDeVentana(),8)
        desplazado=atender_ventanas(inicial,MediaDeVentana(),8,4)
        self.assertEqual(float(fijo[0,8,8,0]),0)
        self.assertGreater(float(desplazado[0,8,8,0]),0)

    def test_no_conecta_bordes_opuestos(self):
        x=torch.zeros(1,16,16,1);x[:,0,0]=1
        y=atender_ventanas(x,MediaDeVentana(),8,4)
        self.assertGreater(float(y[0,0,0,0]),0)
        self.assertEqual(float(y[0,-1,-1,0]),0)
        self.assertEqual(float(y[0,0,-1,0]),0)

    def test_excluye_padding_del_promedio(self):
        x=torch.ones(2,16,16,3)
        y=atender_ventanas(x,MediaDeVentana(),8,4)
        torch.testing.assert_close(y,x)

    def test_identidad_conserva_orden_y_gradiente(self):
        x=torch.randn(2,16,16,3,requires_grad=True)
        def identidad(v,validos=None):return v
        y=atender_ventanas(x,identidad,8,4)
        torch.testing.assert_close(y,x)
        y.sum().backward();torch.testing.assert_close(x.grad,torch.ones_like(x))

    def test_parametros_compatibilidad_y_gradientes_reales(self):
        modelo=DetectorMapas(auxiliar_clasificacion=True,ventanas_desplazadas=True)
        self.assertEqual(sum(p.numel() for p in modelo.parameters()),439309)
        raw=modelo(torch.rand(1,3,384,384));raw.square().mean().backward()
        self.assertEqual(tuple(raw.shape),(1,7,48,48))
        self.assertTrue(bool(torch.isfinite(modelo.detalle[1].atencion.qkv.weight.grad).all()))
        with tempfile.TemporaryDirectory() as t:
            for desplazar in [False,True]:
                modelo=DetectorMapas(ventanas_desplazadas=desplazar)
                config=dict(modelo.configuracion_mapas)
                if not desplazar:config.pop('ventanas_desplazadas')
                path=Path(t)/f'{desplazar}.pt'
                torch.save({'tipo':'ecowaste_mapas_desde_cero_v1','clases':CLASES,'configuracion':config,'pesos':modelo.state_dict()},path)
                copia,_=cargar_detector_mapas(path)
                self.assertEqual(copia.configuracion_mapas['ventanas_desplazadas'],desplazar)


if __name__=='__main__':unittest.main()
