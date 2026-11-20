from pathlib import Path
import sys
import unittest
from PIL import Image
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ecowaste.datasets import geometria_letterbox
from ecowaste.refinamiento import proponer_regiones,refinar_imagen,ventana_contexto


class ClasificadorFicticio(nn.Module):
    def __init__(self):
        super().__init__();self.dummy=nn.Parameter(torch.zeros(()));self.clasificador_auxiliar=True
    def clasificar(self,x):
        raw=torch.full((len(x),6),-10.);raw[:,4]=10
        return raw


class RefinamientoTests(unittest.TestCase):
    def test_clase_procede_del_recorte_no_del_material_pixel(self):
        g=geometria_letterbox(128,64,384)
        raw=torch.full((7,48,48),-10.);raw[0]=10
        raw[0,20:24,10:14]=-10;raw[1,20:24,10:14]=10
        p=refinar_imagen(ClasificadorFicticio(),raw,Image.new('RGB',(128,64)),g)
        self.assertEqual(p['labels'].tolist(),[4])
        torch.testing.assert_close(p['boxes'],torch.tensor([[10/48,20/48,14/48,24/48]]))

    def test_no_propone_regiones_del_padding(self):
        g=geometria_letterbox(128,64,384)
        raw=torch.full((7,48,48),-10.);raw[0]=10
        raw[0,:10]=-10;raw[1,:10]=10
        boxes,_=proponer_regiones(raw,g)
        self.assertEqual(len(boxes),0)

    def test_contexto_limita_a_la_imagen(self):
        self.assertEqual(ventana_contexto([0,0,10,10],100,100),[0,0,12,12])
        self.assertEqual(ventana_contexto([0,0,100,100],100,50),[0,0,100,50])


if __name__=='__main__':unittest.main()
