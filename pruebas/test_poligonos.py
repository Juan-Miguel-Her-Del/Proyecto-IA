from pathlib import Path
import sys
import unittest
import torch

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.datasets import geometria_letterbox
from ecowaste.poligonos import rasterizar_contornos, ContornosTACO


class PoligonosTests(unittest.TestCase):
    def objetivo(self,ancho=32,alto=32):
        return {'geometria':geometria_letterbox(ancho,alto,32),
            'boxes':torch.tensor([[0.,0.,1.,1.]]),'labels':torch.tensor([0]),
            'ignore_boxes':torch.empty(0,4)}

    def test_usa_contorno_y_no_rectangulo(self):
        o=self.objetivo()
        y=rasterizar_contornos(o,[([[0,0,31,0,0,31]],0)],4)
        self.assertEqual(int(y[0,0]),1)
        self.assertEqual(int(y[3,3]),0)
        self.assertEqual(int(y[2,0]),1)

    def test_transformacion_y_padding(self):
        o=self.objetivo(32,16)
        y=rasterizar_contornos(o,[([[0,0,15,0,15,15,0,15]],2)],4)
        self.assertTrue(bool((y[0]==255).all()))
        self.assertTrue(bool((y[3]==255).all()))
        self.assertEqual(int(y[1,0]),3)
        self.assertEqual(int(y[1,3]),0)

    def test_solape_y_fragmento_pequeno_ignorados(self):
        o=self.objetivo()
        y=rasterizar_contornos(o,[([[0,0,15,0,15,15,0,15]],0),
            ([[0,0,15,0,15,15,0,15]],1),([[24,24,25,24,25,25,24,25]],3)],4)
        self.assertEqual(int(y[0,0]),255)
        self.assertEqual(int(y[3,3]),255)
        self.assertEqual(int(y[3,0]),0)

    def test_cualquier_pixel_conserva_objeto_pequeno(self):
        o=self.objetivo()
        y=rasterizar_contornos(o,[([[24,24,25,24,25,25,24,25]],3)],4,1/64)
        self.assertEqual(int(y[3,3]),4)
        self.assertEqual(int(y[3,0]),0)

    def fuente_ejemplo(self):
        c=ContornosTACO.__new__(ContornosTACO)
        c.anotaciones=[{'id':7,'bbox':[0,0,10,10],'category_id':2,'image_id':1,
            'segmentation':[[0,0,10,0,10,10]]}]
        c.categorias={2:'Plastic bottle'};c.imagenes={1:{'width':32,'height':32}}
        c.inventario={'taco:1':{'sha256_pixeles':'abc','grupo':'escena1'}}
        r={'id_original':1,'ancho':32,'alto':32,'sha256_pixeles':'abc','grupo':'escena1'}
        b={'id_interno':1,'id_original':7,'bbox_original':[0,0,10,10],'categoria_original':'Plastic bottle'}
        return c,r,b

    def test_ids_originales_repetidos_no_sustituyen_indice_interno(self):
        c,r,b=self.fuente_ejemplo()
        c.anotaciones.insert(0,dict(c.anotaciones[0],segmentation=[[1,1,4,1,4,4]]))
        b['id_interno']=2
        self.assertEqual(c.contornos(r,b)[0],[0,0,10,0,10,10])
        b['categoria_original']='Glass bottle'
        with self.assertRaises(ValueError):c.contornos(r,b)

    def test_fusion_exige_mismos_pixeles_y_grupo(self):
        c,r,b=self.fuente_ejemplo();r['id_original']=9
        self.assertEqual(len(c.contornos(r,b)),1)
        r['sha256_pixeles']='diferente'
        with self.assertRaises(ValueError):c.contornos(r,b)
        r['sha256_pixeles']='abc';r['grupo']='otro_split'
        with self.assertRaises(ValueError):c.contornos(r,b)

    def test_no_fabrica_mascaras_si_falta_contorno(self):
        c,r,b=self.fuente_ejemplo();c.anotaciones[0]['segmentation']=[]
        with self.assertRaises(ValueError):c.contornos(r,b)


if __name__=='__main__':unittest.main()
