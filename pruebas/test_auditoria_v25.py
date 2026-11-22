"""Contrafactuales analiticos y barrera GT; no entrena modelos."""
import unittest
import numpy as np
import torch
from ecowaste.datasets import geometria_letterbox
from ecowaste.auditoria_datos_representacion import oraculo_rejilla,cruces_propios,iou_np
from scripts.capturar_v25 import verificar_metadatos,metadatos_sin_gt


def escena(boxes,poligonos,ancho=384,alto=384):
    g=geometria_letterbox(ancho,alto,384)
    o={'geometria':g,'boxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
        'labels':torch.zeros(len(boxes),dtype=torch.long),'ignore_boxes':torch.empty(0,4)}
    ss=[{'id':i,'clase':0,'poligonos':[p]} for i,p in enumerate(poligonos)]
    return o,ss


class TestAuditoriaV25(unittest.TestCase):
    def test_gt_no_cruza_captura(self):
        r={'id':'i','ruta':'r','ancho':384,'alto':384,'sha256_archivo':'x','cajas':[1]}
        ms=metadatos_sin_gt([r]);verificar_metadatos(ms)
        with self.assertRaises(ValueError):verificar_metadatos([r])
        with self.assertRaises(ValueError):verificar_metadatos([{**ms[0],'poligonos':[]}])

    def test_fusion_semantica_sin_perdida_de_identidad_gt(self):
        o,ss=escena([[.25,.25,.5,.5],[.5,.25,.75,.5]],
            [[96,96,191,96,191,191,96,191],[192,96,287,96,287,191,192,191]])
        r=oraculo_rejilla(o,ss,48)['filas']
        self.assertTrue(all(f['fusion_mayoritaria'] for f in r))
        self.assertTrue(all(f['iou_instancia_separada_max']>.99 for f in r))
        self.assertTrue(all(f['iou_semantica_max']<=.501 for f in r))
        self.assertEqual([s['id'] for s in ss],[0,1])

    def test_hueco_resuelto_por_mayor_rejilla(self):
        o,ss=escena([[.25,.25,.49,.5],[.51,.25,.75,.5]],
            [[96,96,188,96,188,191,96,191],[196,96,287,96,287,191,196,191]])
        a=oraculo_rejilla(o,ss,48)['filas'];b=oraculo_rejilla(o,ss,96)['filas']
        self.assertTrue(all(f['fusion_mayoritaria'] for f in a))
        self.assertFalse(any(f['fusion_mayoritaria'] for f in b))

    def test_pequeno_puede_perderse_por_area_minima(self):
        o,ss=escena([[.25,.25,99/384,99/384]],[[96,96,98,96,98,98,96,98]])
        r=oraculo_rejilla(o,ss,48)['filas'][0]
        self.assertEqual(r['tokens_tocados'],1);self.assertEqual(r['iou_instancia_separada_max'],0.)
        self.assertGreater(oraculo_rejilla(o,ss,384)['filas'][0]['iou_instancia_separada_max'],.99)

    def test_padding_excluido(self):
        o,ss=escena([[.25,.25,.75,.75]],[[0,0,199,0,199,99,0,99]],200,100)
        r=oraculo_rejilla(o,ss,48)['filas'][0]
        self.assertLessEqual(r['tokens_semanticos_correctos'],48*24)

    def test_material_ambiguo_no_se_fabrica(self):
        o,ss=escena([[.25,.25,.5,.5],[.25,.25,.5,.5]],[[96,96,191,96,191,191,96,191]]*2)
        ss[1]['clase']=1;o['labels'][1]=1
        rr=oraculo_rejilla(o,ss,48)['filas']
        self.assertTrue(all(r['tokens_semanticos_correctos']==0 and r['iou_semantica_max']==0 for r in rr))

    def test_cruces_y_poligono_simple(self):
        self.assertEqual(cruces_propios([0,0,10,0,10,10,0,10]),0)
        self.assertEqual(cruces_propios([0,0,10,10,0,10,10,0]),1)

    def test_iou_vacia_y_escala(self):
        self.assertEqual(iou_np([], [0,0,1,1]).shape,(0,))
        np.testing.assert_array_equal(iou_np([[0,0,1,1]],[0,0,1,1]),[1])


if __name__=='__main__':unittest.main()
