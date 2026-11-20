import unittest
import torch
from ecowaste.auditoria_deteccion import auditar_imagen,tamano_objeto


def objetivo(boxes,labels,ignore=()):
    return {'boxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
            'labels':torch.tensor(labels,dtype=torch.long),'ignore_boxes':torch.tensor(ignore,dtype=torch.float32).reshape(-1,4)}


def pred(boxes,labels,scores=None):return {'boxes':boxes,'labels':labels,'scores':scores if scores is not None else [.8]*len(boxes)}


class TestAuditoriaDeteccion(unittest.TestCase):
    def test_correcto_sin_cobertura_geometria_y_material(self):
        g=objetivo([[.2,.2,.4,.4]],[0])
        casos=[(pred([[.2,.2,.4,.4]],[0]),'correcto'),
               (pred([],[]),'sin_cobertura_suficiente'),
               (pred([[0.,0.,1.,1.]],[0]),'geometria_incorrecta'),
               (pred([[.2,.2,.4,.4]],[3]),'material_equivocado')]
        for p,esperado in casos:self.assertEqual(auditar_imagen(p,g)['objetos'][0]['estado'],esperado)

    def test_parcial_por_debajo_de_cobertura(self):
        g=objetivo([[.2,.2,.6,.6]],[0])
        r=auditar_imagen(pred([[.2,.2,.3,.6]],[0]),g)
        self.assertEqual(r['objetos'][0]['estado'],'sin_cobertura_suficiente')
        self.assertLess(r['objetos'][0]['cobertura_max_gt'],.5)

    def test_material_no_oculta_deteccion_correcta(self):
        g=objetivo([[.2,.2,.4,.4]],[0])
        r=auditar_imagen(pred([[.2,.2,.4,.4]]*2,[3,0],[.9,.8]),g)
        self.assertEqual(r['tp'],1)
        self.assertEqual(r['objetos'][0]['estado'],'correcto')
        self.assertEqual(r['falsos_positivos'][0]['tipo_fp'],'material_equivocado')

    def test_unica_caja_no_cuenta_dos_gt(self):
        g=objetivo([[.2,.2,.4,.4]]*2,[0,0])
        r=auditar_imagen(pred([[.2,.2,.4,.4]],[0]),g)
        self.assertEqual((r['tp'],r['fn']),(1,1))
        self.assertEqual(r['objetos'][1]['subtipo'],'caja_compartida_sin_asignacion_univoca')

    def test_ignoradas_no_generan_cobertura_falsa(self):
        g=objetivo([[.2,.2,.4,.4]],[0],[[0.,0.,1.,1.]])
        r=auditar_imagen(pred([[0.,0.,1.,1.]],[0]),g)
        self.assertEqual(r['ignoradas'],1)
        self.assertEqual(r['objetos'][0]['estado'],'sin_cobertura_suficiente')
        r=auditar_imagen(pred([[.2,.2,.4,.4]],[2]),g)
        self.assertEqual(r['ignoradas'],0)
        self.assertEqual(r['objetos'][0]['estado'],'material_equivocado')

    def test_duplicado_y_fondo_no_desaparecen(self):
        g=objetivo([[.2,.2,.4,.4]],[0])
        r=auditar_imagen(pred([[.2,.2,.4,.4]]*2+[[.7,.7,.8,.8]],[0,0,0],[.9,.8,.7]),g)
        self.assertEqual((r['tp'],r['fp']),(1,2))
        self.assertEqual([x['tipo_fp'] for x in r['falsos_positivos']],['duplicado','fondo_o_solape_debil'])

    def test_tamanos_y_filtro_operativo(self):
        for px,esperado in [(7.9,'menos_8_px'),(8.,'8_a_16_px'),(16.,'16_a_32_px'),(32.,'32_px_o_mas')]:
            self.assertEqual(tamano_objeto(torch.tensor([0.,0.,px/384,px/384]))['grupo_tamano'],esperado)
        g=objetivo([[.2,.2,.4,.4]],[0])
        r=auditar_imagen(pred([[.2,.2,.4,.4]],[0],[.3]),g,umbral=.35)
        self.assertEqual((r['tp'],r['cajas']),(0,0))


if __name__=='__main__':unittest.main()
