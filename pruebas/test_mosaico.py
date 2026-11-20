import unittest
import torch
from ecowaste.datasets import geometria_letterbox, transformar_caja
from ecowaste.cajas import deshacer_letterbox
from ecowaste.mosaico import ventanas, proyectar_al_lienzo
from ecowaste.mosaico import predecir_mosaico
from PIL import Image


class TestMosaico(unittest.TestCase):
    def test_proyeccion_con_escala_padding_y_desplazamiento(self):
        ventana = (200, 75, 500, 300)
        local = geometria_letterbox(300, 225, 384)
        caja = torch.tensor([transformar_caja([50, 25, 100, 100], local)])
        salida = proyectar_al_lienzo(caja, local, ventana, 614, 399)
        original = deshacer_letterbox(salida, geometria_letterbox(614, 399, 384))
        torch.testing.assert_close(original, torch.tensor([[250., 100., 350., 200.]]), atol=3e-5, rtol=1e-6)

    def test_cobertura_completa_sin_ventanas_invalidas(self):
        for ancho, alto in [(100, 70), (1, 1), (524, 524)]:
            zonas = ventanas(ancho, alto)
            self.assertEqual(zonas[0], (0, 0, ancho, alto))
            self.assertEqual(len(zonas), len(set(zonas)))
            cobertura = torch.zeros(alto, ancho, dtype=torch.bool)
            for x0, y0, x1, y1 in zonas[1:] or zonas:
                self.assertTrue(0 <= x0 < x1 <= ancho and 0 <= y0 < y1 <= alto)
                cobertura[y0:y1, x0:x1] = True
            self.assertTrue(bool(cobertura.all()))

    def test_filtro_visual_conserva_cajas_de_evaluacion(self):
        class MapaControl(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(()))
                self.imagen = 384
                self.configuracion_mapas = {'formulacion': 'softmax7'}
            def forward(self, x):
                raw = torch.zeros(len(x), 7, 48, 48) + self.p*0
                raw[:, 1, 16:32, 16:32] = .2
                raw[:, 1, 19:29, 19:29] = 3
                return raw
        torch.set_num_threads(4)
        m = MapaControl().eval()
        imagen = Image.new('RGB', (384, 384))
        base = predecir_mosaico(m, imagen, .05, umbral_propuestas=.05)
        visible = predecir_mosaico(m, imagen, .35, umbral_propuestas=.05)
        keep = base['scores'] >= .35
        self.assertGreater(int(keep.sum()), 0)
        torch.testing.assert_close(visible['boxes'], base['boxes'][keep], atol=0, rtol=0)
        torch.testing.assert_close(visible['scores'], base['scores'][keep], atol=0, rtol=0)
        self.assertTrue(torch.equal(visible['labels'],base['labels'][keep]))


if __name__ == '__main__': unittest.main()
