import tempfile
import unittest
from pathlib import Path
import torch
from ecowaste.datasets import geometria_letterbox, CLASES
from ecowaste.detector_centros import (DetectorCentros, VERSION, objetivos_centros,
    perdida_centros, postprocesar_centros, cargar_detector_centros)
from scripts.entrenar_clasificador import tasa_aprendizaje, aumentar, mezclar_lote


def objetivo(boxes, labels, ignore=(), g=None):
    return {'id': 'control', 'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.tensor(labels), 'ignore_boxes': torch.tensor(ignore, dtype=torch.float32).reshape(-1, 4),
            'geometria': g or geometria_letterbox(384, 384, 384)}


class TestCentros(unittest.TestCase):
    def test_colision_mismo_material_conserva_dos_objetos(self):
        o = objetivo([[.2, .2, .4, .4], [.21, .21, .39, .39]], [0, 0])
        t = objetivos_centros(o, .1)
        self.assertEqual(len(t['indices'].unique()), 2)
        self.assertEqual(int((t['calor'] == 1).sum()), 2)
        self.assertEqual(set(t['objetos'].tolist()), {0, 1})

    def test_colision_excesiva_no_se_oculta(self):
        o = objetivo([[.2, .2, .4, .4]] * 3, [0, 0, 0])
        with self.assertRaises(ValueError): objetivos_centros(o, .1)

    def test_positivos_prevalecen_sobre_ignorados(self):
        o = objetivo([[.2, .2, .4, .4]], [3], [[0, 0, 1, 1]])
        raw = torch.zeros(1, 48, 48, 6, 2, requires_grad=True)
        boxes = torch.ones(1, 48, 48, 6, 2, 4) * torch.tensor([.2, .2, .4, .4])
        loss, d = perdida_centros({'logits': raw, 'boxes': boxes}, [o], .1)
        loss.backward()
        self.assertEqual(d['objetos_asignados'], 1)
        self.assertLess(float(raw.grad.min()), 0)
        self.assertEqual(float(raw.grad.max()), 0)

    def test_sin_objetos_focal_finita(self):
        raw = torch.zeros(1, 48, 48, 6, 2, requires_grad=True)
        boxes = torch.zeros(1, 48, 48, 6, 2, 4)
        loss, d = perdida_centros({'logits': raw, 'boxes': boxes}, [objetivo([], [])], .1)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(d['objetos_asignados'], 0)
        self.assertGreater(float(raw.grad.sum()), 0)

    def test_padding_se_excluye(self):
        g = geometria_letterbox(384, 192, 384)
        t = objetivos_centros(objetivo([], [], g=g), .1)
        self.assertEqual(int(t['valida'].sum()), 48 * 24)

    def test_oraculo_cajas_individuales_y_nms(self):
        o = objetivo([[.1, .1, .2, .2], [.6, .6, .8, .8]], [0, 0])
        t = objetivos_centros(o, .1)
        logits = torch.full((1, 48, 48, 6, 2), -20.)
        boxes = torch.zeros(1, 48, 48, 6, 2, 4)
        logits.view(-1)[t['indices']] = 20
        boxes.view(-1, 4)[t['indices']] = o['boxes'][t['objetos']]
        pred = postprocesar_centros({'logits': logits, 'boxes': boxes}, [o['geometria']], .35)[0]
        self.assertEqual(len(pred['boxes']), 2)
        self.assertEqual(pred['labels'].tolist(), [0, 0])

    def test_modelo_gradiente_carga_sin_convoluciones(self):
        torch.set_num_threads(4)
        m = DetectorCentros()
        self.assertFalse(any(isinstance(x, torch.nn.Conv2d) for x in m.modules()))
        o = objetivo([[.2, .2, .4, .4]], [0])
        raw = m(torch.rand(1, 3, 384, 384))
        loss, d = perdida_centros(raw, [o], .1)
        loss.backward()
        self.assertGreater(float(m.tokenizador_detalle.weight.grad.abs().sum()), 0)
        self.assertEqual(d['objetos_asignados'], 1)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'm.pt'
            torch.save({'tipo': 'ecowaste_centros_desde_cero_v1', 'version': VERSION, 'clases': CLASES,
                        'configuracion': m.configuracion_centros, 'pesos': m.state_dict()}, p)
            cargado, _ = cargar_detector_centros(p)
            for k, v in m.state_dict().items(): self.assertTrue(torch.equal(v, cargado.state_dict()[k]))

    def test_programacion_coseno(self):
        cfg = {'learning_rate': .001, 'warmup_epocas': 2, 'epocas': 10,
               'programacion_lr': 'coseno', 'fraccion_lr_final': .1}
        self.assertAlmostEqual(tasa_aprendizaje(1, cfg), .0005)
        self.assertAlmostEqual(tasa_aprendizaje(2, cfg), .001)
        self.assertAlmostEqual(tasa_aprendizaje(10, cfg), .0001)
        self.assertGreater(tasa_aprendizaje(5, cfg), tasa_aprendizaje(8, cfg))

    def test_programacion_anterior_compatible(self):
        cfg = {'learning_rate': .001, 'warmup_epocas': 2}
        self.assertEqual(tasa_aprendizaje(3, cfg), .001)

    def test_afin_identidad_y_rango(self):
        x = torch.rand(2, 3, 32, 32)
        cfg = {'flip_horizontal': 0, 'contraste': [1, 1], 'brillo': [1, 1],
               'afin': {'grados': 0, 'escala': [1, 1], 'traslacion': 0}}
        torch.testing.assert_close(aumentar(x, cfg), x, atol=2e-6, rtol=1e-5)
        cfg['afin'] = {'grados': 10, 'escala': [.9, 1.1], 'traslacion': .05}
        y = aumentar(x, cfg)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(bool(((y >= 0) & (y <= 1)).all()))

    def test_mixup_respeta_imagenes_y_pesos(self):
        x = torch.arange(8).float()[:, None, None, None].expand(8, 3, 4, 4)
        y = torch.arange(8)
        imagenes, ya, yb, peso = mezclar_lote(x, y, .2)
        torch.testing.assert_close(imagenes[:, 0, 0, 0], peso*ya + (1-peso)*yb)
        self.assertGreaterEqual(peso, 0)
        self.assertLessEqual(peso, 1)
        self.assertEqual(sorted(yb.tolist()), y.tolist())
        sin, _, _, peso = mezclar_lote(x, y, 0)
        self.assertTrue(torch.equal(sin, x))
        self.assertEqual(peso, 1)


if __name__ == '__main__': unittest.main()
