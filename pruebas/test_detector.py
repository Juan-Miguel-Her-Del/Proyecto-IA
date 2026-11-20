from pathlib import Path
import sys
import tempfile
import unittest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.cajas import deshacer_letterbox, giou_alineado, iou_pares, nms_por_clase
from ecowaste.datasets import CLASES, letterbox, transformar_caja
from ecowaste.detector import DetectorResiduos, cargar_detector, decodificar, dividir_ventanas, unir_ventanas
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.ejecucion_detector import aumentar_lote
from ecowaste.inferencia_detector import postprocesar
from ecowaste.perdida_detector import PerdidaDetector, asignar
from PIL import Image


def objetivo(boxes=(), labels=(), ignoradas=()):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.long), "ignore_boxes": torch.tensor(ignoradas, dtype=torch.float32).reshape(-1, 4)}


class DetectorTests(unittest.TestCase):
    def test_ventanas_preservan_orden_y_parametros(self):
        x = torch.arange(2 * 16 * 16 * 3).reshape(2, 16, 16, 3)
        ventanas = dividir_ventanas(x, 8)
        torch.testing.assert_close(ventanas[0], x[0, :8, :8].reshape(64, 3))
        torch.testing.assert_close(unir_ventanas(ventanas, 2, 16, 8), x)
        modelo = DetectorResiduos()
        self.assertEqual(sum(p.numel() for p in modelo.parameters()), 440182)
        with torch.no_grad():
            self.assertEqual(tuple(modelo(torch.rand(1, 3, 384, 384)).shape), (1, 48, 48, 2, 11))

    def test_decodificacion_y_geometria_conocida(self):
        cajas, _, _ = decodificar(torch.zeros(1, 2, 2, 2, 11))
        torch.testing.assert_close(cajas[0, 0, 0, 0], torch.tensor([0., 0., .5, .5]))
        _, g = letterbox(Image.new("RGB", (200, 100)), 384)
        normalizada = torch.tensor([transformar_caja([20, 10, 80, 40], g)])
        torch.testing.assert_close(deshacer_letterbox(normalizada, g), torch.tensor([[20., 10., 100., 50.]]))
        a = torch.tensor([[0., 0., 1., 1.]])
        self.assertAlmostEqual(float(giou_alineado(a, a)), 1)
        self.assertAlmostEqual(float(giou_alineado(a, a + torch.tensor([2., 0., 2., 0.]))), -1/3, places=6)

    def test_asignacion_colisiones_positivos_ganan_a_ignorados(self):
        o = objetivo([[.1, .1, .4, .4], [.15, .15, .35, .35], [.2, .2, .3, .3]], [0, 1, 2], [[0, 0, .5, .5]])
        a = asignar(o, lado=2)
        self.assertEqual(len(a["indices"]), 2)
        self.assertEqual(len(a["colisiones"]), 1)
        raw = torch.zeros(1, 2, 2, 2, 11, requires_grad=True)
        loss, detalle = PerdidaDetector()(raw, [o])
        self.assertEqual(detalle["positivos"], 2)
        self.assertGreater(detalle["negativos"], 0)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))

    def test_sin_objetos_y_regiones_totalmente_ignoradas(self):
        raw = torch.zeros(1, 2, 2, 2, 11, requires_grad=True)
        loss, detalle = PerdidaDetector()(raw, [objetivo(ignoradas=[[0, 0, 1, 1]])])
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(detalle["negativos"], 0)
        loss.backward()
        self.assertEqual(float(raw.grad.abs().sum()), 0)
        raw2 = torch.zeros(1, 2, 2, 2, 11, requires_grad=True)
        loss2, d2 = PerdidaDetector()(raw2, [objetivo()])
        self.assertGreater(float(loss2.detach()), 0)
        self.assertEqual(d2["positivos"], 0)
        loss2.backward()
        self.assertTrue(bool((raw2.grad[..., 0] > 0).all()))

    def test_gradiente_de_negativo_ignorado_es_cero(self):
        raw = torch.zeros(1, 2, 2, 2, 11, requires_grad=True)
        o = objetivo([[.1, .1, .4, .4]], [0], [[0, 0, .5, .5]])
        loss, _ = PerdidaDetector()(raw, [o])
        loss.backward()
        asignacion = asignar(o, lado=2)
        indice = int(asignacion["indices"][0])
        grad = raw.grad[..., 0].flatten()
        self.assertLess(float(grad[indice]), 0)
        self.assertEqual(float(grad[1 - indice]), 0)

    def test_variante_no_penaliza_interiores_de_objetos(self):
        o = objetivo([[0, 0, 1, 1]], [0])
        raw = torch.zeros(1, 2, 2, 2, 11, requires_grad=True)
        loss, detalle = PerdidaDetector(ignorar_interiores_gt=True)(raw, [o])
        self.assertEqual(detalle["positivos"], 1)
        self.assertEqual(detalle["negativos"], 0)
        loss.backward()
        indice = int(asignar(o, lado=2)["indices"][0])
        grad = raw.grad[..., 0].flatten()
        self.assertLess(float(grad[indice]), 0)
        self.assertEqual(int((grad != 0).sum()), 1)

    def test_nms_por_clase_y_ap_perfecta(self):
        cajas = torch.tensor([[0., 0., .5, .5]] * 3)
        keep = nms_por_clase(cajas, torch.tensor([.9, .8, .7]), torch.tensor([0, 0, 1]))
        self.assertEqual(keep.tolist(), [0, 2])
        o = objetivo([[0, 0, .5, .5]], [0], [[.6, .6, 1, 1]])
        pred = {"boxes": torch.tensor([[0., 0., .5, .5], [.6, .6, 1., 1.]]), "scores": torch.tensor([.9, .8]), "labels": torch.tensor([0, 0])}
        metrica = evaluar_detecciones([pred], [o])
        self.assertAlmostEqual(metrica["map50"], 1)
        self.assertEqual(metrica["por_clase"]["plastico"]["por_umbral"][0]["fp"], 0)
        self.assertEqual(metrica["por_clase"]["plastico"]["por_umbral"][0]["predicciones_ignoradas"], 1)
        self.assertIsNone(metrica["por_clase"]["papel"]["ap50"])

    def test_material_equivocado_y_duplicados_no_se_ocultan_en_ignorar(self):
        o = objetivo([[0, 0, .5, .5]], [0], [[0, 0, 1, 1]])
        pred = {"boxes": torch.tensor([[0., 0., .5, .5]] * 3), "scores": torch.tensor([.9, .8, .7]), "labels": torch.tensor([0, 0, 1])}
        m = evaluar_detecciones([pred], [o])
        self.assertEqual(m["por_clase"]["plastico"]["por_umbral"][0]["fp"], 1)
        self.assertEqual(m["por_clase"]["papel"]["por_umbral"][0]["fp"], 1)

    def test_flip_ajusta_cajas_ignoradas_y_relleno_asimetrico(self):
        tensor, g = letterbox(Image.new("RGB", (100, 201)), 384)
        caja = transformar_caja([10, 20, 30, 40], g)
        o = objetivo([caja], [0], [caja])
        o["geometria"] = g
        imagen, objetivos = aumentar_lote(tensor.unsqueeze(0).clone(), [o], {"flip_horizontal": 1.0, "brillo": [1, 1], "contraste": [1, 1]}, 0.1)
        nuevo = objetivos[0]
        self.assertEqual(nuevo["geometria"]["relleno_x"], 97)
        torch.testing.assert_close(deshacer_letterbox(nuevo["boxes"], nuevo["geometria"]), torch.tensor([[60., 20., 90., 60.]]))
        torch.testing.assert_close(nuevo["boxes"], nuevo["ignore_boxes"])
        torch.testing.assert_close(o["boxes"], torch.tensor([caja]))

    def test_padding_no_produce_detecciones_y_checkpoint(self):
        raw = torch.zeros(1, 4, 4, 2, 11)
        raw[..., 0] = 10
        raw[..., 1] = 10
        raw[..., -2:] = -3
        _, g = letterbox(Image.new("RGB", (100, 200)), 384)
        pred = postprocesar(raw, umbral=0.25, geometrias=[g])[0]
        self.assertEqual(len(pred["boxes"]), 8)
        self.assertTrue(bool((pred["boxes"][:, 0] >= .25).all()))
        self.assertTrue(bool((pred["boxes"][:, 2] <= .75).all()))
        modelo = DetectorResiduos(imagen=16, patch_detalle=4, patch_contexto=8, dimension=32, bloques_por_rama=1, mlp=64, ventana=2).eval()
        entrada = torch.rand(1, 3, 16, 16)
        with tempfile.TemporaryDirectory() as temporal:
            path = Path(temporal) / "det.pt"
            torch.save({"tipo": "ecowaste_detector_desde_cero_v1", "clases": CLASES,
                "configuracion": modelo.configuracion, "pesos": modelo.state_dict()}, path)
            restaurado, _ = cargar_detector(path)
            torch.testing.assert_close(modelo(entrada), restaurado(entrada), atol=0, rtol=0)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
