from pathlib import Path
import sys
import tempfile
import unittest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.detector import DetectorResiduos, decodificar, cargar_detector
from ecowaste.datasets import CLASES
from ecowaste.perdida_detector import PerdidaDetector, asignar


def objetivo(boxes):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.zeros(len(boxes), dtype=torch.long),
        "ignore_boxes": torch.empty(0, 4)}


def codificar_cajas_positivas(raw, asignacion, cajas, radio):
    lado, candidatos = raw.shape[1], raw.shape[3]
    vista = raw.view(-1, 11)
    for indice, objeto in zip(asignacion["indices"].tolist(), asignacion["objetos"].tolist()):
        celda = indice // candidatos
        grid = raw.new_tensor([celda % lado, celda // lado])
        centro = (cajas[objeto, :2] + cajas[objeto, 2:]) / 2
        offset = (centro * lado - grid + radio) / (1 + 2 * radio)
        vista[indice, -4:-2] = torch.logit(offset)
        vista[indice, -2:] = torch.logit(cajas[objeto, 2:] - cajas[objeto, :2])


class SupervisionEspacialTests(unittest.TestCase):
    def test_pesos_de_clase_no_se_cancelan_entre_imagenes(self):
        objetivos = [objetivo([[.1, .1, .9, .9]]), objetivo([[.1, .1, .9, .9]])]
        objetivos[1]["labels"][0] = 1
        pesos = torch.tensor([.1, 1., 1., 1., 1., 1.])
        razones = {}
        for modo in ["por_imagen", "global"]:
            raw = torch.zeros(2, 1, 1, 2, 11, requires_grad=True)
            loss, _ = PerdidaDetector(pesos_clase=pesos, normalizacion_clases=modo)(raw, objetivos)
            loss.backward()
            normas = raw.grad[..., 1:7].abs().flatten(1).sum(1)
            razones[modo] = float(normas[1] / normas[0])
        self.assertAlmostEqual(razones["por_imagen"], 1, places=5)
        self.assertAlmostEqual(razones["global"], 10, places=5)

    def test_vecinos_regresan_a_anotacion_real(self):
        o = objetivo([[.2, .2, .8, .8]])
        originales = o["boxes"].clone()
        a = asignar(o, lado=4, radio_centros=1)
        self.assertEqual(len(a["indices"]), 4)
        self.assertEqual(len(a["indices"].unique()), 4)
        raw = torch.zeros(1, 4, 4, 2, 11)
        codificar_cajas_positivas(raw, a, o["boxes"], 1)
        cajas, _, _ = decodificar(raw, radio_centros=1)
        torch.testing.assert_close(cajas.view(-1, 4)[a["indices"]], o["boxes"][a["objetos"]], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(o["boxes"], originales)

    def test_centrales_se_reservan_y_objeto_pequeno_no_supervisa_fondo(self):
        o = objetivo([[.49, .49, .51, .51], [.2, .2, .8, .8]])
        a = asignar(o, lado=4, radio_centros=1)
        self.assertEqual(int((a["objetos"] == 0).sum()), 1)
        self.assertGreater(int((a["objetos"] == 1).sum()), 1)
        self.assertEqual(a["colisiones"], [])
        centro = (2 * 4 + 2) * 2
        self.assertIn(centro, a["indices"].tolist())
        self.assertIn(centro + 1, a["indices"].tolist())

    def test_mismo_peso_por_objeto_con_una_o_nueve_celdas(self):
        o = objetivo([[0, 0, 1, 1]])
        # Mismas predicciones positivas, independientemente de cuántas celdas las supervisan.
        predichas = torch.tensor([[.05, .05, .95, .95]])
        perdidas = []
        for radio in [0, 1]:
            a = asignar(o, lado=3, radio_centros=radio)
            raw = torch.zeros(1, 3, 3, 2, 11)
            codificar_cajas_positivas(raw, a, predichas, radio)
            raw.requires_grad_()
            loss, detalle = PerdidaDetector(ignorar_interiores_gt=True, radio_centros=radio)(raw, [o])
            self.assertEqual(detalle["positivos"], 1 if radio == 0 else 9)
            self.assertEqual(detalle["negativos"], 0)
            loss.backward()
            self.assertTrue(bool(torch.isfinite(raw.grad).all()))
            perdidas.append(float(loss.detach()))
        self.assertAlmostEqual(*perdidas, places=5)

    def test_modelos_anteriores_cargan_sin_cambiar_salida(self):
        modelo = DetectorResiduos(imagen=16, patch_detalle=4, patch_contexto=8, dimension=32, bloques_por_rama=1, mlp=64, ventana=2).eval()
        config_legacy = dict(modelo.configuracion)
        config_legacy.pop("radio_centros")
        config_legacy.pop("escala_posicion")
        with tempfile.TemporaryDirectory() as temporal:
            path = Path(temporal) / "legacy.pt"
            torch.save({"tipo": "ecowaste_detector_desde_cero_v1", "clases": CLASES,
                "configuracion": config_legacy, "pesos": modelo.state_dict()}, path)
            recuperado, _ = cargar_detector(path)
            self.assertEqual(recuperado.configuracion["radio_centros"], 0)
            self.assertEqual(recuperado.configuracion["escala_posicion"], 1)
            entrada = torch.rand(1, 3, 16, 16)
            torch.testing.assert_close(modelo(entrada), recuperado(entrada), atol=0, rtol=0)
        modelo3 = DetectorResiduos(radio_centros=1)
        self.assertEqual(sum(p.numel() for p in modelo3.parameters()), 440182)

    def test_escala_posicional_es_fija_y_preserva_numero_de_parametros(self):
        argumentos = dict(imagen=16, patch_detalle=4, patch_contexto=8, dimension=32, bloques_por_rama=1, mlp=64, ventana=2)
        modelo = DetectorResiduos(**argumentos).eval()
        sin_posicion = DetectorResiduos(**argumentos, escala_posicion=0).eval()
        sin_posicion.load_state_dict(modelo.state_dict())
        entrada = torch.full((1, 3, 16, 16), .5)
        raw = sin_posicion(entrada)
        torch.testing.assert_close(raw, raw[:, :1, :1].expand_as(raw), atol=1e-6, rtol=1e-6)
        self.assertGreater(float(modelo(entrada).detach().std(dim=(1, 2)).mean()), 1e-4)
        self.assertEqual(sum(p.numel() for p in modelo.parameters()), sum(p.numel() for p in sin_posicion.parameters()))


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
