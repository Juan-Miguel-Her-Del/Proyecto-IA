import math
from pathlib import Path
import sys
import tempfile
import unittest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.asistente import Asistente, INTENCIONES, Tokenizador, TransformerIntenciones, responder
from ecowaste.datos import ArbolHash, caja_valida, recortar_desborde_minimo
from ecowaste.datasets import letterbox, transformar_caja
from PIL import Image


class GeometriaTests(unittest.TestCase):
    def test_cajas_limite_y_no_finitas(self):
        self.assertTrue(caja_valida([0, 0, 100, 80], 100, 80))
        for caja in [[-1, 0, 2, 2], [0, 0, 0, 1], [99, 0, 2, 1], [math.nan, 0, 1, 1], [0, 0, math.inf, 1]]:
            self.assertFalse(caja_valida(caja, 100, 80))

    def test_busqueda_hash_contra_fuerza_bruta(self):
        valores = [0, 1, 3, 15, 127, 255, 0xFF00FF, 0x123456, 0x123456]
        arbol = ArbolHash()
        for i, valor in enumerate(valores):
            arbol.agregar(valor, i)
        for consulta in [0, 7, 0x123456]:
            esperado = {i for i, v in enumerate(valores) if (consulta ^ v).bit_count() <= 3}
            self.assertEqual(set(arbol.buscar(consulta, 3)), esperado)

    def test_recorte_conservador_no_inventa_cajas(self):
        self.assertEqual(recortar_desborde_minimo([10, -1, 20, 100], 200, 200), [10, 0, 20, 99])
        self.assertIsNone(recortar_desborde_minimo([-10, 0, 100, 100], 200, 200))
        self.assertIsNone(recortar_desborde_minimo([0, -1, 2, 2], 200, 200))
        self.assertIsNone(recortar_desborde_minimo([math.nan, 0, 10, 10], 200, 200))

    def test_letterbox_caja_y_retorno_a_coordenadas_originales(self):
        tensor, g = letterbox(Image.new("RGB", (200, 100)), 384)
        self.assertEqual(tuple(tensor.shape), (3, 384, 384))
        caja = transformar_caja([20, 10, 80, 40], g)
        self.assertAlmostEqual(caja[0], 0.1)
        self.assertAlmostEqual(caja[1], 0.3)
        self.assertAlmostEqual((caja[2] * 384 - g['relleno_x']) / g['escala_x'], 100)
        self.assertAlmostEqual((caja[3] * 384 - g['relleno_y']) / g['escala_y'], 50)


class AsistenteTests(unittest.TestCase):
    def test_padding_no_cambia_logits_y_gradientes_son_finitos(self):
        torch.manual_seed(42)
        modelo = TransformerIntenciones(12).eval()
        corta = torch.tensor([[2, 3, 4, 0]])
        larga = torch.tensor([[2, 3, 4, 0, 0, 0, 0]])
        torch.testing.assert_close(modelo(corta), modelo(larga), atol=1e-6, rtol=1e-6)
        modelo(larga).square().mean().backward()
        for p in modelo.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(bool(torch.isfinite(p.grad).all()))
        self.assertEqual(modelo.embedding.weight.grad[0].abs().sum().item(), 0)

    def test_vocabulario_solo_train_y_checkpoint(self):
        tokenizador = Tokenizador.ajustar([{"pregunta": "material papel", "contexto_texto": "contexto"}])
        self.assertNotIn("exclusivovalidacion", tokenizador.vocabulario)
        self.assertIn(1, tokenizador.codificar("exclusivovalidacion", "contexto"))
        modelo = TransformerIntenciones(len(tokenizador.vocabulario)).eval()
        with tempfile.TemporaryDirectory() as temporal:
            path = Path(temporal) / "modelo.pt"
            torch.save({"tipo": "ecowaste_intenciones_desde_cero_v1", "configuracion": modelo.configuracion,
                "vocabulario": tokenizador.vocabulario, "pesos": modelo.state_dict(), "umbral": 0.9}, path)
            asistente = Asistente(path)
            ids = torch.tensor([tokenizador.codificar("material papel", "contexto")])
            torch.testing.assert_close(modelo(ids), asistente.modelo(ids))
            self.assertEqual(asistente.consultar("", [])["intencion"], "abstencion")

    def test_respuestas_fundamentadas(self):
        self.assertIn("No tengo detecciones", responder("separacion", []))
        objeto = {"material": "plastico"}
        respuesta = responder("separacion", [objeto])
        self.assertIn("si es aprovechable", respuesta)
        self.assertIn("¿Está limpio", respuesta)
        self.assertNotIn("metal", respuesta)
        self.assertIn("No tengo detecciones", responder("material", [objeto], "vidrio"))
        self.assertIn("gestor especializado", responder("separacion", [{"material": "metal", "peligroso": True}]))
        self.assertIn("no garantiza", responder("incertidumbre", [objeto]))


if __name__ == "__main__":
    unittest.main()
