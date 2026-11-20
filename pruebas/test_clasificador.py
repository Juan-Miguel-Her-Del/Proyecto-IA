from pathlib import Path
import sys
import tempfile
import unittest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.clasificador import ClasificadorResiduos, cargar_clasificador
from ecowaste.datasets import CLASES
from ecowaste.evaluacion import metricas_clasificacion


class ClasificadorTests(unittest.TestCase):
    def test_metricas_con_clases_y_errores_conocidos(self):
        resultado = metricas_clasificacion([0, 0, 1, 2], [0, 1, 1, 2], ["a", "b", "c"])
        self.assertEqual(resultado["accuracy"], 0.75)
        self.assertAlmostEqual(resultado["f1_macro"], 7 / 9)
        self.assertEqual(resultado["matriz_confusion"], [[1, 0, 0], [1, 1, 0], [0, 0, 1]])

    def test_parametros_y_patches_con_orden_espacial(self):
        modelo = ClasificadorResiduos()
        self.assertEqual(sum(p.numel() for p in modelo.parameters()), 629382)
        x = torch.arange(3 * 224 * 224).reshape(1, 3, 224, 224).float()
        patches = modelo.extraer_patches(x)
        self.assertEqual(tuple(patches.shape), (1, 196, 768))
        torch.testing.assert_close(patches[0, 0], x[0, :, :16, :16].reshape(-1))
        torch.testing.assert_close(patches[0, 1], x[0, :, :16, 16:32].reshape(-1))
        torch.testing.assert_close(patches[0, 14], x[0, :, 16:32, :16].reshape(-1))
        self.assertFalse(modelo.posicion.requires_grad)
        with self.assertRaises(ValueError):
            modelo(torch.rand(1, 3, 200, 224))

    def test_gradientes_y_recuperacion_exacta(self):
        torch.manual_seed(42)
        modelo = ClasificadorResiduos()
        x = torch.rand(2, 3, 224, 224)
        salida = modelo(x)
        self.assertEqual(tuple(salida.shape), (2, 6))
        perdida = torch.nn.functional.cross_entropy(salida, torch.tensor([0, 5]))
        perdida.backward()
        for nombre, p in modelo.named_parameters():
            self.assertIsNotNone(p.grad, nombre)
            self.assertTrue(bool(torch.isfinite(p.grad).all()), nombre)
            self.assertGreater(p.grad.abs().sum().item(), 0, nombre)
        with tempfile.TemporaryDirectory() as temporal:
            path = Path(temporal) / "modelo.pt"
            torch.save({"tipo": "ecowaste_clasificador_desde_cero_v1", "configuracion": modelo.configuracion,
                "clases": CLASES, "pesos": modelo.state_dict()}, path)
            recuperado, _ = cargar_clasificador(path)
            torch.testing.assert_close(modelo.eval()(x), recuperado(x), atol=0, rtol=0)

    def test_reanudar_pesos_optimizador_y_rng_reproduce_paso(self):
        torch.manual_seed(12)
        modelo = ClasificadorResiduos(imagen=32, dimension=32, bloques=1, mlp=64)
        optimizador = torch.optim.AdamW(modelo.parameters(), lr=0.001)
        x, y = torch.rand(2, 3, 32, 32), torch.tensor([0, 4])
        def paso(red, opt):
            opt.zero_grad(set_to_none=True)
            perturbada = (x + torch.rand_like(x) * 0.01).clamp(0, 1)
            torch.nn.functional.cross_entropy(red(perturbada), y).backward()
            opt.step()
        paso(modelo, optimizador)
        with tempfile.TemporaryDirectory() as temporal:
            path = Path(temporal) / "continuacion.pt"
            torch.save({"pesos": modelo.state_dict(), "opt": optimizador.state_dict(), "rng": torch.get_rng_state()}, path)
            paso(modelo, optimizador)
            restaurado = ClasificadorResiduos(imagen=32, dimension=32, bloques=1, mlp=64)
            opt_restaurado = torch.optim.AdamW(restaurado.parameters(), lr=0.001)
            datos = torch.load(path, weights_only=True)
            restaurado.load_state_dict(datos["pesos"])
            opt_restaurado.load_state_dict(datos["opt"])
            torch.set_rng_state(datos["rng"])
            paso(restaurado, opt_restaurado)
            for a, b in zip(modelo.parameters(), restaurado.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
