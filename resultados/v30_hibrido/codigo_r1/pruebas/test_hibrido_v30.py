"""Regresiones nuevas V30; las pruebas de los modelos históricos se conservan."""
import copy
from pathlib import Path
import tempfile
import unittest
import torch
from ecowaste.backbone_cnn import BackboneCNN, mascara_a
from ecowaste.detector_hibrido import DetectorHibrido, crear_pareja, mascara_letterbox, cargar_hibrido, TIPO
from ecowaste.detector import patches_rgb, dividir_ventanas, unir_ventanas
from ecowaste.detector_mapas import DetectorMapas, perdida_mapas, cajas_desde_mapas
from ecowaste.datasets import geometria_letterbox, transformar_caja, CLASES
from ecowaste.cajas import deshacer_letterbox
from ecowaste.experimento_v30 import comprobar, planificar


class HibridoV30(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_escalas_cnn(self):
        with torch.no_grad():
            a, b = BackboneCNN()(torch.rand(2, 3, 384, 384))
        self.assertEqual(a.shape, (2, 96, 48, 48))
        self.assertEqual(b.shape, (2, 96, 24, 24))

    def test_formas_y_forward(self):
        with torch.no_grad():
            raw, f = DetectorHibrido()(torch.rand(1, 3, 384, 384), devolver_caracteristicas=True)
        self.assertEqual(raw.shape, (1, 7, 48, 48))
        self.assertEqual(f.shape, (1, 48, 48, 96))
        self.assertTrue(torch.isfinite(raw).all())

    def test_gradientes_cnn_transformer_y_head(self):
        m = DetectorHibrido()
        y = torch.zeros(1, 48, 48, dtype=torch.long)
        y[:, 15:30, 10:30] = 4
        perdida_mapas(m(torch.rand(1, 3, 384, 384)), y).backward()
        for modulo in [m.backbone, m.detalle, m.contexto, m.cabezal]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in modulo.parameters()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in modulo.parameters()))

    def test_transformer_interviene_en_prediccion(self):
        m = DetectorHibrido().eval(); x = torch.rand(1, 3, 384, 384)
        with torch.no_grad():
            antes = m(x)
            for bloque in list(m.detalle)+list(m.contexto):
                for p in bloque.parameters():
                    p.zero_()
            despues = m(x)
        self.assertGreater(float((antes-despues).abs().max()), .001)

    def test_control_no_cnn_y_head_comun(self):
        a, b = crear_pareja()
        self.assertFalse(any(isinstance(x, torch.nn.Conv2d) for x in a.modules()))
        self.assertTrue(any(isinstance(x, torch.nn.Conv2d) for x in b.modules()))
        for n, t in a.state_dict().items():
            if n.startswith(('detalle.', 'contexto.', 'cabezal.', 'fusion_', 'norma_', 'clasificador_')):
                self.assertTrue(torch.equal(t, b.state_dict()[n]), n)

    def test_control_equivale_historico_sin_padding(self):
        a = DetectorHibrido(cnn=False).eval()
        old = DetectorMapas(auxiliar_clasificacion=True).eval()
        old.load_state_dict(a.state_dict(), strict=True)
        x = torch.rand(1, 3, 384, 384)
        with torch.no_grad():
            self.assertTrue(torch.allclose(a(x), old(x), rtol=1e-4, atol=1e-5))

    def test_padding_por_geometria(self):
        g = geometria_letterbox(800, 400, 384)
        m = mascara_letterbox([g])[0]
        self.assertEqual(int(m.sum()), 384*192)
        self.assertFalse(m[:96].any())
        self.assertTrue(m[96:288].all())
        self.assertEqual(int(mascara_a(m[None], (48, 48)).sum()), 48*24)

    def test_padding_no_depende_de_rgb(self):
        m = DetectorHibrido().eval(); x = torch.rand(1, 3, 384, 384)
        v = mascara_letterbox([geometria_letterbox(800, 400, 384)])
        z = x.clone().masked_fill(~v[:, None], 100)
        with torch.no_grad():
            self.assertTrue(torch.equal(m(x, v), m(z, v)))

    def test_todo_padding_finito(self):
        m = DetectorHibrido()
        raw, f = m(torch.zeros(1, 3, 384, 384), torch.zeros(1, 384, 384, dtype=torch.bool), True)
        self.assertTrue(torch.isfinite(raw).all())
        self.assertEqual(int(torch.count_nonzero(f)), 0)

    def test_orden_espacial(self):
        x = torch.arange(48*48).reshape(1, 48, 48, 1)
        self.assertTrue(torch.equal(x, unir_ventanas(dividir_ventanas(x, 8), 1, 48, 8)))
        n = torch.arange(48*48).reshape(1, 1, 48, 48)
        self.assertEqual(int(n.flatten(2).transpose(1, 2)[0, 49]), 49)

    def test_cajas_letterbox_inversa(self):
        g = geometria_letterbox(913, 401, 384)
        xywh = [100, 30, 200, 150]
        box = torch.tensor([transformar_caja(xywh, g)])
        self.assertTrue(torch.allclose(deshacer_letterbox(box, g), torch.tensor([[100., 30., 300., 180.]]), atol=1e-4))

    def test_multiples_instancias_y_vacio(self):
        raw = torch.full((2, 7, 48, 48), -10.)
        raw[:, 0] = 10
        for x in [8, 25]:
            raw[0, 0, 10:15, x:x+5] = -10
            raw[0, 3, 10:15, x:x+5] = 10
        ps = cajas_desde_mapas(raw)
        self.assertEqual(len(ps[0]['boxes']), 2)
        self.assertEqual(len(ps[1]['boxes']), 0)

    def test_limitacion_objetos_tocantes_explicita(self):
        raw = torch.zeros(1, 7, 48, 48); raw[:, 0] = 10
        raw[:, 0, 10:20, 10:30] = -10; raw[:, 3, 10:20, 10:30] = 20
        self.assertEqual(len(cajas_desde_mapas(raw)[0]['boxes']), 1)

    def test_loss_vacia_e_ignore(self):
        raw = torch.randn(2, 7, 48, 48, requires_grad=True)
        y = torch.zeros(2, 48, 48, dtype=torch.long); y[1] = 255
        loss = perdida_mapas(raw, y); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(raw.grad[1].abs().sum()), 0.)

    def test_loss_original_reutilizada(self):
        m = DetectorHibrido()
        self.assertEqual(m.configuracion_mapas['formulacion'], 'softmax7')
        self.assertEqual(m.cabezal[-1].out_features, 7)
        self.assertEqual(m.clasificador_auxiliar.out_features, 6)

    def test_guardado_carga_exactos(self):
        m = DetectorHibrido().eval(); x = torch.rand(1, 3, 384, 384)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'m.pt'
            torch.save({'tipo': TIPO, 'clases': CLASES, 'pesos_externos': False, 'configuracion_hibrido': m.configuracion_hibrido, 'pesos': m.state_dict()}, p)
            nuevo, _ = cargar_hibrido(p)
            with torch.no_grad():
                self.assertTrue(torch.equal(m(x), nuevo(x)))

    def test_rechazar_checkpoint_externo(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'m.pt'; torch.save({'tipo': TIPO, 'clases': CLASES, 'pesos_externos': True}, p)
            with self.assertRaises(ValueError):
                cargar_hibrido(p)

    def test_rechazar_formas_y_mascaras(self):
        m = DetectorHibrido()
        with self.assertRaises(ValueError):
            m(torch.rand(1, 3, 224, 224))
        with self.assertRaises(ValueError):
            m(torch.rand(1, 3, 384, 384), torch.ones(1, 384, 384))

    def test_inicializacion_reproducible(self):
        a, b = crear_pareja(); aa, bb = crear_pareja()
        self.assertTrue(all(torch.equal(t, aa.state_dict()[n]) for n, t in a.state_dict().items()))
        self.assertTrue(all(torch.equal(t, bb.state_dict()[n]) for n, t in b.state_dict().items()))

    def test_no_batchnorm_ni_pesos_descargados(self):
        m = DetectorHibrido()
        self.assertFalse(any(isinstance(x, torch.nn.BatchNorm2d) for x in m.modules()))
        self.assertIsInstance(m.backbone.etapas[0][1], torch.nn.GroupNorm)

    def test_compatibilidad_historica_carga_estricta(self):
        old = DetectorMapas(auxiliar_clasificacion=True)
        nuevo = DetectorHibrido(cnn=False)
        nuevo.load_state_dict(old.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            DetectorHibrido().load_state_dict(old.state_dict(), strict=True)

    def test_geometria_sin_gt(self):
        g = geometria_letterbox(333, 900, 384)
        self.assertEqual(mascara_letterbox([g]).shape, (1, 384, 384))
        with self.assertRaises(ValueError):
            mascara_letterbox([])

    def test_plan_exposicion_y_rng_privado(self):
        from types import SimpleNamespace
        cfg = {'semilla': 3001, 'actualizaciones': 10}
        ht = SimpleNamespace(registros=list(range(9)))
        nu = SimpleNamespace(registros=[{'fuente': 'packwise'}]*3+[{'fuente': 'click'}]*5)
        gl = SimpleNamespace(etiquetas=torch.arange(6).repeat(4))
        antes = torch.get_rng_state().clone()
        p = planificar(cfg, ht, nu, gl)
        q = planificar(cfg, ht, nu, gl)
        self.assertTrue(torch.equal(antes, torch.get_rng_state()))
        self.assertEqual(p['historico'].shape, (10, 8))
        self.assertEqual(p['global'].shape, (10, 16))
        self.assertTrue((p['nuevo'][:, :2] < 3).all())
        self.assertTrue((p['nuevo'][:, 2:] >= 3).all())
        self.assertTrue(all(torch.equal(p[k], q[k]) for k in p))

    def test_reanudacion_rechaza_otro_protocolo_y_rama(self):
        ck = {'tipo': TIPO, 'rama': 'B', 'contrato': {}, 'configuracion_hibrido': {'cnn': True}}
        with self.assertRaises(ValueError): comprobar(ck, 'A', {})
        with self.assertRaises(ValueError): comprobar(ck, 'B', {'otra': 'config'})

    def test_reanudacion_rechaza_presupuesto(self):
        ck = {'tipo': TIPO, 'rama': 'B', 'contrato': {'config': {'actualizaciones': 2}},
              'configuracion_hibrido': {'cnn': True}, 'paso': 3, 'historial': []}
        with self.assertRaises(ValueError): comprobar(ck, 'B', ck['contrato'])

    def test_instancias_forward_backward_sin_componentes(self):
        from ecowaste.detector_hibrido_instancias import DetectorHibridoInstancias, perdida_consultas
        m = DetectorHibridoInstancias(consultas=16, capas=1)
        s = m(torch.rand(1, 3, 384, 384))
        self.assertEqual(s['logits'].shape, (1, 16, 7))
        self.assertEqual(s['boxes'].shape, (1, 16, 4))
        objetivos = [{'boxes': torch.tensor([[.1,.1,.2,.3],[.22,.1,.33,.3]]), 'labels': torch.tensor([2,2]),
                      'ignore_boxes': torch.empty(0,4)}]
        loss, d = perdida_consultas(s, objetivos)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(d['objetos_asignados'], 2)
        for modulo in [m.backbone, m.detalle, m.contexto, m.decodificador, m.material, m.caja]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in modulo.parameters()))

    def test_instancias_no_fusiona_cajas_mismo_material(self):
        from ecowaste.detector_hibrido_instancias import postprocesar_consultas
        logits = torch.full((1, 2, 7), -10.); logits[:, :, 2] = 10
        s = {'logits': logits, 'boxes': torch.tensor([[[.1,.1,.3,.4],[.25,.1,.45,.4]]])}
        p = postprocesar_consultas(s, [geometria_letterbox(384,384,384)])
        self.assertEqual(len(p[0]['boxes']), 2)

    def test_instancias_vacio_y_padding_finito(self):
        from ecowaste.detector_hibrido_instancias import DetectorHibridoInstancias, perdida_consultas
        m = DetectorHibridoInstancias(consultas=16, capas=1)
        s = m(torch.zeros(1,3,384,384), torch.zeros(1,384,384,dtype=torch.bool))
        loss, d = perdida_consultas(s, [{'boxes': torch.empty(0,4), 'labels': torch.empty(0,dtype=torch.long), 'ignore_boxes': torch.empty(0,4)}])
        self.assertTrue(torch.isfinite(s['logits']).all())
        self.assertTrue(torch.isfinite(s['boxes']).all())
        self.assertTrue(torch.isfinite(loss))

    def test_prohibidos_modelos_preentrenados_en_modulos_v30(self):
        import ast
        root = Path(__file__).resolve().parents[1]
        rutas = ['ecowaste/backbone_cnn.py', 'ecowaste/detector_hibrido.py', 'ecowaste/detector_hibrido_instancias.py']
        prohibidos = ['torchvision', 'ultralytics', 'transformers', 'segment_anything', 'sam2', 'timm', 'requests']
        for rel in rutas:
            arbol = ast.parse((root/rel).read_text(encoding='utf8'))
            for n in ast.walk(arbol):
                if isinstance(n, ast.Import):
                    self.assertTrue(all(x.name.split('.')[0] not in prohibidos for x in n.names))
                elif isinstance(n, ast.ImportFrom):
                    self.assertNotIn((n.module or '').split('.')[0], prohibidos)
