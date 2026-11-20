import itertools
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ecowaste.asignacion_optima import asignacion_optima
from ecowaste.datasets import CLASES,geometria_letterbox
from ecowaste.detector_consultas import DetectorConsultas,perdida_consultas,postprocesar_consultas,cargar_detector_consultas,VERSION_CONSULTAS,pesos_materiales,auto_atencion_posicional


class ConsultasTests(unittest.TestCase):
    def test_matching_minimo_global_y_costes_negativos(self):
        rng=np.random.default_rng(42)
        for n,m in [(1,3),(2,3),(3,4)]:
            for _ in range(10):
                a=rng.normal(size=(n,m));indices=asignacion_optima(a)
                optimo=min(sum(a[i,j] for i,j in enumerate(p)) for p in itertools.permutations(range(m),n))
                self.assertAlmostEqual(float(a[np.arange(n),indices].sum()),optimo,places=10)
                self.assertEqual(len(set(indices)),n)
        self.assertEqual(len(asignacion_optima(np.empty((0,10)))),0)
        with self.assertRaises(ValueError):asignacion_optima(np.zeros((3,2)))

    def test_matching_resuelve_caso_donde_falla_greedy(self):
        a=np.array([[1,2],[1.1,100]])
        self.assertEqual(asignacion_optima(a).tolist(),[1,0])

    def test_dos_objetos_cercanos_de_igual_material_tienen_parejas_distintas(self):
        logits=torch.zeros(1,3,7,requires_grad=True)
        boxes=torch.tensor([[[.1,.1,.3,.3],[.3,.1,.5,.3],[.7,.7,.8,.8]]],requires_grad=True)
        objetivo={'boxes':boxes.detach()[0,:2].clone(),'labels':torch.tensor([0,0]),'ignore_boxes':torch.empty(0,4)}
        loss,d=perdida_consultas({'logits':logits,'boxes':boxes},[objetivo]);loss.backward()
        self.assertEqual(d['objetos_asignados'],2)
        self.assertAlmostEqual(d['loss_l1'],0,places=6)
        self.assertAlmostEqual(d['loss_giou'],0,places=6)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertTrue(bool((logits.grad[0,:2,0]<0).all()))
        self.assertLess(float(logits.grad[0,2,6]),0)

    def test_sin_objetos_y_region_ignorada(self):
        for ignorar in [False,True]:
            logits=torch.zeros(1,2,7,requires_grad=True)
            boxes=torch.tensor([[[.1,.1,.2,.2],[.3,.3,.4,.4]]],requires_grad=True)
            o={'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.long),
                'ignore_boxes':torch.tensor([[0.,0.,1.,1.]]) if ignorar else torch.empty(0,4)}
            loss,_=perdida_consultas({'logits':logits,'boxes':boxes},[o]);loss.backward()
            self.assertTrue(bool(torch.isfinite(logits.grad).all()))
            if ignorar:self.assertEqual(float(loss.detach()),0)

    def test_inferencia_filtra_null_padding_y_conserva_instancias(self):
        g=geometria_letterbox(100,200,384)
        logits=torch.full((1,4,7),-10.);logits[0,0,0]=10;logits[0,1,0]=10;logits[0,2,6]=10;logits[0,3,0]=10
        b=torch.tensor([[[.3,.2,.5,.4],[.31,.21,.51,.41],[.5,.5,.6,.6],[0.,0.,.1,.1]]])
        pred=postprocesar_consultas({'logits':logits,'boxes':b},[g])[0]
        self.assertEqual(pred['labels'].tolist(),[0,0])

    def test_ranking_puede_conservar_material_aunque_nula_tenga_mayor_probabilidad(self):
        g=geometria_letterbox(100,100,384)
        raw=torch.full((1,1,7),-10.);raw[0,0,0]=0;raw[0,0,6]=1
        s={'logits':raw,'boxes':torch.tensor([[[.2,.2,.4,.4]]])}
        self.assertEqual(len(postprocesar_consultas(s,[g])[0]['boxes']),0)
        self.assertEqual(postprocesar_consultas(s,[g],descartar_nula=False)[0]['labels'].tolist(),[0])

    def test_modelo_gradientes_sin_convoluciones_y_guardado(self):
        m=DetectorConsultas();salida=m(torch.rand(1,3,384,384))
        self.assertEqual(tuple(salida['logits'].shape),(1,100,7))
        self.assertEqual(tuple(salida['boxes'].shape),(1,100,4))
        self.assertFalse(any(isinstance(a,torch.nn.Conv2d) for a in m.modules()))
        (salida['logits'].square().mean()+salida['boxes'].sum()).backward()
        self.assertTrue(bool(torch.isfinite(m.decodificador[0].cruzada.q.weight.grad).all()))
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'modelo.pt'
            torch.save({'tipo':'ecowaste_consultas_desde_cero_v1','version':VERSION_CONSULTAS,'clases':CLASES,'configuracion':m.configuracion_consultas,'pesos':m.state_dict()},path)
            copia,_=cargar_detector_consultas(path)
            torch.testing.assert_close(copia.referencia,m.referencia)

    def test_centro_por_atencion_transmite_gradiente_geometrico_inicial(self):
        torch.manual_seed(42)
        x=torch.rand(1,3,384,384)
        for usar in [False,True]:
            torch.manual_seed(42);m=DetectorConsultas(centros_atencion=usar)
            salida=m(x);salida['boxes'][0,0].sum().backward()
            grad=m.decodificador[-1].cruzada.q.weight.grad
            self.assertTrue(bool(torch.isfinite(grad).all()))
            if usar:self.assertGreater(float(grad.abs().sum()),0)
            else:self.assertEqual(float(grad.abs().sum()),0)

    def test_peso_de_fondo_refleja_numero_de_consultas_no_emparejadas(self):
        fondos=[];positivos=[]
        for n in [2,20]:
            b=torch.tensor([.2,.2,.4,.4]).reshape(1,1,4).repeat(1,n,1)
            o={'boxes':b[0,:1].clone(),'labels':torch.tensor([0]),'ignore_boxes':torch.empty(0,4)}
            _,d=perdida_consultas({'logits':torch.zeros(1,n,7),'boxes':b},[o],objetivo='densidad_ponderada')
            fondos.append(d['loss_fondo']);positivos.append(d['loss_clase'])
        self.assertAlmostEqual(fondos[1]/fondos[0],19,places=4)
        self.assertAlmostEqual(positivos[0],positivos[1],places=6)

    def test_ponderacion_materiales_usa_frecuencias_y_conserva_media_esperada(self):
        cuenta=torch.tensor([1630,109,149,203,409,5])
        w=pesos_materiales(cuenta)
        self.assertGreater(float(w[5]),float(w[0]))
        self.assertAlmostEqual(float((w*cuenta).sum()/cuenta.sum()),1,places=6)
        self.assertTrue(bool(torch.isfinite(w).all()))

    def test_posicion_se_aplica_a_qk_sin_fabricar_contenido_v(self):
        from ecowaste.transformer import Atencion
        a=Atencion(96,4);x=torch.zeros(1,8,96);pos=torch.randn_like(x)
        y=auto_atencion_posicional(a,x,pos)
        torch.testing.assert_close(y,y[:,:1].expand_as(y),atol=1e-6,rtol=1e-6)

    def test_posicion_por_capa_preserva_parametros_y_gradientes(self):
        m=DetectorConsultas(centros_atencion=True,posicion_por_capa=True)
        self.assertEqual(sum(p.numel() for p in m.parameters()),674129)
        s=m(torch.rand(1,3,384,384));(s['logits'].square().mean()+s['boxes'].sum()).backward()
        self.assertTrue(bool(torch.isfinite(m.consultas.grad).all()))
        self.assertGreater(float(m.consultas.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
