import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from ecowaste.detector_mapas import DetectorMapas,perdida_mapas
from ecowaste.discriminacion_mapas import perdida_discriminativa,ce_seleccionada,comparar_gradientes,descomponer_original
from ecowaste.supervision_regional import regiones_poligonales,regiones_train
from ecowaste.datasets import geometria_letterbox


def region(c=2,mask=None,ident=1):
    return {'pesos':torch.ones(1,2,4) if mask is None else mask[None],
        'labels':torch.tensor([c]),'ids':[ident]}


def mapas():
    z=torch.zeros(1,7,2,4);z[:,0]=5;z[:,1]=2;return z.requires_grad_()


class DiscriminacionTests(unittest.TestCase):
    def test_top25_empates_orden_espacial_sin_rng(self):
        z=mapas();antes=torch.get_rng_state();l,d=perdida_discriminativa(z,[region()])
        self.assertEqual(d['filas'][0]['indices_tokens'],[0,1]);self.assertEqual(d['filas'][0]['rivales'],[0,0])
        self.assertTrue(torch.equal(antes,torch.get_rng_state()));self.assertEqual(l.item(),2.5)

    def test_gradiente_pareado_dirigido_sin_fondo(self):
        z=mapas();l,d=perdida_discriminativa(z,[region()]);g=torch.autograd.grad(l,z)[0]
        self.assertEqual(float(g[:,0].abs().sum()),0)
        self.assertTrue((g[:,3,0,:2]<0).all());self.assertTrue((g[:,1,0,:2]>0).all())
        self.assertEqual(float(g[:,:,1].abs().sum()),0)
        self.assertAlmostEqual(float(g.sum()),0);self.assertLessEqual(float(g.abs().sum()),d['filas'][0]['cota_l1_grad_logits'])
        despues=z.detach()-.1*g
        # Con ocho tokens empatados, top-k puede pasar a otros aún no corregidos.
        self.assertGreater(float((despues[:,3,0,:2]-despues[:,1,0,:2]).mean()),
            float((z[:,3,0,:2]-z[:,1,0,:2]).detach().mean()))
        self.assertLessEqual(perdida_discriminativa(despues,[region()])[0],l)

    def test_no_es_ce_sobre_los_mismos_tokens(self):
        z=mapas();z.data[:,2]=1.;z.data[:,5]=.5
        l,d=perdida_discriminativa(z,[region()]);g=torch.autograd.grad(l,z,retain_graph=True)[0]
        ce7=ce_seleccionada(z,[region()],d);ce6=ce_seleccionada(z,[region()],d,True)
        g7=torch.autograd.grad(ce7,z,retain_graph=True)[0];g6=torch.autograd.grad(ce6,z)[0]
        self.assertGreater(comparar_gradientes(g,g7)['fraccion_ortogonal'],.1)
        self.assertGreater(comparar_gradientes(g,g6)['fraccion_ortogonal'],.1)
        self.assertGreater(comparar_gradientes(g,g6)['producto'],0)

    def test_margen_satisfecho_cero_ce_sigue_positiva(self):
        z=mapas();z.data[:,3]=4.
        l,d=perdida_discriminativa(z,[region()]);self.assertEqual(l.item(),0)
        self.assertEqual(d['filas'][0]['tokens_seleccionados'],0)
        # El fondo sigue ganando: el término no sustituye a la pérdida de presencia.
        self.assertGreater(F.cross_entropy(z,torch.full((1,2,4),3)),0)
        l.backward();self.assertEqual(float(z.grad.abs().sum()),0)

    def test_invarianza_al_fondo_y_traslacion_de_materiales(self):
        z=mapas();l,_=perdida_discriminativa(z,[region()])
        zz=z.detach().clone();zz[:,0]=-100;zz[:,1:]+=21
        self.assertEqual(l.item(),perdida_discriminativa(zz,[region()])[0].item())

    def test_normalizacion_instancia_y_limite_organico(self):
        z=torch.cat([mapas().detach(),mapas().detach()],0).requires_grad_();r0=region(c=5);r1=region(c=2)
        l,d=perdida_discriminativa(z,[r0,r1]);g=torch.autograd.grad(l,z)[0]
        self.assertEqual(d['filas'][0]['peso_clase'],1);self.assertEqual(d['filas'][1]['peso_clase'],2)
        self.assertAlmostEqual(d['filas'][0]['fraccion_peso_instancia'],1/3)
        self.assertLessEqual(float(g[0].abs().sum()),2/3+1e-7)
        self.assertAlmostEqual(sum(f['contribucion'] for f in d['filas']),l.item(),places=6)

    def test_instancia_facil_permanece_en_denominador(self):
        z=torch.cat([mapas().detach(),mapas().detach()],0);z[1,3]=10
        l,d=perdida_discriminativa(z,[region(),region()])
        self.assertEqual(d['denominador_instancias'],4);self.assertEqual(d['filas'][1]['tokens_seleccionados'],0)
        self.assertEqual(l.item(),1.25)

    def test_area_de_instancia_no_multiplica_peso(self):
        z=mapas();a=region(mask=torch.tensor([[1.,0,0,0],[0,0,0,0]]));b=region()
        self.assertEqual(perdida_discriminativa(z,[a])[0].item(),perdida_discriminativa(z,[b])[0].item())

    def test_mascaras_gt_pad_ignore_material_y_flip(self):
        obj={'geometria':geometria_letterbox(32,16,32),'ignore_boxes':torch.tensor([[0,.25,.25,.5]])}
        seg=[{'id':1,'clase':2,'poligonos':[[0,-8,31,-8,31,23,0,23]]}]
        sem=torch.full((4,4),3);sem[2,2]=255
        r=regiones_poligonales(obj,seg,sem,4)
        z=torch.randn(1,7,4,4,requires_grad=True);l,d=perdida_discriminativa(z,[r]);l.backward()
        self.assertEqual(float(z.grad[:,:,0].abs().sum()+z.grad[:,:,3].abs().sum()),0)
        self.assertEqual(float(z.grad[:,:,1,0].abs().sum()),0);self.assertEqual(float(z.grad[:,:,2,2].abs().sum()),0)
        x=z.detach().flip(-1);lf,df=perdida_discriminativa(x,[r],[True]);self.assertAlmostEqual(l.item(),lf.item(),places=6)
        for split,source in [('val','taco'),('train','realwaste')]:
            with self.assertRaises(ValueError):regiones_train({'split':split,'fuente':source},None)

    def test_vacio_invalido_y_fraccion_pequena(self):
        z=mapas();empty={'pesos':torch.empty(0,2,4),'labels':torch.empty(0,dtype=torch.long),'ids':[]}
        l,_=perdida_discriminativa(z,[empty]);l.backward();self.assertEqual(float(z.grad.abs().sum()),0)
        with self.assertRaises(ValueError):perdida_discriminativa(z,[region(mask=torch.zeros(2,4))])
        with self.assertRaises(ValueError):perdida_discriminativa(z,[region(c=6)])
        with self.assertRaises(ValueError):perdida_discriminativa(torch.full_like(z,float('nan')),[region()])
        frac=region(mask=torch.ones(2,4)/64);self.assertTrue(torch.isfinite(perdida_discriminativa(z,[frac])[0]))

    def test_cabeza_principal_extractor_una_pasada_sin_auxiliar(self):
        torch.manual_seed(43);m=DetectorMapas(auxiliar_clasificacion=True)
        r={'pesos':torch.ones(1,48,48),'labels':torch.tensor([2]),'ids':[1]}
        with patch.object(m,'caracteristicas',wraps=m.caracteristicas) as spy:
            raw=m(torch.rand(1,3,384,384));loss,_=perdida_discriminativa(raw,[r]);loss.backward()
            self.assertEqual(spy.call_count,1)
        for p in [m.cabezal[-1].weight,m.cabezal[0].weight,m.tokenizador_detalle.weight,m.tokenizador_contexto.weight]:
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum(),0)
        self.assertIsNone(m.clasificador_auxiliar.weight.grad);self.assertIsNone(m.clasificador_auxiliar.bias.grad)
        self.assertEqual(float(m.cabezal[-1].weight.grad[0].abs().sum()),0)

    def test_control_A_e_inferencia_exactos(self):
        torch.manual_seed(44);a=DetectorMapas(auxiliar_clasificacion=True).eval();b=copy.deepcopy(a)
        x=torch.rand(1,3,384,384);y=torch.ones(1,48,48,dtype=torch.long)
        ra=a(x);rb=b(x);self.assertTrue(torch.equal(ra,rb))
        r={'pesos':torch.ones(1,48,48),'labels':torch.tensor([2]),'ids':[1]}
        lm,_=perdida_discriminativa(rb,[r],con_gradiente=False);self.assertFalse(lm.requires_grad)
        perdida_mapas(ra,y).backward();perdida_mapas(rb,y).backward()
        for pa,pb in zip(a.parameters(),b.parameters()):
            self.assertEqual(pa.grad is None,pb.grad is None)
            if pa.grad is not None:self.assertTrue(torch.equal(pa.grad,pb.grad))
        b.load_state_dict(a.state_dict(),strict=True);self.assertEqual(sum(p.numel() for p in b.parameters()),439309)

    def test_descomposicion_original_exacta(self):
        z=mapas();y=torch.tensor([[[3,3,0,255],[4,1,0,3]]]);w=torch.tensor([1.,2.,3.,.5,1.,3.])
        partes=descomponer_original(z,y,w)
        self.assertTrue(torch.equal(partes['total'],perdida_mapas(z,y,w)))

    def test_limite_binario_puede_ser_colineal_no_promete_ortogonalidad(self):
        z=torch.full((1,7,2,4),-1000.);z[:,0]=8;z[:,1]=2;z[:,3]=0;z.requires_grad_()
        l,d=perdida_discriminativa(z,[region()]);ce=ce_seleccionada(z,[region()],d,True)
        a=torch.autograd.grad(l,z,retain_graph=True)[0];b=torch.autograd.grad(ce,z)[0]
        self.assertAlmostEqual(comparar_gradientes(a,b)['coseno'],1.,places=6)

    def test_estabilidad_bfloat16_y_logits_extremos(self):
        z=mapas().detach().to(torch.bfloat16);z[:,1]=60000;z[:,3]=-60000;z.requires_grad_()
        with torch.autocast('cpu',dtype=torch.bfloat16):l,d=perdida_discriminativa(z,[region()])
        self.assertEqual(l.dtype,torch.float32);self.assertTrue(torch.isfinite(l));l.backward()
        self.assertTrue(torch.isfinite(z.grad).all());self.assertLessEqual(float(z.grad.float().abs().sum()),2.)

    def test_preparacion_bloquea_ab_y_evaluacion_sin_cargar_datos(self):
        from scripts.preparar_v20 import leer_cfg,exigir_autorizacion,validar
        from scripts.evaluar_v20 import exigir_entrenamiento_completo
        cfg=leer_cfg()
        with patch('scripts.preparar_v20.contexto_train') as preparar:
            with self.assertRaisesRegex(ValueError,'deshabilitado'):exigir_autorizacion(cfg)
            with self.assertRaisesRegex(ValueError,'no ejecutado'):exigir_entrenamiento_completo(cfg)
            preparar.assert_not_called()
        for cambio in [{'fraccion_dificiles':.5},{'margen_logits':2},{'ramas':{'A':0,'B':.5}},
            {'pesos_clase_fijos':[1,1,2,2,1,25]}]:
            with self.assertRaises(ValueError):validar({**cfg,**cambio})


if __name__=='__main__':
    torch.set_num_threads(4);unittest.main()
