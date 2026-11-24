import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import torch
from ecowaste.detector_instancias_v31 import DetectorInstanciasV31,pareja,cargar,TIPO
from ecowaste.detector_consultas import perdida_consultas,postprocesar_consultas
from ecowaste.detector_hibrido import mascara_letterbox
from ecowaste.datasets import CLASES,geometria_letterbox
from ecowaste.experimento_v31 import flip_objetivos,planificar,comprobar


def objetivo():
    return {'boxes':torch.tensor([[.10,.20,.11,.21],[.12,.20,.14,.23]]),'labels':torch.tensor([3,3]),
            'ignore_boxes':torch.tensor([[.7,.7,.9,.9]]),'geometria':geometria_letterbox(384,384,384),'id':'fixture'}


class InstanciasV31(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_formas_dos_extractores(self):
        with torch.no_grad():
            for cnn in [False,True]:
                s=DetectorInstanciasV31(cnn=cnn,consultas=16,capas=1)(torch.rand(1,3,384,384))
                self.assertEqual(s['logits'].shape,(1,16,7));self.assertEqual(s['boxes'].shape,(1,16,4))
                self.assertTrue(torch.isfinite(s['boxes']).all())

    def test_gradientes_cnn_atencion_y_localizacion(self):
        m=DetectorInstanciasV31(consultas=16,capas=1)
        s=m(torch.rand(1,3,384,384));l,d=perdida_consultas(s,[objetivo()],'densidad_ponderada',torch.ones(6));l.backward()
        self.assertEqual(d['objetos_asignados'],2)
        for modulo in [m.backbone,m.detalle,m.contexto,m.decodificador,m.material,m.caja]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in modulo.parameters()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in modulo.parameters()))

    def test_gt_subtoken_no_desaparece(self):
        s={'logits':torch.randn(1,8,7,requires_grad=True),'boxes':torch.tensor([[[.1,.2,.11,.21]]*8],requires_grad=True)}
        l,d=perdida_consultas(s,[objetivo()],'densidad_ponderada',torch.ones(6));l.backward()
        self.assertEqual(d['objetos_asignados'],2);self.assertTrue(torch.isfinite(l))

    def test_doble_flip_cajas_y_geometria(self):
        o=objetivo();o['geometria']=geometria_letterbox(233,501,384)
        a=flip_objetivos([o],[True])[0];b=flip_objetivos([a],[True])[0]
        self.assertTrue(torch.allclose(o['boxes'],b['boxes'],atol=1e-7));self.assertEqual(o['geometria'],b['geometria'])
        self.assertTrue(torch.equal(o['ignore_boxes'],b['ignore_boxes']))

    def test_flip_no_modifica_original(self):
        o=objetivo();antes=copy.deepcopy(o);flip_objetivos([o],[True])
        self.assertTrue(torch.equal(o['boxes'],antes['boxes']));self.assertEqual(o['geometria'],antes['geometria'])

    def test_proximos_mismo_material_sin_fusion(self):
        z=torch.full((1,2,7),-20.);z[:,:,3]=20
        p=postprocesar_consultas({'logits':z,'boxes':objetivo()['boxes'][None]},[objetivo()['geometria']])[0]
        self.assertEqual(len(p['boxes']),2)

    def test_sin_objeto_descartado(self):
        z=torch.zeros(1,4,7);z[:,:,6]=10
        p=postprocesar_consultas({'logits':z,'boxes':torch.rand(1,4,4)},[objetivo()['geometria']])[0]
        self.assertEqual(len(p['boxes']),0)

    def test_lote_vacio_finito(self):
        o=objetivo();o['boxes']=torch.empty(0,4);o['labels']=torch.empty(0,dtype=torch.long)
        s=DetectorInstanciasV31(consultas=16,capas=1)(torch.zeros(1,3,384,384))
        l,d=perdida_consultas(s,[o],'densidad_ponderada',torch.ones(6));self.assertTrue(torch.isfinite(l));self.assertEqual(d['objetos_asignados'],0)

    def test_padding_claves_no_influyen(self):
        m=DetectorInstanciasV31(consultas=16,capas=1).eval();x=torch.rand(1,3,384,384)
        v=mascara_letterbox([geometria_letterbox(800,400,384)]);z=x.masked_fill(~v[:,None],99.)
        with torch.no_grad():
            a,b=m(x,v),m(z,v)
        self.assertTrue(torch.equal(a['logits'],b['logits']));self.assertTrue(torch.equal(a['boxes'],b['boxes']))

    def test_inicializacion_comun_y_aleatoria(self):
        a,b=pareja(31)
        for n,t in a.state_dict().items():
            if n in b.state_dict() and b.state_dict()[n].shape==t.shape:self.assertTrue(torch.equal(t,b.state_dict()[n]),n)
        self.assertFalse(any(isinstance(x,torch.nn.Conv2d) for x in a.modules()))
        self.assertTrue(any(isinstance(x,torch.nn.Conv2d) for x in b.modules()))

    def test_guardado_carga_exactos(self):
        m=DetectorInstanciasV31(consultas=16,capas=1).eval();x=torch.rand(1,3,384,384)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'m.pt';torch.save({'tipo':TIPO,'clases':CLASES,'pesos_externos':False,'arquitectura':m.configuracion_v31,'pesos':m.state_dict()},p)
            r,_=cargar(p)
            with torch.no_grad():self.assertTrue(torch.equal(m(x)['boxes'],r(x)['boxes']));self.assertTrue(torch.equal(m(x)['logits'],r(x)['logits']))

    def test_rechazo_preentrenado(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'m.pt';torch.save({'tipo':TIPO,'clases':CLASES,'pesos_externos':True},p)
            with self.assertRaises(ValueError):cargar(p)

    def test_plan_cap_y_fuentes_sin_consumir_rng(self):
        os=[objetivo() for _ in range(6)]
        for i,o in enumerate(os):o['labels']=torch.tensor([i,i])
        ht=SimpleNamespace(objetivos=os);nu=SimpleNamespace(registros=[{'fuente':f} for f in ['packwise']*3+['click']*4],objetivos=os)
        gl=SimpleNamespace(etiquetas=torch.arange(6));antes=torch.get_rng_state().clone()
        p=planificar({'semilla':31,'actualizaciones':10},ht,nu,gl)
        self.assertTrue(torch.equal(antes,torch.get_rng_state()));self.assertLessEqual(float(p['pesos_muestreo'].max()),3)
        self.assertEqual(p['historico'].shape,(10,8));self.assertTrue((p['nuevo'][:,:2]<3).all());self.assertTrue((p['nuevo'][:,2:]>=3).all())

    def test_no_truncar_objetos_por_consultas(self):
        o=objetivo();s={'logits':torch.randn(1,1,7),'boxes':torch.rand(1,1,4)}
        with self.assertRaises(ValueError):perdida_consultas(s,[o])

    def test_reanudacion_rama_y_presupuesto(self):
        c={'cfg':{'actualizaciones':2}};ck={'tipo':TIPO,'contrato':c,'rama':'B','arquitectura':{'cnn':True},'paso':3,'historial':[]}
        with self.assertRaises(ValueError):comprobar(ck,c,'A')
        with self.assertRaises(ValueError):comprobar(ck,c,'B')
