"""Regresión del piloto reducido: ninguna prueba declara revisiones humanas."""
import copy
import inspect
from pathlib import Path
import tempfile
import unittest
import torch
from ecowaste.experimento_v28_2 import indices_ciclicos,crear_plan,lote_plan,Diario,comprobar_estado
from ecowaste.datos_v28_2 import mascara_aux,objetivo_aux
from ecowaste.poligonos import rasterizar_contornos
from ecowaste.reanudacion_regional import publicar
from ecowaste.evaluacion_v28_2 import capturar,geometria_posthoc


class PilotoV282(unittest.TestCase):
    def setUp(self):
        self.cfg={'actualizaciones':300,'semilla_auxiliar':2801,'version':'v28_2_exploratorio_1'}
        self.ck={'rng_loader':torch.Generator().manual_seed(9).get_state(),'rng_auxiliar':torch.Generator().manual_seed(11).get_state()}
        self.p=crear_plan(self.ck,964,28,torch.arange(60)%6,self.cfg)
        self.im={'id':1,'width':384,'height':192,'eco':{'imagen_id_auxiliar':'fixture'}}
        self.a={'bbox':[30,20,60,60],'category_id':1,'eco':{'poligonos_usuario':[[30,20,90,20,90,80,30,80]],'huecos_usuario':[]}}

    def test_presupuesto_y_seis_historicas_comunes(self):
        for k in range(300):
            a,b=lote_plan(self.p,'A',k),lote_plan(self.p,'B',k)
            self.assertTrue(torch.equal(a['historico'][:6],b['historico']))
            self.assertTrue(torch.equal(a['global'],b['global']))
            self.assertEqual(len(a['historico']),8);self.assertEqual(len(b['historico']),6);self.assertEqual(len(b['nuevo']),2)
        with self.assertRaises(ValueError):lote_plan(self.p,'A',300)

    def test_plan_reproducible_sin_consumir_rng_global(self):
        estado=torch.get_rng_state().clone();otro=crear_plan(self.ck,964,28,torch.arange(60)%6,self.cfg)
        self.assertTrue(torch.equal(estado,torch.get_rng_state()))
        for k in ['historico','nuevo','global']:self.assertTrue(torch.equal(self.p[k],otro[k]))

    def test_reanudacion_cursor_conserva_entradas_y_presupuesto(self):
        for rama in ['A','B']:
            completo=[lote_plan(self.p,rama,k) for k in range(300)]
            reanudado=[lote_plan(self.p,rama,k) for k in range(100)]+[lote_plan(self.p,rama,k) for k in range(100,300)]
            for x,y in zip(completo,reanudado):
                for n in x:self.assertTrue(torch.equal(x[n],y[n]))

    def test_materiales_no_provienen_de_etiquetas_globales(self):
        y=mascara_aux(self.im,[self.a],{1:'carton'});self.assertTrue(bool((y==3).any()));self.assertTrue(bool((y==255).any()))
        vacia=mascara_aux(self.im,[],{1:'carton'});self.assertFalse(bool(((vacia>0)&(vacia!=255)).any()))

    def test_rasterizado_sin_huecos_equivale_historico(self):
        o=objetivo_aux(self.im,[self.a],{1:'carton'})
        previo=rasterizar_contornos(o,[(self.a['eco']['poligonos_usuario'],2)],48,1/64)
        self.assertTrue(torch.equal(previo,mascara_aux(self.im,[self.a],{1:'carton'})))

    def test_huecos_y_compuestos_no_fabrican_positivos(self):
        a=copy.deepcopy(self.a);a['eco']['huecos_usuario']=[[40,30,80,30,80,70,40,70]]
        y=mascara_aux(self.im,[a],{1:'organico'});self.assertEqual(int(y[18,7]),0)
        z=mascara_aux(self.im,[self.a],{1:'compuesto'});self.assertFalse(bool(((z>0)&(z!=255)).any()))

    def test_solape_materiales_ignorado(self):
        a=copy.deepcopy(self.a);a['category_id']=2
        y=mascara_aux(self.im,[self.a,a],{1:'papel',2:'plastico'});self.assertFalse(bool(((y>0)&(y!=255)).any()))

    def test_checkpoint_otro_protocolo_o_rama_rechazado(self):
        with self.assertRaises(ValueError):comprobar_estado({'v28_2':{'contrato':{},'rama':'B'}},{},'A')
        with self.assertRaises(ValueError):comprobar_estado({'v28_2':{'contrato':{},'rama':'A'}},{'nuevo':1},'A')
        with self.assertRaises(ValueError):comprobar_estado({'v28_2':{'contrato':{},'rama':'A','paso':301,'historial':[]}},{},'A')

    def test_publicacion_no_sobrescribe(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'e.json';publicar(p,{'humano':'conservar'})
            with self.assertRaises(FileExistsError):publicar(p,{'humano':'alterar'})
            self.assertIn('conservar',p.read_text())

    def test_diario_discontinuo_rechazado(self):
        with tempfile.TemporaryDirectory() as t:
            d=Diario('A',{},t);publicar(d.base/'confirmaciones/paso_0100.json',{})
            with self.assertRaises(ValueError):d.leer()

    def test_captura_no_acepta_gt(self):
        self.assertNotIn('objetivos',inspect.signature(capturar).parameters)
        with self.assertRaises(ValueError):capturar(None,None,[{'id':'x','GT':[]}],'.','.')

    def test_geometria_separa_material_equivocado(self):
        p={'boxes':torch.tensor([[.1,.1,.4,.4]]),'labels':torch.tensor([0]),'scores':torch.tensor([.9])}
        o={'id':'x','boxes':p['boxes'].clone(),'labels':torch.tensor([2])}
        g=geometria_posthoc([p],[o]);self.assertEqual(g['objetos'][0]['causa'],'material_incorrecto_iou50')


if __name__=='__main__':unittest.main()
