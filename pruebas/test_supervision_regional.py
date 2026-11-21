import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ecowaste.datasets import geometria_letterbox
from ecowaste.detector_mapas import DetectorMapas,perdida_mapas
from ecowaste.supervision_regional import (regiones_poligonales,regiones_train,
    pool_regiones,perdida_regional,pesos_clase_regionales,resumen_confusion)


def objetivo(ancho=32,alto=32):
    return {'geometria':geometria_letterbox(ancho,alto,32),'ignore_boxes':torch.empty(0,4)}


def segmento(i=1,c=2,p=None):
    return {'id':i,'clase':c,'poligonos':[p or [0,0,31,0,31,31,0,31]]}


class RegionalTests(unittest.TestCase):
    def test_correspondencia_poligono_no_bbox_y_material(self):
        sem=torch.full((4,4),3)
        r=regiones_poligonales(objetivo(),[segmento(p=[0,0,31,0,0,31])],sem,4)
        self.assertEqual(r['ids'],[1]);self.assertEqual(r['labels'].tolist(),[2])
        self.assertEqual(r['pesos'][0,-1,-1],0)
        self.assertEqual(r['pesos'][0,0,0],1)
        with self.assertRaises(ValueError):regiones_poligonales(objetivo(),[segmento(),segmento()],sem,4)

    def test_excluye_padding_ignoradas_y_255(self):
        o=objetivo(32,16);o['ignore_boxes']=torch.tensor([[0,.25,.25,.5]])
        sem=torch.full((4,4),3);sem[2,2]=255
        r=regiones_poligonales(o,[segmento(p=[0,-16,31,-16,31,31,0,31])],sem,4)
        mask=r['pesos'][0]
        self.assertEqual(mask[0].sum()+mask[3].sum(),0)
        self.assertEqual(mask[1,0],0);self.assertEqual(mask[2,2],0)
        self.assertGreater(mask.sum(),0)

    def test_mezcla_materiales_excluida_incluso_si_semantica_positiva(self):
        r=regiones_poligonales(objetivo(),[segmento(),segmento(2,0)],torch.full((4,4),3),4)
        self.assertEqual(r['pesos'].shape,(0,4,4))
        self.assertTrue(all(d['modo']=='vacia' for d in r['detalles']))

    def test_pequena_fraccion_y_vacia_sin_caja_sustituta(self):
        r=regiones_poligonales(objetivo(),[segmento(p=[1,1,2,1,2,2])],torch.full((4,4),3),4)
        self.assertEqual(r['detalles'][0]['modo'],'fraccion_poligono')
        self.assertGreater(r['pesos'].sum(),0);self.assertLess(r['pesos'].sum(),1)
        vacia=regiones_poligonales(objetivo(),[segmento(p=[90,90,91,90,91,91])],torch.full((4,4),3),4)
        self.assertEqual(len(vacia['labels']),0)
        self.assertEqual(vacia['detalles'][0]['motivo_omision'],'sin_pixeles_rasterizados')

    def test_interiores_preferidos_sobre_bordes(self):
        r=regiones_poligonales(objetivo(),[segmento(p=[0,0,20,0,20,20,0,20])],torch.full((4,4),3),4)
        self.assertEqual(r['detalles'][0]['modo'],'interior')
        self.assertEqual(r['pesos'][0,2,2],0);self.assertEqual(r['pesos'].sum(),4)

    def test_solo_train_y_taco(self):
        for r in [{'split':'val','fuente':'taco'},{'split':'train','fuente':'realwaste'}]:
            with self.assertRaises(ValueError):regiones_train(r,None)

    def test_pool_espejo_ids_y_promedio(self):
        f=torch.arange(16.).reshape(1,4,4,1)
        r=regiones_poligonales(objetivo(),[segmento(9,p=[0,0,7,0,7,7,0,7])],torch.full((4,4),3),4)
        self.assertEqual(pool_regiones(f,[r])['vectores'].item(),0)
        p=pool_regiones(f,[r],[True]);self.assertEqual(p['vectores'].item(),3)
        self.assertEqual(p['ids'],[9]);self.assertEqual(p['indices_imagen'],[0])

    def test_ce_ponderada_contribuciones_y_vacio(self):
        m=DetectorMapas(auxiliar_clasificacion=True)
        f=torch.randn(2,4,4,96,requires_grad=True)
        r=[regiones_poligonales(objetivo(),[segmento(c=c)],torch.full((4,4),c+1),4) for c in [0,2]]
        w=pesos_clase_regionales([100,10,5,20,30,1])
        loss,d,p=perdida_regional(m,f,r,w)
        self.assertTrue(torch.equal(loss,F.cross_entropy(p['logits'],p['labels'],weight=w)))
        self.assertAlmostEqual(sum(x['contribucion_ce'] for x in d['por_material'].values()),loss.item(),places=6)
        vacia={'pesos':torch.empty(0,4,4),'labels':torch.empty(0,dtype=torch.long),'ids':[]}
        l,_,_=perdida_regional(m,f,[vacia,vacia],w);self.assertEqual(l.item(),0);l.backward()
        self.assertTrue(torch.isfinite(f.grad).all())
        with self.assertRaises(ValueError):pesos_clase_regionales([0]*6)

    def test_gradiente_extractor_un_forward_y_cabeza_existente(self):
        torch.manual_seed(12);m=DetectorMapas(auxiliar_clasificacion=True)
        r={'pesos':torch.ones(1,48,48),'labels':torch.tensor([2]),'ids':[1]}
        with patch.object(m,'caracteristicas',wraps=m.caracteristicas) as spy:
            raw,features=m(torch.rand(1,3,384,384),devolver_caracteristicas=True)
            loss,_,_=perdida_regional(m,features,[r],torch.ones(6));loss.backward()
            self.assertEqual(spy.call_count,1)
        self.assertEqual(sum(p.numel() for p in m.parameters()),439309)
        for p in [m.tokenizador_detalle.weight,m.tokenizador_contexto.weight,m.clasificador_auxiliar.weight]:
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum(),0)
        self.assertIsNone(m.cabezal[-1].weight.grad)

    def test_inferencia_y_control_cero_exactos(self):
        torch.manual_seed(42);a=DetectorMapas(auxiliar_clasificacion=True);b=copy.deepcopy(a)
        x=torch.rand(1,3,384,384);y=torch.ones(1,48,48,dtype=torch.long)
        a.eval();b.eval();raw=a(x);nuevo,f=b(x,True)
        self.assertTrue(torch.equal(raw,nuevo))
        r={'pesos':torch.ones(1,48,48),'labels':torch.tensor([0]),'ids':[1]}
        l,_,_=perdida_regional(b,f,[r],torch.ones(6),con_gradiente=False)
        self.assertFalse(l.requires_grad)
        perdida_mapas(raw,y).backward();perdida_mapas(nuevo,y).backward()
        for pa,pb in zip(a.parameters(),b.parameters()):
            self.assertEqual(pa.grad is None,pb.grad is None)
            if pa.grad is not None:self.assertTrue(torch.equal(pa.grad,pb.grad))
        b.load_state_dict(a.state_dict(),strict=True)

    def test_confusion_carton(self):
        cm=torch.zeros(6,6,dtype=torch.long);cm[2,0]=11;cm[2,4]=3;cm[2,2]=6
        self.assertEqual(resumen_confusion(cm)['carton']['tasa_a_plastico_metal'],.7)

    def test_bloqueo_antes_de_cargar_o_entrenar(self):
        import json
        from scripts.supervision_regional_materiales import CONFIG,entrenar,validar_protocolo
        cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
        cfg['entrenamiento_completo_habilitado']=False
        with patch('scripts.supervision_regional_materiales.contexto') as preparar:
            with self.assertRaisesRegex(ValueError,'deshabilitado'):entrenar(cfg)
            preparar.assert_not_called()
        for cambio in [{'ramas':{'A':0,'B':.5}},{'evaluar_test':True},{'actualizaciones_por_rama':1}]:
            with self.assertRaises(ValueError):validar_protocolo({**cfg,**cambio})

    def test_sin_cabeza_compatible_rechaza_sin_anadir_parametros(self):
        m=DetectorMapas();antes=tuple(m.state_dict())
        with self.assertRaises(ValueError):perdida_regional(m,torch.zeros(1,48,48,96),[],torch.ones(6))
        self.assertEqual(antes,tuple(m.state_dict()))


if __name__=='__main__':
    torch.set_num_threads(4);unittest.main()
