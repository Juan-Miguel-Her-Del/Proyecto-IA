import copy
import inspect
import unittest
import torch
from ecowaste.datasets import geometria_letterbox
from ecowaste.traza_propuestas import trazar_logits,nms_observada,iguales
from ecowaste.detector_mapas import cajas_desde_mapas
from ecowaste.mosaico import combinar_vistas,ventanas
from ecowaste.auditoria_propuestas_internas import analizar_imagen,resumir_objetos


class Contornos:
    def contornos(self,r,a):return a['p']


def escena(rectangulos,clase='carton'):
    cajas=[]
    for i,(x,y,w,h) in enumerate(rectangulos):
        cajas.append({'id_interno':i+1,'clase':clase,'bbox':[x,y,w,h],
            'p':[[x,y,x+w,y,x+w,y+h,x,y+h]]})
    return {'id':'sintetica','ancho':384,'alto':384,'cajas':cajas}


def raw_vacio(n=1):
    raw=torch.full((n,7,48,48),-10.);raw[:,0]=10.;return raw


def traza(raw):
    return trazar_logits(raw,[geometria_letterbox(384,384,384)],[(0,0,384,384)],384,384,'sintetica')


class PropuestasInternasTests(unittest.TestCase):
    def test_vacio_e_immutabilidad(self):
        raw=raw_vacio();previo=raw.clone();t=traza(raw)
        self.assertTrue(torch.equal(raw,previo));self.assertEqual(len(t['final']['boxes']),0)
        self.assertEqual(len(t['nodos']),0);self.assertTrue(t['regresion_exacta'])

    def test_una_celda_se_pierde_por_area(self):
        raw=raw_vacio();raw[0,3,10,10]=20;t=traza(raw)
        self.assertEqual(len(t['etapas']['componentes']),1);self.assertEqual(len(t['etapas']['area_minima']),0)
        nodo=next(iter(t['nodos'].values()));self.assertEqual(nodo['descartada_en'],'area_minima')
        filas,_=analizar_imagen(t,escena([(80,80,8,8)]),Contornos())
        self.assertEqual(filas[0]['causa_principal'],'perdida_postprocesamiento')
        self.assertEqual(filas[0]['perdidas_por_etapa'],{'area_minima':1})

    def test_integridad_procedencia_celdas_etapas(self):
        raw=raw_vacio();raw[0,3,8:16,4:8]=20;t=traza(raw)
        nodo=next(iter(t['nodos'].values()))
        self.assertEqual(nodo['material'],'carton');self.assertEqual(nodo['imagen'],'sintetica')
        self.assertEqual(len(t['vistas'][0]['celdas']),32)
        self.assertTrue(all(c['componente']==nodo['id'] for c in t['vistas'][0]['celdas']))
        for e in nodo['etapas']:self.assertIn(nodo['id'],t['etapas'][e])

    def test_padding_excluido_igual_a_produccion(self):
        raw=raw_vacio();raw[0,3,:8,:]=20
        g=geometria_letterbox(384,192,384)
        t=trazar_logits(raw,[g],[(0,0,384,192)],384,192,'pad')
        self.assertEqual(len(t['nodos']),0)
        self.assertTrue(iguales(t['final'],cajas_desde_mapas(raw,.05,2,[g])[0]))

    def test_multivista_igualdad_exacta_aleatoria(self):
        torch.manual_seed(52);zs=ventanas(641,427);gs=[geometria_letterbox(z[2]-z[0],z[3]-z[1],384) for z in zs]
        raw=torch.randn(5,7,48,48);antes=raw.clone()
        t=trazar_logits(raw,gs,zs,641,427,'aleatoria')
        base=combinar_vistas(cajas_desde_mapas(raw,.05,2,gs),gs,zs,641,427,384,.05)
        self.assertTrue(iguales(t['final'],base));self.assertTrue(torch.equal(raw,antes))

    def test_borde_interno_con_origen(self):
        raw=raw_vacio(2);raw[1,3,10:15,0:4]=20
        zs=[(0,0,600,384),(100,0,484,384)];gs=[geometria_letterbox(z[2]-z[0],z[3]-z[1],384) for z in zs]
        t=trazar_logits(raw,gs,zs,600,384,'borde')
        nodo=next(iter(t['nodos'].values()))
        self.assertEqual(nodo['vista'],1);self.assertEqual(nodo['descartada_en'],'bordes')
        self.assertIn('izquierda',nodo['motivo']);self.assertEqual(len(t['final']['boxes']),0)

    def test_nms_empates_clase_y_top100(self):
        b=torch.tensor([[0.,0.,1.,1.]]*3);s=torch.ones(3);l=torch.tensor([0,0,1])
        r,_,d=nms_observada(b,s,l)
        self.assertEqual(r,[0,2]);self.assertEqual(d[1],{'motivo':'nms','supresor':0})
        b=torch.tensor([[i*2.,0.,i*2.+1.,1.] for i in range(105)])
        r,il,d=nms_observada(b,torch.ones(105),torch.zeros(105,dtype=torch.long))
        self.assertEqual(r,list(range(100)));self.assertEqual(il,list(range(105)))
        self.assertEqual(d[104]['motivo'],'top100_sin_examinar')

    def test_nms_despues_tope_no_se_atribuye_supresion_real(self):
        b=torch.tensor([[i*2.,0.,i*2.+1.,1.] for i in range(101)]+[[200.,0.,201.,1.]])
        _,_,d=nms_observada(b,torch.ones(102),torch.zeros(102,dtype=torch.long))
        self.assertEqual(d[101],{'motivo':'top100_sin_examinar','supresor':None})

    def test_fusion_sustentada_por_pixeles_gt(self):
        raw=raw_vacio();raw[0,3,8:16,4:16]=20
        filas,_=analizar_imagen(traza(raw),escena([(32,64,32,64),(96,64,32,64)]),Contornos())
        self.assertTrue(all(f['causa_principal']=='fusion_objetos' for f in filas))
        self.assertTrue(all(f['fusiones'][0]['gt_internos']==[1,2] for f in filas))

    def test_fragmentacion_con_union_oracle(self):
        raw=raw_vacio();raw[0,3,8:16,4:8]=20;raw[0,3,8:16,12:16]=20
        filas,_=analizar_imagen(traza(raw),escena([(32,64,96,64)]),Contornos())
        f=filas[0];self.assertEqual(f['causa_principal'],'fragmentacion')
        self.assertGreaterEqual(f['fragmentos'][0]['iou_envolvente_oracle'],.5)
        self.assertLess(f['fragmentos'][0]['max_iou_individual'],.5)

    def test_ausencia_y_material_incorrecto(self):
        r=escena([(32,64,32,64)]);raw=raw_vacio()
        filas,_=analizar_imagen(traza(raw),r,Contornos());self.assertEqual(filas[0]['causa_principal'],'ausencia_respuesta_fg')
        raw[0,1,8:16,4:8]=20
        filas,_=analizar_imagen(traza(raw),r,Contornos());self.assertEqual(filas[0]['causa_principal'],'material_incorrecto')

    def test_gt_solo_posthoc_no_modifica_salida(self):
        self.assertNotIn('gt',inspect.signature(trazar_logits).parameters)
        raw=raw_vacio();raw[0,3,8:16,4:8]=20;t=traza(raw);antes=copy.deepcopy(t['final'])
        a,_=analizar_imagen(t,escena([(32,64,32,64)]),Contornos())
        b,_=analizar_imagen(t,escena([(240,240,32,64)]),Contornos())
        self.assertEqual(a[0]['causa_principal'],'correcto');self.assertNotEqual(b[0]['causa_principal'],'correcto')
        self.assertTrue(iguales(antes,t['final']));self.assertEqual(resumir_objetos(a)['total']['gt'],1)

    def test_forma_o_umbrales_ajenos_rechazados(self):
        with self.assertRaises(ValueError):trazar_logits(torch.zeros(1,6,48,48),[],[],384,384,'x')
        with self.assertRaises(ValueError):trazar_logits(raw_vacio(),[{}],[[]],384,384,'x',umbral=.1)


if __name__=='__main__':
    torch.set_num_threads(4);unittest.main()
