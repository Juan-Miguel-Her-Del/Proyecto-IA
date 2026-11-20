import copy
import unittest
import torch
from ecowaste.ablacion_fondo import huella,estado_aleatorio,restaurar_aleatorio,metricas_semanticas,pasos_adam
from ecowaste.detector_mapas import perdida_mapas


class TestAblacionFondo(unittest.TestCase):
    def test_solo_cambia_el_termino_de_fondo(self):
        torch.manual_seed(4)
        x=torch.randn(2,7,4,4,requires_grad=True);y=torch.zeros(2,4,4,dtype=torch.long);y[:,1:3,1:3]=2;y[:,0,0]=255
        a=perdida_mapas(x,y,coeficiente_fondo=3)
        b=perdida_mapas(x,y,coeficiente_fondo=1)
        ce=torch.nn.functional.cross_entropy(x,y,ignore_index=255,reduction='none')
        torch.testing.assert_close(a-b,2*ce[y==0].mean())
        torch.testing.assert_close(a,perdida_mapas(x,y),atol=0,rtol=0)

    def test_sin_fondo_o_todo_ignorado(self):
        x=torch.randn(1,7,3,3,requires_grad=True)
        for y in [torch.ones(1,3,3,dtype=torch.long),torch.full((1,3,3),255,dtype=torch.long)]:
            torch.testing.assert_close(perdida_mapas(x,y,coeficiente_fondo=3),perdida_mapas(x,y,coeficiente_fondo=1))
        with self.assertRaises(ValueError):perdida_mapas(x,y,coeficiente_fondo=-1)

    def test_huella_y_restauracion_completa_adam_rng(self):
        m=torch.nn.Linear(3,2);opt=torch.optim.AdamW(m.parameters(),lr=.0005)
        torch.nn.functional.cross_entropy(m(torch.ones(2,3)),torch.tensor([0,1])).backward();opt.step()
        ck=copy.deepcopy({'pesos':m.state_dict(),'adam':opt.state_dict()})
        self.assertEqual(huella(ck),huella(copy.deepcopy(ck)))
        ck['adam']['param_groups'][0]['lr']=.01
        self.assertNotEqual(huella(ck),huella({'pesos':m.state_dict(),'adam':opt.state_dict()}))
        self.assertEqual(pasos_adam(opt),1)
        rng=torch.Generator().manual_seed(6);aux=torch.Generator().manual_seed(7)
        estado=estado_aleatorio(rng,aux)
        antes=[torch.rand(5),torch.rand(5,generator=rng),torch.rand(5,generator=aux)]
        restaurar_aleatorio(estado,rng,aux)
        despues=[torch.rand(5),torch.rand(5,generator=rng),torch.rand(5,generator=aux)]
        for a,b in zip(antes,despues):torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_iou_y_confusiones_incluyen_fondo(self):
        cm=torch.zeros(7,7,dtype=torch.long);cm[0,0]=8;cm[0,1]=2;cm[1,0]=3;cm[1,1]=7
        r=metricas_semanticas(cm)
        self.assertEqual(r['confusion_binaria_fondo_material'],[[8,2],[3,7]])
        self.assertAlmostEqual(r['por_clase']['plastico']['iou'],7/12)
        self.assertEqual(r['material_predicho_fondo'],3)
        self.assertIsNone(r['por_clase']['organico']['iou'])


if __name__=='__main__':unittest.main()
