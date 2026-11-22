"""Regresión del resumen GT bajo determinismo estricto, sin entrenar ningún modelo."""
import copy
import unittest
from unittest.mock import patch
import torch
from scripts.evaluar_v20 import agregar_regiones_cpu,validar_compatibilidad_codigo


class AgregacionEvaluacionCPU(unittest.TestCase):
    def test_logits_cuda_agregan_en_cpu_bajo_determinismo(self):
        self.assertTrue(torch.cuda.is_available());antes=torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            z=torch.zeros(1,7,2,2,device='cuda',dtype=torch.float16);z[:,3]=5;z[0,0,1,0]=20
            r={'pesos':torch.tensor([[[1.,.5],[0.,1.]]]),'labels':torch.tensor([2]),'ids':[1]}
            real=torch.bincount;vistos=[]
            def contar(x,**kw):
                vistos.append(x.device.type);self.assertEqual(x.device.type,'cpu')
                if 'weights' in kw:self.assertEqual(kw['weights'].device.type,'cpu')
                return real(x,**kw)
            with patch('torch.bincount',side_effect=contar):a=agregar_regiones_cpu(z,[r]);b=agregar_regiones_cpu(z,[r])
            self.assertEqual(a,b);self.assertEqual(vistos,['cpu']*4)
            self.assertEqual(a['filas'][0]['fraccion_correcta'],1.);self.assertEqual(a['filas'][0]['fraccion_fondo'],0.)
            self.assertTrue(a['filas'][0]['condicional_correcta_mayoria'])
        finally:torch.use_deterministic_algorithms(antes)

    def test_regiones_vacias_no_inventan_instancias(self):
        r={'pesos':torch.empty(0,2,2),'labels':torch.empty(0,dtype=torch.long),'ids':[]}
        a=agregar_regiones_cpu(torch.zeros(1,7,2,2),[r]);self.assertEqual(a['instancias'],0)
        self.assertTrue(all(x['instancias']==0 for x in a['por_material'].values()))

    def test_compatibilidad_no_admite_cambios_de_perdida_o_entrenador(self):
        a={'scripts/evaluar_v20.py':'1','ecowaste/discriminacion_mapas.py':'2'};b={**a,'scripts/evaluar_v20.py':'3'}
        c={'aprobada':True,'hashes_entrenamiento':a,'hashes_evaluacion':b,'protocolo':{'id':24}}
        validar_compatibilidad_codigo(a,b,c,{'id':24})
        mal={**b,'ecowaste/discriminacion_mapas.py':'4'};cc=copy.deepcopy(c);cc['hashes_evaluacion']=mal
        with self.assertRaises(ValueError):validar_compatibilidad_codigo(a,mal,cc,{'id':24})


if __name__=='__main__':unittest.main()
