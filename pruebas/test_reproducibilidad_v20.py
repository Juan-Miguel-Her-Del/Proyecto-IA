"""El observador numérico conserva tolerancia y captura todos los tensores."""
import copy
import json
from pathlib import Path
import unittest
import torch
from ecowaste.reproducibilidad_v20 import comparar_arbol,comparar_envolvente,cpu,adam_nominal,RTOL,ATOL
from scripts.preparar_v20 import leer_cfg,huellas_codigo


class ReproducibilidadV20(unittest.TestCase):
    def test_no_detiene_comparacion_en_primer_tensor(self):
        a={'primero':torch.tensor([1.,2.]),'ultimo':{'exp_avg':torch.tensor([3.])}}
        b={'primero':torch.tensor([1.01,2.]),'ultimo':{'exp_avg':torch.tensor([3.01])}}
        r=comparar_arbol(a,b);self.assertEqual(r['tensores'],2);self.assertEqual(r['elementos_fuera_tolerancia'],2)
        self.assertEqual({f['nombre'] for f in r['filas']},{'primero','ultimo.exp_avg'})

    def test_isclose_igual_a_piloto_original_y_tipos(self):
        a=torch.tensor([0.,1.,-2.,1.00012],dtype=torch.float64);b=torch.tensor([1e-6,1.,-2.01,1.],dtype=torch.float64)
        r=comparar_arbol(a,b)
        self.assertEqual((RTOL,ATOL),(1e-4,1e-5))
        self.assertEqual(r['elementos_fuera_tolerancia'],int((~torch.isclose(a,b,rtol=RTOL,atol=ATOL)).sum()))
        self.assertEqual(r['filas'][0]['dtype_a'],'torch.float64')

    def test_ceros_no_finitos_y_estructura_se_registran(self):
        r=comparar_arbol({'x':torch.tensor([1.,float('nan')]),'entero':torch.tensor([1])},
            {'x':torch.tensor([0.,0.]),'entero':torch.tensor([2]),'otra':None})
        self.assertFalse(r['dentro_tolerancia']);json.dumps(r,allow_nan=False)
        self.assertTrue(any(f.get('relativa_infinita') for f in r['filas']))
        self.assertTrue(any(f['tipo']=='clave_ausente' for f in r['filas']))

    def test_envolvente_no_amplia_tolerancia(self):
        c=comparar_arbol(torch.tensor([1.]),torch.tensor([1.001]))
        r=comparar_arbol(torch.tensor([1.]),torch.tensor([1.002]))
        self.assertFalse(comparar_envolvente(c,r)['cumple']);self.assertTrue(comparar_envolvente(c,c)['cumple'])

    def test_snapshot_es_independiente_y_adam_tiene_nombres(self):
        m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters());m(torch.ones(1,2)).sum().backward();o.step()
        s=cpu(m.state_dict());antes=s['weight'].clone()
        with torch.no_grad():m.weight.add_(1)
        self.assertTrue(torch.equal(s['weight'],antes));self.assertNotIn(0,adam_nominal(m,o)['estado'])
        self.assertIn('weight',adam_nominal(m,o)['estado']);self.assertIn('exp_avg',adam_nominal(m,o)['estado']['weight'])

    def test_solo_diagnostico_y_huellas(self):
        self.assertFalse(leer_cfg()['entrenamiento_completo_habilitado'])
        for p in ['ecowaste/reproducibilidad_v20.py','scripts/probar_entrenamiento_v20.py',
            'scripts/ejecutar_diagnostico_reproducibilidad_v20.py','pruebas/test_reproducibilidad_v20.py']:
            self.assertIn(p,huellas_codigo())


if __name__=='__main__':unittest.main()
