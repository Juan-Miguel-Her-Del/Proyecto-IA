from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ecowaste.auditoria_instancias import analizar_componentes


class AuditoriaInstanciasTests(unittest.TestCase):
    def test_dos_objetos_del_mismo_material_se_funden(self):
        a=np.zeros((8,8),bool);a[2:4,1:3]=True
        b=np.zeros((8,8),bool);b[2:4,3:5]=True
        y=(a|b).astype(np.uint8)
        d=analizar_componentes(y,[(10,0,a),(11,0,b)])
        self.assertEqual(d['instancias_en_componentes_compartidos'],[10,11])
        self.assertEqual(len(d['fusiones']),1)

    def test_componentes_separados_no_se_funden_y_un_objeto_puede_fragmentarse(self):
        a=np.zeros((8,8),bool);a[2:4,1:3]=True
        b=np.zeros((8,8),bool);b[2:4,5:7]=True
        y=(a|b).astype(np.uint8)
        d=analizar_componentes(y,[(10,0,a),(11,0,b)])
        self.assertEqual(d['instancias_en_componentes_compartidos'],[])
        d=analizar_componentes(y,[(10,0,a|b)])
        self.assertEqual(d['instancias_fragmentadas'],[10])

    def test_objeto_de_una_celda_no_supera_minimo(self):
        a=np.zeros((8,8),bool);a[0,0]=True
        d=analizar_componentes(a.astype(np.uint8),[(10,0,a)])
        self.assertEqual(d['instancias_sin_componente_minimo'],[10])


if __name__=='__main__':unittest.main()
