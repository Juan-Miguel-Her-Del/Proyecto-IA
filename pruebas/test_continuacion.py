from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ecowaste.datasets import CLASES
from ecowaste.continuacion import validar_continuacion


class ContinuacionTests(unittest.TestCase):
    def caso(self):
        cfg={'nombre':'original','epocas':16,'min_epocas':8,'paciencia':6,'learning_rate':.0005,
            'clasificacion_auxiliar':{'coeficiente_perdida':.5,'batch_size':16}}
        previo={'tipo':'ecowaste_mapas_desde_cero_v1','pesos_externos':False,'clases':CLASES,
            'hashes_manifiestos':{'train':'abc','val':'def','clasificacion_auxiliar_train':'ghi'},
            'config_entrenamiento':cfg,'epoca':16}
        nuevo=dict(cfg,nombre='continuacion',epocas=48,min_epocas=32,paciencia=10)
        return previo,nuevo,deepcopy(previo['hashes_manifiestos'])

    def test_acepta_solo_ampliar_presupuesto_y_parada(self):
        previo,cfg,hashes=self.caso();validar_continuacion(previo,cfg,hashes)

    def test_rechaza_cambiar_optimizacion_o_peso_auxiliar(self):
        previo,cfg,hashes=self.caso();cfg['learning_rate']=.001
        with self.assertRaises(ValueError):validar_continuacion(previo,cfg,hashes)
        previo,cfg,hashes=self.caso();cfg=deepcopy(cfg);cfg['clasificacion_auxiliar']['coeficiente_perdida']=1
        with self.assertRaises(ValueError):validar_continuacion(previo,cfg,hashes)

    def test_rechaza_cambios_en_cualquiera_de_las_fuentes(self):
        for key in ['train','val','clasificacion_auxiliar_train']:
            previo,cfg,hashes=self.caso();hashes[key]='otro'
            with self.assertRaises(ValueError):validar_continuacion(previo,cfg,hashes)

    def test_rechaza_reutilizar_nombre_no_ampliar_o_pesos_ajenos(self):
        previo,cfg,hashes=self.caso();cfg['nombre']='original'
        with self.assertRaises(ValueError):validar_continuacion(previo,cfg,hashes)
        previo,cfg,hashes=self.caso();cfg['epocas']=16
        with self.assertRaises(ValueError):validar_continuacion(previo,cfg,hashes)
        previo,cfg,hashes=self.caso();previo['pesos_externos']=True
        with self.assertRaises(ValueError):validar_continuacion(previo,cfg,hashes)


if __name__=='__main__':unittest.main()
