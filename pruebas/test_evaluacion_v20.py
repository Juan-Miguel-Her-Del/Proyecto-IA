"""Configuración CLI, coherencia numérica y separación de la referencia histórica."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from ecowaste.backend_numerico import politica,aplicar_backend,registro_efectivo,VERSION
from scripts.preparar_v20 import leer_cfg,protocolo,huellas_codigo
from scripts.evaluar_v20 import (argumentos,contexto_evaluacion,comprobar_checkpoint_backend,
    comparar_predicciones_historicas)


class EvaluacionV20(unittest.TestCase):
    def cfg(self):
        c=leer_cfg();c.update(backend_numerico=politica('minimo'),version_protocolo=VERSION);return c

    def test_cli_acepta_configuracion_y_etapas_exclusivas(self):
        a=argumentos(['--config','configuracion/v20_backend_v23_1_completo_ep19.json','--verificar-etapa-20'])
        self.assertEqual(a.config,Path('configuracion/v20_backend_v23_1_completo_ep19.json'));self.assertTrue(a.verificar_etapa_20)
        with patch('sys.stderr'):
            with self.assertRaises(SystemExit):argumentos(['--referencia-historica','--verificar-etapa-20'])

    def test_contexto_numero_inferencia_aplica_y_restaura(self):
        antes=registro_efectivo()
        with contexto_evaluacion(self.cfg()):
            r=registro_efectivo();self.assertTrue(r['cudnn_deterministic']);self.assertEqual(r['sdpa'],{'flash':False,'efficient':False,'math':True,'cudnn':False})
        self.assertEqual(antes,registro_efectivo())

    def test_compatibilidad_checkpoint_inferencia_sin_optimizador_activo(self):
        cfg=self.cfg()
        with contexto_evaluacion(cfg):
            ef=registro_efectivo();guardado={**ef,'adam':[{'foreach':None,'fused':None}]}
            ck={'protocolo_v20':protocolo(cfg),'hashes_codigo_v20':huellas_codigo(),
                'sha256_origen':cfg['sha256_checkpoint_origen'],'backend_numerico':cfg['backend_numerico'],
                'backend_efectivo':guardado,'optimizador':{'param_groups':[{'foreach':None,'fused':None}]}}
            comprobar_checkpoint_backend(ck,cfg,ef)
            for mutar in [lambda c:c.update(backend_numerico=politica('completo')),
                lambda c:c['backend_efectivo'].update(torch='otra'),
                lambda c:c['optimizador']['param_groups'][0].update(foreach=True),
                lambda c:c.update(hashes_codigo_v20={})]:
                malo=copy.deepcopy(ck);mutar(malo)
                with self.assertRaises(ValueError):comprobar_checkpoint_backend(malo,cfg,ef)

    def test_historico_exige_identidad_pero_recalculo_math_registra_diferencia(self):
        a=[{'id':'train:1','boxes':[[0.,0.,1.,1.]],'scores':[.9],'labels':[2]}]
        b=copy.deepcopy(a);b[0]['scores'][0]=.90001
        with self.assertRaises(AssertionError):comparar_predicciones_historicas(a,b,True)
        r=comparar_predicciones_historicas(a,b,False)
        self.assertEqual(r['imagenes_exactas'],0);self.assertEqual(r['ambito'],'comparacion_entre_backends');self.assertFalse(r['identidad_exigida'])

    def test_no_acepta_referencias_de_otras_imagenes(self):
        with self.assertRaises(ValueError):comparar_predicciones_historicas([{'id':'a'}],[{'id':'b'}],False)

    def test_origen_sin_backend_es_admisible_solo_como_referencia_ep19(self):
        comprobar_checkpoint_backend({'epoca':19},self.cfg(),{},origen=True)
        with self.assertRaises(ValueError):comprobar_checkpoint_backend({'epoca':20},self.cfg(),{},origen=True)


if __name__=='__main__':unittest.main()
