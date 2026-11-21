"""Contrato numérico, aislamiento y evidencia del control V23."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ecowaste.backend_numerico import (politica,validar_politica,aplicar_backend,registro_efectivo,
    verificar_registro,configurar_optimizador,VERSION)
from ecowaste.reanudacion_v20 import validar_estado_v20
from ecowaste.observacion_backend import ObservadorBackend
from scripts.preparar_v20 import leer_cfg,validar,protocolo


class BackendNumerico(unittest.TestCase):
    def test_minimo_conserva_amp_adam_y_clipping(self):
        p=politica('minimo');self.assertTrue(p['amp']);self.assertFalse(p['deterministic_algorithms'])
        self.assertIsNone(p['adam_foreach']);self.assertIsNone(p['adam_fused']);self.assertIsNone(p['clip_foreach'])
        self.assertTrue(p['tf32_cudnn']);self.assertIsNone(p['CUBLAS_WORKSPACE_CONFIG'])
        with self.assertRaises(ValueError):validar_politica({**p,'adam_foreach':False})

    def test_contexto_minimo_aplica_solo_flags_declarados_y_restaura(self):
        antes=registro_efectivo()
        with aplicar_backend(politica('minimo')):
            r=registro_efectivo();verificar_registro(politica('minimo'),r)
            for k in antes:
                if k not in ['sdpa','cudnn_deterministic','cudnn_benchmark']:self.assertEqual(antes[k],r[k])
        self.assertEqual(antes,registro_efectivo())

    def test_rechaza_opciones_y_entorno_de_reanudacion_distintos(self):
        with aplicar_backend(politica('minimo')):
            r=registro_efectivo();previo=copy.deepcopy(r);previo['torch']='otra'
            with self.assertRaises(ValueError):verificar_registro(politica('minimo'),r,previo)
            malo=copy.deepcopy(r);malo['sdpa']['efficient']=True
            with self.assertRaises(ValueError):verificar_registro(politica('minimo'),malo)

    def test_completo_exige_cublas_antes_del_proceso(self):
        with patch.dict('os.environ',{'CUBLAS_WORKSPACE_CONFIG':':16:8'}):
            with self.assertRaises(ValueError):
                with aplicar_backend(politica('completo')):pass

    def test_adam_minimo_rechaza_cambiar_defaults(self):
        m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters(),foreach=False)
        with self.assertRaises(ValueError):configurar_optimizador(politica('minimo'),o)
        configurar_optimizador(politica('completo'),o);self.assertIs(o.param_groups[0]['fused'],False)

    def test_protocolo_versionado_y_no_mezcla_checkpoint_anterior(self):
        cfg=leer_cfg();cfg.update(backend_numerico=politica('minimo'),version_protocolo=VERSION);validar(cfg)
        mal=copy.deepcopy(cfg);mal['version_protocolo']='22'
        with self.assertRaises(ValueError):validar(mal)
        ck={'protocolo_v20':protocolo(cfg),'hashes_codigo_v20':{},'reanudacion':{},'historial':[],'epoca':20}
        with self.assertRaisesRegex(ValueError,'backend'):validar_estado_v20(ck,cfg,'A',{}, {},None)

    def test_observador_conserva_instantanea_si_cambia_una_huella(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters());m(torch.ones(1,2)).sum().backward();o.step()
            sc=torch.amp.GradScaler('cpu',enabled=False)
            s={'loss_original':1.,'loss_nueva':0.,'loss_global':1.,'norma_antes_clip':1.,'huella_entrada':'x','detalle':{}}
            a=ObservadorBackend(m,o,sc,None,None,root/'C1',root/'C1',2);a.d={'valor':torch.ones(2)};a.finalizar(s,3)
            b=ObservadorBackend(m,o,sc,None,None,root/'C2',root/'C1',2);b.d={'valor':torch.ones(2)};b.finalizar(s,3)
            self.assertFalse((root/'C2/paso_0003.pt').exists())
            c=ObservadorBackend(m,o,sc,None,None,root/'R',root/'C1',2);c.d={'valor':torch.zeros(2)};c.finalizar(s,3)
            self.assertTrue((root/'R/paso_0003.pt').exists())


if __name__=='__main__':unittest.main()
