"""Estados y presupuesto V20 con fixture CPU; CUDA se comprueba en el piloto real."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from pruebas.test_reanudacion_regional import origen,epoca
from ecowaste.reanudacion_regional import publicar,sha,huellas_estado
from ecowaste.reanudacion_v20 import DiarioV20,validar_estado_v20
from scripts.preparar_v20 import leer_cfg,protocolo,huellas_codigo,exigir_autorizacion


def envolver(s,cfg,rama,d,previo):
    c=copy.deepcopy(s)
    for h in c['historial']:
        h['coeficiente_discriminativo']=h.pop('coeficiente_regional',cfg['ramas'][rama])
        h['region_organica']={'presencias':1,'eventos':[{'imagen':'train:0','id_region':1,'peso_clase':1.,
            'contribucion':.1,'contribucion_loss_total':cfg['ramas'][rama]*.1}]}
    c.update(protocolo_v20=protocolo(cfg),hashes_codigo_v20={'fixture':'cpu'},
        sha256_origen=cfg['sha256_checkpoint_origen'],huella_regiones='regiones',hashes_manifiestos={'train':'train'},
        config_entrenamiento={'v20':{'coeficiente':cfg['ramas'][rama],'margen':1.,'fraccion':.25,
            'delta_huber':1.,'pesos':[1.,1.,2.,2.,1.,1.]}})
    c['reanudacion']={'tipo':'v20_margen_huber','rama':rama,'epoca_completa':True,'commit_previo_sha256':previo,
        'sha256_diagnostico':'diagnostico','estado_inicial_origen':d['estado_inicial'],'huellas_estado':huellas_estado(c)}
    return c


class EntrenamientoV20(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);cls.cfg=leer_cfg();cls.base=origen();cls.estados={};trazas={}
        for rama in ['A','B']:
            s=cls.base;ss=[]
            for e in range(20,32):s=epoca(s,rama,e);ss.append(s)
            cls.estados[rama]=ss
            trazas[rama]=[{'huella_orden_y_aumentos':s['historial'][-1]['huella_orden_y_aumentos']} for s in ss]
        cls.d={'sha256_archivo':'diagnostico','estado_inicial':huellas_estado(cls.base),'huella_regiones':'regiones',
            'hashes_datos':{'train':'train'},'organico_elegible':{'imagen':'train:0','id_region':1},'trazas_preparadas':trazas}

    def test_reanudacion_dos_ramas_preserva_1452_entradas_y_estados(self):
        for rama in ['A','B']:
            with self.subTest(rama=rama),tempfile.TemporaryDirectory() as tmp:
                d=DiarioV20(tmp,self.cfg,rama,self.d,{'fixture':'cpu'});previo=None
                for e in [20,21]:
                    c=envolver(self.estados[rama][e-20],self.cfg,rama,self.d,previo);previo=d.confirmar(c,previo)
                anteriores={p:sha(p) for p in Path(tmp).rglob('*') if p.is_file()}
                epoca(c,rama,22,interrumpir=3) # estos pasos sin commit se descartan
                c,regs,previo=d.leer(recuperar=True)
                self.assertEqual(c['actualizaciones_desde_origen'],242)
                for e in range(22,32):
                    c=envolver(epoca(c,rama,e),self.cfg,rama,self.d,previo);previo=d.confirmar(c,previo)
                c,regs,_=d.leer();d.actualizar_alias(regs)
                esperado=envolver(self.estados[rama][-1],self.cfg,rama,self.d,None)
                self.assertEqual(c['historial'],esperado['historial'])
                self.assertEqual(huellas_estado(c),huellas_estado(esperado)) # solo esta fixture CPU
                self.assertEqual(c['actualizaciones_desde_origen'],1452);self.assertEqual(len(regs),12)
                self.assertEqual(sha(Path(tmp)/'checkpoints'/f'{d.nombre}_ultimo.pt'),sha(d.checkpoint(31)))
                self.assertTrue(all(sha(p)==h for p,h in anteriores.items()))
        self.assertEqual(self.d['trazas_preparadas']['A'],self.d['trazas_preparadas']['B'])

    def test_rechaza_protocolo_rama_codigo_presupuesto_rng_y_organico(self):
        original=envolver(self.estados['A'][0],self.cfg,'A',self.d,None)
        cambios=[lambda c:c['protocolo_v20'].update(coeficiente_fondo=1),
            lambda c:c['reanudacion'].update(rama='B'),lambda c:c['reanudacion'].update(tipo='v18'),
            lambda c:c['hashes_codigo_v20'].update(fixture='otro'),lambda c:c.update(actualizaciones_desde_origen=1453),
            lambda c:c.update(rng_torch=torch.zeros_like(c['rng_torch'])),lambda c:c['historial'][0].update(huella_orden_y_aumentos='otra'),
            lambda c:c['config_entrenamiento']['v20'].update(pesos=[1.]*6),
            lambda c:c['historial'][0]['region_organica']['eventos'][0].update(peso_clase=25),
            lambda c:c['historial'][0]['region_organica']['eventos'][0].update(contribucion_loss_total=.1),
            lambda c:c['historial'][0]['region_organica'].update(presencias=0)]
        for cambio in cambios:
            c=copy.deepcopy(original);cambio(c)
            with self.assertRaises(ValueError):validar_estado_v20(c,self.cfg,'A',self.d,{'fixture':'cpu'},None)

    def test_recuperacion_commit_inmutabilidad_y_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=DiarioV20(tmp,self.cfg,'A',self.d,{'fixture':'cpu'})
            c=envolver(self.estados['A'][0],self.cfg,'A',self.d,None)
            publicar(d.checkpoint(20),c,tensor=True);antes=sha(d.checkpoint(20))
            with self.assertRaises(ValueError):d.leer()
            _,regs,_=d.leer(recuperar=True)
            self.assertTrue(regs[0]['recuperado_tras_interrupcion']);self.assertEqual(sha(d.checkpoint(20)),antes)
            with self.assertRaises(FileExistsError):d.confirmar(c,None)
            alias=Path(tmp)/'checkpoints'/f'{d.nombre}_ultimo.pt';alias.write_bytes(b'ajeno')
            with self.assertRaises(ValueError):d.actualizar_alias(regs)
            self.assertEqual(alias.read_bytes(),b'ajeno')

    def test_huellas_incluyen_evaluador_observador_y_dependencias(self):
        h=huellas_codigo()
        for p in ['scripts/entrenar_v20.py','scripts/probar_entrenamiento_v20.py','scripts/evaluar_v20.py',
            'scripts/auditar_propuestas_internas.py','ecowaste/traza_propuestas.py',
            'ecowaste/auditoria_propuestas_internas.py','ecowaste/evaluacion_detector.py','ecowaste/detector_centros.py']:
            self.assertIn(p,h)
        self.assertTrue(all(len(x)==64 for x in h.values()))

    def test_deshabilitado_antes_de_cargar_datos(self):
        cfg=leer_cfg();cfg['entrenamiento_completo_habilitado']=False
        with self.assertRaises(ValueError):exigir_autorizacion(cfg)

    def test_inferencia_no_recibe_anotaciones(self):
        import inspect
        from scripts.evaluar_v20 import inferir_sin_gt
        self.assertEqual(list(inspect.signature(inferir_sin_gt).parameters),['modelo','imagenes','metadatos','carpeta'])


if __name__=='__main__':unittest.main()
