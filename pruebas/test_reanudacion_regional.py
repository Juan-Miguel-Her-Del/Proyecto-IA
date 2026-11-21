"""Modelo pequeño CPU: prueba estado, entradas y presupuesto; no afirma equivalencia CUDA."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch.nn import functional as F
from ecowaste.ablacion_fondo import huella,estado_aleatorio
from ecowaste.reanudacion_regional import (DiarioRama,publicar,huellas_estado,validar_estado,
    restaurar_estados,sha,bloqueo)
from scripts.supervision_regional_materiales import CONFIG,loaders
from scripts.ablacion_fondo import aumentar


class Aux:
    etiquetas=torch.arange(24)%6
    def __len__(self):return 24


def crear():
    m=torch.nn.Sequential(torch.nn.Linear(2,6),torch.nn.Dropout(.1))
    opt=torch.optim.AdamW(m.parameters(),lr=.0005)
    sc=torch.amp.GradScaler('cpu',init_scale=16)
    return m,opt,sc,torch.Generator(),torch.Generator()


def origen():
    torch.manual_seed(29);m,o,s,r,a=crear();r.manual_seed(21);a.manual_seed(22)
    sc=F.cross_entropy(m(torch.ones(2,2)),torch.tensor([0,1]));sc.backward();o.step();o.zero_grad()
    for v in o.state.values():v['step'].fill_(2299)
    return copy.deepcopy({'pesos':m.state_dict(),'optimizador':o.state_dict(),'scaler':s.state_dict(),
        **estado_aleatorio(r,a),'historial':[],'epoca':19})


def epoca(estado,rama,ep,interrumpir=None):
    m,o,s,r,a=crear();restaurar_estados(m,o,s,estado,r,a)
    dl,da=loaders(range(964),Aux(),r,a,solo_indices=True);itaux=iter(da);h=hashlib.sha256()
    for paso,(indices,) in enumerate(dl,1):
        _,_,aug=aumentar(torch.zeros(len(indices),1,1,1))
        (ia,)=next(itaux);_,_,auga=aumentar(torch.zeros(len(ia),1,1,1))
        x=torch.stack([indices.float()/964,torch.ones(len(indices))],1)
        xa=torch.stack([ia.float()/24,torch.ones(len(ia))],1)
        o.zero_grad(set_to_none=True)
        loss=F.cross_entropy(m(x),indices%6)*(1+(0.25 if rama=='B' else 0))+.5*F.cross_entropy(m(xa),ia%6)
        s.scale(loss).backward();s.unscale_(o);torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
        s.step(o);s.update()
        h.update(huella({'indices_train':indices,'aumentos_train':aug,'indices_aux':ia,'aumentos_aux':auga}).encode())
        if interrumpir==paso:return None
    hist=copy.deepcopy(estado['historial']);hist.append({'epoca':ep,'actualizaciones':121*(ep-19),
        'paso_adamw':2299+121*(ep-19),'learning_rate':.0005,'coeficiente_regional':0. if rama=='A' else .25,
        'map50_val':.1,'huella_orden_y_aumentos':h.hexdigest()})
    hist[-1]['region_organica']={'presencia':1,'peso':1.,'eventos':[{'imagen':'train:0','id_region':1,'indice_train':0,
        'contribucion_ce':.1,'contribucion_loss_total':(.25 if rama=='B' else 0)*.1}]}
    return copy.deepcopy({'pesos':m.state_dict(),'optimizador':o.state_dict(),'scaler':s.state_dict(),
        **estado_aleatorio(r,a),'epoca':ep,'historial':hist,'actualizaciones_desde_origen':121*(ep-19),
        'mejor_epoca':20,'mejor_map':.1})


def envolver(estado,cfg,rama,diag,previo):
    ck=copy.deepcopy(estado)
    ck.update({'protocolo_regional':cfg,'sha256_origen':cfg['sha256_checkpoint_origen'],
        'hashes_codigo':{'fixture':'cpu'},'huella_regiones':'regiones','hashes_manifiestos':{'train':'train'},
        'config_entrenamiento':{'supervision_regional':{'coeficiente':cfg['ramas'][rama],'pesos_train':[1.]*6}}})
    ck['reanudacion']={'version':1,'rama':rama,'commit_previo_sha256':previo,'sha256_diagnostico':'diagnostico',
        'epoca_completa':True,'huellas_estado':huellas_estado(ck),
        'estado_inicial_origen':diag['coste_diagnostico'][rama]['estado_inicial']}
    return ck


class ReanudacionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);cls.cfg=json.loads(CONFIG.read_text(encoding='utf-8'));cls.base=origen()
        cls.continuos={};cls.diags={}
        for rama in ['A','B']:
            estados=[];actual=cls.base
            for e in range(20,32):actual=epoca(actual,rama,e);estados.append(actual)
            cls.continuos[rama]=estados
            cls.diags[rama]={'huella_regiones':'regiones','hashes_datos':{'train':'train'},
                'sha256_archivo':'diagnostico','estadistica_regiones':{'pesos_clase':[1.]*6},
                'region_organica_elegible':{'imagen':'train:0','id_region':1,'indice_train':0,'peso':1.},
                'coste_diagnostico':{rama:{'estado_inicial':huellas_estado(cls.base)}},
                'trazas_preparadas':{rama:[{'huella_orden_y_aumentos':s['historial'][-1]['huella_orden_y_aumentos']} for s in estados]}}

    def test_interrupcion_y_reanudacion_ambas_ramas_1452(self):
        for rama in ['A','B']:
            with self.subTest(rama=rama),tempfile.TemporaryDirectory() as tmp:
                d=DiarioRama(tmp,self.cfg,rama,self.diags[rama],{'fixture':'cpu'});previo=None
                # Dos épocas confirmadas, luego pérdida de tres pasos sin commit.
                for e in [20,21]:
                    ck=envolver(self.continuos[rama][e-20],self.cfg,rama,self.diags[rama],previo)
                    previo=d.confirmar(ck,previo)
                hashes={p:sha(p) for p in Path(tmp).rglob('*') if p.is_file()}
                epoca(ck,rama,22,interrumpir=3)
                actual,regs,previo=d.leer(recuperar=True)
                self.assertEqual(actual['actualizaciones_desde_origen'],242)
                for e in range(22,32):
                    actual=epoca(actual,rama,e);actual=envolver(actual,self.cfg,rama,self.diags[rama],previo)
                    previo=d.confirmar(actual,previo)
                final,regs,_=d.leer();d.actualizar_alias(regs)
                self.assertEqual(final['actualizaciones_desde_origen'],1452)
                self.assertEqual(len(final['historial']),12)
                self.assertEqual(final['historial'],self.continuos[rama][-1]['historial'])
                # Identidad comprobada solo para esta fixture CPU.
                self.assertEqual(huellas_estado(final),huellas_estado(self.continuos[rama][-1]))
                self.assertTrue(all(sha(p)==v for p,v in hashes.items()))
                self.assertEqual(sha(Path(tmp)/'checkpoints'/f'{d.nombre}_ultimo.pt'),sha(d.checkpoint(31)))
                self.assertEqual(sha(Path(tmp)/'checkpoints'/f'{d.nombre}_mejor.pt'),sha(d.checkpoint(20)))

    def test_mismas_entradas_A_B(self):
        a=self.continuos['A'][-1]['historial'];b=self.continuos['B'][-1]['historial']
        self.assertEqual([h['huella_orden_y_aumentos'] for h in a],[h['huella_orden_y_aumentos'] for h in b])

    def test_recupera_checkpoint_publicado_sin_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=DiarioRama(tmp,self.cfg,'A',self.diags['A'],{'fixture':'cpu'})
            ck=envolver(self.continuos['A'][0],self.cfg,'A',self.diags['A'],None)
            publicar(d.checkpoint(20),ck,tensor=True);antes=sha(d.checkpoint(20))
            with self.assertRaises(ValueError):d.leer()
            actual,reg,_=d.leer(recuperar=True)
            self.assertTrue(reg[0]['recuperado_tras_interrupcion'])
            self.assertEqual(sha(d.checkpoint(20)),antes);self.assertEqual(actual['epoca'],20)

    def test_rechaza_rama_protocolo_codigo_rng_y_presupuesto(self):
        original=envolver(self.continuos['A'][0],self.cfg,'A',self.diags['A'],None)
        cambios=[lambda c:c['reanudacion'].update(rama='B'),lambda c:c['protocolo_regional'].update(coeficiente_fondo=1),
            lambda c:c['hashes_codigo'].update(fixture='otro'),lambda c:c.update(rng_torch=torch.zeros_like(c['rng_torch'])),
            lambda c:c.update(actualizaciones_desde_origen=1453),lambda c:c['historial'][0].update(huella_orden_y_aumentos='otra'),
            lambda c:c['reanudacion'].update(epoca_completa=False),lambda c:c.update(mejor_epoca=21)]
        cambios.append(lambda c:c['historial'][0]['region_organica'].update(presencia=0))
        for cambio in cambios:
            c=copy.deepcopy(original);cambio(c)
            with self.assertRaises(ValueError):validar_estado(c,self.cfg,'A',self.diags['A'],{'fixture':'cpu'},None)

    def test_inmutabilidad_y_alias_ajeno(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=DiarioRama(tmp,self.cfg,'A',self.diags['A'],{'fixture':'cpu'})
            c=envolver(self.continuos['A'][0],self.cfg,'A',self.diags['A'],None);d.confirmar(c,None)
            with self.assertRaises(FileExistsError):d.confirmar(c,None)
            _,regs,_=d.leer();alias=Path(tmp)/'checkpoints'/f'{d.nombre}_ultimo.pt'
            alias.write_bytes(b'otro experimento')
            with self.assertRaises(ValueError):d.actualizar_alias(regs)
            self.assertEqual(alias.read_bytes(),b'otro experimento')

    def test_rechaza_commit_corrupto_y_hueco(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=DiarioRama(tmp,self.cfg,'A',self.diags['A'],{'fixture':'cpu'})
            c=envolver(self.continuos['A'][0],self.cfg,'A',self.diags['A'],None);d.confirmar(c,None)
            r=json.loads(d.registro(20).read_text(encoding='utf-8'));r['sha256_checkpoint']='otro'
            d.registro(20).write_text(json.dumps(r),encoding='utf-8')
            with self.assertRaises(ValueError):d.leer()
        with tempfile.TemporaryDirectory() as tmp:
            d=DiarioRama(tmp,self.cfg,'A',self.diags['A'],{'fixture':'cpu'})
            publicar(d.checkpoint(21),c,tensor=True)
            with self.assertRaises(ValueError):d.leer(recuperar=True)

    def test_bloqueo_exclusivo(self):
        with tempfile.TemporaryDirectory() as tmp:
            with bloqueo(Path(tmp)/'rama.lock'):
                with self.assertRaises((OSError,BlockingIOError)):
                    with bloqueo(Path(tmp)/'rama.lock'):pass
            with bloqueo(Path(tmp)/'rama.lock'):pass


if __name__=='__main__':unittest.main()
