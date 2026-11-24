"""Protocolo V30 independiente: mismos datos, pérdidas y postproceso A/B."""
import copy
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from ecowaste.detector_hibrido import crear_pareja, DetectorHibrido, mascara_letterbox, TIPO
from ecowaste.datasets import CLASES
from ecowaste.datos_v28_2 import historico_train, auxiliar_global_train
from ecowaste.datos_v29 import cargar_coleccion
from ecowaste.experimento_v28_2 import indices_ciclicos
from ecowaste.ablacion_fondo import huella, estado_aleatorio, restaurar_aleatorio
from ecowaste.backend_numerico import politica, registro_efectivo, verificar_registro
from ecowaste.incorporacion import leer, sha
from ecowaste.reanudacion_regional import publicar, huellas_estado, restaurar_estados
from ecowaste.reproducibilidad_v20 import cpu, gradientes, adam_nominal
from scripts.ablacion_fondo import aumentar

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ/'resultados/v30_hibrido_revision02'
CONFIG = RAIZ/'configuracion/entrenamiento_hibrido_v30_revision02.json'


def codigo():
    files = list((RAIZ/'ecowaste').glob('*.py'))
    files += list((RAIZ/'scripts').glob('*v30*.py')) + list((RAIZ/'pruebas').glob('*v30*.py'))
    files += [RAIZ/'scripts'/n for n in ['ablacion_fondo.py', 'entrenar_detector_mapas.py', 'evaluar_v28_2.py', 'auditar_propuestas_internas.py']]
    return {p.relative_to(RAIZ).as_posix(): sha(p) for p in sorted(set(files))}


def contexto():
    cfg = leer(CONFIG)
    if cfg['version'] != 'v30_cnn_mapas_desde_cero_2_amp1024' or cfg['grad_scaler_init_scale'] != 1024. or cfg['actualizaciones'] != 2400 or cfg['test'] or cfg['adopcion_automatica']:
        raise ValueError('Protocolo incompatible')
    if sha(RAIZ/cfg['checkpoint_referencia']) != cfg['sha256_referencia'] or sha(RAIZ/cfg['coleccion']/'MANIFIESTO.json') != cfg['sha256_coleccion']:
        raise ValueError('Referencia o colección alteradas')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Este experimento requiere CUDA')
    ht = historico_train(RAIZ)
    nu = cargar_coleccion(RAIZ, RAIZ/cfg['coleccion'], 'train')
    gl = auxiliar_global_train(RAIZ)
    for ds in [ht, nu]:
        ds.validos = mascara_letterbox([o['geometria'] for o in ds.objetivos])
    n = torch.bincount(ht.mascaras[ht.mascaras != 255].flatten(), minlength=7)[1:].float()
    pm = n.clamp(min=1e-6).rsqrt()
    pm = (pm/pm.mean()).clamp(max=3).cuda()
    return cfg, ht, nu, gl, pm


def planificar(cfg, ht, nu, gl):
    g = torch.Generator().manual_seed(cfg['semilla']+10)
    pasos = cfg['actualizaciones']
    hist = indices_ciclicos(len(ht.registros), pasos*8, g).reshape(pasos, 8)
    ns = []
    for fuente in ['packwise', 'click']:
        idx = torch.tensor([i for i, r in enumerate(nu.registros) if r['fuente'] == fuente])
        ns.append(idx[indices_ciclicos(len(idx), pasos*2, g)].reshape(pasos, 2))
    counts = torch.bincount(gl.etiquetas, minlength=6)
    aux = torch.multinomial(1/counts[gl.etiquetas].double(), pasos*16, True, generator=g).reshape(pasos, 16)
    return {'historico': hist, 'nuevo': torch.cat(ns, 1), 'global': aux}


def contrato(cfg, plan):
    datos = ['datos/procesados/deteccion/train.jsonl', 'datos/procesados/deteccion/val.jsonl',
             'datos/procesados/clasificacion/train.jsonl', cfg['coleccion']+'/MANIFIESTO.json']
    return {'config': cfg, 'codigo': codigo(), 'plan': huella(plan), 'datos': {p: sha(RAIZ/p) for p in datos}}


def inicializar(rama, cfg, ck=None):
    if rama not in ['A', 'B']:
        raise ValueError('Rama desconocida')
    if ck is not None:
        if ck.get('tipo') != TIPO or ck.get('rama') != rama or ck.get('contrato', {}).get('config') != cfg:
            raise ValueError('Solo reanudar estados V30 de esta rama; prohibida inicialización ep19/externa')
    a, b = crear_pareja(cfg['semilla'])
    m = (a if rama == 'A' else b).cuda()
    del a, b
    opt = torch.optim.AdamW(m.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'], foreach=False, fused=False)
    sc = torch.amp.GradScaler('cuda', init_scale=cfg['grad_scaler_init_scale'])
    rg, ra = torch.Generator().manual_seed(3002), torch.Generator().manual_seed(3003)
    torch.manual_seed(cfg['semilla']+20)
    torch.cuda.manual_seed_all(cfg['semilla']+20)
    if ck is not None:
        restaurar_estados(m, opt, sc, ck, rg, ra)
        random.setstate(ck['rng_python'])
        nr = ck['rng_numpy']
        np.random.set_state((nr[0], np.asarray(nr[1], dtype=np.uint32), *nr[2:]))
        real_np = np.random.get_state()
        if random.getstate() != ck['rng_python'] or huella((real_np[0], real_np[1].tolist(), *real_np[2:])) != huella(ck['rng_numpy']):
            raise ValueError('RNG adicional no restaurado exactamente')
        verificar_registro(politica('completo'), registro_efectivo(opt), ck['backend'])
    else:
        random.seed(cfg['semilla']+20)
        np.random.seed(cfg['semilla']+20)
    return m, opt, sc, rg, ra


def paso(m, opt, sc, ht, nu, gl, pm, plan, k, rg, ra, observar=False, global_activa=True):
    from ecowaste.detector_mapas import perdida_mapas
    m.train(); opt.zero_grad(set_to_none=True)
    pruebas = {} if observar else None
    perdidas = {}
    entradas = {}
    for nombre, ds, peso in [('historico', ht, 1.), ('nuevo', nu, .5)]:
        ix = plan[nombre][k]
        x, y, aug = aumentar(ds.imagenes[ix].cuda().float()/255, ds.mascaras[ix].cuda())
        validos = ds.validos[ix].cuda()
        validos = torch.where(aug['flip'][:, 0].cuda(), validos.flip(-1), validos)
        with torch.autocast('cuda', dtype=torch.float16):
            raw = m(x, validos)
            loss = perdida_mapas(raw, y, pm, coeficiente_fondo=3.)
        if not torch.isfinite(loss):
            raise RuntimeError('Loss no finita; detener')
        sc.scale(peso*loss).backward()
        perdidas[nombre] = float(loss.detach())
        entradas[nombre] = {'indices': ix, 'aumentos': aug, 'valida': huella(validos)}
        if observar:
            pruebas[nombre] = {'entrada': cpu(x), 'target': cpu(y), 'padding': cpu(validos), 'logits': cpu(raw), 'loss': cpu(loss)}
    if global_activa:
        ix = plan['global'][k]
        x = torch.from_numpy(np.array(gl.imagenes[ix.numpy()], copy=True)).cuda().float()/255
        x, _, aug = aumentar(x)
        with torch.autocast('cuda', dtype=torch.float16):
            raw = m.clasificar(x)
            lg = F.cross_entropy(raw.float(), gl.etiquetas[ix].cuda())
        if not torch.isfinite(lg):
            raise RuntimeError('Loss global no finita')
        sc.scale(.5*lg).backward()
        perdidas['global'] = float(lg.detach())
        entradas['global'] = {'indices': ix, 'aumentos': aug}
    sc.unscale_(opt)
    if observar:
        pruebas['gradientes'] = gradientes(m)
    normas = {}
    for nombre in ['backbone', 'detalle', 'contexto', 'cabezal']:
        gs = [p.grad.detach().float().norm()**2 for n, p in m.named_parameters() if n.startswith(nombre+'.') and p.grad is not None]
        normas[nombre] = float(torch.stack(gs).sum().sqrt()) if gs else 0.
    norma = torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True, foreach=False)
    if observar:
        pruebas['gradientes_clip'] = gradientes(m)
    previo = sc.get_scale()
    sc.step(opt); sc.update()
    steps = {int(s['step']) for s in opt.state.values()}
    if sc.get_scale() < previo or steps != {k+1}:
        raise RuntimeError(f'Actualización omitida/AdamW inconsistente: {steps}')
    tr = {'paso': k+1, 'loss': perdidas, 'loss_total': perdidas['historico']+.5*perdidas['nuevo']+.5*perdidas.get('global', 0),
          'norma_pre_clip': float(norma), 'normas_modulos': normas, 'scaler': sc.state_dict(),
          'entradas_sha256': huella(entradas), 'indices': {n: p[k].tolist() for n, p in plan.items()},
          'rng_sha256': huella(estado_aleatorio(rg, ra)), 'paso_adamw': k+1}
    if observar:
        pruebas.update(pesos=cpu(m.state_dict()), adamw=adam_nominal(m, opt), scaler=copy.deepcopy(sc.state_dict()),
                       rng=cpu(estado_aleatorio(rg, ra)), norma=cpu(norma), rng_python=random.getstate(),
                       rng_numpy=(np.random.get_state()[0], np.random.get_state()[1].tolist(), *np.random.get_state()[2:]))
    return tr, pruebas


def snapshot(m, opt, sc, rg, ra, rama, k, historial, c, previo):
    nr = np.random.get_state()
    ck = cpu({'tipo': TIPO, 'clases': CLASES, 'pesos_externos': False,
        'configuracion_hibrido': m.configuracion_hibrido, 'pesos': m.state_dict(), 'optimizador': opt.state_dict(),
        'scaler': sc.state_dict(), **estado_aleatorio(rg, ra), 'rng_python': random.getstate(),
        'rng_numpy': (nr[0], nr[1].tolist(), *nr[2:]), 'rama': rama, 'paso': k, 'historial': historial,
        'contrato': c, 'previo': previo, 'backend': registro_efectivo(opt),
        'config_entrenamiento': {'umbral_evaluacion': .05}, 'epoca': None})
    ck['huellas'] = huellas_estado(ck)
    ck['huella_rng_adicional'] = huella([ck['rng_python'], ck['rng_numpy']])
    return ck


def comprobar(ck, rama, c):
    if ck['tipo'] != TIPO or ck['rama'] != rama or ck['contrato'] != c or ck['configuracion_hibrido']['cnn'] != (rama == 'B'):
        raise ValueError('Checkpoint de otro modelo/protocolo/rama')
    k = ck['paso']
    if not 0 <= k <= c['config']['actualizaciones'] or len(ck['historial']) != k or [x['paso'] for x in ck['historial']] != list(range(1, k+1)):
        raise ValueError('Presupuesto/historial inválidos')
    if ck['huellas'] != huellas_estado(ck) or ck['huella_rng_adicional'] != huella([ck['rng_python'], ck['rng_numpy']]):
        raise ValueError('Estado alterado')
    if k and {int(s['step']) for s in ck['optimizador']['state'].values()} != {k}:
        raise ValueError('AdamW incoherente')
    verificar_registro(politica('completo'), ck['backend'])


class Diario:
    def __init__(self, rama, c):
        self.rama, self.c = rama, c
        self.base = SALIDA/'entrenamiento'/rama

    def leer(self):
        ck, previo, hueco = None, None, False
        for k in self.c['config']['checkpoints']:
            p = self.base/'confirmaciones'/f'paso_{k:04d}.json'
            if not p.exists():
                hueco = True
                continue
            if hueco:
                raise ValueError('Diario discontinuo')
            ev = leer(p)
            if ev['previo'] != previo or sha(RAIZ/ev['checkpoint']) != ev['sha256']:
                raise ValueError('Cadena/checkpoint alterado')
            ck = torch.load(RAIZ/ev['checkpoint'], map_location='cpu', weights_only=True)
            comprobar(ck, self.rama, self.c)
            if ck['paso'] != k or ck['previo'] != previo:
                raise ValueError('Commit ajeno')
            for h in ck['historial']:
                if sha(RAIZ/h['traza']) != h['sha256_traza']:
                    raise ValueError('Traza modificada')
            previo = sha(p)
        return ck, previo

    def guardar(self, ck):
        comprobar(ck, self.rama, self.c)
        k = ck['paso']
        if k not in self.c['config']['checkpoints']:
            raise ValueError('Punto de guardado no autorizado')
        cp = RAIZ/f'checkpoints/v30_hibrido_revision02/{self.rama}/paso_{k:04d}.pt'
        publicar(cp, ck, tensor=True)
        p = self.base/'confirmaciones'/f'paso_{k:04d}.json'
        publicar(p, {'checkpoint': cp.relative_to(RAIZ).as_posix(), 'sha256': sha(cp), 'previo': ck['previo']})
        return sha(p)
