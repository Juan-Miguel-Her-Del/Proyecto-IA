"""Entrenamiento de instancias desde cero, mismo protocolo por rama y diario inmutable."""
import copy
import importlib.metadata
import platform
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from ecowaste.detector_instancias_v31 import pareja,TIPO
from ecowaste.detector_hibrido import mascara_letterbox
from ecowaste.detector_consultas import perdida_consultas,pesos_materiales,xyxy_a_cxcywh
from ecowaste.datasets import CLASES
from ecowaste.datos_v28_2 import auxiliar_global_train
from ecowaste.ejecucion_detector import CacheDeteccion
from ecowaste.incorporacion import leer,sha
from ecowaste.reanudacion_regional import publicar,huellas_estado,restaurar_estados
from ecowaste.ablacion_fondo import huella,estado_aleatorio
from ecowaste.backend_numerico import politica,registro_efectivo,verificar_registro
from ecowaste.reproducibilidad_v20 import cpu,gradientes,adam_nominal
from ecowaste.experimento_v28_2 import indices_ciclicos
from scripts.ablacion_fondo import aumentar

RAIZ=Path(__file__).resolve().parents[1]
OUT=RAIZ/'resultados/v31_instancias'
CONFIG=RAIZ/'configuracion/instancias_v31.json'


def codigo():
    files=list((RAIZ/'ecowaste').glob('*.py'))+list((RAIZ/'scripts').glob('*v31*.py'))+list((RAIZ/'pruebas').glob('*v31*.py'))
    files += [RAIZ/'scripts'/n for n in ['ablacion_fondo.py','entrenar_detector_mapas.py','entrenar_clasificador.py']]
    return {p.relative_to(RAIZ).as_posix():sha(p) for p in sorted(set(files))}


def contexto():
    cfg=leer(CONFIG);torch.set_num_threads(4)
    if cfg['version']!='v31_instancias_control_extractor_1' or cfg['actualizaciones']!=2400 or cfg['test'] or cfg['pesos_externos']:
        raise ValueError('Protocolo no autorizado')
    if sha(RAIZ/cfg['referencia'])!=cfg['sha256_referencia'] or sha(RAIZ/cfg['coleccion']/'MANIFIESTO.json')!=cfg['sha256_coleccion']:
        raise ValueError('Referencia/datos alterados')
    if not torch.cuda.is_available():raise RuntimeError('CUDA requerida')
    ht0=CacheDeteccion(RAIZ,'train',.1)
    claves=['boxes','labels','ignore_boxes','geometria','id','formato_boxes']
    ht=SimpleNamespace(imagenes=ht0.imagenes,objetivos=[{k:o[k] for k in claves if k in o} for o in ht0.objetivos],registros=ht0.origen.registros)
    meta=leer(RAIZ/cfg['coleccion']/'MANIFIESTO.json');p=RAIZ/cfg['coleccion']/'train.pt'
    if sha(p)!=meta['archivos']['train.pt']:raise ValueError('Caché nueva modificada')
    nu=SimpleNamespace(**torch.load(p,map_location='cpu',weights_only=True))
    for ds in [ht,nu]:ds.validos=mascara_letterbox([o['geometria'] for o in ds.objetivos])
    gl=auxiliar_global_train(RAIZ)
    conteos=torch.bincount(torch.cat([o['labels'] for ds in [ht,nu] for o in ds.objetivos]),minlength=6)
    if max(len(o['boxes']) for ds in [ht,nu] for o in ds.objetivos)>cfg['consultas']:raise ValueError('Consultas insuficientes; no truncar GT')
    return cfg,ht,nu,gl,pesos_materiales(conteos).clamp(max=10.).cuda()


def planificar(cfg,ht,nu,gl):
    g=torch.Generator().manual_seed(cfg['semilla']+1);n=cfg['actualizaciones']
    conteos=torch.bincount(torch.cat([o['labels'] for o in ht.objetivos]),minlength=6)
    pc=pesos_materiales(conteos)
    w=torch.tensor([float(pc[o['labels'].unique()].mean()) if len(o['labels']) else 1. for o in ht.objetivos])
    w=(w/w.mean()).clamp(max=3.)
    hist=torch.multinomial(w.double(),n*8,True,generator=g).reshape(n,8)
    fuentes=[]
    for f in ['packwise','click']:
        idx=torch.tensor([i for i,r in enumerate(nu.registros) if r['fuente']==f])
        fuentes.append(idx[indices_ciclicos(len(idx),n*2,g)].reshape(n,2))
    cuenta=torch.bincount(gl.etiquetas,minlength=6)
    glob=torch.multinomial(1/cuenta[gl.etiquetas].double(),n*16,True,generator=g).reshape(n,16)
    priors=xyxy_a_cxcywh(torch.cat([o['boxes'] for ds in [ht,nu] for o in ds.objetivos]))[:,2:].median(0).values
    return {'historico':hist,'nuevo':torch.cat(fuentes,1),'global':glob,'pesos_muestreo':w,'priores_wh':priors}


def contrato(cfg,plan):
    paths=['datos/procesados/deteccion/train.jsonl','datos/procesados/deteccion/val.jsonl','datos/procesados/clasificacion/train.jsonl',cfg['coleccion']+'/MANIFIESTO.json',cfg['coleccion']+'/train.pt']
    return {'cfg':cfg,'codigo':codigo(),'plan_sha256':huella(plan),'datos':{p:sha(RAIZ/p) for p in paths},
        'entorno':{'python':platform.python_version(),'paquetes':{n:importlib.metadata.version(n) for n in ['torch','numpy','pillow','opencv-python']}}}


def iniciar(rama,cfg,plan,ck=None):
    if rama not in ['A','B']:raise ValueError('Rama inválida')
    if ck is not None and (ck.get('tipo')!=TIPO or ck.get('rama')!=rama or ck['contrato']['cfg']!=cfg):raise ValueError('Solo reanudar V31, nunca ep19/preentrenados')
    a,b=pareja(cfg['semilla']);m=(a if rama=='A' else b).cuda();del a,b
    with torch.no_grad():m.prior_wh.copy_(plan['priores_wh'].cuda())
    opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=.01,foreach=False,fused=False)
    sc=torch.amp.GradScaler('cuda',init_scale=1024)
    rg,ra=torch.Generator().manual_seed(3111),torch.Generator().manual_seed(3112)
    torch.manual_seed(cfg['semilla']+2);torch.cuda.manual_seed_all(cfg['semilla']+2);random.seed(cfg['semilla']+2);np.random.seed(cfg['semilla']+2)
    if ck is not None:
        restaurar_estados(m,opt,sc,ck,rg,ra)
        random.setstate(ck['rng_python']);r=ck['rng_numpy'];np.random.set_state((r[0],np.asarray(r[1],dtype=np.uint32),*r[2:]))
        verificar_registro(politica('completo'),registro_efectivo(opt),ck['backend'])
    return m,opt,sc,rg,ra


def flip_objetivos(objetivos,flip):
    out=[]
    for o,f in zip(objetivos,flip):
        z=dict(o)
        if bool(f):
            for key in ['boxes','ignore_boxes']:
                b=o[key].clone();b[:,0]=1-o[key][:,2];b[:,2]=1-o[key][:,0];z[key]=b
            g=dict(o['geometria']);g['relleno_x']=384-(g['relleno_x']+round(g['ancho_original']*g['escala_x']));z['geometria']=g
        out.append(z)
    return out


def paso(m,opt,sc,ht,nu,gl,pm,plan,k,rg,ra,observar=False):
    m.train();opt.zero_grad(set_to_none=True);entradas={};ls={};detalles={};obs={}
    for nombre,ds,coef in [('historico',ht,1.),('nuevo',nu,.5)]:
        ix=plan[nombre][k];x,_,aug=aumentar(ds.imagenes[ix].cuda().float()/255)
        os=flip_objetivos([ds.objetivos[i] for i in ix.tolist()],aug['flip'].flatten().tolist())
        validos=mascara_letterbox([o['geometria'] for o in os],'cuda')
        with torch.autocast('cuda',dtype=torch.float16):s=m(x,validos)
        l,d=perdida_consultas(s,os,'densidad_ponderada',pm)
        if not torch.isfinite(l):raise RuntimeError('Pérdida no finita')
        if d['objetos_asignados']!=sum(len(o['boxes']) for o in os):raise RuntimeError('Se perdió un objeto GT')
        sc.scale(coef*l).backward();ls[nombre]=float(l.detach());detalles[nombre]=d
        entradas[nombre]={'indices':ix,'aug':aug,'cajas':huella([o['boxes'] for o in os]),'padding':huella(validos)}
        if observar:obs[nombre]={'x':cpu(x),'salida':cpu(s),'loss':cpu(l),'objetivos':cpu(os)}
    ix=plan['global'][k];x=torch.from_numpy(np.array(gl.imagenes[ix.numpy()],copy=True)).cuda().float()/255
    x,_,aug=aumentar(x)
    with torch.autocast('cuda',dtype=torch.float16):la=m.clasificar(x)
    lg=F.cross_entropy(la.float(),gl.etiquetas[ix].cuda())
    if not torch.isfinite(lg):raise RuntimeError('Loss global no finita')
    sc.scale(.5*lg).backward();ls['global']=float(lg.detach());entradas['global']={'indices':ix,'aug':aug}
    sc.unscale_(opt)
    if observar:obs['gradientes']=gradientes(m)
    normas={}
    for pref in ['backbone','detalle','contexto','decodificador','material','caja']:
        gs=[p.grad.float().norm().square() for n,p in m.named_parameters() if n.startswith(pref+'.') and p.grad is not None]
        normas[pref]=float(torch.stack(gs).sum().sqrt()) if gs else 0.
    norma=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True,foreach=False)
    if observar:obs['gradientes_clip']=gradientes(m)
    antes=sc.get_scale();sc.step(opt);sc.update()
    if sc.get_scale()<antes or {int(s['step']) for s in opt.state.values()}!={k+1}:raise RuntimeError('Paso AdamW omitido/incoherente')
    tr={'paso':k+1,'loss':ls,'loss_total':ls['historico']+.5*ls['nuevo']+.5*ls['global'],'componentes':detalles,
        'norma':float(norma),'normas':normas,'scaler':sc.state_dict(),'entradas_sha256':huella(entradas),
        'indices':{n:plan[n][k].tolist() for n in ['historico','nuevo','global']},'rng':huella(estado_aleatorio(rg,ra))}
    if observar:obs.update(pesos=cpu(m.state_dict()),adamw=adam_nominal(m,opt),scaler=copy.deepcopy(sc.state_dict()),rng=cpu(estado_aleatorio(rg,ra)))
    return tr,obs


def instantanea(m,opt,sc,rg,ra,rama,k,hist,c,previo):
    nr=np.random.get_state()
    ck=cpu({'tipo':TIPO,'clases':CLASES,'pesos_externos':False,'arquitectura':m.configuracion_v31,'rama':rama,'paso':k,'historial':hist,
        'pesos':m.state_dict(),'optimizador':opt.state_dict(),'scaler':sc.state_dict(),**estado_aleatorio(rg,ra),
        'rng_python':random.getstate(),'rng_numpy':(nr[0],nr[1].tolist(),*nr[2:]),'contrato':c,'previo':previo,'backend':registro_efectivo(opt)})
    ck['huellas']=huellas_estado(ck);ck['rng_extra']=huella([ck['rng_python'],ck['rng_numpy']]);return ck


def comprobar(ck,c,rama):
    if ck.get('tipo')!=TIPO or ck['contrato']!=c or ck['rama']!=rama or ck['arquitectura']['cnn']!=(rama=='B'):raise ValueError('Protocolo/rama incompatible')
    k=ck['paso']
    if not 0<=k<=c['cfg']['actualizaciones'] or len(ck['historial'])!=k or [h['paso'] for h in ck['historial']]!=list(range(1,k+1)):raise ValueError('Presupuesto/historial discontinuo')
    if ck['huellas']!=huellas_estado(ck) or ck['rng_extra']!=huella([ck['rng_python'],ck['rng_numpy']]):raise ValueError('Estado alterado')
    if k and {int(s['step']) for s in ck['optimizador']['state'].values()}!={k}:raise ValueError('AdamW incorrecto')
    verificar_registro(politica('completo'),ck['backend'])


class Diario:
    def __init__(self,rama,c):self.rama,self.c=rama,c;self.base=OUT/'entrenamiento'/rama
    def leer(self):
        ck=None;previo=None;hueco=False
        for k in self.c['cfg']['checkpoints']:
            p=self.base/f'confirmaciones/{k:04d}.json'
            if not p.exists():hueco=True;continue
            if hueco:raise ValueError('Diario discontinuo')
            ev=leer(p)
            if ev['previo']!=previo or sha(RAIZ/ev['checkpoint'])!=ev['sha256']:raise ValueError('Commit alterado')
            ck=torch.load(RAIZ/ev['checkpoint'],map_location='cpu',weights_only=True);comprobar(ck,self.c,self.rama)
            if ck['paso']!=k or ck['previo']!=previo:raise ValueError('Commit ajeno')
            for h in ck['historial']:
                if sha(RAIZ/h['archivo'])!=h['sha256']:raise ValueError('Traza modificada')
            previo=sha(p)
        return ck,previo
    def guardar(self,ck):
        comprobar(ck,self.c,self.rama);k=ck['paso']
        if k not in self.c['cfg']['checkpoints']:raise ValueError('Punto de guardado no autorizado')
        cp=RAIZ/f'checkpoints/v31_instancias/{self.rama}/{k:04d}.pt';publicar(cp,ck,tensor=True)
        p=self.base/f'confirmaciones/{k:04d}.json';publicar(p,{'checkpoint':cp.relative_to(RAIZ).as_posix(),'sha256':sha(cp),'previo':ck['previo']});return sha(p)
