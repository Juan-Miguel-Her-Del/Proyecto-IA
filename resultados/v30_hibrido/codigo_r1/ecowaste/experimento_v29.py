"""A/B de datos V29: exposición histórica idéntica y microbatch nuevo adicional."""
from contextlib import contextmanager
import copy
from pathlib import Path
import time,uuid
import importlib.metadata,platform
import numpy as np
import torch
from torch.nn import functional as F
from ecowaste.incorporacion import leer,sha,digest
from ecowaste.reanudacion_regional import publicar,huellas_estado
from ecowaste.ablacion_fondo import huella,estado_aleatorio,pasos_adam
from ecowaste.backend_numerico import politica,registro_efectivo,verificar_registro
from ecowaste.experimento_v28_2 import restaurar,indices_ciclicos
from ecowaste.reproducibilidad_v20 import cpu,gradientes,adam_nominal
from ecowaste.datos_v28_2 import historico_train,auxiliar_global_train
from ecowaste.datos_v29 import cargar_coleccion
from ecowaste.detector_mapas import perdida_mapas
from scripts.ablacion_fondo import aumentar

RAIZ=Path(__file__).resolve().parents[1]
SALIDA=RAIZ/'resultados/v29_packwise_click'
CONFIG=RAIZ/'configuracion/entrenamiento_packwise_click_v29.json'
VERSION='v29_datos_preserva_historico_1'


def validar(cfg):
    fijo={'version':VERSION,'actualizaciones':600,'historico_batch':8,'nuevo_batch':4,
        'coeficiente_nuevos':.5,'auxiliar_batch':16,'coeficiente_auxiliar_global':.5,
        'learning_rate':.0005,'weight_decay':.01,'clipping':1.,'coeficiente_fondo':3.,
        'backend_numerico':politica('completo'),'warmup_nuevo':False,'parada_temprana':False,
        'semilla_nuevos':2901,'checkpoints':[0,100,300,600],'seleccion_por_validacion':False,
        'test':False,'generalizacion':False,'adopcion_automatica':False,'vistas':[1,5],
        'umbral':.05,'politica':'argmax7','nms':.5,'top':100,'area_minima':2}
    if any(cfg.get(k)!=v for k,v in fijo.items()):raise ValueError('Protocolo V29 alterado')
    if cfg['checkpoint_origen']!='checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt':raise ValueError('Origen distinto')
    if cfg['sha256_origen']!='eb8fd00ee536ead21e3a2a0ecd5b06baad04acb2e437b22e0eac3686a7bf36b9' or sha(RAIZ/cfg['checkpoint_origen'])!=cfg['sha256_origen']:raise ValueError('Checkpoint original alterado')
    if sha(RAIZ/cfg['coleccion']/'MANIFIESTO.json')!=cfg['sha256_coleccion']:raise ValueError('Datos no congelados o alterados')


def codigo():
    archivos=list((RAIZ/'ecowaste').glob('*.py'))
    archivos+=list((RAIZ/'scripts').glob('*v29*.py'))+list((RAIZ/'pruebas').glob('*v29*.py'))
    archivos+=[RAIZ/'scripts'/n for n in ['ablacion_fondo.py','entrenar_detector_mapas.py','entrenar_clasificador.py','auditar_propuestas_internas.py']]
    return {p.relative_to(RAIZ).as_posix():sha(p) for p in sorted(set(archivos))}


def crear_plan(ck,n_hist,fuentes,etiquetas,cfg):
    """Generadores privados: crear el plan nunca consume RNG global."""
    rg=torch.Generator();rg.set_state(ck['rng_loader']);ra=torch.Generator();ra.set_state(ck['rng_auxiliar'])
    rn=torch.Generator().manual_seed(cfg['semilla_nuevos']);n=cfg['actualizaciones']
    hist=indices_ciclicos(n_hist,n*8,rg).reshape(n,8)
    clases=torch.bincount(etiquetas,minlength=6)
    if bool((clases==0).any()):raise ValueError('Tarea global sin seis clases')
    glob=torch.multinomial(1/clases[etiquetas].double(),n*16,True,generator=ra).reshape(n,16)
    columnas=[]
    for fuente in ['packwise','click']:
        idx=torch.tensor([i for i,s in enumerate(fuentes) if s==fuente],dtype=torch.long)
        if not len(idx):raise ValueError('Falta una de las dos fuentes autorizadas')
        columnas.append(idx[indices_ciclicos(len(idx),n*2,rn)].reshape(n,2))
    return {'historico':hist,'nuevo':torch.cat(columnas,1),'global':glob,
        'version':cfg['version'],'regla':'A/B mismas ocho históricas; B añade 2 PackWISE + 2 CLICK; global16 idéntico',
        'rng_plan_final':{'historico':rg.get_state(),'global':ra.get_state(),'nuevo':rn.get_state()}}


def indices(plan,rama,k):
    if rama not in ['A','B'] or not 0<=k<len(plan['historico']):raise ValueError('Rama/paso fuera de presupuesto')
    return {'historico':plan['historico'][k],'global':plan['global'][k],
        'nuevo':plan['nuevo'][k] if rama=='B' else torch.empty(0,dtype=torch.long)}


def rng_nuevo_inicial(cfg):
    cpu_gen=torch.Generator().manual_seed(cfg['semilla_nuevos']+1)
    cuda_gen=torch.Generator(device='cuda').manual_seed(cfg['semilla_nuevos']+2)
    return {'cpu':cpu_gen.get_state(),'cuda':[cuda_gen.get_state()]}


@contextmanager
def flujo_auxiliar(estado):
    """Dropout/aumentos nuevos no desplazan el RNG de las imágenes históricas."""
    previo={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
    torch.set_rng_state(estado['cpu']);torch.cuda.set_rng_state_all(estado['cuda'])
    try:yield
    finally:
        estado['cpu']=torch.get_rng_state();estado['cuda']=torch.cuda.get_rng_state_all()
        torch.set_rng_state(previo['cpu']);torch.cuda.set_rng_state_all(previo['cuda'])


def contexto():
    cfg=leer(CONFIG);validar(cfg);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA requerida')
    ck=torch.load(RAIZ/cfg['checkpoint_origen'],map_location='cpu',weights_only=True)
    ht=historico_train(RAIZ);nu=cargar_coleccion(RAIZ,RAIZ/cfg['coleccion'],'train');gl=auxiliar_global_train(RAIZ)
    n=torch.bincount(ht.mascaras[ht.mascaras!=255].flatten(),minlength=7)[1:].float()
    pm=n.clamp(min=1e-6).rsqrt();pm=(pm/pm.mean()).clamp(max=3)
    return cfg,ck,ht,nu,gl,pm.cuda()


def paso(m,opt,sc,ht,nu,gl,pm,plan,rama,k,rg,ra,rn,observar=False):
    inicio=time.perf_counter();ix=indices(plan,rama,k);m.train();opt.zero_grad(set_to_none=True)
    rng0=huella(estado_aleatorio(rg,ra));prueba={} if observar else None
    x,y,aug=aumentar(ht.imagenes[ix['historico']].cuda().float()/255,ht.mascaras[ix['historico']].cuda())
    with torch.autocast('cuda',dtype=torch.float16):
        raw=m(x);lh=perdida_mapas(raw,y,pm,coeficiente_fondo=3.)
    if not torch.isfinite(lh):raise RuntimeError('Loss histórica no finita')
    if observar:prueba.update(entrada_historica=cpu(x),etiquetas_historicas=cpu(y),logits_historico=cpu(raw),loss_historico=cpu(lh))
    sc.scale(lh).backward()
    ia=ix['global'];xa=torch.from_numpy(np.array(gl.imagenes[ia.numpy()],copy=True));ya=gl.etiquetas[ia].cuda()
    xa,_,auga=aumentar(xa.cuda().float()/255)
    with torch.autocast('cuda',dtype=torch.float16):rawa=m.clasificar(xa)
    lg=F.cross_entropy(rawa.float(),ya)
    if not torch.isfinite(lg):raise RuntimeError('Loss global no finita')
    sc.scale(.5*lg).backward();ln=lh.detach()*0;augn=None
    if observar:prueba.update(entrada_global=cpu(xa),etiquetas_globales=cpu(ya),logits_global=cpu(rawa),loss_global=cpu(lg))
    if rama=='B':
        with flujo_auxiliar(rn):
            xn,yn,augn=aumentar(nu.imagenes[ix['nuevo']].cuda().float()/255,nu.mascaras[ix['nuevo']].cuda())
            with torch.autocast('cuda',dtype=torch.float16):
                rawn=m(xn);ln=perdida_mapas(rawn,yn,pm,coeficiente_fondo=3.)
            if not torch.isfinite(ln):raise RuntimeError('Loss nueva no finita')
            sc.scale(.5*ln).backward()
            if observar:prueba.update(entradas_nuevas=cpu(xn),etiquetas_nuevas=cpu(yn),logits_nuevos=cpu(rawn),loss_nuevos=cpu(ln))
    sc.unscale_(opt)
    if observar:prueba['gradientes']=gradientes(m)
    norma=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True,foreach=False)
    if observar:prueba['gradientes_clip']=gradientes(m)
    anterior=pasos_adam(opt);escala=copy.deepcopy(sc.state_dict());sc.step(opt);sc.update()
    if pasos_adam(opt)!=anterior+1:raise RuntimeError('AdamW omitió un paso; detener, no ampliar presupuesto')
    comun={'indices':{'historico':ix['historico'],'global':ix['global']},'mapas':aug,'global':auga}
    entradas={'comunes':comun,'indices_nuevos':ix['nuevo'],'aumentos_nuevos':augn}
    torch.cuda.synchronize()
    tr={'rama':rama,'paso':k+1,'loss_historico':float(lh),'loss_global':float(lg),'loss_nuevos':float(ln),
        'loss_total':float(lh+.5*lg+.5*ln),'norma_gradiente_pre_clip':float(norma),
        'scaler_antes':escala,'scaler_despues':sc.state_dict(),'paso_adamw':pasos_adam(opt),
        'indices':{s:t.tolist() for s,t in ix.items()},'huella_entradas_comunes':huella(comun),
        'huella_entradas_todas':huella(entradas),'rng_comun_antes':rng0,
        'rng_comun_despues':huella(estado_aleatorio(rg,ra)),'rng_nuevo_despues':huella(rn),
        'gradientes_finitos':True,'segundos':time.perf_counter()-inicio}
    if observar:prueba.update(entradas=cpu(entradas),pesos=cpu(m.state_dict()),adamw=adam_nominal(m,opt),
        scaler=copy.deepcopy(sc.state_dict()),norma=cpu(norma),rng=cpu(estado_aleatorio(rg,ra)),rng_nuevo=cpu(rn))
    return tr,prueba


def contrato(cfg,plan):
    ht=sha(RAIZ/'datos/procesados/deteccion/train.jsonl')
    gl=sha(RAIZ/'datos/procesados/clasificacion/train.jsonl')
    return {'config':cfg,'codigo':codigo(),'plan_sha256':huella(plan),
        'train_historico_sha256':ht,'global_train_sha256':gl,
        'cache_historico_sha256':sha(RAIZ/f'datos/cache/deteccion_384_train_{ht[:12]}.pt'),
        'metadata_global_train_sha256':sha(RAIZ/f'datos/cache/clasificacion_auxiliar_{gl[:12]}/metadata.json'),
        'versiones_ejecucion':{'python':platform.python_version(),
            'paquetes':{n:importlib.metadata.version(n) for n in ['torch','numpy','pillow','opencv-python']}}}


def instantanea(origen,m,opt,sc,rg,ra,rn,rama,k,historial,c,previo):
    ck=cpu({**origen,'pesos':m.state_dict(),'optimizador':opt.state_dict(),'scaler':sc.state_dict(),
        **estado_aleatorio(rg,ra),'v29':{'contrato':c,'rama':rama,'paso':k,'historial':historial,
            'rng_nuevo':rn,'rng_nuevo_sha256':huella(rn),'commit_previo_sha256':previo,'backend_efectivo':registro_efectivo(opt)}})
    ck['v29']['huellas_estado']=huellas_estado(ck)
    return ck


def comprobar(ck,c,rama):
    r=ck.get('v29',{})
    if r.get('contrato')!=c or r.get('rama')!=rama:raise ValueError('Checkpoint de otro protocolo/rama')
    k=r['paso'];hist=r['historial']
    if not 0<=k<=c['config']['actualizaciones'] or len(hist)!=k:raise ValueError('Presupuesto/historial incompatible')
    if [h['paso'] for h in hist]!=list(range(1,k+1)):raise ValueError('Historial discontinuo')
    if any(h['rama']!=rama or h['paso_adamw']!=2299+h['paso'] for h in hist):raise ValueError('AdamW/historial incompatible')
    if {int(s['step']) for s in ck['optimizador']['state'].values()}!={2299+k}:raise ValueError('Contador AdamW incorrecto')
    if r['huellas_estado']!=huellas_estado(ck) or r['rng_nuevo_sha256']!=huella(r['rng_nuevo']):raise ValueError('Estado alterado')
    verificar_registro(c['config']['backend_numerico'],r['backend_efectivo'])


class Diario:
    def __init__(self,rama,c,raiz=RAIZ):
        if rama not in ['A','B']:raise ValueError('Rama inválida')
        self.rama,self.c,self.raiz=rama,c,Path(raiz)
        self.base=self.raiz/'resultados/v29_packwise_click/entrenamiento'/rama
        self.cps=self.raiz/'checkpoints'/('v29_'+digest(c)[:12])/rama

    def leer(self):
        ck=None;previo=None;encontrado=False
        for k in self.c['config']['checkpoints']:
            p=self.base/'confirmaciones'/f'paso_{k:04d}.json'
            if not p.exists():encontrado=True;continue
            if encontrado:raise ValueError('Diario discontinuo')
            ev=leer(p);cp=self.raiz/ev['checkpoint']
            if sha(cp)!=ev['sha256_checkpoint'] or ev['previo']!=previo:raise ValueError('Cadena de commits alterada')
            ck=torch.load(cp,map_location='cpu',weights_only=True);comprobar(ck,self.c,self.rama)
            if ck['v29']['paso']!=k or ck['v29']['commit_previo_sha256']!=previo:raise ValueError('Checkpoint ajeno al commit')
            esperadas={h['archivo_traza']:h['sha256_traza'] for h in ck['v29']['historial']}
            if ev['trazas']!=esperadas:raise ValueError('Trazas/historial distintos')
            for tr,h in ev['trazas'].items():
                if sha(self.raiz/tr)!=h:raise ValueError('Traza alterada')
            previo=sha(p)
        return ck,previo

    def confirmar(self,ck,previo):
        comprobar(ck,self.c,self.rama);k=ck['v29']['paso']
        if k not in self.c['config']['checkpoints']:raise ValueError('Punto de guardado no autorizado')
        cp=self.cps/f'paso_{k:04d}_{uuid.uuid4().hex[:10]}.pt';publicar(cp,ck,tensor=True)
        ev={'checkpoint':cp.relative_to(self.raiz).as_posix(),'sha256_checkpoint':sha(cp),'previo':previo,
            'trazas':{h['archivo_traza']:h['sha256_traza'] for h in ck['v29']['historial']}}
        p=self.base/'confirmaciones'/f'paso_{k:04d}.json';publicar(p,ev);return sha(p)
