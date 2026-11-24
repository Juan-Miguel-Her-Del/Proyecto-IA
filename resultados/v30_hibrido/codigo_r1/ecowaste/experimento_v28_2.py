"""Protocolo exploratorio independiente: plan fijo, estados completos y diarios inmutables."""
import copy
import json
from pathlib import Path
import time
import uuid
import numpy as np
import torch
from torch.nn import functional as F
from ecowaste.incorporacion import leer,sha,digest
from ecowaste.reanudacion_regional import publicar,restaurar_estados,huellas_estado
from ecowaste.ablacion_fondo import huella,estado_aleatorio,pasos_adam
from ecowaste.backend_numerico import politica,configurar_optimizador,registro_efectivo,verificar_registro
from ecowaste.detector_mapas import DetectorMapas,perdida_mapas
from ecowaste.reproducibilidad_v20 import cpu,gradientes,adam_nominal
from ecowaste.datos_v28_2 import historico_train,AuxiliarExperimental,auxiliar_global_train
from scripts.ablacion_fondo import aumentar

RAIZ=Path(__file__).resolve().parents[1]
SALIDA=RAIZ/'resultados/v28_2_exploratorio'
CONFIG=RAIZ/'configuracion/entrenamiento_exploratorio_v28_2.json'
CERTIFICADOS=SALIDA/'controles_inferencia_cuda_v2'


def validar(cfg):
    fijo={'version':'v28_2_exploratorio_1','entrenamiento_autorizado':True,'actualizaciones':300,
        'batch_size':8,'historicas_comunes':6,'ranuras_sustituidas':2,'semilla_auxiliar':2801,
        'learning_rate':.0005,'weight_decay':.01,'coeficiente_fondo':3.,'coeficiente_auxiliar_global':.5,
        'auxiliar_batch':16,'clipping':1.,'nueva_perdida':False,'backend_numerico':politica('completo'),
        'warmup_nuevo':False,'parada_temprana':False,'vistas':[1,5],'umbral':.05,'visual':.35,
        'area_minima':2,'nms':.5,'top':100,'politica':'argmax7','pasos_checkpoint':[0,100,200,300],
        'seleccion_mejor_checkpoint':False,'evaluar_val':False,'evaluar_test':False,
        'generalizacion':False,'adopcion_automatica':False,'presupuesto_no_ampliable':True}
    if any(cfg.get(k)!=v for k,v in fijo.items()):raise ValueError('Protocolo V28.2 alterado')
    if cfg['checkpoint_origen']!='checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt':raise ValueError('Origen distinto')
    if sha(RAIZ/cfg['checkpoint_origen'])!=cfg['sha256_origen']:raise ValueError('Pesos originales alterados')
    if sha(RAIZ/cfg['coleccion']/'MANIFIESTO.json')!=cfg['sha256_coleccion']:raise ValueError('Colección distinta')


def codigo():
    # Todas las dependencias propias del entrenador, lector, evaluador y observador.
    paths=list((RAIZ/'ecowaste').glob('*.py'))
    paths += [RAIZ/'scripts'/s for s in ['preparar_v28_2.py','entrenar_v28_2.py','evaluar_v28_2.py',
        'verificar_datos_v28_2.py','ablacion_fondo.py','entrenar_detector_mapas.py','entrenar_clasificador.py','auditar_propuestas_internas.py']]
    paths += [RAIZ/'pruebas/test_experimento_v28_2.py']
    return {p.relative_to(RAIZ).as_posix():sha(p) for p in paths}


def indices_ciclicos(n,cantidad,rng):
    if n<1:raise ValueError('Colección vacía')
    return torch.cat([torch.randperm(n,generator=rng) for _ in range((cantidad+n-1)//n)])[:cantidad]


def crear_plan(ck,n_hist,n_aux,etiquetas,cfg):
    rg=torch.Generator();rg.set_state(ck['rng_loader'])
    ra=torch.Generator();ra.set_state(ck['rng_auxiliar'])
    rn=torch.Generator().manual_seed(cfg['semilla_auxiliar']);n=cfg['actualizaciones']
    hist=indices_ciclicos(n_hist,n*8,rg).reshape(n,8)
    aux=indices_ciclicos(n_aux,n*2,rn).reshape(n,2)
    cuentas=torch.bincount(etiquetas,minlength=6)
    if bool((cuentas==0).any()):raise ValueError('Auxiliar global sin clase')
    glob=torch.multinomial(1/cuentas[etiquetas].double(),n*16,True,generator=ra).reshape(n,16)
    return {'historico':hist,'nuevo':aux,'global':glob,'rng_plan_final':{'historico':rg.get_state(),
        'nuevo':rn.get_state(),'global':ra.get_state()},'regla':'barajado cíclico sin lote corto; A8, B6+2; global16 balanceado',
        'version':cfg['version']}


def lote_plan(plan,rama,k):
    if rama not in ['A','B'] or not 0<=k<len(plan['historico']):raise ValueError('Rama o presupuesto inválido')
    return {'historico':plan['historico'][k,:8 if rama=='A' else 6],
        'nuevo':plan['nuevo'][k] if rama=='B' else torch.empty(0,dtype=torch.long),'global':plan['global'][k]}


def contexto():
    cfg=leer(CONFIG);validar(cfg);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise ValueError('El piloto requiere CUDA')
    ck=torch.load(RAIZ/cfg['checkpoint_origen'],map_location='cpu',weights_only=True)
    if ck['epoca']!=19 or ck['recortes_train'] is not None or ck['pesos_externos']:raise ValueError('Origen incompatible')
    ht=historico_train(RAIZ);nu=AuxiliarExperimental(RAIZ,RAIZ/cfg['coleccion']);gl=auxiliar_global_train(RAIZ)
    n=torch.bincount(ht.mascaras[ht.mascaras!=255].flatten(),minlength=7)[1:].float()
    pm=n.clamp(min=1e-6).rsqrt();pm=(pm/pm.mean()).clamp(max=3)
    return cfg,ck,ht,nu,gl,pm.cuda()


def restaurar(ck,cfg):
    m=DetectorMapas(**ck['configuracion']).cuda()
    opt=torch.optim.AdamW(m.parameters(),lr=.0005,weight_decay=.01)
    sc=torch.amp.GradScaler('cuda');rg=torch.Generator();ra=torch.Generator()
    exacto=restaurar_estados(m,opt,sc,ck,rg,ra)
    configurar_optimizador(cfg['backend_numerico'],opt)
    verificar_registro(cfg['backend_numerico'],registro_efectivo(opt))
    return m,opt,sc,rg,ra,exacto


def paso(m,opt,sc,ht,nu,gl,pm,plan,rama,k,rg,ra,observar=False):
    t=time.perf_counter();ix=lote_plan(plan,rama,k);ih,ine,ia=ix.values()
    x=ht.imagenes[ih];y=ht.mascaras[ih]
    if len(ine):x=torch.cat([x,nu.imagenes[ine]]);y=torch.cat([y,nu.mascaras[ine]])
    m.train();opt.zero_grad(set_to_none=True)
    rng0=huella(estado_aleatorio(rg,ra))
    x,y,aug=aumentar(x.cuda().float()/255,y.cuda())
    with torch.autocast('cuda',dtype=torch.float16):
        logits=m(x);lm=perdida_mapas(logits,y,pm,formulacion='softmax7',coeficiente_fondo=3.)
    if not bool(torch.isfinite(lm)):raise RuntimeError('Loss mapas no finita')
    prueba={'logits_mapas':cpu(logits),'loss_mapas':cpu(lm)} if observar else None
    sc.scale(lm).backward()
    # Copia explícita desde mmap de solo lectura: jamás escribir la caché histórica.
    xa=torch.from_numpy(np.array(gl.imagenes[ia.numpy()],copy=True));ya=gl.etiquetas[ia].cuda()
    xa,_,auga=aumentar(xa.cuda().float()/255)
    with torch.autocast('cuda',dtype=torch.float16):la_raw=m.clasificar(xa)
    la=F.cross_entropy(la_raw.float(),ya)
    if not bool(torch.isfinite(la)):raise RuntimeError('Loss auxiliar no finita')
    sc.scale(.5*la).backward();sc.unscale_(opt)
    if observar:prueba.update(logits_global=cpu(la_raw),loss_global=cpu(la),gradientes=gradientes(m))
    norma=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True,foreach=False)
    if observar:prueba['gradientes_clip']=gradientes(m)
    previo=pasos_adam(opt);escala=sc.state_dict();sc.step(opt);sc.update()
    if pasos_adam(opt)!=previo+1:raise RuntimeError('AdamW omitió una actualización; detener sin ampliar presupuesto')
    entradas={'indices':ix,'aumentos_mapas':aug,'aumentos_global':auga}
    torch.cuda.synchronize()
    r={'paso':k+1,'rama':rama,'loss_mapas':float(lm),'loss_global':float(la),'loss_total':float(lm+.5*la),
        'norma_gradiente_pre_clip':float(norma),'scaler_antes':escala,'scaler_despues':sc.state_dict(),
        'paso_adamw':pasos_adam(opt),'rng_antes':rng0,'rng_despues':huella(estado_aleatorio(rg,ra)),
        'entradas':{s:v.tolist() for s,v in ix.items()},'huella_entradas_y_aumentos':huella(entradas),
        'huella_aumentos_comunes':huella({'mapas':aug,'global':auga}),
        'logits_sha256':huella(logits),'gradientes_finitos':True,'segundos':time.perf_counter()-t}
    if observar:prueba.update(entradas=cpu(entradas),pesos=cpu(m.state_dict()),adamw=adam_nominal(m,opt),
        scaler=copy.deepcopy(sc.state_dict()),rng=cpu(estado_aleatorio(rg,ra)),norma=cpu(norma))
    return r,prueba


def instantanea(origen,m,opt,sc,rg,ra,rama,k,historial,contrato,previo):
    ck=cpu({**origen,'pesos':m.state_dict(),'optimizador':opt.state_dict(),'scaler':sc.state_dict(),
        **estado_aleatorio(rg,ra),'v28_2':{'contrato':contrato,'rama':rama,'paso':k,
        'historial':historial,'commit_previo_sha256':previo,'backend_efectivo':registro_efectivo(opt)}})
    ck['v28_2']['huellas_estado']=huellas_estado(ck)
    return ck


def comprobar_estado(ck,contrato,rama):
    r=ck.get('v28_2',{})
    if r.get('contrato')!=contrato or r.get('rama')!=rama:raise ValueError('Checkpoint de otro protocolo o rama')
    k=r['paso']
    if k not in [0,100,200,300] or len(r['historial'])!=k:raise ValueError('Presupuesto o historial inválido')
    if [h['paso'] for h in r['historial']]!=list(range(1,k+1)):raise ValueError('Pasos de historial incompletos')
    if any(h['rama']!=rama or h['paso_adamw']!=2299+h['paso'] for h in r['historial']):raise ValueError('Contadores incompatibles')
    if {int(s['step']) for s in ck['optimizador']['state'].values()}!={2299+k}:raise ValueError('AdamW incompatible')
    if r['huellas_estado']!=huellas_estado(ck):raise ValueError('Estado alterado')
    verificar_registro(contrato['config']['backend_numerico'],r['backend_efectivo'])


class Diario:
    def __init__(self,rama,contrato,raiz=RAIZ):
        if rama not in ['A','B']:raise ValueError('Rama inválida')
        self.rama=rama;self.contrato=contrato;self.raiz=Path(raiz)
        self.base=self.raiz/'resultados/v28_2_exploratorio'/rama
        self.cps=self.raiz/'checkpoints'/('v28_2_'+digest(contrato)[:12])/rama

    def leer(self):
        ck=None;previo=None
        for k in [0,100,200,300]:
            p=self.base/'confirmaciones'/f'paso_{k:04d}.json'
            if not p.exists():
                if any((self.base/'confirmaciones'/f'paso_{j:04d}.json').exists() for j in [0,100,200,300] if j>k):raise ValueError('Diario discontinuo')
                break
            ev=leer(p);cp=self.raiz/ev['checkpoint']
            if sha(cp)!=ev['sha256_checkpoint'] or ev['previo']!=previo:raise ValueError('Cadena de checkpoint alterada')
            for tr,h in ev['trazas'].items():
                if sha(self.raiz/tr)!=h:raise ValueError('Traza alterada')
            ck=torch.load(cp,map_location='cpu',weights_only=True);comprobar_estado(ck,self.contrato,self.rama)
            if ck['v28_2']['paso']!=k or ck['v28_2']['commit_previo_sha256']!=previo:raise ValueError('Otro commit')
            if ev['trazas']!={h['archivo_traza']:h['sha256_traza'] for h in ck['v28_2']['historial']}:raise ValueError('Trazas e historial distintos')
            previo=sha(p)
        return ck,previo

    def confirmar(self,ck,previo):
        comprobar_estado(ck,self.contrato,self.rama);k=ck['v28_2']['paso']
        cp=self.cps/f'paso_{k:04d}_{uuid.uuid4().hex[:12]}.pt';publicar(cp,ck,tensor=True)
        ev={'checkpoint':cp.relative_to(self.raiz).as_posix(),'sha256_checkpoint':sha(cp),'previo':previo,
            'trazas':{h['archivo_traza']:h['sha256_traza'] for h in ck['v28_2']['historial']}}
        p=self.base/'confirmaciones'/f'paso_{k:04d}.json';publicar(p,ev);return sha(p)
