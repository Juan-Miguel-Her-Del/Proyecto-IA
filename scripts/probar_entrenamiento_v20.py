"""V22: piloto forense train-only por proceso; nunca publica aprobación de A/B."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time
import traceback
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from ecowaste.ablacion_fondo import huella,estado_aleatorio,pasos_adam
from ecowaste.reanudacion_regional import publicar,sha,huellas_estado
from ecowaste.reproducibilidad_v20 import cpu,gradientes,adam_nominal,comparar_arbol
from scripts.preparar_v20 import leer_cfg,protocolo,huellas_codigo,contexto_train,calcular_lote
from scripts.supervision_regional_materiales import restaurar,loaders
from scripts.ablacion_fondo import aumentar

CONFIG=RAIZ/'configuracion/diagnostico_reproducibilidad_v20.json'


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def validar_diagnostico():
    cfg=leer_cfg();p=leer(CONFIG);d=leer(RAIZ/p['diagnostico_regenerado'])
    if cfg['entrenamiento_completo_habilitado']:raise ValueError('A/B debe permanecer deshabilitado')
    if not d['aprobada'] or d['hashes_codigo']!=huellas_codigo() or d['protocolo']!=protocolo(cfg):raise ValueError('Regenerar diagnóstico antes del piloto')
    return cfg,p,d


def crear_plan():
    cfg,p,d=validar_diagnostico();ck,train,aux,*_=contexto_train(cfg)
    m,opt,sc,rng,ra,real=restaurar(ck)
    if real!=d['estado_inicial']:raise ValueError('Origen distinto')
    dl,da=loaders(train,aux,rng,ra,solo_indices=True);itaux=iter(da);plan=[]
    for (ix,) in dl:
        (ia,)=next(itaux);plan.append({'train':ix,'auxiliar':ia})
        if len(plan)==4:break
    publicar(RAIZ/p['salida']/'plan.pt',{'lotes':plan,'rng_loader':rng.get_state(),'rng_auxiliar':ra.get_state(),
        'huellas_origen':real,'sha256_origen':cfg['sha256_checkpoint_origen'],'sha256_diagnostico':sha(RAIZ/p['diagnostico_regenerado']),
        'hashes_codigo':huellas_codigo()},tensor=True)


def opciones(modo,opt):
    """Cada cambio es diagnóstico y queda limitado al proceso worker."""
    if modo=='determinista':
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Configurar cuBLAS antes de iniciar Python')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
    if modo in ['adam_serial','determinista']:
        for g in opt.param_groups:g.update(foreach=False,fused=False)
    return sdpa_kernel(SDPBackend.MATH) if modo in ['sdpa_math','determinista'] else nullcontext()


def entorno(opt):
    return {'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'gpu':torch.cuda.get_device_name(),'python':sys.version,'pid':os.getpid(),
        'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'sdpa':{'flash':torch.backends.cuda.flash_sdp_enabled(),'efficient':torch.backends.cuda.mem_efficient_sdp_enabled(),
            'math':torch.backends.cuda.math_sdp_enabled(),'cudnn':torch.backends.cuda.cudnn_sdp_enabled()},
        'adamw_grupos':[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],
        'threads':torch.get_num_threads(),'threads_interop':torch.get_num_interop_threads()}


def estado(ck,m,opt,sc,rng,ra,plan,k):
    return cpu({**ck,'pesos':m.state_dict(),'optimizador':opt.state_dict(),'scaler':sc.state_dict(),
        **estado_aleatorio(rng,ra),'diagnostico_reproducibilidad':{'plan':plan,'siguiente_paso':k+1}})


def paso_observado(m,opt,sc,x,y,ix,xa,ya,ia,regiones,pm,coef,modo,rng,ra,k,salida):
    """Mismo orden que paso_adamw; copias CPU y profiler hacen observable cada etapa."""
    inicio=time.perf_counter();m.train();rng0=estado_aleatorio(rng,ra)
    x,y,aug=aumentar(x.cuda().float()/255,y.cuda());opt.zero_grad(set_to_none=True)
    amp=modo!='sin_autocast';pre={'pesos':cpu(m.state_dict()),'adamw':adam_nominal(m,opt),'scaler':sc.state_dict(),
        'rng_inicio':{n:huella(v) for n,v in rng0.items()}}
    with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
        loss,raw,lo,ln,d=calcular_lote(m,x,y,[regiones[i] for i in ix.tolist()],pm,coef,aug['flip'].flatten())
    if not torch.isfinite(loss):raise RuntimeError('Loss TACO no finita')
    pre['logits_taco']=cpu(raw);pre['losses']={'original':cpu(lo),'discriminativa':cpu(ln),'taco_total':cpu(loss)}
    sc.scale(loss).backward();pre['gradientes_taco_escalados']=gradientes(m)
    xa,_,auga=aumentar(xa.cuda().float()/255)
    with torch.autocast('cuda',dtype=torch.float16,enabled=amp):logits=m.clasificar(xa)
    la=F.cross_entropy(logits.float(),ya.cuda())
    if not torch.isfinite(la):raise RuntimeError('Loss global no finita')
    pre['logits_auxiliar']=cpu(logits);pre['losses']['global']=cpu(la)
    sc.scale(.5*la).backward();pre['gradientes_total_escalados']=gradientes(m)
    sc.unscale_(opt);pre['gradientes_sin_escala']=gradientes(m)
    norma=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True,
        foreach=False if modo in ['clip_serial','determinista'] else None)
    pre['norma_antes_clip']=cpu(norma);pre['gradientes_clip']=gradientes(m)
    traza={'indices_train':ix,'aumentos_train':aug,'indices_aux':ia,'aumentos_aux':auga}
    pre['huella_entrada']=huella(traza);pre['entradas']=cpu(traza)
    # La evidencia pre-step se publica incluso si la actualización falla después.
    publicar(salida/f'paso_{k:02d}_pre.pt',pre,tensor=True)
    antes=pasos_adam(opt);sc.step(opt);sc.update()
    if pasos_adam(opt)!=antes+1:raise RuntimeError('Actualización omitida')
    post={'pesos':cpu(m.state_dict()),'adamw':adam_nominal(m,opt),'scaler':sc.state_dict(),
        'rng_fin':{n:huella(v) for n,v in estado_aleatorio(rng,ra).items()}}
    publicar(salida/f'paso_{k:02d}_post.pt',post,tensor=True)
    resumen={'paso':k,'losses':{n:float(v) for n,v in pre['losses'].items()},'norma_antes_clip':float(norma),
        'scaler_antes':pre['scaler'],'scaler_despues':post['scaler'],'paso_adamw':pasos_adam(opt),
        'huella_entrada':pre['huella_entrada'],'rng_inicio':pre['rng_inicio'],'rng_fin':post['rng_fin'],
        'huellas_adamw':{'antes':huella(pre['adamw']),'despues':huella(post['adamw'])},
        'sha256_pre':sha(salida/f'paso_{k:02d}_pre.pt'),'sha256_post':sha(salida/f'paso_{k:02d}_post.pt'),
        'segundos':time.perf_counter()-inicio,'detalle_regional':d}
    publicar(salida/f'paso_{k:02d}.json',resumen)
    return resumen


def worker(modo,rama,trayectoria,fase):
    cfg,p,d=validar_diagnostico();salida=RAIZ/p['salida']/modo/rama/trayectoria
    if fase=='inicio':
        if salida.exists():raise FileExistsError('Trayectoria existente')
        salida.mkdir(parents=True)
    elif not salida.is_dir() or (salida/'restauracion.json').exists():raise FileExistsError('Reanudación inexistente o repetida')
    ck,train,aux,regiones,pm,est=contexto_train(cfg);plan=torch.load(RAIZ/p['salida']/'plan.pt',weights_only=True)
    base=ck if fase=='inicio' else torch.load(salida/'checkpoint_paso_02.pt',map_location='cpu',weights_only=True)
    m,opt,sc,rng,ra,real=restaurar(base)
    if fase=='inicio':
        if real!=d['estado_inicial']:raise ValueError('Origen no exacto')
        rng.set_state(plan['rng_loader']);ra.set_state(plan['rng_auxiliar'])
    else:
        recargado=estado(base,m,opt,sc,rng,ra,plan,2)
        comp=comparar_arbol(base,recargado)
        publicar(salida/'restauracion.json',{'exacta':comp['exacto'],'comparacion':comp,'huellas_antes':huellas_estado(base),
            'huellas_recargadas':real,'sha256_checkpoint':sha(salida/'checkpoint_paso_02.pt'),'pid':os.getpid()})
        if not comp['exacto']:raise ValueError('Restauración no exacta')
    with opciones(modo,opt):
        publicar(salida/f'entorno_{fase}.json',entorno(opt));torch.cuda.reset_peak_memory_stats()
        pasos=range(1,3) if trayectoria=='R' and fase=='inicio' else range(3,5) if fase=='reanudar' else range(1,5)
        for k in pasos:
            ix=plan['lotes'][k-1]['train'];ia=plan['lotes'][k-1]['auxiliar']
            perfil=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if k==1 else nullcontext()
            with perfil as prof:
                r=paso_observado(m,opt,sc,train.base.imagenes[ix],train.mascaras[ix],ix,
                    torch.stack([aux[int(i)][0] for i in ia]),aux.etiquetas[ia],ia,regiones,pm,cfg['ramas'][rama],modo,rng,ra,k,salida)
            if k==1:
                publicar(salida/'operadores.json',[{'operador':ev.key,'llamadas':ev.count} for ev in prof.key_averages()
                    if any(t in ev.key.lower() for t in ['scaled_dot','flash','attention','foreach','adam','norm'])])
            if k==2:publicar(salida/'checkpoint_paso_02.pt',estado(ck,m,opt,sc,rng,ra,plan,k),tensor=True)
            print(modo,rama,trayectoria,k,r['losses'],flush=True)
        publicar(salida/f'fin_{fase}.json',{'pasos':list(pasos),'ultimo_adamw':pasos_adam(opt),
            'memoria_gpu_max_allocated_bytes':torch.cuda.max_memory_allocated(),'memoria_gpu_max_reserved_bytes':torch.cuda.max_memory_reserved(),
            'pesos_solo_diagnosticos_no_adoptables':True})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--crear-plan',action='store_true')
    p.add_argument('--modo',choices=leer(CONFIG)['modos']);p.add_argument('--rama',choices=['A','B'])
    p.add_argument('--trayectoria',choices=['C1','C2','R']);p.add_argument('--fase',choices=['inicio','reanudar'],default='inicio');a=p.parse_args()
    if a.crear_plan:crear_plan();return
    if not all([a.modo,a.rama,a.trayectoria]):p.error('Usar --crear-plan o modo/rama/trayectoria; no ejecuta A/B')
    try:worker(a.modo,a.rama,a.trayectoria,a.fase)
    except Exception:
        cfg=leer(CONFIG);out=RAIZ/cfg['salida']/f'error_{a.modo}_{a.rama}_{a.trayectoria}_{a.fase}.json'
        publicar(out,{'error':traceback.format_exc(),'pid':os.getpid(),'AB_habilitado':False});raise


if __name__=='__main__':main()
