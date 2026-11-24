"""Comparación exhaustiva de tensores; nunca detiene el registro por un isclose falso."""
import math
import torch

RTOL=1e-4
ATOL=1e-5


def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return x


def comparar_arbol(a,b,raiz=''):
    filas=[]
    def visitar(x,y,n):
        if torch.is_tensor(x) and torch.is_tensor(y):
            f={'nombre':n,'tipo':'tensor','dtype_a':str(x.dtype),'dtype_b':str(y.dtype),'forma':list(x.shape)}
            if x.shape!=y.shape or x.dtype!=y.dtype:
                filas.append({**f,'exacto':False,'estructura_compatible':False,'fuera_tolerancia':max(x.numel(),y.numel())});return
            x=x.cpu();y=y.cpu();f.update(estructura_compatible=True,elementos=x.numel(),exacto=torch.equal(x,y))
            if x.is_floating_point():
                aa=x.double();bb=y.double();delta=(aa-bb).abs();tol=ATOL+RTOL*bb.abs()
                finitos=torch.isfinite(aa)&torch.isfinite(bb);close=finitos&(delta<=tol)
                positivos=bb.abs()>0;rel=delta[positivos]/bb[positivos].abs()
                infinito_rel=bool(((~positivos)&(delta>0)).any());no_finitos=int((~finitos).sum())
                maxd=float(delta.max()) if delta.numel() and not no_finitos else (None if no_finitos else 0.)
                maxn=float((delta/tol).max()) if delta.numel() and not no_finitos else (None if no_finitos else 0.)
                f.update(fuera_tolerancia=int((~close).sum()),no_finitos=no_finitos,
                    max_diferencia_absoluta=maxd,max_diferencia_relativa=None if infinito_rel or no_finitos else (float(rel.max()) if rel.numel() else 0.),
                    relativa_infinita=infinito_rel,max_error_normalizado=maxn,
                    indice_max_absoluto=int(delta.flatten().argmax()) if delta.numel() else None,
                    norma_l2_diferencia=float(delta.norm()) if not no_finitos else None)
            else:f.update(fuera_tolerancia=int((x!=y).sum()),max_diferencia_absoluta=float((x.double()-y.double()).abs().max()) if x.numel() else 0.,max_error_normalizado=0. if f['exacto'] else None)
            filas.append(f)
        elif isinstance(x,dict) and isinstance(y,dict):
            for k in sorted(set(x)|set(y),key=str):
                if k not in x or k not in y:filas.append({'nombre':n+'.'+str(k),'tipo':'clave_ausente','exacto':False,'fuera_tolerancia':1})
                else:visitar(x[k],y[k],n+'.'+str(k) if n else str(k))
        elif isinstance(x,(tuple,list)) and isinstance(y,type(x)):
            if len(x)!=len(y):filas.append({'nombre':n,'tipo':'longitud','exacto':False,'fuera_tolerancia':1})
            for i,(xx,yy) in enumerate(zip(x,y)):visitar(xx,yy,f'{n}[{i}]')
        else:
            # Los float escalares también se comparan con la tolerancia original.
            if isinstance(x,float) and isinstance(y,float):
                visitar(torch.tensor(x,dtype=torch.float64),torch.tensor(y,dtype=torch.float64),n)
            else:filas.append({'nombre':n,'tipo':type(x).__name__,'exacto':x==y,'fuera_tolerancia':int(x!=y),'a':x,'b':y})
    visitar(a,b,raiz)
    return {'exacto':all(f['exacto'] for f in filas),'dentro_tolerancia':all(f['fuera_tolerancia']==0 for f in filas),
        'tensores':sum(f['tipo']=='tensor' for f in filas),'elementos_fuera_tolerancia':sum(f['fuera_tolerancia'] for f in filas),
        'max_diferencia_absoluta':max((f.get('max_diferencia_absoluta') or 0. for f in filas),default=0.),
        'filas':filas,'rtol':RTOL,'atol':ATOL}


def comparar_envolvente(control,reanudado):
    """Envolvente observada, conservadora; no es un intervalo estadístico."""
    aa={f['nombre']:f for f in control['filas']};fallos=[]
    for f in reanudado['filas']:
        c=aa.get(f['nombre']);n=f.get('max_error_normalizado')
        if c is None or f['tipo']!=c['tipo']:fallos.append(f['nombre']);continue
        if f['fuera_tolerancia']>c['fuera_tolerancia'] or (n is not None and n>max(1.,c.get('max_error_normalizado') or 0.)):
            fallos.append(f['nombre'])
        elif n is None and f['fuera_tolerancia']>0:fallos.append(f['nombre'])
    return {'cumple':not fallos and len(aa)==len(reanudado['filas']),'tensores_o_estados_que_exceden':fallos}


def adam_nominal(modelo,opt):
    nombres={id(p):n for n,p in modelo.named_parameters()}
    return {'estado':{nombres[id(p)]:cpu(v) for p,v in opt.state.items()},
        'grupos':[{**{k:cpu(v) for k,v in g.items() if k!='params'},'params':[nombres[id(p)] for p in g['params']]} for g in opt.param_groups]}


def gradientes(modelo):return {n:cpu(p.grad) if p.grad is not None else None for n,p in modelo.named_parameters()}
