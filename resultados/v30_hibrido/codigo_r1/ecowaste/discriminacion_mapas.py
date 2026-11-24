"""V20: margen material sobre logits de la cabeza principal, nunca auxiliar."""
import math
import torch
from torch.nn import functional as F
from ecowaste.datasets import CLASES

VERSION='margen_huber_rival_max_top025_instancia_v1'
PESOS=(1.,1.,2.,2.,1.,1.)


def perdida_discriminativa(logits,regiones,flips=None,con_gradiente=True):
    """Top 25 % de violaciones (margen=1); empate por índice espacial.

    Selección sin gradiente. Huber hinge con pendiente <=1. Se incluyen las
    instancias fáciles en el denominador; no se renormaliza por activas.
    """
    if not con_gradiente:
        with torch.no_grad():return perdida_discriminativa(logits.detach(),regiones,flips,True)
    if logits.ndim!=4 or logits.shape[1]!=7 or len(logits)!=len(regiones):raise ValueError('Mapas o regiones incompatibles')
    if flips is not None and len(flips)!=len(logits):raise ValueError('Flip incompatible')
    with torch.autocast(device_type=logits.device.type,enabled=False):
        raw=logits.float();cero=raw.sum()*0;numerador=cero;den=0.;filas=[]
        if not torch.isfinite(raw).all():raise ValueError('Logits no finitos')
        for b,r in enumerate(regiones):
            masks=r['pesos'].to(raw.device,dtype=torch.float32)
            if tuple(masks.shape[1:])!=tuple(raw.shape[2:]) or len(masks)!=len(r['labels']) or len(masks)!=len(r['ids']):
                raise ValueError('No corresponden máscaras, IDs y etiquetas')
            if not torch.isfinite(masks).all() or (masks<0).any() or (masks>1).any():raise ValueError('Máscaras inválidas')
            if len(set(r['ids']))!=len(r['ids']):raise ValueError('IDs regionales duplicados')
            if flips is not None and bool(flips[b]):masks=masks.flip(-1)
            mapa=raw[b,1:].permute(1,2,0).reshape(-1,6)
            for mask,y,ident in zip(masks,r['labels'].tolist(),r['ids']):
                if not 0<=y<6:raise ValueError('Material fuera de rango')
                indices=torch.where(mask.flatten()>0)[0]
                if not len(indices):raise ValueError('Región elegible vacía')
                z=mapa[indices];sin_gt=z.detach().clone();sin_gt[:,y]=-torch.inf
                rivales=sin_gt.argmax(1)
                gap=z.gather(1,rivales[:,None]).squeeze(1)-z[:,y]
                violacion=1.+gap
                # Ranking detached: no RNG y no derivada a través de top-k/argmax.
                orden=torch.argsort(violacion.detach(),descending=True,stable=True)
                k=max(1,math.ceil(.25*len(indices)));seleccion=orden[:k]
                seleccion=seleccion[violacion.detach()[seleccion]>0]
                if len(seleccion):
                    v=violacion[seleccion];rho=torch.where(v<=1,.5*v.square(),v-.5)
                    pesos=mask.flatten()[indices[seleccion]]
                    loss=(rho*pesos).sum()/pesos.sum()
                else:loss=cero
                peso=PESOS[y];numerador=numerador+peso*loss;den+=peso
                filas.append({'imagen_lote':b,'id_region':ident,'clase':y,'material':CLASES[y],
                    'peso_clase':peso,'tokens_validos':len(indices),'tokens_seleccionados':len(seleccion),
                    'indices_tokens':indices[seleccion].detach().cpu().tolist(),
                    'rivales':rivales[seleccion].detach().cpu().tolist(),
                    'loss_instancia':float(loss.detach()),'numerador_ponderado':float((peso*loss).detach())})
        loss=numerador/den if den else cero
    for f in filas:
        f['fraccion_peso_instancia']=f['peso_clase']/den
        f['contribucion']=f['numerador_ponderado']/den
        f['cota_l1_grad_logits']=2*f['peso_clase']/den
    return loss,{'version':VERSION,'instancias':len(filas),'denominador_instancias':den,'filas':filas,
        'por_material':{c:{'instancias':sum(f['material']==c for f in filas),
            'activas':sum(f['material']==c and f['tokens_seleccionados']>0 for f in filas),
            'tokens_seleccionados':sum(f['tokens_seleccionados'] for f in filas if f['material']==c),
            'contribucion':sum(f['contribucion'] for f in filas if f['material']==c)} for c in CLASES}}


def ce_seleccionada(logits,regiones,detalle,solo_materiales=False,flips=None):
    """Control de redundancia con exactamente los mismos tokens/pesos, no otra propuesta."""
    raw=logits.float();num=raw.sum()*0;den=detalle['denominador_instancias']
    for f in detalle['filas']:
        if not f['indices_tokens']:continue
        b=f['imagen_lote'];r=regiones[b];pos=r['ids'].index(f['id_region'])
        mask=r['pesos'][pos].to(raw.device)
        if flips is not None and bool(flips[b]):mask=mask.flip(-1)
        ix=torch.tensor(f['indices_tokens'],device=raw.device);c=f['clase']
        z=raw[b].permute(1,2,0).reshape(-1,7)[ix]
        if solo_materiales:z=z[:,1:]
        ce=F.cross_entropy(z,torch.full((len(ix),),c if solo_materiales else c+1,dtype=torch.long,device=raw.device),reduction='none')
        w=mask.flatten()[ix];num+=f['peso_clase']*(ce*w).sum()/w.sum()
    return num/den if den else num


def comparar_gradientes(a,b):
    a=a.detach().float().flatten();b=b.detach().float().flatten()
    na=float(a.norm());nb=float(b.norm());dot=float(a@b)
    alpha=dot/(nb*nb) if nb else 0.;residuo=a-alpha*b
    return {'norma_nuevo':na,'norma_referencia':nb,'coseno':dot/(na*nb) if na and nb else None,
        'alpha_proyeccion':alpha,'fraccion_ortogonal':float(residuo.norm())/na if na else None,
        'producto':dot}


def descomponer_original(logits,etiquetas,pesos_clase):
    """Auditoría numérica; no reemplaza ni modifica perdida_mapas."""
    raw=logits.float();valida=etiquetas!=255;fg=(etiquetas>0)&valida;bg=(etiquetas==0)&valida
    cero=raw.sum()*0;ce=F.cross_entropy(raw,etiquetas,ignore_index=255,reduction='none')
    w=pesos_clase[etiquetas[fg]-1] if bool(fg.any()) else raw.new_empty(0)
    foreground=(ce[fg]*w).sum()/w.sum() if bool(fg.any()) else cero
    fondo=ce[bg].mean() if bool(bg.any()) else cero
    probs=raw.softmax(1)[:,1:]*valida[:,None]
    gt=F.one_hot(etiquetas.clamp(max=6),7)[...,1:].permute(0,3,1,2).float()*valida[:,None]
    presentes=gt.sum((0,2,3))>0
    ds=1-(2*(probs*gt).sum((0,2,3))+1e-6)/(probs.square().sum((0,2,3))+gt.square().sum((0,2,3))+1e-6)
    dice=ds[presentes].mean() if presentes.any() else cero
    return {'ce_foreground':foreground,'ce_fondo':fondo,'dice':dice,'total':foreground+3*fondo+dice}


@torch.inference_mode()
def metricas_instancias(logits,regiones):
    """Solo cabeza principal: votos por token e instancias, material condicionado y fondo."""
    raw=logits.float();cm=torch.zeros(7,7,dtype=torch.float64);cc=torch.zeros(6,6,dtype=torch.float64)
    ci=torch.zeros(7,7,dtype=torch.long);cci=torch.zeros(6,6,dtype=torch.long);filas=[]
    for b,r in enumerate(regiones):
        z=raw[b].permute(1,2,0).reshape(-1,7)
        for mask,c,ident in zip(r['pesos'],r['labels'].tolist(),r['ids']):
            w=mask.flatten().to(z.device);ix=w>0;w=w[ix];zz=z[ix]
            pred=zz.argmax(1);cond=zz[:,1:].argmax(1)
            votos=torch.bincount(pred,weights=w,minlength=7)/w.sum()
            vc=torch.bincount(cond,weights=w,minlength=6)/w.sum()
            cm[c+1]+=votos.cpu().double();cc[c]+=vc.cpu().double()
            ip=int(votos.argmax());ic=int(vc.argmax());ci[c+1,ip]+=1;cci[c,ic]+=1
            filas.append({'imagen_lote':b,'id_region':ident,'clase':c,'pred_principal':ip,
                'pred_condicional':ic,'fraccion_correcta':float(votos[c+1]),'fraccion_fondo':float(votos[0]),
                'fraccion_material_equivocado':float(votos[1:].sum()-votos[c+1]),
                'recuperada_mayoria_gt':bool(votos[c+1]>=.5),'condicional_correcta_mayoria':bool(vc[c]>=.5)})
    return {'orden_siete':['fondo',*CLASES],'orden_materiales':CLASES,
        'confusion_token_normalizada_por_instancia':cm.tolist(),'confusion_condicional_token_normalizada_por_instancia':cc.tolist(),
        'confusion_instancia_voto_principal':ci.tolist(),'confusion_instancia_voto_condicional':cci.tolist(),
        'instancias':len(filas),'filas':filas,
        'por_material':{c:{'instancias':sum(f['clase']==i for f in filas),
            'recuperadas':sum(f['clase']==i and f['recuperada_mayoria_gt'] for f in filas),
            'condicionales_recuperadas':sum(f['clase']==i and f['condicional_correcta_mayoria'] for f in filas)} for i,c in enumerate(CLASES)}}
