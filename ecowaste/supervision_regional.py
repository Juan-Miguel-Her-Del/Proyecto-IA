"""Pooling GT poligonal por instancia sobre características existentes, solo para train."""
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from ecowaste.datasets import CLASES, objetivo_deteccion

VERSION = 'regional_poligono_interior_fraccion_sin_ambiguos_v1'


def regiones_poligonales(objetivo, segmentos, semantica, lado=48):
    """Prefiere patches interiores; fallback de cobertura real sin fabricar regiones.

    Cada segmento tiene id, clase 0..5 y polígonos originales de una instancia.
    Padding/ignore y patches con más de un material se excluyen incluso en fallback.
    """
    g=objetivo['geometria'];res=g['lado']
    if res%lado or tuple(semantica.shape)!=(lado,lado):raise ValueError('Rejilla regional incompatible')
    if len({s['id'] for s in segmentos})!=len(segmentos):raise ValueError('Instancia regional duplicada')
    paso=res//lado
    valido=torch.zeros(res,res,dtype=torch.bool)
    x,y=g['relleno_x'],g['relleno_y']
    w,h=round(g['ancho_original']*g['escala_x']),round(g['alto_original']*g['escala_y'])
    valido[y:y+h,x:x+w]=True
    for b in objetivo['ignore_boxes'].cpu():
        x0,y0=torch.floor(b[:2]*res).long().clamp(0,res).tolist()
        x1,y1=torch.ceil(b[2:]*res).long().clamp(0,res).tolist()
        valido[y0:y1,x0:x1]=False
    # Se excluye entero un token que mezcla contenido con padding/ignore.
    tokens_validos=F.avg_pool2d(valido.float()[None,None],paso)[0,0]==1
    coberturas=[]
    for s in segmentos:
        if not 0<=s['clase']<6 or not s['poligonos']:raise ValueError('Material o polígonos regionales inválidos')
        lienzo=Image.new('L',(res,res));pincel=ImageDraw.Draw(lienzo)
        for p in s['poligonos']:
            if len(p)<6 or len(p)%2 or not np.isfinite(p).all():raise ValueError('Polígono regional inválido')
            xy=[(px*g['escala_x']+x,py*g['escala_y']+y) for px,py in zip(p[::2],p[1::2])]
            pincel.polygon(xy,fill=1)
        pixels=torch.from_numpy(np.array(lienzo,copy=True)).float()
        coberturas.append(F.avg_pool2d(pixels[None,None],paso)[0,0])
    por_material=torch.zeros(6,lado,lado,dtype=torch.bool)
    for s,c in zip(segmentos,coberturas):por_material[s['clase']]|=c>0
    ambiguos=por_material.sum(0)>1
    pesos=[];etiquetas=[];ids=[];detalles=[]
    for s,c in zip(segmentos,coberturas):
        candidatos=(c>0)&tokens_validos&~ambiguos&(semantica.cpu()==s['clase']+1)
        interior=candidatos&(c==1)
        modo='interior' if bool(interior.any()) else 'fraccion_poligono' if bool(candidatos.any()) else 'vacia'
        mascara=interior.float() if modo=='interior' else c*candidatos
        n=int((mascara>0).sum())
        razon=None
        if not n:
            razon='sin_pixeles_rasterizados' if not bool((c>0).any()) else 'sin_tokens_no_ambiguos_validos'
        detalles.append({'id':s['id'],'clase':s['clase'],'modo':modo,'tokens':n,
                         'cobertura_total_tokens':float(mascara.sum()),'motivo_omision':razon})
        if n:
            pesos.append(mascara);etiquetas.append(s['clase']);ids.append(s['id'])
    return {'pesos':torch.stack(pesos) if pesos else torch.empty(0,lado,lado),
            'labels':torch.tensor(etiquetas,dtype=torch.long),'ids':ids,'detalles':detalles}


def regiones_train(registro,contornos,lado=48):
    if registro.get('split')!='train':raise ValueError('La supervisión regional solo admite TACO train')
    if registro.get('fuente')!='taco':raise ValueError('No se generan regiones desde etiquetas globales')
    segmentos=[{'id':a['id_interno'],'clase':CLASES.index(a['clase']),
                'poligonos':contornos.contornos(registro,a)} for a in registro['cajas'] if a['clase']]
    return regiones_poligonales(objetivo_deteccion(registro),segmentos,contornos.rasterizar(registro,lado),lado)


def pesos_clase_regionales(conteos):
    n=torch.as_tensor(conteos,dtype=torch.float32)
    if n.shape!=(6,) or not bool(torch.isfinite(n).all()) or bool((n<0).any()) or n.sum()<=0:
        raise ValueError('Conteos regionales train inválidos')
    pesos=torch.where(n>0,n.clamp(min=1).rsqrt(),torch.zeros_like(n))
    # Contribución esperada media uno sobre regiones elegibles de train.
    return pesos/((pesos*n).sum()/n.sum())


def pool_regiones(caracteristicas,regiones,flips=None):
    if caracteristicas.ndim!=4 or len(regiones)!=len(caracteristicas):raise ValueError('Lote de regiones incompatible')
    b,h,w,d=caracteristicas.shape
    if flips is not None and len(flips)!=b:raise ValueError('Espejos regionales incompatibles')
    valores=[];labels=[];ids=[];imagenes=[]
    for i,r in enumerate(regiones):
        mask=r['pesos'].to(device=caracteristicas.device,dtype=torch.float32)
        if tuple(mask.shape[1:])!=(h,w) or len(mask)!=len(r['labels']) or len(mask)!=len(r['ids']):
            raise ValueError('Máscaras, materiales e IDs no corresponden')
        if not len(mask):continue
        if not bool(torch.isfinite(mask).all()) or bool((mask<0).any()):raise ValueError('Pesos regionales inválidos')
        if flips is not None and bool(flips[i]):mask=mask.flip(-1)
        den=mask.sum((1,2))
        if bool((den<=0).any()):raise ValueError('Una región elegible tiene máscara vacía')
        with torch.autocast(device_type=caracteristicas.device.type,enabled=False):
            v=mask.flatten(1)@caracteristicas[i].float().reshape(h*w,d)
            v=v/den[:,None]
        valores.append(v);labels.append(r['labels'].to(caracteristicas.device));ids.extend(r['ids']);imagenes.extend([i]*len(mask))
    vacio=caracteristicas.new_empty((0,d),dtype=torch.float32)+caracteristicas.float().sum()*0
    return {'vectores':torch.cat(valores) if valores else vacio,
            'labels':torch.cat(labels) if labels else torch.empty(0,dtype=torch.long,device=caracteristicas.device),
            'ids':ids,'indices_imagen':imagenes}


def perdida_regional(modelo,caracteristicas,regiones,pesos_clase,flips=None,con_gradiente=True):
    cabeza=modelo.clasificador_auxiliar
    if cabeza is None or cabeza.in_features!=caracteristicas.shape[-1] or cabeza.out_features!=6:
        raise ValueError('No hay cabeza auxiliar compatible; no se crea una arquitectura alternativa')
    if not con_gradiente:
        with torch.no_grad():return perdida_regional(modelo,caracteristicas.detach(),regiones,pesos_clase,flips,True)
    p=pool_regiones(caracteristicas,regiones,flips)
    logits=cabeza(p['vectores']).float();y=p['labels']
    w=torch.as_tensor(pesos_clase,dtype=torch.float32,device=logits.device)
    if w.shape!=(6,) or not bool(torch.isfinite(w).all()) or bool((w<0).any()):raise ValueError('Pesos de material inválidos')
    if len(y) and (int(y.min())<0 or int(y.max())>5 or bool((w[y]<=0).any())):raise ValueError('Material sin peso regional train')
    ce=F.cross_entropy(logits,y,reduction='none') if len(y) else logits.new_empty(0)
    den=w[y].sum() if len(y) else logits.new_tensor(1.)
    loss=(ce*w[y]).sum()/den if len(y) else logits.sum()*0
    detalle={'regiones':len(y),'denominador_pesos':float(den.detach()),'por_material':{}}
    for c,nombre in enumerate(CLASES):
        ix=y==c;n=int(ix.sum())
        detalle['por_material'][nombre]={'regiones':n,'ce_media':float(ce[ix].detach().mean()) if n else None,
            'numerador_ponderado':float((ce[ix]*w[c]).detach().sum()),
            'contribucion_ce':float((ce[ix]*w[c]).detach().sum()/den)}
    return loss,detalle,{'logits':logits,'labels':y,'ids':p['ids'],'indices_imagen':p['indices_imagen']}


def preparar_regiones(registros,contornos):
    """Instancias elegibles y frecuencias exclusivamente de los registros train."""
    regiones=[regiones_train(r,contornos) for r in registros]
    conteos=torch.zeros(6,dtype=torch.long);originales=conteos.clone()
    modos={c:Counter() for c in CLASES};omitidas=[]
    for registro,r in zip(registros,regiones):
        conteos+=torch.bincount(r['labels'],minlength=6)
        for d in r['detalles']:
            originales[d['clase']]+=1;modos[CLASES[d['clase']]][d['modo']]+=1
            if d['motivo_omision']:omitidas.append({'imagen':registro['id'],**d})
    pesos=pesos_clase_regionales(conteos)
    return regiones,pesos,{'version':VERSION,'imagenes_train':len(registros),
        'orden_materiales':CLASES,'instancias_originales':originales.tolist(),
        'instancias_elegibles':conteos.tolist(),'pesos_clase':pesos.tolist(),
        'formula_pesos':'1/sqrt(n_elegibles_train), normalizado a media por instancia 1',
        'modos_por_material':{c:dict(v) for c,v in modos.items()},'omitidas':omitidas,
        'todas_las_clases_elegibles':bool((conteos>0).all())}


def resumen_confusion(matriz):
    cm=torch.as_tensor(matriz,dtype=torch.long).cpu()
    if cm.shape!=(6,6) or bool((cm<0).any()):raise ValueError('Confusión regional inválida')
    n=int(cm.sum());carton=int(cm[2].sum())
    return {'orden':CLASES,'matriz':cm.tolist(),'regiones':n,
        'accuracy':int(cm.diag().sum())/n if n else None,
        'carton':{'regiones':carton,'a_plastico':int(cm[2,0]),'a_metal':int(cm[2,4]),
            'tasa_a_plastico_metal':int(cm[2,0]+cm[2,4])/carton if carton else None},
        'por_material':{c:{'regiones':int(cm[i].sum()),
            'recall':int(cm[i,i])/int(cm[i].sum()) if cm[i].sum() else None}
            for i,c in enumerate(CLASES)}}
