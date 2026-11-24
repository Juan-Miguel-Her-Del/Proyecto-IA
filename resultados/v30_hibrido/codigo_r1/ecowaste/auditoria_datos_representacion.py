"""V25: censos y contrafactuales GT offline. Nunca modifica el detector."""
from collections import Counter
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
from ecowaste.datasets import CLASES,objetivo_deteccion
from ecowaste.poligonos import rasterizar_contornos
from ecowaste.supervision_regional import regiones_poligonales
from ecowaste.auditoria_deteccion import tamano_objeto

VERSION='v25_censo_gt_48_96_384_sin_entrenamiento_v1'


def iou_np(a,b):
    a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(4)
    it=np.maximum(0,np.minimum(a[:,2:],b[2:])-np.maximum(a[:,:2],b[:2])).prod(1)
    return it/np.maximum(1e-12,(a[:,2:]-a[:,:2]).clip(0).prod(1)+(b[2:]-b[:2]).clip(0).prod()-it)


def cruces_propios(p):
    """Cruces estrictos entre aristas no adyacentes; no certifica poligonos simples."""
    q=np.asarray(p,float).reshape(-1,2)
    if len(q)>1 and np.array_equal(q[0],q[-1]):q=q[:-1]
    a=q;b=np.roll(q,-1,axis=0);n=len(q);total=0
    def cruz(x,y):return x[...,0]*y[...,1]-x[...,1]*y[...,0]
    for i in range(n):
        js=np.arange(i+2,n)
        if i==0:js=js[js!=n-1]
        if not len(js):continue
        c,d=a[js],b[js]
        total+=int(((cruz(b[i]-a[i],c-a[i])*cruz(b[i]-a[i],d-a[i])<-1e-9)&
            (cruz(d-c,a[i]-c)*cruz(d-c,b[i]-c)<-1e-9)).sum())
    return total


def mascara_instancia(poligonos,g):
    im=Image.new('L',(384,384));d=ImageDraw.Draw(im)
    for p in poligonos:
        d.polygon([(x*g['escala_x']+g['relleno_x'],y*g['escala_y']+g['relleno_y']) for x,y in zip(p[::2],p[1::2])],fill=1)
    return np.asarray(im,dtype=np.uint8)


def componentes_semanticos(y,clase,g,area_minima=2):
    n,cc,stats,_=cv2.connectedComponentsWithStats((y==clase+1).astype(np.uint8),connectivity=8)
    ids=[j for j in range(1,n) if stats[j,cv2.CC_STAT_AREA]>=area_minima]
    lado=y.shape[0];boxes=[]
    x0,y0=g['relleno_x']/384,g['relleno_y']/384
    x1=x0+g['ancho_original']*g['escala_x']/384;y1=y0+g['alto_original']*g['escala_y']/384
    for j in ids:
        x,y,w,h,_=stats[j];boxes.append([max(x/lado,x0),max(y/lado,y0),min((x+w)/lado,x1),min((y+h)/lado,y1)])
    return cc,ids,np.asarray(boxes).reshape(-1,4)


def oraculo_rejilla(objetivo,segmentos,lado):
    """Contrafactual semantico GT -> C8, sin scores/NMS. No es techo absoluto de AP.

    La entrada permanece 384. 96/384 solo cambian discretizacion, no crean detalle RGB.
    Conserva politica original de etiquetas y exclusión por centro de padding.
    """
    if lado not in [48,96,384]:raise ValueError('Rejilla de diagnostico no prefijada')
    g=objetivo['geometria'];y=rasterizar_contornos(objetivo,[(s['poligonos'],s['clase']) for s in segmentos],lado,1/64).numpy()
    yy,xx=np.meshgrid((np.arange(lado)+.5)/lado,(np.arange(lado)+.5)/lado,indexing='ij')
    x0,y0=g['relleno_x']/384,g['relleno_y']/384
    x1=x0+g['ancho_original']*g['escala_x']/384;y1=y0+g['alto_original']*g['escala_y']/384
    valido=(xx>=x0)&(xx<=x1)&(yy>=y0)&(yy<=y1)
    fuera=int(((y>0)&(y<7)&~valido).sum());y[~valido]=255
    componentes={c:componentes_semanticos(y,c,g) for c in range(6)}
    filas=[];por_componente={}
    for k,s in enumerate(segmentos):
        m=mascara_instancia(s['poligonos'],g)
        cobertura=cv2.resize(m.astype(np.float32),(lado,lado),interpolation=cv2.INTER_AREA)
        cc,ids,boxes=componentes[s['clase']];den=float(cobertura.sum())
        asociaciones=[j for j in ids if float(cobertura[cc==j].sum())>0]
        mayoritarias=[j for j in ids if den>0 and float(cobertura[cc==j].sum())/den>=.5]
        for j in mayoritarias:por_componente.setdefault((s['clase'],j),[]).append(k)
        propia=np.zeros_like(y);propia[(cobertura>0)&valido]=s['clase']+1
        _,_,bprop=componentes_semanticos(propia,s['clase'],g)
        box=objetivo['boxes'][k].numpy()
        filas.append({'id':s['id'],'clase':s['clase'],'tokens_tocados':int((cobertura>0).sum()),
            'tokens_semanticos_correctos':int(((cobertura>0)&(y==s['clase']+1)).sum()),
            'masa_poligono_tokens':den,'componentes_tocados':len(asociaciones),
            'iou_semantica_max':float(iou_np(boxes,box).max()) if len(boxes) else 0.,
            'iou_instancia_separada_max':float(iou_np(bprop,box).max()) if len(bprop) else 0.,
            'fusion_mayoritaria':False,'fusion_toca':False})
    # Distinguimos contacto incidental (informe 10) de >=50% de cobertura (informe 19).
    por_contacto={}
    for k,s in enumerate(segmentos):
        cov=cv2.resize(mascara_instancia(s['poligonos'],g).astype(np.float32),(lado,lado),interpolation=cv2.INTER_AREA)
        cc,ids,_=componentes[s['clase']]
        for j in ids:
            if (cov[cc==j]>0).any():por_contacto.setdefault((s['clase'],j),[]).append(k)
    for grupos,campo in [(por_componente,'fusion_mayoritaria'),(por_contacto,'fusion_toca')]:
        for ks in grupos.values():
            if len(ks)>1:
                for k in ks:filas[k][campo]=True
    return {'lado':lado,'positivos_fuera_contenido_antes_recorte':fuera,'filas':filas}


def censar_escena(r,contornos):
    """Incluye positivas e ignoradas, conserva ID y categoria original y valida procedencia."""
    o=objetivo_deteccion(r);segmentos=[];filas=[]
    for a in r['cajas']:
        ps=contornos.contornos(r,a);original=contornos.anotaciones[a['id_interno']-1]
        pts=np.concatenate([np.asarray(p).reshape(-1,2) for p in ps]);lo=pts.min(0);hi=pts.max(0)
        x,y,w,h=a['bbox'];pb=[*lo,*hi];bb=[x,y,x+w,y+h]
        fuera=bool((pts<0).any() or (pts[:,0]>r['ancho']).any() or (pts[:,1]>r['alto']).any())
        area=sum(abs(float(cv2.contourArea(np.asarray(p,np.float32).reshape(-1,2)))) for p in ps)
        fila={'imagen':r['id'],'split':r['split'],'id':a['id_interno'],'id_original':a['id_original'],
            'clase':a['clase'],'categoria':a['categoria_original'],'ruta':r['ruta'],
            'grupo':r['grupo'],'adquisicion':r['grupo_adquisicion'],'fuente':r['fuente'],
            'ancho_imagen':r['ancho'],'alto_imagen':r['alto'],'bbox':a['bbox'],
            'imagen_anotacion_original':original['image_id'],'procedencia_verificada':True,
            'iscrowd':a['iscrowd'],'poligonos':len(ps),'vertices':sum(len(p)//2 for p in ps),
            'poligono_fuera_imagen':fuera,'bbox_fuera_imagen':x<0 or y<0 or x+w>r['ancho'] or y+h>r['alto'],
            'bbox_recortada_en_preparacion':a['bbox']!=a['bbox_original'],
            'bbox_poligono_iou':float(iou_np([pb],bb)[0]),'area_poligono_geometrica':area,
            'area_coco':original.get('area'),'fraccion_poligono_en_bbox':area/max(w*h,1e-12),
            'cruces_estrictos':sum(cruces_propios(p) for p in ps),
            'poligono_degenerado':area<=0,'toca_borde_imagen':bool((pts[:,0]<=1).any() or (pts[:,1]<=1).any() or
                (pts[:,0]>=r['ancho']-1).any() or (pts[:,1]>=r['alto']-1).any()),
            'posible_multimaterial_por_categoria':a['categoria_original'] in ['Meal carton','Pizza box','Other carton','Other plastic wrapper','Aluminium foil'],
            'revision_material':'sin_adjudicacion_humana_no_reetiquetado'}
        if a['clase']:
            indice=len(segmentos);fila.update(tamano_objeto(o['boxes'][indice]))
            segmentos.append({'id':a['id_interno'],'clase':CLASES.index(a['clase']),'poligonos':ps})
        filas.append(fila)
    sem=contornos.rasterizar(r);reg=regiones_poligonales(o,segmentos,sem)
    detalles={f['id']:f for f in reg['detalles']}
    for f in filas:
        if f['clase']:f['region_elegible']=detalles[f['id']]
    solapes=[]
    aa=[a for a in r['cajas'] if a['clase']]
    for i,a in enumerate(aa):
        for j in range(i+1,len(aa)):
            iu=float(iou_np(o['boxes'][j:j+1].numpy(),o['boxes'][i].numpy())[0])
            if iu>=.5:solapes.append({'imagen':r['id'],'a':a['id_interno'],'b':aa[j]['id_interno'],'iou':iu,'mismo_material':a['clase']==aa[j]['clase']})
    oraculos=[oraculo_rejilla(o,segmentos,n) for n in [48,96,384]]
    # Mide la politica real sin cambiarla: positivos que pisan ignoradas y padding.
    vacio=dict(o,boxes=torch.empty(0,4),labels=torch.empty(0,dtype=torch.long))
    from ecowaste.detector_mapas import rasterizar
    ignorado=rasterizar(vacio)==255
    sobreescritos=int(((sem>0)&(sem<7)&ignorado).sum())
    return filas,oraculos,solapes,{'tokens_positivos_en_zonas_previamente_ignoradas':sobreescritos}


def resumir_censo(filas):
    def resumir(ff):
        return {'instancias':len(ff),'escenas':len({f['imagen'] for f in ff}),
            'categorias':dict(Counter(f['categoria'] for f in ff)),
            'tamanos':dict(Counter(f.get('grupo_tamano') for f in ff)),
            'tocan_borde':sum(f['toca_borde_imagen'] for f in ff),
            'cruces_estrictos':sum(f['cruces_estrictos']>0 for f in ff),
            'bbox_poligono_iou_menor_090':sum(f['bbox_poligono_iou']<.9 for f in ff),
            'fuera_imagen':sum(f['poligono_fuera_imagen'] or f['bbox_fuera_imagen'] for f in ff),
            'degenerados':sum(f['poligono_degenerado'] for f in ff),
            'elegibles':sum(f.get('region_elegible',{}).get('tokens',0)>0 for f in ff),
            'omitidas_por_razon':dict(Counter(f.get('region_elegible',{}).get('motivo_omision') for f in ff if f['clase'] and not f.get('region_elegible',{}).get('tokens',0))),
            'lotes':dict(Counter(f['adquisicion'] for f in ff))}
    return {'total':resumir(filas),'por_material':{c:resumir([f for f in filas if f['clase']==c]) for c in CLASES},
        'ignoradas':resumir([f for f in filas if f['clase'] is None])}
