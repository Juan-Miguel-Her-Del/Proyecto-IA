"""COCO de autores, QA y caché V29. Nunca modifica etiquetas históricas/humanas."""
from collections import Counter
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cv2
from PIL import Image
import torch
from ecowaste.datasets import CLASES,letterbox,geometria_letterbox,objetivo_deteccion
from ecowaste.incorporacion import leer,sha
from ecowaste.validacion_anotaciones import celdas

VERSION='v29_autores_coco_1'


def decodificar_coco(seg,w,h):
    """Codec oficial COCO; una instancia puede contener varias partes/huecos."""
    from pycocotools import mask as cm
    if isinstance(seg,list):
        if not seg:raise ValueError('Polígono vacío')
        for p in seg:
            a=np.asarray(p,dtype=np.float64)
            if len(a)<6 or len(a)%2 or not np.isfinite(a).all():raise ValueError('Polígono inválido')
            if (a[::2]<-.1).any() or (a[::2]>w+.1).any() or (a[1::2]<-.1).any() or (a[1::2]>h+.1).any():raise ValueError('Polígono fuera de imagen')
        r=cm.merge(cm.frPyObjects(seg,h,w))
    elif isinstance(seg,dict):
        if seg.get('size')!=[h,w]:raise ValueError('Dimensión RLE incorrecta')
        r=dict(seg)
        if isinstance(r.get('counts'),list):
            if any(not isinstance(x,int) or x<0 for x in r['counts']) or sum(r['counts'])!=w*h:raise ValueError('Runs RLE inválidos')
            r=cm.frPyObjects(r,h,w)
        elif isinstance(r.get('counts'),str):r['counts']=r['counts'].encode('ascii')
        elif not isinstance(r.get('counts'),bytes):raise ValueError('RLE sin counts válidos')
    else:raise ValueError('Segmentación no COCO')
    m=cm.decode(r)
    if m.ndim!=2 or m.shape!=(h,w) or not m.any():raise ValueError('Máscara vacía/incompatible')
    return np.asarray(m,dtype=np.uint8)


def bbox_mascara(m):
    if m.flags.c_contiguous:x,y,w,h=cv2.boundingRect(m)
    else:y,x,h,w=cv2.boundingRect(m.T)
    if not w or not h:raise ValueError('Máscara vacía')
    return [x,y,w,h]


def bbox_iou(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=(4,) or b.shape!=(4,) or not np.isfinite(a).all() or not np.isfinite(b).all():return 0.
    wh=np.maximum(0,np.minimum(a[:2]+a[2:],b[:2]+b[2:])-np.maximum(a[:2],b[:2]));inter=float(wh.prod())
    return inter/max(1e-9,float(a[2:].prod()+b[2:].prod()-inter))


def ajustar_mascara(m,g):
    """Nearest sin inventar contornos; pooling 8x8 por cobertura como V6."""
    nw,nh=round(g['ancho_original']*g['escala_x']),round(g['alto_original']*g['escala_y'])
    if m.flags.c_contiguous:z=np.asarray(Image.fromarray(m).resize((nw,nh),Image.Resampling.NEAREST),dtype=np.uint8)
    else:
        # COCO devuelve Fortran: trasponer antes del resize evita copiar/recorrer
        # por filas 16M píxeles por objeto. Es exactamente el mismo nearest.
        z=np.asarray(Image.fromarray(m.T).resize((nh,nw),Image.Resampling.NEAREST),dtype=np.uint8).T
    out=np.zeros((384,384),dtype=np.uint8);x,y=g['relleno_x'],g['relleno_y'];out[y:y+nh,x:x+nw]=z
    return out


def rasterizar_instancias(reg,anotaciones,mascaras,*,mascaras_384=False):
    """Padding, ignorados y mezcla de materiales no aportan supervisión."""
    g=geometria_letterbox(reg['ancho'],reg['alto'],384)
    cover=np.zeros((6,48,48),dtype=np.float32);ig=np.zeros((48,48),dtype=bool);rs=[]
    for a,m in zip(anotaciones,mascaras):
        cov=celdas(m if mascaras_384 else ajustar_mascara(m,g));c=a['material']
        if c in CLASES:
            ci=CLASES.index(c);cover[ci]=np.maximum(cover[ci],cov);rs.append((ci,cov>=1/64,a['id']))
        else:ig|=cov>0
    yy,xx=np.meshgrid((np.arange(48)+.5)*8,(np.arange(48)+.5)*8,indexing='ij')
    valido=(xx>=g['relleno_x'])&(yy>=g['relleno_y'])&(xx<g['relleno_x']+round(reg['ancho']*g['escala_x']))&(yy<g['relleno_y']+round(reg['alto']*g['escala_y']))
    y=np.zeros((48,48),dtype=np.int64);y[~valido|ig|(cover.sum(0)>0)]=255
    presentes=cover>=1/64;unico=presentes.sum(0)==1
    for c in range(6):y[presentes[c]&unico&valido&~ig]=c+1
    return torch.from_numpy(y),[(c,m&valido&~ig&(y==c+1),aid) for c,m,aid in rs]


def validar_imagen(ruta,im,anns,mapeo):
    p=Path(ruta)
    if not p.exists():raise FileNotFoundError(p)
    with Image.open(p) as pil:
        pil.load()
        if pil.size!=(im['width'],im['height']):raise ValueError('Dimensiones de foto/anotación distintas')
        # No rotar automáticamente la foto con EXIF: las coordenadas son del archivo.
        rgb=pil.convert('RGB');pix=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        mini=rgb.resize((9,8),Image.Resampling.LANCZOS).convert('L');v=np.asarray(mini)
        dhash=f'{sum(int(b)<<i for i,b in enumerate((v[:,1:]>v[:,:-1]).flatten())):016x}'
        gris=np.asarray(rgb.resize((32,32),Image.Resampling.LANCZOS).convert('L'),dtype=np.float32)
        coef=cv2.dct(gris)[:8,:8].flatten();bits=coef>np.median(coef[1:]);bits[0]=False
        phash=f'{sum(int(b)<<i for i,b in enumerate(bits)):016x}'
        x,_=letterbox(rgb,384)
    out=[];masks=[];avisos=[];vistos=set();g=geometria_letterbox(im['width'],im['height'],384)
    for a in anns:
        ident=a['id']
        if ident in vistos:raise ValueError('ID de instancia duplicado')
        vistos.add(ident)
        if a['category_id'] not in mapeo:raise ValueError('Categoría sin política explícita')
        m=decodificar_coco(a['segmentation'],im['width'],im['height']);bb=bbox_mascara(m)
        iou=bbox_iou(a.get('bbox',[]),bb)
        if iou<.9:raise ValueError(f'Caja/máscara inconsistente en objeto {ident}: IoU={iou}')
        material=mapeo[a['category_id']]
        if a.get('iscrowd',0):material=None
        out.append({'id':ident,'material':material,'bbox':bb,'bbox_fuente':a.get('bbox'),
            'category_id_original':a['category_id'],'segmentation':a['segmentation'],'area':int(cv2.countNonZero(m if m.flags.c_contiguous else m.T)),
            'etiqueta_origen':'autores_del_dataset','revision_independiente_proyecto':False,'aceptada_V26':False})
        masks.append(ajustar_mascara(m,g))
        if iou<.999:avisos.append({'objeto':ident,'bbox_derivada_de_mascara':True,'iou_bbox_fuente':iou})
    reg={'id':im['eco_id'],'ancho':im['width'],'alto':im['height'],
        'cajas':[{'bbox':a['bbox'],'clase':a['material']} for a in out]}
    y,rs=rasterizar_instancias(reg,out,masks,mascaras_384=True)
    stats=[]
    for a in out:
        rr=next((m for c,m,aid in rs if aid==a['id']),None)
        stats.append({'id':a['id'],'material':a['material'],'area_px':a['area'],'tokens_48':int(rr.sum()) if rr is not None else 0})
    return {'registro':reg,'anotaciones':out,'sha256_archivo':sha(p),'sha256_pixeles':pix,'dhash':dhash,'phash':phash,
        'avisos':avisos,'objetos_48':stats},(x*255).round().to(torch.uint8),y,rs


def cargar_coleccion(raiz,carpeta,split):
    if split not in ['train','val']:raise ValueError('Solo cohortes train/val autorizadas')
    raiz,carpeta=Path(raiz),Path(carpeta);meta=leer(carpeta/'MANIFIESTO.json')
    if meta['version']!=VERSION or meta['aceptadas_V26']!=0 or meta['estado']!='EXPERIMENTAL_ETIQUETAS_AUTORES':raise ValueError('Contrato de datos incompatible')
    for rel,h in meta['archivos'].items():
        if sha(carpeta/rel)!=h:raise ValueError('Colección/caché modificada: '+rel)
    d=torch.load(carpeta/f'{split}.pt',map_location='cpu',weights_only=True)
    for r in d['registros']:
        if sha(raiz/r['ruta'])!=r['sha256_archivo']:raise ValueError('Imagen fuente modificada')
    return SimpleNamespace(**d,meta=meta)


def conflicto_duplicado(a,b,distancia=6):
    exacto=a['sha256_archivo']==b.get('sha256_archivo') or a['sha256_pixeles']==b.get('sha256_pixeles')
    candidato=bool(a.get('dhash') and b.get('dhash') and (int(a['dhash'],16)^int(b['dhash'],16)).bit_count()<=distancia)
    if a.get('phash') and b.get('phash'):
        candidato=candidato and (int(a['phash'],16)^int(b['phash'],16)).bit_count()<=4
    return 'exacto' if exacto else 'perceptual_candidato' if candidato else None


def resumen_materiales(filas):
    cuentas=Counter(a['material'] or 'ignore' for f in filas for a in f['anotaciones'])
    return dict(cuentas)
