"""Puente mínimo V27: conserva propuestas públicas y metadatos desconocidos.

No crea identidades físicas, no convierte bbox en polígonos y no acepta GT.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from ecowaste.incorporacion import poligonos_validos,caja_derivada,qa_geometria


def imagen_verificada(path):
    path=Path(path);blob=path.read_bytes()
    with Image.open(path) as im:
        im.load();rgb=im.convert('RGB');w,h=rgb.size
        gris=np.asarray(rgb.resize((32,32),Image.Resampling.LANCZOS).convert('L'),dtype=np.float32)
        dct=cv2.dct(gris)[:8,:8].flatten();b=dct>np.median(dct[1:]);b[0]=False
        return {'width':w,'height':h,'formato':im.format,'orientacion_exif':im.getexif().get(274,1),
            'sha256_archivo':hashlib.sha256(blob).hexdigest(),'sha256_pixeles':hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest(),
            'phash':f'{sum(int(v)<<i for i,v in enumerate(b)):016x}'}


def mapear_material(fuente,categoria):
    if fuente=='zerowaste':
        return {'rigid_plastic':'plastico','soft_plastic':'plastico','cardboard':'carton','metal':'metal'}.get(categoria)
    if fuente=='warp':
        if categoria.startswith('glass-'):return 'vidrio'
        if categoria.startswith('bottle-'):return 'plastico'  # Los autores definen estos 17 tipos como botellas plásticas.
        # Cans/detergent/canister requieren confirmación de composición; leche/zumo multicapa no son cartón simple.
        return None
    return None


def caja_yolo(valores,w,h):
    if len(valores)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in valores):
        raise ValueError('Caja YOLO no finita o mal formada')
    x,y,bw,bh=valores
    if not 0<=x<=1 or not 0<=y<=1 or not 0<bw<=1 or not 0<bh<=1:raise ValueError('Caja YOLO fuera de rango')
    b=[(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h]
    if b[0]<-1e-6 or b[1]<-1e-6 or b[0]+b[2]>w+1e-6 or b[1]+b[3]>h+1e-6:raise ValueError('Caja YOLO excede imagen; no recortar silenciosamente')
    return b


def propuesta(fuente,iid,aid,categoria,dimensiones,*,poligonos=None,bbox=None):
    material=mapear_material(fuente,categoria)
    r={'id_fuente':str(aid),'imagen_id_fuente':str(iid),'categoria_original':categoria,'material_propuesto':material,
        'poligonos_originales':poligonos,'bbox_original':bbox,'bbox_v26_derivada':None,'poligonos_normalizados_v26':None,
        'objeto_fisico_id':None,'sesion_id':None,'sitio_id':None,'dispositivo_id':None,'capturado_en':None,
        'oclusion':None,'truncado':None,'transparente':None,'visible_only_confirmado':False,
        'estado':'propuesta_pendiente_revision_humana','aceptada':False,'errores_geometricos':[],
        'mapeo_pendiente':material is None,'evidencia_identidad_fisica':None}
    if poligonos is not None:
        try:
            norm=poligonos_validos(poligonos,*dimensiones);r['poligonos_normalizados_v26']=norm
            r['bbox_v26_derivada']=caja_derivada(norm)
            r['bbox_original_coincide_v26']=bbox is not None and bool(np.allclose(bbox,r['bbox_v26_derivada'],rtol=0,atol=1e-6))
        except (ValueError,TypeError) as e:r['errores_geometricos'].append(str(e))
    else:r['requiere_poligono_humano']=True
    if bbox is not None:
        try:
            x,y,w,h=bbox
            if not all(math.isfinite(v) for v in bbox) or w<=0 or h<=0 or min(x,y)<-1e-6 or x+w>dimensiones[0]+1e-6 or y+h>dimensiones[1]+1e-6:
                raise ValueError('BBox original fuera de imagen o no finita')
        except (ValueError,TypeError) as e:r['errores_geometricos'].append(str(e))
    return r


def diagnostico_regional(imagen,propuestas):
    """GT de fuente como diagnóstico offline, sin asignar IDs físicos ni aceptarlo.

    Si falta un polígono o material, se omite la cifra: no ignorar objetos sin máscara
    para declarar regiones orgánicas libres de ambigüedad.
    """
    if not propuestas:return {'calculado':False,'motivo':'sin_anotaciones_espaciales_originales','organicos_utilizables':None}
    if any(p['poligonos_normalizados_v26'] is None or p['material_propuesto'] is None for p in propuestas):
        return {'calculado':False,'motivo':'poligonos_o_materiales_incompletos_no_inventar_mascaras','organicos_utilizables':None}
    inst=[]
    for p in propuestas:
        ps=p['poligonos_normalizados_v26']
        from PIL import ImageDraw
        pil=Image.new('1',(imagen['width'],imagen['height']));d=ImageDraw.Draw(pil)
        for pol in ps:d.polygon(list(zip(pol[::2],pol[1::2])),fill=1)
        inst.append({'id':p['id_fuente'],'objeto_fisico_id':None,'material':p['material_propuesto'],'poligonos':ps,
            'bbox':p['bbox_v26_derivada'],'area_mask_px':int(np.asarray(pil).sum()),'area_declarada':None,'solapa_con':[]})
    qa=qa_geometria({**imagen,'instancias':inst});qa['calculado']=True
    qa['organicos_utilizables']=sum(x['organico_utilizable_48'] for x in qa['instancias'])
    qa['cuotas_acreditadas']=0;qa['no_es_revision_humana']=True
    return qa
