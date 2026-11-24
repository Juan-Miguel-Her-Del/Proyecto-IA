"""Contrato separado para correcciones IA: nunca una revisión humana ni GT aceptada.

Las máscaras RLE son canónicas en esta copia. Los polígonos históricos se
conservan como procedencia; no se proyecta RLE al contrato anterior a escondidas.
"""
from copy import deepcopy
import numpy as np
from PIL import Image
from ecowaste.auxiliares_publicos import rle_coco
from ecowaste.validacion_anotaciones import mask_poligonos
from ecowaste.incorporacion import digest

VERSION='ajustes_asistente_IA_v28_3_1'
CLASES=['plastico','papel','carton','vidrio','metal','organico']


def decodificar(r):
    h,w=r['size'];c=r['counts']
    if h<=0 or w<=0 or any(type(x) is not int or x<0 for x in c) or sum(c)!=h*w:
        raise ValueError('RLE no válido')
    return np.repeat(np.arange(len(c))%2,c).astype(bool).reshape((h,w),order='F')


def caja(m):
    y,x=np.where(m)
    if not len(x):raise ValueError('Máscara vacía')
    return [int(x.min()),int(y.min()),int(x.max()-x.min()+1),int(y.max()-y.min()+1)]


def guardar_mask(a,m):
    a['segmentacion_canonica_RLE']=rle_coco(m)
    a['bbox']=caja(m);a['area_px']=int(m.sum())
    return a


def anotar_cambio(a,accion,motivo,**extra):
    a.setdefault('ajustes_IA',[]).append({'accion':accion,'motivo':motivo,**extra})
    a.update(actor_ultimo_ajuste='asistente_IA',revision_humana_del_ajuste=False,
             revision_independiente=False,aceptada_V26=False)


def canonizar(a,w,h):
    a=deepcopy(a)
    if a.get('poligonos') and not a.get('descartada'):
        # Preservar la semántica evenodd del guardado, no llenar sus huecos por unión.
        m=mask_poligonos(a,w,h,paridad=True).astype(bool)
        if m.any():
            u=mask_poligonos(a,w,h).astype(bool)
            guardar_mask(a,m)
            if not np.array_equal(m,u):
                anotar_cambio(a,'relleno_canonico_explicito',
                    'Se conserva el relleno por paridad mostrado en el editor. Los polígonos originales permanecen intactos.',
                    pixeles_union_vs_paridad=int(np.count_nonzero(m!=u)))
    return a


def ambiguo(a,motivo):
    if 'segmentacion_canonica_RLE' not in a:raise ValueError('No se puede ignorar una región sin máscara')
    anterior=a['material'];a['material']='ambiguo';a['motivo']=motivo;a['ignore']=True
    anotar_cambio(a,'material_a_ignore',motivo,material_anterior=anterior)


def quitar_solape(anns,ids,motivo,frente=None):
    """Conserva exactamente la unión; jamás rellena o conecta regiones invisibles."""
    pares={a['id']:a for a in anns};a,b=[pares[i] for i in ids]
    if any('segmentacion_canonica_RLE' not in x for x in (a,b)):raise ValueError('Solape sin máscara')
    ma,mb=[decodificar(x['segmentacion_canonica_RLE']) for x in (a,b)];s=ma&mb
    if not s.any():return None
    if frente is not None:
        if frente not in ids:raise ValueError('Objeto frontal ajeno al par')
        detras=b if frente==a['id'] else a;md=mb if detras is b else ma
        if not (md&~s).any():raise ValueError('La sustracción eliminaría todo el objeto')
        guardar_mask(detras,md&~s);anotar_cambio(detras,'occlusion_visible',motivo,frente=frente,pixeles=int(s.sum()))
        return None
    if any(not m[~s].any() for m in (ma,mb)):raise ValueError('Un objeto quedaría vacío; requiere adjudicación')
    for obj,m in ((a,ma),(b,mb)):
        guardar_mask(obj,m&~s);anotar_cambio(obj,'solape_ambiguo_a_ignore',motivo,pixeles=int(s.sum()))
    obj={'id':'ia_ignore_'+digest([ids,rle_coco(s)])[:20],'material':'ambiguo',
         'motivo':motivo,'ignore':True,'descartada':False,'origen':'ajuste_asistente_IA',
         'poligonos':[],'huecos':[],'propuesta_ia_id':None,'aceptada_V26':False,
         'actor_ultimo_ajuste':'asistente_IA','revision_humana_del_ajuste':False,
         'revision_independiente':False,'no_acredita_nueva_instancia':True}
    guardar_mask(obj,s);anns.append(obj);return obj


def nueva_propuesta(sam,material,detalle,m=None):
    m=decodificar(sam['resultado']['rle_ia_original']) if m is None else np.asarray(m,dtype=bool)
    if material not in [*CLASES,'ambiguo']:raise ValueError('Material no admitido')
    a={'id':sam['id'],'material':material,'poligonos':[],'huecos':[],
       'origen':'SAM2_comprobado_por_asistente_IA_NO_humano','propuesta_ia_id':None,
       'sam_IA_id':sam['id'],'descartada':False,'motivo':detalle,
       'actor_ultimo_ajuste':'asistente_IA','revision_humana_del_ajuste':False,
       'revision_independiente':False,'aceptada_V26':False,'ignore':material=='ambiguo'}
    return guardar_mask(a,m)


def componentes_dudosos_a_ignore(a,motivo):
    """Sólo para objetos inspeccionados: aislar restos sin inventar su identidad."""
    import cv2
    m=decodificar(a['segmentacion_canonica_RLE'])
    n,lab,stats,_=cv2.connectedComponentsWithStats(np.uint8(m),8)
    if n<=2:return None
    principal=lab==(1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])))
    resto=m&~principal
    if principal.sum()<.5*m.sum():raise ValueError('No hay componente dominante; adjudicar material/identidad')
    guardar_mask(a,principal);anotar_cambio(a,'componentes_dudosos_a_ignore',motivo,pixeles=int(resto.sum()))
    otro={'id':'ia_ignore_'+digest([a['id'],rle_coco(resto)])[:20],'material':'ambiguo',
        'motivo':motivo,'ignore':True,'descartada':False,'origen':'ajuste_asistente_IA',
        'poligonos':[],'huecos':[],'aceptada_V26':False,'actor_ultimo_ajuste':'asistente_IA',
        'revision_humana_del_ajuste':False,'revision_independiente':False,
        'no_acredita_nueva_instancia':True}
    return guardar_mask(otro,resto)


def elegibilidad48(m,w,h):
    """Sólo ocupación geométrica aproximada; no aprobación ni identidad física."""
    lado=384;s=min(lado/w,lado/h);nw,nh=round(w*s),round(h*s)
    p=Image.fromarray(np.uint8(m)*255).resize((nw,nh),Image.Resampling.NEAREST)
    l=Image.new('L',(lado,lado));l.paste(p,((lado-nw)//2,(lado-nh)//2))
    return int(np.count_nonzero(np.asarray(l.resize((48,48),Image.Resampling.NEAREST))))
