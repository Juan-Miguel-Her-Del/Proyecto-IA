"""QA V28 de instantáneas humanas: lectura, sin reparar contornos ni aceptar V26."""
from copy import deepcopy
import hashlib
import numpy as np
import cv2
from PIL import Image,ImageDraw
from ecowaste.auxiliares_publicos import normalizar,mascara_poligonos
from ecowaste.incorporacion import CLASES,leer,sha,digest,caja_derivada,poligonos_validos
from ecowaste.datasets import geometria_letterbox


def revision_humana_final(eventos):
    revisiones=[e for e in eventos if e['modo']=='asistida']
    return revisiones[-1] if revisiones else None


def iniciales(taller,iid):
    im=taller.imagen(iid);anns=deepcopy(im['anotaciones'])
    idx=leer(taller.path/'PROPUESTAS_IA_INICIALES.json')
    if idx['proyecto_sha256']!=taller.base['huella']:raise ValueError('Índice IA de otro proyecto')
    for a in anns:
        pid=idx['por_imagen'].get(iid,{}).get(a['id'])
        if pid:
            r=leer(taller.path/'propuestas_ia'/(pid+'.json'))
            if r['imagen_id']!=iid or r['huella']!=digest({k:v for k,v in r.items() if k!='huella'}):raise ValueError('Propuesta inicial alterada')
            a.update(poligonos=r['resultado']['poligonos'],huecos=r['resultado']['huecos'],
                     bbox=caja_derivada(r['resultado']['poligonos']),propuesta_ia_id=pid)
    return anns


def mask_poligonos(a,w,h,*,paridad=False,geometria=None):
    if geometria is None:
        sx=sy=1.;dx=dy=0.;res=(w,h)
    else:
        sx,sy=geometria['escala_x'],geometria['escala_y'];dx,dy=geometria['relleno_x'],geometria['relleno_y'];res=(384,384)
    salida=np.zeros((res[1],res[0]),dtype=np.uint8)
    for holes,ps in [(False,a.get('poligonos',[])),(True,a.get('huecos',[]))]:
        for p in ps:
            im=Image.new('1',res);ImageDraw.Draw(im).polygon([(x*sx+dx,y*sy+dy) for x,y in zip(p[::2],p[1::2])],fill=1)
            m=np.asarray(im,dtype=np.uint8)
            if paridad:salida^=m
            elif holes:salida[m>0]=0
            else:salida|=m
    return salida


def celdas(mask):
    if mask.shape!=(384,384):raise ValueError('Se requiere máscara 384×384')
    return mask.reshape(48,8,48,8).mean(axis=(1,3))


def max_componente(mask):
    n,_,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    return int(stats[1:,cv2.CC_STAT_AREA].max()) if n>1 else 0


def verificar_objeto(a,im,completa):
    r={'id':a.get('id'),'material':a.get('material'),'descartada':a.get('descartada') is True,
       'motivo':a.get('motivo',''),'errores':[],'avisos':[],'valido_geometria':False,
       'imagen_cobertura_completa':completa,'poligonos_exteriores':len(a.get('poligonos',[])),
       'huecos':len(a.get('huecos',[])),'origen':a.get('origen'),'propuesta_ia_id':a.get('propuesta_ia_id')}
    if r['descartada']:
        if len(r['motivo'].strip())<5:r['errores'].append('descarte_sin_motivo')
        return r,None,None
    try:
        normalizar(a,im,completa=False)
        if not a.get('poligonos'):raise ValueError('sin_contorno')
        bbox=caja_derivada(a['poligonos']);r['bbox_derivada']=bbox
        if a.get('bbox') is None or not np.allclose(a['bbox'],bbox,atol=1e-6,rtol=0):r['errores'].append('bbox_no_corresponde_a_poligonos')
        m=mask_poligonos(a,im['width'],im['height'])
        par=mask_poligonos(a,im['width'],im['height'],paridad=True)
        delta=int((m!=par).sum());r['pixeles_discrepancia_editor_exportador']=delta
        if delta:r['errores'].append('relleno_paridad_editor_difiere_union_exportador')
        r['area_px']=int(m.sum());r['mask_sha256']=hashlib.sha256(m.tobytes()).hexdigest()
        n,_,stats,_=cv2.connectedComponentsWithStats(m,connectivity=8)
        areas=sorted(map(int,stats[1:,cv2.CC_STAT_AREA]),reverse=True)
        r['componentes_totales']=n-1;r['areas_componentes']=areas
        r['componentes_significativos']=sum(s>=max(4,.01*max(1,r['area_px'])) for s in areas)
        if r['componentes_significativos']>1:r['avisos'].append('fragmentacion_o_objetos_fusionados_requiere_inspeccion')
        if not r['area_px']:r['errores'].append('mascara_vacia')
        if a['material']=='sin_asignar':r['errores'].append('material_sin_asignar')
        r['valido_geometria']=not any(x not in ['material_sin_asignar'] for x in r['errores'])
        return r,m,mask_poligonos(a,im['width'],im['height'],geometria=geometria_letterbox(im['width'],im['height'],384))
    except (ValueError,TypeError,KeyError) as e:
        r['errores'].append(str(e));return r,None,None


def analizar_imagen(im,anns,completa):
    objetos=[];mascaras={};maps={};errores=[];avisos=[];solapes=[]
    for a in anns:
        r,m,m384=verificar_objeto(a,im,completa);objetos.append(r)
        if m is not None and not r['descartada']:
            mascaras[a['id']]=m;maps[a['id']]=m384
        if r['errores']:errores.append({'objeto':a.get('id'),'motivos':r['errores']})
        if r['avisos']:avisos.append({'objeto':a.get('id'),'motivos':r['avisos']})
    activos=[r for r in objetos if r['id'] in mascaras]
    for j,a in enumerate(activos):
        for b in activos[:j]:
            ma,mb=mascaras[a['id']],mascaras[b['id']];inter=int((ma&mb).sum())
            frac=inter/max(1,min(a['area_px'],b['area_px']))
            if inter>4 and frac>.01:
                x={'a':a['id'],'b':b['id'],'pixeles':inter,'fraccion_menor':frac,
                   'materiales_distintos':a['material']!=b['material'],'iou':inter/max(1,int((ma|mb).sum()))}
                solapes.append(x)
                avisos.append({'objetos':[a['id'],b['id']],'motivos':['solape_visible_requiere_decision_no_prueba_de_occlusion'],'datos':x})
                if np.array_equal(ma,mb):errores.append({'objetos':[a['id'],b['id']],'motivos':['mascaras_identicas_instancias_duplicadas_o_conflicto_material']})
    g=geometria_letterbox(im['width'],im['height'],384)
    contenido=np.zeros((384,384),dtype=np.uint8)
    x,y=g['relleno_x'],g['relleno_y'];w=round(g['ancho_original']*g['escala_x']);h=round(g['alto_original']*g['escala_y'])
    contenido[y:y+h,x:x+w]=1;tokens_validos=celdas(contenido)==1
    coberturas={aid:celdas(m) for aid,m in maps.items()}
    materiales=np.zeros((6,48,48),dtype=bool);ignoradas=np.zeros((48,48),dtype=bool)
    for a in activos:
        if a['material'] in CLASES:materiales[CLASES.index(a['material'])]|=coberturas[a['id']]>0
        else:ignoradas|=coberturas[a['id']]>0
    mezcladas=materiales.sum(0)>1
    for a in objetos:
        a['tokens_regionales']=0;a['componente_maximo_48']=0;a['organico_utilizable_48']=False
        if a['id'] not in coberturas or a['material'] not in CLASES or not a['valido_geometria']:continue
        c=coberturas[a['id']];candidatos=(c>=1/64)&tokens_validos&~mezcladas&~ignoradas
        interior=candidatos&(c==1);region=interior if interior.any() else candidatos
        a['tokens_regionales']=int(region.sum());a['componente_maximo_48']=max_componente(region)
        a['modo_regional']='interior' if interior.any() else 'fraccion_poligono'
        a['organico_utilizable_48']=bool(completa and a['material']=='organico' and a['componente_maximo_48']>=2)
        if a['componente_maximo_48']<2:a['avisos'].append('sin_componente_de_dos_celdas_48')
    fusiones=[]
    for ci,c in enumerate(CLASES):
        n,labels,_,_=cv2.connectedComponentsWithStats((materiales[ci]&tokens_validos&~mezcladas&~ignoradas).astype(np.uint8),connectivity=8)
        for k in range(1,n):
            ids=[a['id'] for a in activos if a['material']==c and int(((coberturas[a['id']]>0)&(labels==k)).sum())>=2]
            if len(ids)>1:fusiones.append({'material':c,'objetos':ids,'componente':k,'tipo':'fusion_representacion_semantica_no_error_GT_demostrado'})
    return {'objetos':objetos,'errores':errores,'avisos_visuales':avisos,'solapes':solapes,'fusiones_48':fusiones,
            'tokens_mezcla_material':int(mezcladas.sum()),'tokens_padding_excluidos':int((~tokens_validos).sum()),
            'organicos_utilizables_48':sum(a['organico_utilizable_48'] for a in objetos)}


def comparar(inicial,final,im):
    ini={a['id']:a for a in inicial};fin={a['id']:a for a in final}
    r={'agregados':[],'descartados':[],'reclasificados':[],'contornos_cambiados':[],
       'sin_cambio_coordenadas_material':[],'originales_ausentes_sin_registro':sorted(set(ini)-set(fin))}
    for aid,a in fin.items():
        if aid not in ini:r['agregados'].append(aid)
        if a.get('descartada'):r['descartados'].append(aid)
        if aid in ini:
            b=ini[aid]
            if a['material']!=b['material']:r['reclasificados'].append({'id':aid,'antes':b['material'],'despues':a['material']})
            geo=any(a.get(k,[])!=b.get(k,[]) for k in ['poligonos','huecos'])
            if geo:r['contornos_cambiados'].append(aid)
            if not geo and a['material']==b['material'] and not a.get('descartada'):r['sin_cambio_coordenadas_material'].append(aid)
    return r
