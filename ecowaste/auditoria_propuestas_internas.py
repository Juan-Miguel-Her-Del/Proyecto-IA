"""Análisis GT posterior a la inferencia. Ninguna decisión vuelve al detector."""
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
from ecowaste.datasets import CLASES,objetivo_deteccion
from ecowaste.cajas import iou_pares,cubiertas_por_ignorar
from ecowaste.auditoria_deteccion import auditar_imagen,tamano_objeto
from ecowaste.traza_propuestas import ETAPAS

CAUSAS=['correcto','perdida_postprocesamiento','fusion_objetos','fragmentacion',
    'material_incorrecto','ausencia_respuesta_fg','geometria_insuficiente','sin_soporte_gt_no_ambiguo']
VERSION='gt_posthoc_poligonos_visibilidad080_fragmentos010_union050_fusion050_v1'


def cobertura_poligono(poligonos,g,z,lado):
    res=g['lado'];im=Image.new('L',(res,res));d=ImageDraw.Draw(im)
    for p in poligonos:
        d.polygon([((x-z[0])*g['escala_x']+g['relleno_x'],(y-z[1])*g['escala_y']+g['relleno_y'])
            for x,y in zip(p[::2],p[1::2])],fill=1)
    a=np.array(im,copy=True);valida=np.zeros_like(a)
    x,y=g['relleno_x'],g['relleno_y'];w=round(g['ancho_original']*g['escala_x']);h=round(g['alto_original']*g['escala_y'])
    valida[y:y+h,x:x+w]=1;a*=valida
    return F.avg_pool2d(torch.from_numpy(a).float()[None,None],res//lado)[0,0].numpy()


def visibilidad_bbox(b,z):
    x,y,w,h=b
    return max(0,min(x+w,z[2])-max(x,z[0]))*max(0,min(y+h,z[3])-max(y,z[1]))/max(w*h,1e-9)


def diagnosticar_objeto(tp,etapas,raw_correctos,raw_otros,vistas,fusiones,fragmentos,nodos):
    """Causa principal ordenada; indicadores no excluyentes se conservan aparte."""
    if tp:return 'correcto',None
    if raw_correctos and not etapas['final']['correctos_iou50']:
        perdidas=Counter(nodos[i]['descartada_en'] or 'final' for i in raw_correctos)
        return 'perdida_postprocesamiento',dict(perdidas)
    if fusiones:return 'fusion_objetos',None
    if fragmentos:return 'fragmentacion',None
    utiles=[v for v in vistas if v['util'] and v['soporte_gt_tokens']>0]
    if not utiles:return 'sin_soporte_gt_no_ambiguo',None
    correcto=max(v['fraccion_material_correcto'] for v in utiles)
    fg=max(v['fraccion_fg'] for v in utiles)
    if raw_otros or (correcto==0 and fg>0):return 'material_incorrecto',None
    if fg==0:return 'ausencia_respuesta_fg',None
    return 'geometria_insuficiente',None


def analizar_imagen(traza,registro,contornos):
    objetivo=objetivo_deteccion(registro);anotaciones=[a for a in registro['cajas'] if a['clase']]
    poligonos=[contornos.contornos(registro,a) for a in anotaciones]
    n=len(anotaciones);por_gt=[[] for _ in range(n)];fusiones=[[] for _ in range(n)];fragmentos=[[] for _ in range(n)]
    nodos=traza['nodos'];ids=list(nodos)
    cajas=torch.tensor([nodos[i]['caja_global'] for i in ids],dtype=torch.float32).reshape(-1,4)
    labels=torch.tensor([nodos[i]['clase'] for i in ids],dtype=torch.long)
    ious=iou_pares(cajas,objetivo['boxes'])
    ignorar=cubiertas_por_ignorar(cajas,objetivo['ignore_boxes'])
    ignorar&=(ious.max(1).values<.5) if n else torch.ones(len(cajas),dtype=torch.bool)
    id_indice={ident:i for i,ident in enumerate(ids)}
    etapas={e:[id_indice[i] for i in traza['etapas'][e]] for e in ETAPAS}
    for vista in traza['vistas']:
        g,z=vista['geometria'],vista['ventana'];lado=vista['ganador'].shape[0]
        mascaras=[cobertura_poligono(p,g,z,lado) for p in poligonos]
        por_clase=np.zeros((6,lado,lado),dtype=bool)
        for a,m in zip(anotaciones,mascaras):por_clase[CLASES.index(a['clase'])]|=m>0
        ambiguo=por_clase.sum(0)>1
        ignoradas=np.zeros((lado,lado),dtype=bool)
        for a in registro['cajas']:
            if a['clase']:continue
            x,y,w,h=a['bbox'];p=[[x,y,x+w,y,x+w,y+h,x,y+h]]
            ignoradas|=cobertura_poligono(p,g,z,lado)>0
        coberturas=[]
        for k,(a,m) in enumerate(zip(anotaciones,mascaras)):
            peso=m*vista['validas']*~ambiguo*~ignoradas;den=float(peso.sum())
            c=CLASES.index(a['clase'])+1;ganador=vista['ganador'];pr=vista['probabilidades']
            cuentas=np.bincount(vista['componentes'].ravel(),weights=peso.ravel(),minlength=len(vista['ids_componentes'])+1)
            cobertura=cuentas[1:]/den if den else np.zeros(len(cuentas)-1);coberturas.append(cobertura)
            util=visibilidad_bbox(a['bbox'],z)>=.8
            detalle={'vista':vista['vista'],'util':util,'visibilidad_bbox':visibilidad_bbox(a['bbox'],z),
                'soporte_gt_tokens':den,'tokens_intersectados':int((peso>0).sum()),
                'lado_menor_entrada_px':min(a['bbox'][2]*g['escala_x'],a['bbox'][3]*g['escala_y']),
                'fraccion_material_correcto':float((peso*(ganador==c)*(pr[c]>=.05)).sum()/den) if den else 0.,
                'fraccion_fg':float((peso*(ganador>0)).sum()/den) if den else 0.,
                'fraccion_fondo':float((peso*(ganador==0)).sum()/den) if den else 0.,
                'prob_gt_media':float((peso*pr[c]).sum()/den) if den else None,
                'prob_gt_max':float(pr[c][peso>0].max()) if den else None,
                'fraccion_prob_gt_mayor_005':float((peso*(pr[c]>=.05)).sum()/den) if den else None,
                'componentes_tocan_gt':[{'id':i,'fraccion_gt':float(cv)} for i,cv in zip(vista['ids_componentes'],cobertura) if cv>0]}
            por_gt[k].append(detalle)
            partes=[i for i,cv in zip(vista['ids_componentes'],cobertura) if cv>=.1 and nodos[i]['clase']==c-1]
            if util and len(partes)>=2:
                b=torch.tensor([nodos[i]['caja_global'] for i in partes]);union=torch.cat([b[:,:2].min(0).values,b[:,2:].max(0).values])[None]
                iu=float(iou_pares(union,objetivo['boxes'][k:k+1])[0,0]);individual=float(iou_pares(b,objetivo['boxes'][k:k+1]).max())
                cubre=sum(float(cobertura[vista['ids_componentes'].index(i)]) for i in partes)
                if cubre>=.5 and iu>=.5 and individual<.5:
                    fragmentos[k].append({'vista':vista['vista'],'ids':partes,'cobertura_gt':cubre,
                        'iou_envolvente_oracle':iu,'max_iou_individual':individual,'caja_union_oracle':union[0].tolist()})
        for j,ident in enumerate(vista['ids_componentes']):
            tocados=[k for k in range(n) if coberturas[k][j]>=.5 and por_gt[k][-1]['util']]
            if len(tocados)>=2:
                for k in tocados:
                    if nodos[ident]['clase']==int(objetivo['labels'][k]):
                        fusiones[k].append({'vista':vista['vista'],'id':ident,
                            'gt_internos':[anotaciones[t]['id_interno'] for t in tocados],
                            'coberturas':[float(coberturas[t][j]) for t in tocados]})
    final=auditar_imagen(traza['final'],objetivo);filas=[]
    for k,a in enumerate(anotaciones):
        c=int(objetivo['labels'][k]);good=(ious[:,k]>=.5)&~ignorar;correctos=good&(labels==c)
        raw_correctos=[ids[i] for i in torch.where(correctos)[0].tolist()]
        raw_otros=[ids[i] for i in torch.where(good&(labels!=c))[0].tolist()]
        resumen={}
        for e,ix in etapas.items():
            resumen[e]={'candidatos':len(ix),'correctos_iou50':sum(bool(correctos[i]) for i in ix),
                'cualquier_material_iou50':sum(bool(good[i]) for i in ix),
                'max_iou_correcto':max((float(ious[i,k]) for i in ix if int(labels[i])==c and not ignorar[i]),default=0.)}
        tp=final['objetos'][k]['estado']=='correcto'
        causa,perdidas=diagnosticar_objeto(tp,resumen,raw_correctos,raw_otros,por_gt[k],fusiones[k],fragmentos[k],nodos)
        utiles=[v for v in por_gt[k] if v['util'] and v['soporte_gt_tokens']>0]
        mejor=max(utiles,key=lambda v:(v['fraccion_material_correcto'],v['lado_menor_entrada_px'],-v['vista'])) if utiles else por_gt[k][0]
        filas.append({'imagen':registro['id'],'id_gt':a['id_interno'],'material':a['clase'],'clase':c,
            'caja_gt':objetivo['boxes'][k].tolist(),'bbox_original':a['bbox'],**tamano_objeto(objetivo['boxes'][k]),
            'max_lado_menor_vistas_utiles_px':max((v['lado_menor_entrada_px'] for v in por_gt[k] if v['util']),default=0.),
            'causa_principal':causa,'perdidas_por_etapa':perdidas,'etapas':resumen,'vistas':por_gt[k],
            'vista_testigo':mejor['vista'],'fusiones':fusiones[k],'fragmentos':fragmentos[k],
            'ids_correctos_pre_filtros':raw_correctos,'ids_otros_materiales_iou50':raw_otros,
            'diagnostico_caja_final':final['objetos'][k]})
    return filas,final


def resumir_objetos(filas):
    def grupo(items):
        return {'gt':len(items),'causas':{c:sum(f['causa_principal']==c for f in items) for c in CAUSAS},
            'indicadores_fragmentacion':sum(bool(f['fragmentos']) for f in items),
            'indicadores_fusion':sum(bool(f['fusiones']) for f in items),
            'oportunidades_por_etapa':{e:sum(f['etapas'][e]['correctos_iou50']>0 for f in items) for e in ETAPAS},
            'ganancia_resolucion_tokens':sum(f['lado_menor_384']<8<=f['max_lado_menor_vistas_utiles_px'] for f in items)}
    tamanos=['menos_8_px','8_a_16_px','16_a_32_px','32_px_o_mas']
    return {'total':grupo(filas),'por_material':{c:grupo([f for f in filas if f['material']==c]) for c in CLASES},
        'por_tamano':{t:grupo([f for f in filas if f['grupo_tamano']==t]) for t in tamanos},
        'por_material_y_tamano':{c:{t:grupo([f for f in filas if f['material']==c and f['grupo_tamano']==t]) for t in tamanos} for c in CLASES}}
