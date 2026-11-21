"""Observador argmax7 sin GT: reconstruye etapas y exige igualdad con producción."""
import cv2
import numpy as np
import torch
from ecowaste.cajas import iou_pares,nms_por_clase,deshacer_letterbox
from ecowaste.detector_mapas import cajas_desde_mapas
from ecowaste.mosaico import proyectar_al_lienzo,combinar_vistas
from ecowaste.datasets import CLASES

VERSION='observador_argmax7_c8_area2_nms050_top100_v1'
ETAPAS=['componentes','area_minima','recorte_contenido','nms_vista','top100_vista',
    'bordes','nms_global','top100_global','final','visual035']


def nms_observada(boxes,scores,labels,limite=100):
    """La lista ilimitada es contrafactual; los primeros 100 son el NMS real."""
    orden=scores.argsort(descending=True,stable=True);elegidos=[];supresiones={}
    while len(orden):
        i=int(orden[0]);elegidos.append(i);resto=orden[1:]
        iou=iou_pares(boxes[i:i+1],boxes[resto]).squeeze(0)
        descartar=(labels[resto]==labels[i])&(iou>.5)
        for j in resto[descartar].tolist():supresiones[j]=i
        orden=resto[~descartar]
    reales=elegidos[:limite];set_real=set(reales)
    descartes={j:{'motivo':'nms','supresor':i} if i in set_real else
        {'motivo':'top100_sin_examinar','supresor':None} for j,i in supresiones.items()}
    for j in elegidos[limite:]:descartes[j]={'motivo':'top100_sin_examinar','supresor':None}
    tensor=torch.tensor(reales,dtype=torch.long)
    if not torch.equal(tensor,nms_por_clase(boxes,scores,labels,.5,limite)):
        raise AssertionError('NMS observado no conserva el prefijo de producción')
    return reales,elegidos,descartes


def empaquetar(nodos,ids):
    return {'boxes':torch.tensor([nodos[i]['caja_global'] for i in ids],dtype=torch.float32).reshape(-1,4),
        'scores':torch.tensor([nodos[i]['puntuacion'] for i in ids]),
        'labels':torch.tensor([nodos[i]['clase'] for i in ids],dtype=torch.long)}


def iguales(a,b):
    return all(torch.equal(a[k],b[k]) for k in ['boxes','scores','labels'])


@torch.inference_mode()
def trazar_logits(logits,geometrias,zonas,ancho,alto,id_imagen,umbral=.05,area_minima=2):
    if logits.ndim!=4 or logits.shape[1]!=7 or len(logits)!=len(geometrias) or len(logits)!=len(zonas):
        raise ValueError('Logits/vistas/geometrías incompatibles')
    if umbral!=.05 or area_minima!=2:raise ValueError('Se preserva el protocolo seleccionado')
    raw=logits.float();prob=raw.softmax(1).cpu();ganador=raw.argmax(1).cpu()
    h,w=raw.shape[-2:];nodos={};etapas={e:[] for e in ETAPAS};vistas=[];parciales=[];ids_bordes=[]
    yy,xx=torch.meshgrid((torch.arange(h)+.5)/h,(torch.arange(w)+.5)/w,indexing='ij')
    for v,(g,z) in enumerate(zip(geometrias,zonas)):
        x0,y0=g['relleno_x']/g['lado'],g['relleno_y']/g['lado']
        x1=x0+g['ancho_original']*g['escala_x']/g['lado'];y1=y0+g['alto_original']*g['escala_y']/g['lado']
        valida=(xx>=x0)&(xx<=x1)&(yy>=y0)&(yy<=y1)
        datos={'vista':v,'ventana':list(z),'geometria':g,'celdas':[],
            'validas':valida.numpy(),'ganador':ganador[v].numpy(),'probabilidades':prob[v].numpy(),
            'componentes':np.zeros((h,w),dtype=np.int32),'ids_componentes':[]}
        candidatos=[]
        for c in range(1,7):
            mask=((prob[v,c]>=umbral)&valida&(ganador[v]==c)).numpy().astype(np.uint8)
            n,cc,estadisticas,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
            for j in range(1,n):
                x,y,bw,bh,area=estadisticas[j].tolist();seleccion=cc==j
                identificador=f'{id_imagen}/v{v}/m{c-1}/cc{j}'
                local=[x/w,y/h,(x+bw)/w,(y+bh)/h]
                boxclip=[max(local[0],x0),max(local[1],y0),min(local[2],x1),min(local[3],y1)]
                score=float(prob[v,c][torch.from_numpy(seleccion)].mean())
                global_box=proyectar_al_lienzo(torch.tensor([local],dtype=torch.float32),g,z,ancho,alto)[0].tolist()
                nodo={'id':identificador,'imagen':id_imagen,'vista':v,'clase':c-1,'material':CLASES[c-1],
                    'puntuacion':score,'caja_vista_sin_recorte':local,'caja_vista':boxclip,'caja_global':global_box,
                    'area_celdas':area,'etapas':['componentes'],'descartada_en':None,'motivo':None,'supresor':None}
                nodos[identificador]=nodo;etapas['componentes'].append(identificador)
                datos['ids_componentes'].append(identificador);ordinal=len(datos['ids_componentes'])
                datos['componentes'][seleccion]=ordinal
                # Celdas son las propuestas anteriores a CC, con relación a su componente.
                for cy,cx in np.argwhere(seleccion):
                    datos['celdas'].append({'id':f'{id_imagen}/v{v}/y{cy}x{cx}','imagen':id_imagen,'vista':v,
                        'clase':c-1,'material':CLASES[c-1],'puntuacion':float(prob[v,c,cy,cx]),
                        'caja_vista':[int(cx)/w,int(cy)/h,(int(cx)+1)/w,(int(cy)+1)/h],
                        'etapa':'antes_componentes','componente':identificador})
                if area<area_minima:
                    nodo.update(descartada_en='area_minima',motivo='menos_de_dos_celdas');continue
                etapas['area_minima'].append(identificador);nodo['etapas'].append('area_minima')
                if boxclip[2]<=boxclip[0] or boxclip[3]<=boxclip[1]:
                    nodo.update(descartada_en='recorte_contenido',motivo='caja_vacia');continue
                nodo['caja_global']=proyectar_al_lienzo(torch.tensor([boxclip],dtype=torch.float32),g,z,ancho,alto)[0].tolist()
                etapas['recorte_contenido'].append(identificador);nodo['etapas'].append('recorte_contenido');candidatos.append(identificador)
        boxes=torch.tensor([nodos[i]['caja_vista'] for i in candidatos],dtype=torch.float32).reshape(-1,4)
        scores=torch.tensor([nodos[i]['puntuacion'] for i in candidatos]);labels=torch.tensor([nodos[i]['clase'] for i in candidatos],dtype=torch.long)
        reales,ilimitados,descartes=nms_observada(boxes,scores,labels)
        for i in ilimitados:
            etapas['nms_vista'].append(candidatos[i]);nodos[candidatos[i]]['etapas'].append('nms_vista')
        for i,d in descartes.items():
            nodos[candidatos[i]].update(descartada_en='nms_vista' if d['motivo']=='nms' else 'top100_vista',
                motivo=d['motivo'],supresor=candidatos[d['supresor']] if d['supresor'] is not None else None)
        pred={'boxes':boxes[reales],'scores':scores[reales],'labels':labels[reales]}
        parciales.append(pred)
        for i in reales:
            identificador=candidatos[i];nodo=nodos[identificador]
            etapas['top100_vista'].append(identificador);nodo['etapas'].append('top100_vista')
            b=deshacer_letterbox(boxes[i:i+1],g)[0];bordes=[]
            if z[0]>0 and b[0]<=1:bordes.append('izquierda_interna')
            if z[1]>0 and b[1]<=1:bordes.append('superior_interno')
            if z[2]<ancho and b[2]>=z[2]-z[0]-1:bordes.append('derecha_interna')
            if z[3]<alto and b[3]>=z[3]-z[1]-1:bordes.append('inferior_interno')
            if bordes:nodo.update(descartada_en='bordes',motivo=','.join(bordes));continue
            etapas['bordes'].append(identificador);nodo['etapas'].append('bordes');ids_bordes.append(identificador)
        vistas.append(datos)
    originales=cajas_desde_mapas(raw,umbral=.05,area_minima=2,geometrias=geometrias,politica='argmax7')
    if not all(iguales(a,b) for a,b in zip(parciales,originales)):raise AssertionError('Cambió la salida de alguna vista')
    p=empaquetar(nodos,ids_bordes);reales,ilimitados,descartes=nms_observada(p['boxes'],p['scores'],p['labels'])
    etapas['nms_global']=[ids_bordes[i] for i in ilimitados]
    for identificador in etapas['nms_global']:nodos[identificador]['etapas'].append('nms_global')
    for i,d in descartes.items():
        nodos[ids_bordes[i]].update(descartada_en='nms_global' if d['motivo']=='nms' else 'top100_global',
            motivo=d['motivo'],supresor=ids_bordes[d['supresor']] if d['supresor'] is not None else None)
    for i in reales:
        identificador=ids_bordes[i];nodo=nodos[identificador];etapas['top100_global'].append(identificador)
        nodo['etapas'].append('top100_global')
        if nodo['puntuacion']>=umbral:
            etapas['final'].append(identificador);nodo['etapas'].append('final')
            if nodo['puntuacion']>=.35:etapas['visual035'].append(identificador);nodo['etapas'].append('visual035')
        else:nodo.update(descartada_en='final',motivo='puntuacion_menor_005')
    final=empaquetar(nodos,etapas['final'])
    base=combinar_vistas(originales,geometrias,zonas,ancho,alto,384,.05)
    if not iguales(final,base):raise AssertionError('La traza modificó las predicciones finales')
    return {'imagen':id_imagen,'nodos':nodos,'etapas':etapas,'vistas':vistas,
        'final':final,'regresion_exacta':True,'nms_sin_limite_es_contrafactual':True}
