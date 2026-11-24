"""Propone regiones de residuo y clasifica sus recortes con la cabeza auxiliar propia."""
import math
import cv2
import numpy as np
import torch
from ecowaste.datasets import letterbox
from ecowaste.cajas import deshacer_letterbox,nms_por_clase

VERSION='regiones_fg050_recorte_contexto12_auxiliar_v1'


def proponer_regiones(logits,geometria,area_minima=2,max_propuestas=40):
    raw=logits.float().cpu()
    if raw.ndim!=3 or raw.shape[0]!=7:raise ValueError('Se espera un mapa [7,H,W]')
    presencia=1-raw.softmax(0)[0];h,w=presencia.shape;g=geometria
    yy,xx=torch.meshgrid((torch.arange(h)+.5)/h,(torch.arange(w)+.5)/w,indexing='ij')
    x0,y0=g['relleno_x']/g['lado'],g['relleno_y']/g['lado']
    x1=x0+g['ancho_original']*g['escala_x']/g['lado'];y1=y0+g['alto_original']*g['escala_y']/g['lado']
    valida=(xx>=x0)&(xx<=x1)&(yy>=y0)&(yy<=y1)
    mascara=((presencia>=.5)&valida).numpy().astype(np.uint8)
    n,componentes,stats,_=cv2.connectedComponentsWithStats(mascara,connectivity=8)
    cajas=[];puntuaciones=[]
    for i in range(1,n):
        x,y,cw,ch,area=stats[i].tolist()
        if area<area_minima:continue
        cajas.append([max(x/w,x0),max(y/h,y0),min((x+cw)/w,x1),min((y+ch)/h,y1)])
        puntuaciones.append(float(presencia[torch.from_numpy(componentes==i)].mean()))
    boxes=torch.tensor(cajas,dtype=torch.float32).reshape(-1,4);scores=torch.tensor(puntuaciones)
    indices=torch.argsort(scores,descending=True)[:max_propuestas]
    return boxes[indices],scores[indices]


def ventana_contexto(caja,ancho,alto):
    x0,y0,x1,y1=caja
    lado=max(x1-x0,y1-y0)*1.2
    cw,ch=min(ancho,max(1,math.ceil(lado))),min(alto,max(1,math.ceil(lado)))
    x=min(max(0,round((x0+x1-cw)/2)),ancho-cw)
    y=min(max(0,round((y0+y1-ch)/2)),alto-ch)
    return [x,y,x+cw,y+ch]


@torch.inference_mode()
def refinar_imagen(modelo,logits,imagen,geometria,umbral=.05,amp=False):
    if modelo.clasificador_auxiliar is None:raise ValueError('Se requiere cabeza auxiliar entrenada')
    if getattr(modelo,'configuracion_mapas',{}).get('formulacion','softmax7')!='softmax7':
        raise ValueError('El refinamiento requiere mapas softmax7')
    boxes,presencia=proponer_regiones(logits,geometria)
    if not len(boxes):return {'boxes':boxes,'scores':presencia,'labels':torch.empty(0,dtype=torch.long)}
    originales=deshacer_letterbox(boxes,geometria)
    entradas=[]
    for b in originales.tolist():
        ventana=ventana_contexto(b,imagen.width,imagen.height)
        x,_=letterbox(imagen.crop(ventana),384);entradas.append(x)
    device=next(modelo.parameters()).device;probabilidades=[]
    for i in range(0,len(entradas),8):
        x=torch.stack(entradas[i:i+8]).to(device)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):raw=modelo.clasificar(x)
        probabilidades.append(raw.float().softmax(1).cpu())
    confianza,labels=torch.cat(probabilidades).max(1)
    scores=presencia*confianza;keep=scores>=umbral
    boxes,scores,labels=boxes[keep],scores[keep],labels[keep]
    indices=nms_por_clase(boxes,scores,labels,.5,100)
    return {'boxes':boxes[indices],'scores':scores[indices],'labels':labels[indices]}
