"""Inferencia experimental persistente y E/S local de MP4 o cámara explícita."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
from ecowaste.datasets import CLASES,letterbox
from ecowaste.cajas import deshacer_letterbox
from ecowaste.detector_mapas import cargar_detector_mapas,cajas_desde_mapas
from ecowaste.mosaico import predecir_mosaico

ADVERTENCIA='EXPERIMENTAL - DETECCION NO FIABLE'
LIMITE='Detecciones experimentales con errores frecuentes. Puntuaciones no calibradas. Recuentos por frame, sin seguimiento de objetos.'


class MotorVideo:
    def __init__(self,raiz,checkpoint=None,vistas=5,umbral=.35,dispositivo=None):
        if vistas not in [1,5] or not 0<=umbral<=1:raise ValueError('Vistas o umbral inválidos')
        self.raiz=Path(raiz)
        seleccion=json.loads((self.raiz/'configuracion/arquitectura.json').read_text(encoding='utf-8'))['detector']
        self.checkpoint=Path(checkpoint) if checkpoint else self.raiz/seleccion['checkpoint']
        if not self.checkpoint.is_file():raise FileNotFoundError('Checkpoint ausente')
        torch.set_num_threads(4)
        self.device=dispositivo or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelo,self.ck=cargar_detector_mapas(self.checkpoint,self.device)
        if self.modelo.configuracion_mapas['formulacion']!='softmax7':raise ValueError('Video requiere mapas softmax7')
        self.modelo.requires_grad_(False)
        self.vistas,self.umbral=vistas,float(umbral)
        self.umbral_propuestas=float(self.ck['config_entrenamiento'].get('umbral_evaluacion',.05))
        es_seleccion=self.checkpoint.resolve()==(self.raiz/seleccion['checkpoint']).resolve()
        self.metadatos={'modelo':'EcoWasteFormer-DetMapas','checkpoint':str(self.checkpoint.resolve()),
            'sha256_checkpoint':hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),'epoca_checkpoint':self.ck['epoca'],
            'vistas':vistas,'politica':'argmax7','umbral_formacion_propuestas':self.umbral_propuestas,
            'umbral_puntuacion':self.umbral,'estado_detector':'experimental_validacion_insuficiente',
            'map50_validacion_inferencia':seleccion['map50_validacion'] if es_seleccion and vistas==5 else None,
            'dispositivo':self.device,'limite':LIMITE,'seguimiento_de_objetos':False}

    @torch.inference_mode()
    def detectar(self,frame_bgr):
        imagen=Image.fromarray(cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB))
        entrada,g=letterbox(imagen,self.modelo.imagen)
        amp=self.device=='cuda'
        if self.vistas==5:
            p=predecir_mosaico(self.modelo,imagen,self.umbral,amp=amp,
                umbral_propuestas=self.umbral_propuestas,politica_mapas='argmax7')
        else:
            with torch.autocast(device_type=self.device,dtype=torch.float16,enabled=amp):
                raw=self.modelo(entrada.unsqueeze(0).to(self.device))
            p=cajas_desde_mapas(raw,umbral=self.umbral_propuestas,area_minima=2,geometrias=[g],politica='argmax7')[0]
            keep=p['scores']>=self.umbral;p={k:v[keep] for k,v in p.items()}
        boxes=deshacer_letterbox(p['boxes'],g)
        det=[{'material':CLASES[int(l)],'puntuacion':float(s),'bbox_xyxy':[round(float(v),2) for v in b]}
             for b,s,l in zip(boxes,p['scores'],p['labels']) if b[2]>b[0] and b[3]>b[1]]
        return {'detecciones':det,'estado_detector':'experimental_validacion_insuficiente','limite':LIMITE,
                'coordenadas':'xyxy_pixeles_frame','ancho':imagen.width,'alto':imagen.height}


def anotar(frame,resultado,indice,tiempo_fuente,latencia_ms):
    salida=frame.copy();h,w=salida.shape[:2]
    colores=[(115,160,10),(210,110,20),(0,140,220),(180,70,110),(100,50,220),(40,140,70)]
    for d in resultado['detecciones']:
        x0,y0,x1,y1=[int(round(v)) for v in d['bbox_xyxy']]
        color=colores[CLASES.index(d['material'])]
        cv2.rectangle(salida,(max(0,x0),max(0,y0)),(min(w-1,x1),min(h-1,y1)),color,2)
        cv2.putText(salida,f"{d['material']} {d['puntuacion']:.2f}",(max(0,x0),min(h-5,max(75,y0))),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    alto=min(h,62)
    cv2.rectangle(salida,(0,0),(w-1,alto-1),(15,15,15),-1)
    escala=max(.3,min(.65,(w-16)/750))
    cv2.putText(salida,ADVERTENCIA,(8,min(23,h-3)),cv2.FONT_HERSHEY_SIMPLEX,escala,(0,210,255),1,cv2.LINE_AA)
    texto=f'Frame {indice} | t={tiempo_fuente:.2f}s | {len(resultado["detecciones"])} propuestas | {latencia_ms:.0f} ms'
    cv2.putText(salida,texto,(8,min(47,h-3)),cv2.FONT_HERSHEY_SIMPLEX,escala,(235,235,235),1,cv2.LINE_AA)
    return salida


def procesar_fuente(fuente,motor,salida,tipo='archivo',max_frames=None,mostrar=False,guardar_video=True):
    salida=Path(salida)
    if tipo not in ['archivo','camara']:raise ValueError('Tipo de fuente inválido')
    if salida.exists():raise FileExistsError('La carpeta de salida debe ser nueva para conservar resultados')
    if max_frames is not None and max_frames<=0:raise ValueError('max_frames debe ser positivo')
    if tipo=='archivo':
        fuente=Path(fuente)
        if not fuente.is_file():raise FileNotFoundError('Video de entrada ausente')
    elif not isinstance(fuente,int) or fuente<0:raise ValueError('Índice de cámara inválido')
    captura=None;writer=None;log=None;ventana_abierta=False
    latencias=[];n=0;ultimo=None;resumen=None;razon='fin_archivo';error=None
    try:
        captura=cv2.VideoCapture(str(fuente)) if tipo=='archivo' else cv2.VideoCapture(fuente,cv2.CAP_DSHOW)
        if not captura.isOpened():raise RuntimeError(f'No se puede abrir la fuente {tipo}')
        ok,frame=captura.read()
        if not ok or frame is None:raise RuntimeError('La fuente no entrega un primer frame decodificable')
        fps_reportado=float(captura.get(cv2.CAP_PROP_FPS))
        fps=fps_reportado if math.isfinite(fps_reportado) and 0<fps_reportado<=1000 else 30.
        conteo=float(captura.get(cv2.CAP_PROP_FRAME_COUNT)) if tipo=='archivo' else 0.
        declarados=int(conteo) if tipo=='archivo' and math.isfinite(conteo) and conteo>=0 else None
        h,w=frame.shape[:2];ww,hh=w+w%2,h+h%2
        salida.mkdir(parents=True,exist_ok=False)
        log=(salida/'frames.jsonl').open('w',encoding='utf-8')
        cabecera={'tipo':'cabecera','fecha_utc':datetime.now(timezone.utc).isoformat(),
            'fuente_tipo':tipo,'fuente':str(fuente),'ancho':w,'alto':h,'ancho_video_salida':ww,'alto_video_salida':hh,
            'fps_reportado':fps_reportado if math.isfinite(fps_reportado) else None,'fps_salida':fps,
            'frames_declarados':declarados,'audio_exportado':False,**motor.metadatos}
        log.write(json.dumps(cabecera,ensure_ascii=False)+'\n')
        if guardar_video:
            writer=cv2.VideoWriter(str(salida/'anotado.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(ww,hh))
            if not writer.isOpened():raise RuntimeError('El codificador MP4 no está disponible')
        inicio=time.perf_counter();previo=-1.
        while ok and frame is not None:
            if frame.shape[:2]!=(h,w):raise RuntimeError('La resolución de la fuente cambió durante la lectura')
            ms=float(captura.get(cv2.CAP_PROP_POS_MSEC))
            if tipo=='camara':t=time.perf_counter()-inicio;metodo='reloj_captura'
            elif math.isfinite(ms) and ms>=0 and ms/1000>previo:t=ms/1000;metodo='pos_msec'
            else:t=n/fps;metodo='indice_sobre_fps'
            previo=t;ti=time.perf_counter();resultado=motor.detectar(frame)
            latencia=(time.perf_counter()-ti)*1000;latencias.append(latencia)
            ultimo={'tipo':'frame','indice_frame':n,'tiempo_fuente_segundos':t,'origen_tiempo':metodo,
                    'latencia_inferencia_ms':latencia,**resultado}
            log.write(json.dumps(ultimo,ensure_ascii=False)+'\n');log.flush()
            anotado=anotar(frame,resultado,n,t,latencia)
            if writer is not None:
                if (hh,ww)!=(h,w):anotado=cv2.copyMakeBorder(anotado,0,hh-h,0,ww-w,cv2.BORDER_CONSTANT,value=(114,114,114))
                writer.write(anotado)
            n+=1
            if mostrar:
                cv2.imshow('EcoWasteFormer experimental',anotado);ventana_abierta=True
                if cv2.waitKey(1)&0xFF in [27,ord('q')]:razon='detenido_por_usuario';break
            if max_frames is not None and n>=max_frames:razon='limite_frames';break
            ok,frame=captura.read()
            if not ok and tipo=='camara':razon='fin_lectura_camara'
        duracion=time.perf_counter()-inicio
        if razon=='fin_archivo' and declarados is not None and n<declarados:razon='lectura_incompleta'
        resumen={'estado':'incompleto' if razon=='lectura_incompleta' else 'procesado','motivo_fin':razon,'frames_procesados':n,'duracion_procesamiento_segundos':duracion,
            'fps_procesamiento':n/duracion if duracion else None,'latencia_inferencia_mediana_ms':float(np.median(latencias)),
            'latencia_inferencia_p95_ms':float(np.percentile(latencias,95)),
            'frames_declarados':declarados,'fps_salida':fps,'audio_exportado':False,
            'archivo_video':'anotado.mp4' if guardar_video else None,'archivo_frames':'frames.jsonl',
            'recuento_acumulado_objetos_unicos':None,'ultimo_frame':ultimo,**motor.metadatos}
    except KeyboardInterrupt:
        resumen={'estado':'interrumpido','frames_procesados':n,'ultimo_frame':ultimo,**motor.metadatos}
    except Exception as e:
        error=e
        if salida.is_dir():resumen={'estado':'error','error':str(e),'frames_procesados':n,**motor.metadatos}
    finally:
        if captura is not None:captura.release()
        if writer is not None:writer.release()
        if ventana_abierta:cv2.destroyWindow('EcoWasteFormer experimental')
        if log is not None:
            if resumen:log.write(json.dumps({'tipo':'resumen',**resumen},ensure_ascii=False)+'\n')
            log.close()
        if resumen is not None and salida.is_dir():
            (salida/'resumen.json').write_text(json.dumps(resumen,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if error is not None:raise error
    return resumen
