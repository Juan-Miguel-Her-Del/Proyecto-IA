"""Foto/MP4/webcam multirresiduo y asistente. No entrena ni modifica la selección histórica."""
from datetime import datetime,timezone
import json,math,time,unicodedata
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageOps
from ecowaste.integracion_d2 import MotorD2,leer_config,RAIZ
from ecowaste.incorporacion import sha

VENTANA='EcoWasteFormer - D2 + clasificador | Q/Esc salir | A asistente'
def guardar_json(p,r):
    with Path(p).open('x',encoding='utf8') as f:json.dump(r,f,indent=2,ensure_ascii=False,allow_nan=False)
def proteger_imagen(p):
    p=Path(p)
    if any(x.casefold() in ['test','generalizacion','generalization'] for x in p.parts):raise ValueError('Test/generalización permanecen reservados')
    reservados=RAIZ/'checkpoints/vision_equipo/reserved_hashes.json'
    if reservados.exists() and sha(p) in set(json.loads(reservados.read_text(encoding='utf8'))):raise ValueError('Imagen reservada por Vision-IA: no procesar')
    particiones=RAIZ/'checkpoints/vision_equipo/particiones_taco.json'
    if particiones.exists():
        clave='/'.join(p.parts[-2:]).casefold();sp={k.casefold():v for k,v in json.loads(particiones.read_text(encoding='utf8'))['por_archivo'].items()}.get(clave)
        if sp=='test':raise ValueError('Fotografía TACO reservada por D2, incluso a otra resolución')
def ascii_text(s):return unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode()
def dibujar(frame,r,latencia=0,respuesta=None):
    out=frame.copy();h,w=out.shape[:2];esc=max(.4,min(.75,w/1200));th=max(1,round(w/1000))
    for d in r['detecciones']:
        x1,y1,x2,y2=map(lambda x:int(round(x)),d['bbox_xyxy']);low=d.get('clasificacion_baja_confianza',False);color=(0,170,255) if low else (50,210,80)
        cv2.rectangle(out,(max(0,x1),max(0,y1)),(min(w-1,x2),min(h-1,y2)),color,max(2,th))
        if 'confianza_detector' in d and 'confianza_clasificador' in d:text=f'{d["material"]}{" ?" if low else ""} | D:{d["confianza_detector"]:.2f} C:{d["confianza_clasificador"]:.2f}'
        else:text=f'{d["material"]}{" ?" if low else ""} | S:{d["puntuacion"]:.2f}'
        cv2.putText(out,text,(max(0,x1),min(h-5,max(85,y1-7))),cv2.FONT_HERSHEY_SIMPLEX,esc,color,th,cv2.LINE_AA)
    cv2.rectangle(out,(0,0),(w-1,min(65,h-1)),(20,20,20),-1)
    for y,t in [(22,'EcoWasteFormer | material predicho - confianza no calibrada'),(48,f'{len(r["detecciones"])} objetos | {latencia:.0f} ms | Q/Esc salir | A: asistente')]:cv2.putText(out,t,(8,y),cv2.FONT_HERSHEY_SIMPLEX,esc,(240,240,240),th,cv2.LINE_AA)
    if respuesta:
        import textwrap
        lines=textwrap.wrap(ascii_text(respuesta),width=max(35,int(w/(9*esc))))[:5];alto=len(lines)*24+10;top=max(68,h-alto);cv2.rectangle(out,(0,top),(w-1,h-1),(20,20,20),-1)
        for k,t in enumerate(lines):cv2.putText(out,t,(8,top+23+k*24),cv2.FONT_HERSHEY_SIMPLEX,esc,(230,240,240),th,cv2.LINE_AA)
    return out
def crear_motor(nombre=None,clasificador=None,dispositivo=None,config=None,checkpoint=None):
    cfg=leer_config(config);nombre=nombre or cfg['motor']
    if nombre in ['d2-transformer','d2-vision']:
        modo=clasificador or ('transformer' if nombre=='d2-transformer' else 'combinado');return MotorD2(cfg,modo,dispositivo)
    if clasificador is not None:raise ValueError('El clasificador intercambiable corresponde a D2')
    if nombre=='original':
        from ecowaste.video import MotorVideo
        m=MotorVideo(RAIZ,vistas=5)
    elif nombre=='hibrido-v30':
        from ecowaste.video_v30 import MotorHibrido
        if checkpoint is None:raise ValueError('hibrido-v30 requiere --checkpoint propio compatible')
        m=MotorHibrido(RAIZ,checkpoint,vistas=5)
    else:raise ValueError('Motor desconocido')
    class Adaptador:
        def __init__(self):self.cfg=cfg;self.asistente=None;self.metadatos={**m.metadatos,'motor':nombre,'clasificador':'cabeza_original_del_detector'}
        def detectar(self,frame):
            r=m.detectar(frame)
            for d in r['detecciones']:d['clasificacion_baja_confianza']=d['puntuacion']<.65;d['tipo_puntuacion']='score_original_del_detector; no hay dos confianzas independientes'
            return r
        def consultar(self,pregunta,r):return MotorD2.consultar(self,pregunta,r)
    return Adaptador()
def fotografia(path,motor,salida,pregunta=None,mostrar=False):
    path=Path(path);proteger_imagen(path);salida=Path(salida)
    if salida.exists():raise FileExistsError('Usar una carpeta de salida nueva')
    with Image.open(path) as im:rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
    frame=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);t=time.perf_counter();r=motor.detectar(frame);lat=(time.perf_counter()-t)*1000;salida.mkdir(parents=True)
    result={**r,'fuente':str(path),'sha256_fuente':sha(path),'latencia_ms':lat,**motor.metadatos};guardar_json(salida/'detecciones.json',result)
    if not cv2.imwrite(str(salida/'original.png'),frame):raise RuntimeError('No se guardó original')
    respuesta=motor.consultar(pregunta,r) if pregunta else None
    if respuesta:guardar_json(salida/'asistente.json',respuesta)
    annotated=dibujar(frame,r,lat,respuesta['respuesta'] if respuesta else None)
    if not cv2.imwrite(str(salida/'anotada.jpg'),annotated):raise RuntimeError('No se guardó foto anotada')
    if mostrar:
        try:
            cv2.namedWindow(VENTANA,cv2.WINDOW_NORMAL);cv2.resizeWindow(VENTANA,1100,760);cv2.imshow(VENTANA,annotated)
            while cv2.waitKey(50)&255 not in [27,ord('q')]:
                if cv2.getWindowProperty(VENTANA,cv2.WND_PROP_VISIBLE)<1:break
        finally:cv2.destroyAllWindows()
    return result
def video(fuente,motor,salida,camara=False,mostrar=False,max_frames=None,pregunta=None,guardar_entrada=False):
    salida=Path(salida)
    if salida.exists():raise FileExistsError('Usar una carpeta de salida nueva')
    if max_frames is not None and max_frames<1:raise ValueError('max_frames debe ser positivo')
    if not camara and not Path(fuente).is_file():raise FileNotFoundError(fuente)
    cap=writer=entrada=log=None;n=0;latencias=[];ultimo=None;respuesta=None;reason='fin_archivo';error=None;inicio=time.perf_counter();resumen={}
    try:
        cap=cv2.VideoCapture(fuente,cv2.CAP_DSHOW) if camara else cv2.VideoCapture(str(fuente))
        if not cap.isOpened():raise RuntimeError('No se pudo abrir la cámara física' if camara else 'No se pudo abrir el video')
        ok,frame=cap.read()
        if not ok or frame is None:raise RuntimeError('La fuente no entrega fotogramas')
        h,w=frame.shape[:2];ww,hh=w+w%2,h+h%2;fps=float(cap.get(cv2.CAP_PROP_FPS));fps=fps if math.isfinite(fps) and 0<fps<=1000 else 30.
        salida.mkdir(parents=True);log=(salida/'frames.jsonl').open('x',encoding='utf8');log.write(json.dumps({'tipo':'cabecera','fecha_utc':datetime.now(timezone.utc).isoformat(),'fuente':'camara' if camara else str(fuente),'fps_salida':fps,'audio_exportado':False,**motor.metadatos})+'\n')
        def escritor(p):
            q=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),fps,(ww,hh))
            if not q.isOpened():raise RuntimeError('Codec MP4 no disponible')
            return q
        writer=escritor(salida/'anotado.mp4')
        if guardar_entrada:entrada=escritor(salida/'entrada_capturada.mp4')
        if mostrar:cv2.namedWindow(VENTANA,cv2.WINDOW_NORMAL);cv2.resizeWindow(VENTANA,1100,760)
        inicio=time.perf_counter()
        while ok and frame is not None:
            if frame.shape[:2]!=(h,w):raise RuntimeError('La resolución cambió durante la captura')
            t=time.perf_counter();r=motor.detectar(frame);lat=(time.perf_counter()-t)*1000;latencias.append(lat);ultimo={'tipo':'frame','indice_frame':n,'tiempo_fuente_segundos':time.perf_counter()-inicio if camara else n/fps,'latencia_ms':lat,**r};log.write(json.dumps(ultimo,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
            anotado=dibujar(frame,r,lat,respuesta['respuesta'] if respuesta else None)
            if (hh,ww)!=(h,w):anotado=cv2.copyMakeBorder(anotado,0,hh-h,0,ww-w,cv2.BORDER_CONSTANT,value=(114,114,114))
            writer.write(anotado)
            if entrada is not None:entrada.write(cv2.copyMakeBorder(frame,0,hh-h,0,ww-w,cv2.BORDER_CONSTANT,value=(114,114,114)))
            n+=1
            if mostrar:
                cv2.imshow(VENTANA,anotado);key=cv2.waitKey(1)&255
                if key in [27,ord('q')]:reason='detenido_por_usuario';break
                if key in [ord('a'),ord('A')]:
                    respuesta=motor.consultar(pregunta or '¿Cómo separo estos residuos?',ultimo);log.write(json.dumps({'tipo':'asistente','indice_frame':n-1,**respuesta},ensure_ascii=False)+'\n');log.flush();print(respuesta['respuesta'],flush=True)
                if cv2.getWindowProperty(VENTANA,cv2.WND_PROP_VISIBLE)<1:reason='ventana_cerrada';break
            if max_frames is not None and n>=max_frames:reason='limite_frames';break
            ok,frame=cap.read()
            if not ok and camara:reason='fin_lectura_camara'
        if pregunta and ultimo:respuesta=motor.consultar(pregunta,ultimo);guardar_json(salida/'asistente_ultimo_frame.json',respuesta)
    except KeyboardInterrupt:reason='interrupcion_usuario'
    except Exception as exc:error=exc;reason='error'
    finally:
        if cap is not None:cap.release()
        if writer is not None:writer.release()
        if entrada is not None:entrada.release()
        if mostrar:cv2.destroyAllWindows()
        elapsed=time.perf_counter()-inicio;resumen={'estado':'error' if error else 'procesado','motivo_fin':reason,'error':str(error) if error else None,'frames_procesados':n,'segundos':elapsed,'fps_procesamiento':n/elapsed if elapsed else None,'latencia_mediana_ms':float(np.median(latencias)) if latencias else None,'latencia_p95_ms':float(np.percentile(latencias,95)) if latencias else None,'ultimo_frame':ultimo,'objetos_unicos_seguidos':None,'audio_exportado':False,'camara_fisica':camara,**motor.metadatos}
        if log is not None:log.write(json.dumps({'tipo':'resumen',**resumen},ensure_ascii=False)+'\n');log.close()
        if salida.exists():guardar_json(salida/'resumen.json',resumen)
    if error:raise error
    return resumen
