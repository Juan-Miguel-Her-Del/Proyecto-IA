"""Prueba técnica con montaje de fotos val: no es video independiente ni evaluación de calidad."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import cv2
from PIL import Image,ImageOps
import numpy as np
from scripts.entrenar_clasificador import escribir_json


def main():
    destino=RAIZ/'resultados/demo_video_experimental'
    entrada=RAIZ/'resultados/demo_video_entrada.mp4'
    if entrada.exists() or destino.exists():raise FileExistsError('Prueba real existente; conservar evidencia')
    config=RAIZ/'configuracion/arquitectura.json'
    antes=hashlib.sha256(config.read_bytes()).hexdigest()
    registros=[json.loads(l) for l in (RAIZ/'datos/procesados/deteccion/val.jsonl').read_text(encoding='utf-8').splitlines()[:4]]
    writer=cv2.VideoWriter(str(entrada),cv2.VideoWriter_fourcc(*'mp4v'),4,(320,240))
    if not writer.isOpened():raise RuntimeError('No se pudo crear MP4 de control')
    for r in registros:
        with Image.open(RAIZ/r['ruta']) as im:
            im=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(320,240))
            lienzo=Image.new('RGB',(320,240),(114,114,114));lienzo.paste(im,((320-im.width)//2,(240-im.height)//2))
            frame=cv2.cvtColor(np.array(lienzo),cv2.COLOR_RGB2BGR)
        writer.write(frame);writer.write(frame)
    writer.release()
    ejec=subprocess.run([sys.executable,'scripts/detectar_video.py','--video',str(entrada),'--salida',str(destino),
        '--pregunta','¿Cómo separo estos residuos?'],cwd=RAIZ,capture_output=True,text=True,encoding='utf-8',
        env=os.environ|{'PYTHONIOENCODING':'utf-8'})
    if ejec.returncode:raise RuntimeError(ejec.stderr)
    resumen=json.loads((destino/'resumen.json').read_text(encoding='utf-8'))
    lineas=[json.loads(l) for l in (destino/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    frames=[r for r in lineas if r['tipo']=='frame']
    if resumen['frames_procesados']!=8 or len(frames)!=8:raise ValueError('No se procesaron todos los frames')
    if any(f['estado_detector']!='experimental_validacion_insuficiente' for f in frames):raise ValueError('Advertencia perdida')
    if resumen['politica']!='argmax7' or resumen['vistas']!=5:raise ValueError('No se usó la configuración conservada')
    captura=cv2.VideoCapture(str(destino/'anotado.mp4'));n=0
    while True:
        ok,frame=captura.read()
        if not ok:break
        if frame.shape[:2]!=(240,320):raise ValueError('Resolución inesperada')
        if frame[:60].max()<100:raise ValueError('Banner no renderizado')
        n+=1
    captura.release()
    if n!=8:raise ValueError('MP4 de salida incompleto')
    respuesta=json.loads((destino/'asistente_ultimo_frame.json').read_text(encoding='utf-8'))
    if respuesta['indice_frame']!=7 or respuesta['estado_detector']!=resumen['estado_detector']:raise ValueError('Asistente no vinculado al último frame')
    if antes!=hashlib.sha256(config.read_bytes()).hexdigest():raise ValueError('Se cambió la selección')
    informe={'correcto':True,'entrada':entrada.relative_to(RAIZ).as_posix(),'salida':destino.relative_to(RAIZ).as_posix(),
        'frames_entrada':8,'frames_inferidos':8,'frames_mp4_exportados':n,'vistas':5,'politica':'argmax7',
        'camara_real_activada':False,'video_independiente':False,'fuentes_fotos_val':[r['id'] for r in registros],
        'proposito':'Prueba de lectura, inferencia real, anotación, exportación y conexión al asistente; no mide precisión de video.',
        'mediana_inferencia_ms':resumen['latencia_inferencia_mediana_ms'],'p95_inferencia_ms':resumen['latencia_inferencia_p95_ms'],
        'fps_procesamiento':resumen['fps_procesamiento'],'asistente_ultimo_frame_verificado':True,
        'sha256_entrada':hashlib.sha256(entrada.read_bytes()).hexdigest(),
        'sha256_salida':hashlib.sha256((destino/'anotado.mp4').read_bytes()).hexdigest(),
        'sha256_seleccion_inalterada':antes,'audio_exportado':False,'test_evaluado':False}
    escribir_json(RAIZ/'resultados/verificacion_video_real.json',informe)
    print(json.dumps(informe))


if __name__=='__main__':main()
