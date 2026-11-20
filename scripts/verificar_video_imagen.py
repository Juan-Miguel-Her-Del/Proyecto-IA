"""Contrasta el primer frame decodificado con la consulta de imagen existente."""
import json
import os
from pathlib import Path
import subprocess
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import cv2
from scripts.entrenar_clasificador import escribir_json


def main():
    origen=RAIZ/'resultados/demo_video_entrada.mp4'
    cap=cv2.VideoCapture(str(origen));ok,frame=cap.read();cap.release()
    if not ok:raise RuntimeError('No se pudo leer el primer frame')
    png=RAIZ/'resultados/demo_video_experimental/entrada_frame0.png'
    cv2.imwrite(str(png),frame)
    salida=RAIZ/'resultados/demo_video_experimental/imagen_frame0.json'
    r=subprocess.run([sys.executable,'scripts/detectar_imagen.py',str(png),'--json-salida',str(salida)],
        cwd=RAIZ,capture_output=True,text=True,encoding='utf-8',env=os.environ|{'PYTHONIOENCODING':'utf-8'})
    if r.returncode:raise RuntimeError(r.stderr)
    imagen=json.loads(salida.read_text(encoding='utf-8'))
    registros=[json.loads(l) for l in (RAIZ/'resultados/demo_video_experimental/frames.jsonl').read_text(encoding='utf-8').splitlines()]
    video=next(r for r in registros if r['tipo']=='frame')
    if imagen['detecciones']!=video['detecciones']:raise ValueError('Imagen y video discrepan en el mismo frame')
    informe={'correcto':True,'frame_contrastado':0,'detecciones_identicas':len(video['detecciones']),
        'coordenadas_materiales_y_scores_identicos':True,'vistas':5,'politica':'argmax7',
        'camara_real_activada':False,'test_evaluado':False}
    escribir_json(RAIZ/'resultados/verificacion_video_imagen.json',informe);print(json.dumps(informe))


if __name__=='__main__':main()
