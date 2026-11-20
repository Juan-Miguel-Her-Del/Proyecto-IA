"""Procesa MP4 o una cámara solicitada explícitamente con el detector experimental."""
import argparse
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.video import MotorVideo,procesar_fuente


def main():
    p=argparse.ArgumentParser(description=__doc__)
    fuente=p.add_mutually_exclusive_group(required=True)
    fuente.add_argument('--video',type=Path);fuente.add_argument('--camara',type=int)
    p.add_argument('--salida',type=Path,required=True,help='Carpeta nueva para MP4, JSONL y resumen')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--vistas',type=int,choices=[1,5],default=5)
    p.add_argument('--umbral',type=float,default=.35);p.add_argument('--max-frames',type=int)
    p.add_argument('--mostrar',action='store_true');p.add_argument('--sin-video',action='store_true')
    p.add_argument('--pregunta',help='Consulta al asistente usando solo las detecciones del último frame')
    a=p.parse_args()
    if not 0<=a.umbral<=1 or a.max_frames is not None and a.max_frames<=0:p.error('Umbral o límite de frames inválidos')
    if a.salida.exists():p.error('La carpeta de salida debe ser nueva')
    if a.video is not None and not a.video.is_file():p.error('El archivo de entrada no existe')
    if a.camara is not None and a.camara<0:p.error('Índice de cámara inválido')
    motor=MotorVideo(RAIZ,a.checkpoint,a.vistas,a.umbral)
    r=procesar_fuente(a.video if a.video is not None else a.camara,motor,a.salida,
        tipo='archivo' if a.video is not None else 'camara',max_frames=a.max_frames,
        mostrar=a.mostrar,guardar_video=not a.sin_video)
    if a.pregunta and r.get('ultimo_frame'):
        from ecowaste.asistente import Asistente
        respuesta=Asistente(RAIZ/'checkpoints/asistente_v1.pt').consultar(a.pregunta,r['ultimo_frame']['detecciones'])
        respuesta.update(estado_detector=r['estado_detector'],limite_detector=r['limite'],
            origen_detecciones='solo_ultimo_frame_video_experimental',indice_frame=r['ultimo_frame']['indice_frame'])
        (a.salida/'asistente_ultimo_frame.json').write_text(json.dumps(respuesta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:r[k] for k in ['estado','frames_procesados','estado_detector','vistas']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
