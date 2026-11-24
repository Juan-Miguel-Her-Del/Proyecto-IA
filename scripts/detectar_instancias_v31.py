"""Foto/MP4/webcam con selección explícita de un checkpoint propio V31."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,sys
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
import cv2
from ecowaste.video_instancias_v31 import MotorInstanciasV31
from ecowaste.video import procesar_fuente,anotar
from ecowaste.incorporacion import publicar_bytes
from ecowaste.reanudacion_regional import publicar


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--foto',type=Path);g.add_argument('--video',type=Path);g.add_argument('--camara',type=int)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--salida',type=Path,required=True)
    p.add_argument('--vistas',type=int,choices=[1,5],default=1);p.add_argument('--max-frames',type=int);p.add_argument('--mostrar',action='store_true');p.add_argument('--pregunta');a=p.parse_args()
    if a.salida.exists():p.error('La salida debe ser nueva')
    m=MotorInstanciasV31(RAIZ,a.checkpoint,a.vistas)
    if a.foto:
        frame=cv2.imread(str(a.foto))
        if frame is None:raise ValueError('Foto no legible')
        r=m.detectar(frame);publicar(a.salida/'detecciones.json',{**r,**m.metadatos})
        ok,jpg=cv2.imencode('.jpg',anotar(frame,r,0,0,0))
        if not ok:raise RuntimeError('No se codificó la salida')
        publicar_bytes(a.salida/'anotada.jpg',jpg.tobytes())
    else:
        r=procesar_fuente(a.video if a.video else a.camara,m,a.salida,tipo='archivo' if a.video else 'camara',max_frames=a.max_frames,mostrar=a.mostrar)
        r=r.get('ultimo_frame',{'detecciones':[]})
    if a.pregunta:
        from ecowaste.asistente import Asistente
        publicar(a.salida/'asistente.json',Asistente(RAIZ/'checkpoints/asistente_v1.pt').consultar(a.pregunta,r['detecciones']))
    print(r,flush=True)


if __name__=='__main__':main()
