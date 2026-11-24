"""Entrada integrada predeterminada D2 + Transformer; foto, MP4, webcam y asistente."""
import argparse,json,sys
from datetime import datetime
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.aplicacion_integrada import crear_motor,fotografia,video
from ecowaste.integracion_d2 import MODOS
def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--foto',type=Path);g.add_argument('--video',type=Path);g.add_argument('--camara',type=int)
    p.add_argument('--motor',choices=['d2-transformer','d2-vision','original','hibrido-v30']);p.add_argument('--clasificador',choices=MODOS);p.add_argument('--dispositivo',choices=['cpu','cuda','auto']);p.add_argument('--config',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--salida',type=Path);p.add_argument('--max-frames',type=int);p.add_argument('--mostrar',action='store_true');p.add_argument('--sin-ventana',action='store_true');p.add_argument('--guardar-entrada',action='store_true');p.add_argument('--pregunta');a=p.parse_args()
    if a.camara is not None and a.camara<0:p.error('Índice de cámara inválido')
    out=a.salida or RAIZ/'datos/nuevos/integracion_d2/sesiones'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if out.exists():p.error('La salida debe ser una carpeta nueva')
    m=crear_motor(a.motor,a.clasificador,a.dispositivo,a.config,a.checkpoint);print('Motor cargado:',m.metadatos,flush=True);mostrar=(a.mostrar or a.camara is not None) and not a.sin_ventana
    if a.foto:r=fotografia(a.foto,m,out,a.pregunta,mostrar)
    else:r=video(a.camara if a.camara is not None else a.video,m,out,a.camara is not None,mostrar,a.max_frames,a.pregunta,a.guardar_entrada)
    print(json.dumps({'salida':str(out),'detecciones_ultimo_frame':len(r.get('detecciones',r.get('ultimo_frame',{}).get('detecciones',[]))),'estado':r.get('estado','fotografia_procesada')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
