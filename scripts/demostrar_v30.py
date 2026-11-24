"""Fotografía/MP4/webcam y asistente: híbrido opcional, referencia predeterminada."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import sys
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import cv2
from ecowaste.video import MotorVideo, procesar_fuente, anotar
from ecowaste.video_v30 import MotorHibrido
from ecowaste.reanudacion_regional import publicar
from ecowaste.incorporacion import publicar_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--foto', type=Path); src.add_argument('--video', type=Path); src.add_argument('--camara', type=int)
    p.add_argument('--modelo', choices=['original', 'hibrido'], default='original')
    p.add_argument('--vistas', type=int, choices=[1, 5], default=5)
    p.add_argument('--salida', type=Path, required=True); p.add_argument('--max-frames', type=int)
    p.add_argument('--pregunta'); p.add_argument('--mostrar', action='store_true')
    a = p.parse_args()
    if a.salida.exists(): p.error('Usar una carpeta nueva; no sobrescribir resultados')
    m = MotorVideo(RAIZ, vistas=a.vistas) if a.modelo == 'original' else MotorHibrido(RAIZ, RAIZ/'checkpoints/v30_hibrido_revision02/B/paso_2400.pt', vistas=a.vistas)
    if a.foto:
        frame = cv2.imread(str(a.foto))
        if frame is None: raise ValueError('Fotografía no legible')
        r = m.detectar(frame)
        publicar(a.salida/'detecciones.json', {**r, **m.metadatos})
        ok, jpg = cv2.imencode('.jpg', anotar(frame, r, 0, 0, 0))
        if not ok: raise RuntimeError('No se puede codificar fotografía')
        publicar_bytes(a.salida/'anotada.jpg', jpg.tobytes())
    else:
        r = procesar_fuente(a.video if a.video else a.camara, m, a.salida, tipo='archivo' if a.video else 'camara',
                            max_frames=a.max_frames, mostrar=a.mostrar)
        r = r.get('ultimo_frame', {'detecciones': []})
    if a.pregunta:
        from ecowaste.asistente import Asistente
        publicar(a.salida/'asistente.json', Asistente(RAIZ/'checkpoints/asistente_v1.pt').consultar(a.pregunta, r['detecciones']))
    print(json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
