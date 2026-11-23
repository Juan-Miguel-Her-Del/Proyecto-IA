"""Visor local de sólo lectura para una publicación IA ya verificada."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifiesto',type=Path,required=True)
    p.add_argument('--puerto',type=int,default=8770);a=p.parse_args()
    m=leer(a.manifiesto);paths=[RAIZ/x for x in m['archivos']]
    carpetas={x.parent.resolve() for x in paths}
    if len(carpetas)!=1:raise ValueError('Sólo una carpeta publicada')
    carpeta=next(iter(carpetas))
    if not carpeta.is_relative_to((RAIZ/'datos/nuevos/v28_3/gestion').resolve()):raise ValueError('Ruta fuera de la colección IA')
    for nombre,h in m['archivos'].items():
        if sha(RAIZ/nombre)!=h:raise ValueError('Publicación alterada: '+nombre)
    class SoloLectura(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            super().end_headers()
        def list_directory(self,path):self.send_error(404,'Sin listado de directorios')
    app=ThreadingHTTPServer(('127.0.0.1',a.puerto),partial(SoloLectura,directory=str(carpeta)))
    print(f'Visor IA: http://127.0.0.1:{a.puerto}/ · sólo lectura · sin entrenamiento',flush=True)
    try:app.serve_forever()
    finally:app.server_close()


if __name__=='__main__':main()
