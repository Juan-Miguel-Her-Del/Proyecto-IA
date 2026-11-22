"""Editor local de máscaras y SAM 2 CPU. No sube imágenes ni habilita entrenamiento."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
import sys
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser
import urllib.request
import urllib.error

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import numpy as np
from PIL import Image
from ecowaste.auxiliares_publicos import Taller, MATERIALES, VERSION
from ecowaste.asistente_sam2 import AsistenteSAM2

PROYECTO = RAIZ / 'datos/nuevos/v27_2/proyecto/seleccion_72'
ESTATICOS = RAIZ / 'interfaz/anotacion_v27_2'


def servidor(taller, motor, puerto=8767):
    csrf = secrets.token_urlsafe(32)
    vistas = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(fmt % args, flush=True)

        def enviar(self, blob, tipo='application/json; charset=utf-8', status=200, descargar=False):
            self.send_response(status)
            self.send_header('Content-Type', tipo)
            self.send_header('Content-Length', str(len(blob)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'")
            if descargar:
                self.send_header('Content-Disposition', 'attachment; filename="ecowaste_auxiliar_COCO_pendiente.json"')
            self.end_headers()
            self.wfile.write(blob)

        def json(self, data, status=200):
            self.enviar(json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8'), status=status)

        def seguridad(self, post=False):
            hosts = [f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}']
            if self.headers.get('Host') not in hosts:
                raise ValueError('Host local requerido')
            if post:
                if self.headers.get('Origin') not in [None, *['http://' + h for h in hosts]]:
                    raise ValueError('Origen externo rechazado')
                if self.headers.get('X-Eco-Token') != csrf:
                    raise ValueError('Token local incorrecto')
                if self.headers.get_content_type() != 'application/json':
                    raise ValueError('JSON requerido')

        def do_GET(self):
            try:
                self.seguridad()
                url = urlsplit(self.path)
                q = parse_qs(url.query)
                if url.path in ['/', '/app.js', '/estilo.css']:
                    nombre, tipo = {'/': ('index.html', 'text/html; charset=utf-8'),
                                    '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                                    '/estilo.css': ('estilo.css', 'text/css; charset=utf-8')}[url.path]
                    self.enviar((ESTATICOS / nombre).read_bytes(), tipo)
                elif url.path == '/api/config':
                    self.json({'version': VERSION, 'token': csrf, 'materiales': MATERIALES, 'dispositivo': 'cpu',
                               'GPU_utilizada': False, 'proyecto_sha256': taller.base['huella'], 'entrenamiento': False})
                elif url.path == '/api/lista':
                    self.json([{'id': i['id'], 'fuente': i['fuente'], 'estado': taller.tarea(i['id'])['estado']}
                               for i in taller.imagenes.values()])
                elif url.path == '/api/tarea':
                    iid = q['id'][0]
                    t = taller.tarea(iid, independiente=q.get('modo', ['asistida'])[0] == 'independiente')
                    token = secrets.token_urlsafe(24)
                    with lock:
                        # Límite de memoria; una página vieja puede recargar sin perder sus diarios.
                        if len(vistas) >= 1000:
                            vistas.pop(next(iter(vistas)))
                        vistas[token] = {'id': iid, 'modo': t['modo'], 'base': t['revision_base'], 'revision': t['revision']}
                    self.json({**t, 'token_vista': token})
                elif url.path == '/api/imagen':
                    path = taller.archivo_imagen(q['id'][0])
                    self.enviar(path.read_bytes(), 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg')
                elif url.path == '/api/semantica':
                    path = taller.archivo_imagen(q['id'][0], semantica=True)
                    with Image.open(path) as im:
                        a = np.asarray(im)
                    colores = np.array([[i * 67 % 256, i * 137 % 256, i * 199 % 256] for i in range(256)], dtype=np.uint8)
                    out = io.BytesIO()
                    Image.fromarray(colores[a]).save(out, format='PNG')
                    self.enviar(out.getvalue(), 'image/png')
                else:
                    self.json({'error': 'Ruta inexistente'}, 404)
            except (ValueError, KeyError, OSError) as e:
                self.json({'error': str(e)}, 400)

        def do_POST(self):
            try:
                self.seguridad(post=True)
                n = int(self.headers.get('Content-Length', '0'))
                if not 0 < n <= 12 * 1024 * 1024:
                    raise ValueError('Tamaño de solicitud inválido')
                def invalido(v):
                    raise ValueError('Número JSON no finito: ' + v)
                d = json.loads(self.rfile.read(n), parse_constant=invalido)
                ruta = urlsplit(self.path).path
                if ruta in ['/api/guardar', '/api/sam']:
                    vista = vistas.get(d.get('token_vista'))
                    if not vista or vista['id'] != d.get('imagen_id'):
                        raise ValueError('Vista inválida; recargar tarea')
                    if ruta == '/api/guardar':
                        if d.get('revision') != vista['revision']:
                            raise ValueError('No corresponde a la versión visualizada')
                        taller.guardar(d['imagen_id'], d, modo=vista['modo'], revision_base=vista['base'])
                        with lock:
                            vistas.pop(d['token_vista'], None)
                        self.json({'guardado': True, 'aceptada_V26': False})
                    else:
                        self.json(motor.proponer(taller, d['imagen_id'], d['indicaciones']))
                elif ruta == '/api/exportar':
                    self.enviar(json.dumps(taller.exportar_coco(), ensure_ascii=False, allow_nan=False).encode('utf-8'), descargar=True)
                elif ruta == '/api/liberar':
                    motor.liberar()
                    self.json({'liberado': True, 'GPU_utilizada': False})
                else:
                    self.json({'error': 'Ruta inexistente'}, 404)
            except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as e:
                self.json({'error': str(e)}, 400)
            except Exception as e:
                # Fallos de SAM también se muestran al usuario sin guardar una revisión.
                self.json({'error': f'{type(e).__name__}: {e}'}, 500)

    return ThreadingHTTPServer(('127.0.0.1', puerto), Handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--puerto', type=int, default=8767)
    ap.add_argument('--proyecto', type=Path, default=PROYECTO)
    ap.add_argument('--abrir', action='store_true')
    args = ap.parse_args()
    taller = Taller(args.proyecto, RAIZ)
    if args.abrir:
        existente = None
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.puerto}/api/config',timeout=2) as r:
                existente = json.load(r)
        except (OSError, ValueError):
            pass
        if existente:
            if existente.get('proyecto_sha256') != taller.base['huella']:
                raise ValueError('Puerto ocupado por otro proyecto; no reutilizar silenciosamente')
            webbrowser.open(f'http://127.0.0.1:{args.puerto}')
            return
    app = servidor(taller, AsistenteSAM2(RAIZ), args.puerto)
    url = f'http://127.0.0.1:{app.server_port}'
    print('Editor local:', url, '| CPU; sin entrenamiento ni aceptación V26', flush=True)
    if args.abrir:
        webbrowser.open(url)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.server_close()


if __name__ == '__main__':
    main()
