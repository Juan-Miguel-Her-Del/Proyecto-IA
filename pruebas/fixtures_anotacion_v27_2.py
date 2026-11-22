"""Figuras descartables para tests; jamás se incorporan al piloto real ni cuotas."""
from pathlib import Path
from PIL import Image, ImageDraw
from ecowaste.incorporacion import publicar, digest, sha
from ecowaste.auxiliares_publicos import VERSION, DESCONOCIDOS, Taller


def fixture(root):
    root = Path(root)
    img = Image.new('RGB', (240, 180), '#eee')
    ImageDraw.Draw(img).rectangle([20, 25, 110, 90], fill='#d4a229')
    img.save(root / 'fixture.png')
    im = {'id': 'fixture:1', 'fuente': 'fixture_descartable_NO_dataset', 'archivo': 'fixture.png',
          'width': 240, 'height': 180, 'sha256_archivo': sha(root / 'fixture.png'),
          'licencia': 'figura unitaria generada solo para pruebas y eliminada',
          **{k: None for k in DESCONOCIDOS},
          'anotaciones': [{'id': 'objeto_1', 'material': 'carton', 'poligonos': [[20,25,110,25,110,90,20,90]],
                          'bbox': [20,25,90,65], 'huecos': [], 'motivo': '', 'origen': 'fixture_sin_validez_real',
                          'descartada': False, 'aceptada_V26': False, 'propuesta_ia_id': None}]}
    p = {'version': VERSION, 'coleccion_id': 'ensayo_fixture', 'imagenes': [im],
         'entrenamiento_autorizado': False, 'aceptadas_V26': 0, 'generalizacion_desbloqueada': False,
         'ensayo_descartable_no_cuotas': True}
    p['huella'] = digest(p)
    proyecto = root / 'proyecto'
    publicar(proyecto / 'PROYECTO.json', p)
    return Taller(proyecto, root)
