"""Convierte cajas públicas del piloto en PROPUESTAS SAM, sin revisiones humanas."""
import json
from pathlib import Path
import sys
import time
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from ecowaste.auxiliares_publicos import Taller
from ecowaste.asistente_sam2 import AsistenteSAM2
from ecowaste.incorporacion import publicar, leer, digest, poligonos_validos
from scripts.fuentes_v27_2 import SALIDA

PROYECTO = RAIZ / 'datos/nuevos/v27_2/proyecto/seleccion_72'


def main():
    taller = Taller(PROYECTO, RAIZ)
    indice = PROYECTO / 'PROPUESTAS_IA_INICIALES.json'
    if indice.exists():
        raise FileExistsError('Índice IA existente; no reemplazar propuestas/correcciones')
    sam = AsistenteSAM2(RAIZ)
    por_imagen, trazas, errores = {}, [], []
    inicio = time.perf_counter()
    for im in taller.imagenes.values():
        if im['fuente'] != 'warp':
            continue
        for a in im['anotaciones']:
            if a['poligonos']:
                continue
            x, y, w, h = a['bbox']
            try:
                p = sam.proponer(taller, im['id'], {'caja': [x, y, x + w, y + h]})
                # Una propuesta inválida se conserva en evidencia, pero no se ofrece
                # como polígono listo para revisar: se mantiene la caja original.
                poligonos_validos(p['resultado']['poligonos'], im['width'], im['height'])
                por_imagen.setdefault(im['id'], {})[a['id']] = p['id']
                trazas.append({'imagen': im['id'], 'instancia_auxiliar': a['id'], 'propuesta': p['id'],
                               **{k: v for k, v in p['resultado'].items() if k not in ['poligonos', 'huecos', 'rle_ia_original']}})
            except ValueError as e:
                errores.append({'imagen': im['id'], 'instancia_auxiliar': a['id'], 'error': str(e)})
            print(im['id'], a['id'], 'propuestas', len(trazas), 'incidencias', len(errores), flush=True)
    publicar(indice, {'proyecto_sha256': taller.base['huella'], 'por_imagen': por_imagen,
                     'anotaciones_humanas': 0, 'aceptadas_V26': 0})
    publicar(SALIDA / 'piloto_sam2_cpu.json', {'modelo': 'SAM 2.1 Hiera Tiny', 'propuestas_ofrecidas': len(trazas),
             'incidencias': errores, 'trazas': trazas, 'segundos_totales': time.perf_counter() - inicio,
             'memoria_GPU_bytes': 0, 'entrenamiento': False, 'anotaciones_humanas': 0, 'aceptadas_V26': 0})
    sam.liberar()


if __name__ == '__main__':
    main()
