"""Evidencia pública V27.2, acotada y separada de datos históricos."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from ecowaste.incorporacion import sha, leer, publicar, publicar_bytes
from scripts.investigar_v27 import obtener

SALIDA = RAIZ / 'resultados/v27_2_anotacion'
DATOS = RAIZ / 'datos/nuevos/v27_2/externos'

FUENTES = {
    'wsc_pagina': 'https://data.mendeley.com/datasets/3wnf3n242s/4',
    'wsc_snapshot': 'https://api.data.mendeley.com/datasets/3wnf3n242s/snapshot?version=4',
    'wsc_frontend': 'https://data.mendeley.com/public-api/datasets/3wnf3n242s/versions/4',
    'dwsd_pagina': 'https://data.mendeley.com/datasets/gr99ny6b8p/1',
    'dwsd_snapshot': 'https://api.data.mendeley.com/datasets/gr99ny6b8p/snapshot?version=1',
    'dwsd_frontend': 'https://data.mendeley.com/public-api/datasets/gr99ny6b8p/versions/1',
    'realwaste_uci': 'https://archive.ics.uci.edu/dataset/908/realwaste',
    'sam2_readme': 'https://raw.githubusercontent.com/facebookresearch/sam2/main/README.md',
    'sam2_licencia': 'https://raw.githubusercontent.com/facebookresearch/sam2/main/LICENSE',
    'sam2_instalacion': 'https://raw.githubusercontent.com/facebookresearch/sam2/main/INSTALL.md',
    'sam2_version': 'https://api.github.com/repos/facebookresearch/sam2/commits/main',
    'cvat_instalacion': 'https://docs.cvat.ai/docs/administration/community/advanced/installation_automatic_annotation/',
    'mendeley_bundle': 'https://data.mendeley.com/datasets/bundle.js?c2c2cda645a5424c6f37',
    'wsc_archivos': 'https://data.mendeley.com/public-api/datasets/3wnf3n242s/files?version=4',
    'dwsd_archivos': 'https://data.mendeley.com/public-api/datasets/gr99ny6b8p/files?version=1',
    'wsc_root': 'https://data.mendeley.com/public-api/datasets/3wnf3n242s/files?folder_id=root&version=4&$start=0&$limit=1000',
    'dwsd_root': 'https://data.mendeley.com/public-api/datasets/gr99ny6b8p/files?folder_id=root&version=1&$start=0&$limit=1000',
    'sam2_pagina': 'https://github.com/facebookresearch/sam2',
}


def proteger():
    if (SALIDA / 'historicos_protegidos.json').exists():
        raise FileExistsError('La evidencia previa no se reinicializa')
    nombres = subprocess.check_output(['git', 'ls-files', '-z'], cwd=RAIZ).decode('utf-8').split('\0')
    paths = [RAIZ / n for n in nombres if n]
    paths.extend((RAIZ / 'datos/procesados').rglob('*.jsonl'))
    paths.extend([RAIZ / 'checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt',
                  RAIZ / 'datos/originales/taco/annotations.json', RAIZ / 'datos/reportes/inventario_imagenes.jsonl'])
    publicar(SALIDA / 'historicos_protegidos.json', {p.relative_to(RAIZ).as_posix(): sha(p) for p in paths if p.is_file()})
    publicar(SALIDA / 'checkpoints_previos.json', {p.relative_to(RAIZ).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
                                                for p in (RAIZ / 'checkpoints').rglob('*.pt')})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preparar', action='store_true')
    ap.add_argument('--reintentar', action='store_true')
    args = ap.parse_args()
    if args.preparar:
        proteger()
    for nombre, url in FUENTES.items():
        recibo = SALIDA / 'consultas' / (nombre + '.json')
        if recibo.exists():
            if not args.reintentar or 'error' not in leer(recibo):
                continue
            recibo = SALIDA / 'consultas' / (nombre + '_revision02.json')
            if recibo.exists():
                continue
        try:
            data, r = obtener(url)
            p = DATOS / 'fuentes' / (nombre + '.bin')
            publicar_bytes(p, data)
            r['archivo_local'] = p.relative_to(RAIZ).as_posix()
            publicar(recibo, r)
            print(nombre, len(data), flush=True)
            if 'snapshot' in nombre or 'frontend' in nombre:
                j = json.loads(data)
                print(nombre, str(j)[:2000].encode('ascii', 'backslashreplace').decode(), flush=True)
        except Exception as e:
            publicar(recibo, {'url': url, 'error': str(e)})
            print(nombre, str(e), flush=True)


if __name__ == '__main__':
    main()
