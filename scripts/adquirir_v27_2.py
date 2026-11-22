"""Descargas públicas verificadas y originales de anotación; no importa GT."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path, PurePosixPath
import sys
import ssl
import urllib.request
import zipfile

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from ecowaste.incorporacion import leer, publicar, publicar_bytes, sha
from scripts.fuentes_v27_2 import SALIDA, DATOS


def descargar(url, destino, limite, esperado=None):
    destino = Path(destino)
    if destino.exists():
        raise FileExistsError('No sobrescribir originales: ' + str(destino))
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + '.partial')
    h = hashlib.sha256()
    total = 0
    req = urllib.request.Request(url, headers={'User-Agent': 'EcoWasteFormer-data-audit/27.2'})
    contexto = ssl.create_default_context()
    # Agregar raíces públicas actualizadas; nunca desactivar validación TLS.
    bundle = Path.home() / '.venvs/EcoWasteAnotacion/Lib/site-packages/certifi/cacert.pem'
    if bundle.is_file():
        contexto.load_verify_locations(cafile=str(bundle))
    with urllib.request.urlopen(req, timeout=60, context=contexto) as r, parcial.open('xb') as f:
        final = r.url
        if int(r.headers.get('Content-Length', '0')) > limite:
            raise ValueError('Archivo excede presupuesto')
        while True:
            b = r.read(min(1024 * 1024, limite - total + 1))
            if not b:
                break
            total += len(b)
            if total > limite:
                raise ValueError('Archivo excede presupuesto')
            h.update(b)
            f.write(b)
            if total % (20 * 1024 * 1024) == 0:
                print(destino.name, total, 'bytes', flush=True)
    if esperado is not None and h.hexdigest() != esperado:
        raise ValueError('SHA256 no coincide con proveedor; conservar partial y detener')
    parcial.rename(destino)
    return {'url': url, 'url_final': final, 'archivo': destino.relative_to(RAIZ).as_posix(),
            'bytes': total, 'sha256': h.hexdigest(), 'sha256_proveedor_comprobado': esperado is not None}


def archivos():
    for nombre in ['wsc', 'dwsd']:
        path = SALIDA / ('archivo_' + nombre + '.json')
        if path.exists():
            continue
        detalles = leer(DATOS / 'fuentes' / (nombre + '_root.bin'))[0]['content_details']
        rec = descargar(detalles['download_url'], DATOS / nombre / 'original.zip',
                        300 * 1024 * 1024, detalles['sha256_hash'])
        if rec['bytes'] != detalles['size']:
            raise ValueError('Tamaño distinto al catálogo')
        publicar(path, rec)
    for nombre in ['wsc', 'dwsd']:
        destino = SALIDA / ('indice_' + nombre + '.json')
        if destino.exists():
            continue
        with zipfile.ZipFile(DATOS / nombre / 'original.zip') as z:
            miembros = [{'nombre': x.filename, 'bytes': x.file_size, 'crc32': f'{x.CRC:08x}'} for x in z.infolist()]
            publicar(DATOS / nombre / 'indice_zip.json', miembros)
            jsons = [x for x in z.infolist() if x.filename.lower().endswith(('.json', '.txt', '.csv')) and not x.is_dir()]
            total = 0
            for x in jsons:
                partes = PurePosixPath(x.filename).parts
                if '..' in partes or PurePosixPath(x.filename).is_absolute() or any(':' in s for s in partes):
                    raise ValueError('Ruta de archivo insegura')
                # No abrir anotaciones externas de evaluación si la fuente las separa.
                if any(s.lower() in ['test', 'val', 'valid', 'validation'] for s in partes):
                    continue
                total += x.file_size
                if x.file_size > 40 * 1024 * 1024 or total > 100 * 1024 * 1024:
                    raise ValueError('Presupuesto de anotaciones excedido')
                publicar_bytes(DATOS / nombre / 'anotaciones_originales' / x.filename, z.read(x))
            exts = Counter(PurePosixPath(x['nombre']).suffix.lower() for x in miembros)
            resumen = {'miembros': len(miembros), 'extensiones': dict(exts),
                       'archivos_anotacion_en_indice': len(jsons), 'bytes_anotaciones_leidas': total,
                       'fotografias_extraidas': 0, 'GT_aceptado': 0, 'test_original_evaluado': False}
            publicar(destino, resumen)
            print(nombre, resumen, 'primeros miembros', [x['nombre'] for x in miembros][:18], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fase', choices=['archivos'], default='archivos')
    ap.parse_args()
    archivos()


if __name__ == '__main__':
    main()
