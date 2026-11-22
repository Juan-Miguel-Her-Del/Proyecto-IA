"""SAM 2.1 Tiny en entorno CPU separado, sin cambiar EcoWasteFormer ni sus pesos."""
import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import venv
import zipfile

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from ecowaste.incorporacion import leer, publicar, publicar_bytes, sha
from scripts.adquirir_v27_2 import descargar
from scripts.fuentes_v27_2 import SALIDA

REVISION = '2b90b9f5ceec907a1c18123530e92e794ad901a4'
MODELOS = RAIZ / 'datos/nuevos/v27_2/modelos'
ENTORNO = Path.home() / '.venvs/EcoWasteAnotacion'


def ejecutar(cmd):
    print('Ejecutando:', cmd[:5], flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--instalar', action='store_true', required=True)
    ap.add_argument('--pruebas-ui', action='store_true')
    args = ap.parse_args()
    python = ENTORNO / 'Scripts/python.exe'
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(ENTORNO)
    if args.pruebas_ui:
        ejecutar([str(python), '-m', 'pip', 'install', 'playwright==1.56.0'])
        publicar(SALIDA / 'dependencias_pruebas_ui.json', {'pip_freeze': subprocess.check_output(
            [str(python), '-m', 'pip', 'freeze'], text=True), 'navegador': 'Edge instalado, headless; no nuevo navegador'})
        return
    if not (SALIDA / 'dependencias_sam_cpu.json').exists():
        ejecutar([str(python), '-m', 'pip', 'install', 'torch==2.5.1', 'torchvision==0.20.1',
                  '--index-url', 'https://download.pytorch.org/whl/cpu'])
        ejecutar([str(python), '-m', 'pip', 'install', 'numpy>=2,<3', 'pillow>=11,<14',
                  'opencv-python==4.10.0.84', 'hydra-core==1.3.2', 'iopath==0.1.10',
                  'tqdm==4.67.1', 'huggingface-hub==0.26.2'])
        freeze = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
        publicar(SALIDA / 'dependencias_sam_cpu.json', {'entorno': str(ENTORNO), 'pip_freeze': freeze,
                 'entorno_historico_modificado': False, 'dispositivo_predeterminado': 'cpu', 'entrenamiento': False})
    source = MODELOS / 'sam2_fuente.zip'
    if not (SALIDA / 'sam2_fuente.json').exists():
        rec = descargar('https://codeload.github.com/facebookresearch/sam2/zip/' + REVISION,
                        source, 180 * 1024 * 1024)
        rec['revision'] = REVISION
        publicar(SALIDA / 'sam2_fuente.json', rec)
    destino = MODELOS / 'fuente'
    if not (destino / 'sam2').exists():
        with zipfile.ZipFile(source) as z:
            for info in z.infolist():
                rel = PurePosixPath(*PurePosixPath(info.filename).parts[1:])
                if '..' in rel.parts or rel.is_absolute():
                    raise ValueError('Ruta de fuente insegura')
                if (str(rel).startswith('sam2/') and rel.suffix in ['.py', '.yaml']) or str(rel) in ['LICENSE', 'README.md', 'INSTALL.md']:
                    publicar_bytes(destino / str(rel), z.read(info))
    model = MODELOS / 'sam2.1_hiera_tiny.pt'
    if not (SALIDA / 'sam2_archivos_fuente.json').exists():
        publicar(SALIDA / 'sam2_archivos_fuente.json', {'revision': REVISION,
            'archivos': {p.relative_to(destino).as_posix(): sha(p) for p in destino.rglob('*')
                         if p.is_file() and (p.suffix in ['.py', '.yaml'] or p.name in ['LICENSE','README.md','INSTALL.md'])}})
    if not (SALIDA / 'sam2_pesos.json').exists():
        rec = descargar('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
                        model, 180 * 1024 * 1024)
        rec['licencia'] = 'Apache-2.0, pesos y código según repositorio oficial'
        rec['no_es_checkpoint_EcoWasteFormer'] = True
        publicar(SALIDA / 'sam2_pesos.json', rec)
    ejecutar([str(python), '-c', 'import torch,torchvision; print(torch.__version__, torchvision.__version__, "CUDA:",torch.cuda.is_available())'])
    print('SAM 2.1 Tiny preparado en CPU. No se carga al arrancar el editor.', flush=True)


if __name__ == '__main__':
    main()
