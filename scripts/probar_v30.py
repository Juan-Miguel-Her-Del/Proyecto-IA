"""Ejecuta suite completa sin contaminar el entorno de los tests históricos."""
import os
from pathlib import Path
import re
import subprocess
import sys
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
from ecowaste.experimento_v30 import SALIDA, codigo
from ecowaste.reanudacion_regional import publicar
from ecowaste.incorporacion import publicar_bytes, sha


def main():
    env = os.environ.copy(); env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    cmds = {'suite': [sys.executable, '-m', 'unittest', 'discover', '-s', 'pruebas'],
            'codec_cpu': [str(Path.home()/'.venvs/EcoWasteAnotacion/Scripts/python.exe'), '-m', 'unittest', 'pruebas.test_datos_y_entrenamiento_v29', '-v']}
    rs = {}
    for name, cmd in cmds.items():
        r = subprocess.run(cmd, cwd=RAIZ, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf8', errors='replace')
        p = SALIDA/f'pruebas/{name}.log'; publicar_bytes(p, r.stdout.encode('utf8'))
        n = re.search(r'Ran (\d+) tests', r.stdout)
        rs[name] = {'returncode': r.returncode, 'pruebas': int(n[1]) if n else 0, 'sha256_log': sha(p)}
        print(name, rs[name], flush=True)
    ok = rs['suite']['returncode'] == 0 and rs['suite']['pruebas'] >= 450 and rs['codec_cpu']['returncode'] == 0
    publicar(SALIDA/'pruebas.json', {'aprobada': ok, 'resultados': rs, 'codigo': codigo(), 'anteriores': 428})
    if not ok: raise RuntimeError('Suite fallida; entrenamiento bloqueado')


if __name__ == '__main__': main()
