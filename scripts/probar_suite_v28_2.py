"""Conserva la salida y el código real de retorno de la suite completa."""
from pathlib import Path
import json
import re
import subprocess
import sys
RAIZ=Path(__file__).resolve().parents[1]
SALIDA=RAIZ/'resultados/v28_2_exploratorio'


def main():
    log=SALIDA/'pruebas_finales.log'
    if log.exists():raise FileExistsError('Conservar evidencia previa')
    r=subprocess.run([sys.executable,'-m','unittest','discover','-s','pruebas'],cwd=RAIZ,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace')
    with log.open('x',encoding='utf-8') as f:f.write(r.stdout)
    n=re.search(r'Ran (\d+) tests',r.stdout);cantidad=int(n.group(1)) if n else 0
    cert={'aprobada':r.returncode==0 and cantidad>=322,'pruebas':cantidad,'codigo_retorno':r.returncode,
        'python':sys.executable,'anteriores_conservadas':310,'nuevas':12}
    with (SALIDA/'pruebas_finales.json').open('x',encoding='utf-8') as f:json.dump(cert,f,ensure_ascii=False,indent=2)
    print(cert,flush=True)
    if not cert['aprobada']:raise RuntimeError('Suite fallida; revisar log completo')


if __name__=='__main__':main()
