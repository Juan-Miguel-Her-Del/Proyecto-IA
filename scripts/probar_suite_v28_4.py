"""Suite completa y recibo nuevo; conserva las 371 pruebas anteriores."""
from pathlib import Path
import re
import subprocess
import sys
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import publicar,publicar_bytes,sha
from ecowaste.revision_ia_v28_4 import sello


def main():
    carpeta=RAIZ/'resultados/v28_4_revision_IA/pruebas'/sello()
    r=subprocess.run([sys.executable,'-m','unittest','discover','-s','pruebas'],cwd=RAIZ,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf8',errors='replace')
    publicar_bytes(carpeta/'suite.log',r.stdout.encode('utf8'));m=re.search(r'Ran (\d+) tests',r.stdout);n=int(m.group(1)) if m else 0
    recibo={'aprobada':r.returncode==0 and n>=409,'cantidad_pruebas':n,'anteriores_conservadas':371,
        'nuevas':n-371,'codigo_retorno':r.returncode,'sha256_log':sha(carpeta/'suite.log'),'python':sys.executable}
    publicar(carpeta/'pruebas.json',recibo);print(recibo,flush=True)
    if not recibo['aprobada']:raise RuntimeError('Revisar '+str(carpeta/'suite.log'))


if __name__=='__main__':main()
