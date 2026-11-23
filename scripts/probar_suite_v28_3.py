"""Suite completa conservada en un recibo nuevo por ejecución."""
from datetime import datetime,timezone
from pathlib import Path
import re
import subprocess
import sys
import uuid
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import publicar,publicar_bytes,sha


def main():
    marca=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    carpeta=RAIZ/'resultados/v28_3_correcciones/pruebas'/marca
    r=subprocess.run([sys.executable,'-m','unittest','discover','-s','pruebas'],cwd=RAIZ,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        encoding='utf-8',errors='replace');publicar_bytes(carpeta/'suite.log',r.stdout.encode('utf-8'))
    n=re.search(r'Ran (\d+) tests',r.stdout);n=int(n.group(1)) if n else 0
    recibo={'aprobada':r.returncode==0 and n>=350,'cantidad_pruebas':n,'codigo_retorno':r.returncode,
        'anteriores_conservadas':322,'nuevas':n-322,'sha256_log':sha(carpeta/'suite.log'),'python':sys.executable}
    publicar(carpeta/'pruebas.json',recibo);print(recibo,flush=True)
    if not recibo['aprobada']:raise RuntimeError('Revisar '+str(carpeta/'suite.log'))


if __name__=='__main__':main()
