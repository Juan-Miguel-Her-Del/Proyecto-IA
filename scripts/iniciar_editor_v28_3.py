"""Inicia el servicio local sin ventana y sin cambiar políticas de PowerShell."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import uuid
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,publicar
from ecowaste.correcciones_v28_3 import VERSION
from scripts.anotar_v27_2 import PROYECTO


def main():
    url='http://127.0.0.1:8768';base=leer(PROYECTO/'PROYECTO.json')
    try:
        with urllib.request.urlopen(url+'/api/config',timeout=2) as r:actual=json.load(r)
    except OSError:actual=None
    if actual:
        if actual['version']!=VERSION or actual['proyecto_sha256']!=base['huella']:raise ValueError('Puerto de otra versión/proyecto; no detener ni sustituir')
        print('Editor ya disponible:',url,flush=True);return
    marca=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    carpeta=RAIZ/'datos/nuevos/v28_3/gestion/logs';carpeta.mkdir(parents=True,exist_ok=True)
    out=carpeta/(marca+'.log');err=carpeta/(marca+'.err.log')
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'DETACHED_PROCESS',0)
    with out.open('xb') as f,err.open('xb') as e:
        p=subprocess.Popen([sys.executable,'-u','scripts/anotar_v28_3.py','--puerto','8768'],cwd=RAIZ,
            stdout=f,stderr=e,stdin=subprocess.DEVNULL,creationflags=flags,close_fds=True)
    publicar(carpeta/(marca+'_inicio.json'),{'pid':p.pid,'url':url,'python':sys.executable,
        'proyecto_sha256':base['huella'],'stdout':out.name,'stderr':err.name,'GPU_utilizada':False})
    print('Servicio CPU iniciado:',url,'PID',p.pid,flush=True)


if __name__=='__main__':main()
