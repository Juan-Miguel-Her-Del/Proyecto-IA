"""Lanzador local oculto; no sustituye los servicios anteriores ni abre entrenamiento."""
from pathlib import Path
import subprocess
import sys
import urllib.request
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,publicar
from ecowaste.revision_ia_v28_4 import VERSION,sello


def main():
    cfg=leer(RAIZ/'configuracion/revision_ia_v28_4.json');base=leer(RAIZ/cfg['proyecto']/'catalogo.json')
    url=f"http://127.0.0.1:{cfg['puerto']}"
    try:
        with urllib.request.urlopen(url+'/api/config',timeout=3) as r:actual=__import__('json').load(r)
    except OSError:actual=None
    if actual:
        if actual.get('version')!=VERSION or actual.get('proyecto_sha256')!=base['huella']:raise ValueError('Puerto ocupado por otro proyecto; no sustituir')
        if actual.get('reinicio_necesario',True):raise ValueError('El servicio anterior necesita reinicio; no se reemplaza ni se pierden borradores automáticamente')
        print('Editor disponible:',url,flush=True);return
    carpeta=RAIZ/cfg['proyecto']/'logs';carpeta.mkdir(exist_ok=True);marca=sello()
    out=carpeta/(marca+'.log');err=carpeta/(marca+'.err.log')
    with out.open('xb') as f,err.open('xb') as e:
        p=subprocess.Popen([sys.executable,'-u','scripts/anotar_v28_4.py'],cwd=RAIZ,
            stdin=subprocess.DEVNULL,stdout=f,stderr=e,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'DETACHED_PROCESS',0))
    publicar(carpeta/(marca+'_inicio.json'),{'pid':p.pid,'url':url,'python':sys.executable,
        'proyecto_sha256':base['huella'],'stdout':out.name,'stderr':err.name,'SAM_modelo_cargado':False})
    print('Editor iniciado:',url,'PID',p.pid,flush=True)


if __name__=='__main__':main()
