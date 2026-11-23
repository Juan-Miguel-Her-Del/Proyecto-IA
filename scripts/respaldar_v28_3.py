"""Respaldo íntegro antes de V28.3, sin crear revisiones humanas."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
import uuid
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from scripts.respaldar_anotaciones_v28 import respaldar,ORIGEN
from ecowaste.incorporacion import leer,publicar,sha
from ecowaste.auxiliares_publicos import Taller
SALIDA=RAIZ/'resultados/v28_3_correcciones'


def main():
    if (SALIDA/'respaldo.json').exists():raise FileExistsError('Respaldo ya documentado; no reemplazar')
    antiguo=leer(RAIZ/'resultados/v28_validacion_anotaciones/respaldo_verificado.json')
    for p,h in antiguo['archivos'].items():
        if sha(RAIZ/antiguo['destino']/p)!=h or sha(ORIGEN/p)!=h:raise ValueError('Cambió un archivo de la instantánea humana: '+p)
    marca=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    destino=RAIZ/'datos/nuevos/v28_3/respaldos'/marca/'proyecto'
    hashes=respaldar(ORIGEN,destino);t=Taller(destino,RAIZ)
    eventos=sum(len(t.diario(i)) for i in t.imagenes)
    if eventos<134:raise ValueError('Faltan revisiones históricas')
    publicar(SALIDA/'respaldo.json',{'origen':ORIGEN.relative_to(RAIZ).as_posix(),'destino':destino.relative_to(RAIZ).as_posix(),
        'archivos':hashes,'cantidad_archivos':len(hashes),'eventos_humanos':eventos,'cadenas_verificadas':True,
        'proyecto_sha256':t.base['huella'],'sin_nuevas_revisiones_humanas':True})
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=RAIZ).decode().split('\0')
    paths=[RAIZ/p for p in tracked if p]
    paths += list((RAIZ/'checkpoints').rglob('*.pt'))+list((RAIZ/'datos/procesados').rglob('*.jsonl'))
    paths += [p for p in (RAIZ/'resultados').rglob('*') if p.is_file() and not p.is_relative_to(SALIDA)]
    publicar(SALIDA/'historicos_protegidos.json',{p.relative_to(RAIZ).as_posix():sha(p) for p in paths if p.is_file()})
    print('Respaldo V28.3: '+str(len(hashes))+' archivos, '+str(eventos)+' eventos humanos; históricos protegidos',flush=True)


if __name__=='__main__':main()
