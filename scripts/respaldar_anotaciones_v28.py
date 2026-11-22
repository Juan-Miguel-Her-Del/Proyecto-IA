"""Respaldo de bytes completo y verificable antes de interpretar las revisiones."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import sha,publicar,bloqueo

ORIGEN=RAIZ/'datos/nuevos/v27_2/proyecto/seleccion_72'
SALIDA=RAIZ/'resultados/v28_validacion_anotaciones'


def inventariar(path,*,sin_lock=False):
    return {p.relative_to(path).as_posix():sha(p) for p in path.rglob('*') if p.is_file() and not (sin_lock and p==path/'registro.lock')}


def respaldar(origen,destino):
    origen,destino=Path(origen).resolve(),Path(destino).resolve()
    if destino.exists() or destino.is_relative_to(origen) or origen.is_relative_to(destino):
        raise ValueError('Respaldo debe ser nuevo y separado del origen')
    # Este es el mismo bloqueo de los guardados del editor. No se vacía el archivo.
    contenido_lock=(origen/'registro.lock').read_bytes() if (origen/'registro.lock').exists() else b'0'
    with bloqueo(origen/'registro.lock'):
        antes=inventariar(origen,sin_lock=True)
        shutil.copytree(origen,destino,copy_function=shutil.copy2,ignore=shutil.ignore_patterns('registro.lock'))
        (destino/'registro.lock').write_bytes(contenido_lock)
        copia=inventariar(destino,sin_lock=True)
        despues=inventariar(origen,sin_lock=True)
    if antes!=copia or antes!=despues or (origen/'registro.lock').read_bytes()!=contenido_lock:
        raise ValueError('Cambió el proyecto durante la copia; conservar intento y repetir en otra ruta')
    return inventariar(destino)


def main():
    if SALIDA.exists():raise FileExistsError('No reemplazar evidencia V28 existente')
    marca=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    destino=RAIZ/'datos/nuevos/v28/respaldos'/marca/'proyecto'
    hashes=respaldar(ORIGEN,destino)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=RAIZ).decode('utf-8').split('\0')
    paths=[RAIZ/p for p in tracked if p]
    paths.extend((RAIZ/'datos/procesados').rglob('*.jsonl'))
    paths.extend([RAIZ/'checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt',RAIZ/'datos/originales/taco/annotations.json',
                  RAIZ/'datos/reportes/inventario_imagenes.jsonl'])
    publicar(SALIDA/'historicos_protegidos.json',{p.relative_to(RAIZ).as_posix():sha(p) for p in paths if p.is_file()})
    publicar(SALIDA/'checkpoints_previos.json',{p.relative_to(RAIZ).as_posix():[p.stat().st_size,p.stat().st_mtime_ns]
            for p in (RAIZ/'checkpoints').rglob('*.pt')})
    publicar(SALIDA/'respaldo_verificado.json',{'origen':ORIGEN.relative_to(RAIZ).as_posix(),
        'destino':destino.relative_to(RAIZ).as_posix(),'archivos':hashes,'cantidad_archivos':len(hashes),
        'bytes':sum((destino/p).stat().st_size for p in hashes),'copias_y_origen_identicos':True,
        'incluye_todas_las_revisiones_y_propuestas_IA':True,'fecha_UTC':datetime.now(timezone.utc).isoformat(),
        'entrenamiento':False,'test_evaluado':False})
    # Logs activos: copia de los bytes observados, no se exige que dejen de crecer.
    logs={}
    for p in (ORIGEN.parent/'logs').glob('*'):
        if not p.is_file():continue
        copia=destino.parent/'logs_observados'/p.name
        copia.parent.mkdir(parents=True,exist_ok=True)
        blob=p.read_bytes();copia.write_bytes(blob)
        logs[p.name]={'archivo':copia.relative_to(RAIZ).as_posix(),'bytes':len(blob),'sha256':sha(copia)}
    publicar(SALIDA/'logs_respaldados.json',logs)
    print({'respaldo':str(destino),'archivos':len(hashes),'eventos_revision':sum(p.startswith('revisiones/') for p in hashes)},flush=True)


if __name__=='__main__':main()
