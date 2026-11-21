"""Verifica la entrega detenida V20, sin entrenar ni inferir; manifiesto inmutable."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.reanudacion_regional import sha,publicar
from scripts.preparar_v20 import huellas_codigo,leer_cfg

MANIFIESTO=RAIZ/'resultados/v20_comparacion/manifiesto_integridad.json'


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def comprobar():
    cfg=leer_cfg();previo=leer(RAIZ/'resultados/v20_ejecucion/estado_previo.json')
    d=leer(RAIZ/cfg['diagnostico']);i=leer(RAIZ/'resultados/v20_ejecucion/incidencia_prueba_corta.json')
    for p,h in previo['historicos'].items():
        if sha(RAIZ/p)!=h:raise ValueError('Histórico alterado: '+p)
    if not d['aprobada'] or d['hashes_codigo']!=huellas_codigo():raise ValueError('Diagnóstico no vigente')
    if d['pruebas']['ejecutadas']!=171 or not d['pruebas']['aprobadas']:raise ValueError('Pruebas no aprobadas')
    if cfg['entrenamiento_completo_habilitado'] or (RAIZ/cfg['prueba_corta']).exists():raise ValueError('Se habilitó entrenamiento tras fallo')
    if list((RAIZ/'checkpoints').glob('v20*.pt')) or list((RAIZ/'datos/cache').glob('v20_piloto_*')):raise ValueError('Pesos V20 inesperados')
    if (RAIZ/f"resultados/{cfg['nombre']}.json").exists() or (RAIZ/f"resultados/{cfg['nombre']}_evaluacion.json").exists():raise ValueError('Resultados completos inesperados')
    if i['actualizaciones_experimento_completo']!={'A':0,'B':0} or i['actualizaciones_reales_del_piloto']['total']!=8:raise ValueError('Presupuesto distinto')
    revisada=previo['configuracion_v20_aprobada']
    cambios={k for k in set(cfg)|set(revisada) if cfg.get(k)!=revisada.get(k)}
    if cambios!={'diagnostico','prueba_corta'}:raise ValueError('Cambio científico o de habilitación inesperado')
    return previo,d,i


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--crear',action='store_true');a=p.parse_args()
    previo,d,i=comprobar()
    if a.crear:
        if MANIFIESTO.exists():raise FileExistsError('Manifiesto existente; no sobrescribir')
        paths=set(huellas_codigo())|{'README.md','configuracion/aprendizaje_discriminativo_mapas.json',
            'scripts/informar_resultados_v20.py','scripts/verificar_entrega_v20.py',
            'documentacion/21_resultados_aprendizaje_discriminativo.md'}
        paths.update(p.relative_to(RAIZ).as_posix() for p in RAIZ.glob('requirements*.txt'))
        for carpeta in ['v20_ejecucion','v20_diagnostico_ejecucion','v20_comparacion']:
            paths.update(p.relative_to(RAIZ).as_posix() for p in (RAIZ/'resultados'/carpeta).rglob('*') if p.is_file() and p!=MANIFIESTO)
        versiones={}
        for nombre in ['torch','numpy','pillow','opencv-python','opencv-python-headless']:
            try:versiones[nombre]=importlib.metadata.version(nombre)
            except importlib.metadata.PackageNotFoundError:pass
        publicar(MANIFIESTO,{'estado':i['estado'],'fecha_utc':datetime.now(timezone.utc).isoformat(),
            'commit_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=RAIZ,text=True).strip(),
            'sha256_checkpoint_predeterminado':leer_cfg()['sha256_checkpoint_origen'],
            'historicos_intactos':len(previo['historicos']),'indice_historicos':'resultados/v20_ejecucion/estado_previo.json',
            'dependencias_entrenador_evaluador_observador':d['hashes_codigo'],
            'archivos':{p:sha(RAIZ/p) for p in sorted(paths)},'python':platform.python_version(),'paquetes_entorno_cuda':versiones,
            'pruebas':d['pruebas'],'piloto_aprobado':False,'actualizaciones_AB':{'A':0,'B':0},
            'test_evaluado':False,'cambio_predeterminado':False,'identidad_numerica_CUDA_no_demostrada':True,
            'sin_firma_criptografica_externa':True})
    m=leer(MANIFIESTO)
    for p,h in m['archivos'].items():
        if sha(RAIZ/p)!=h:raise ValueError('Entrega alterada: '+p)
    print(json.dumps({'estado':m['estado'],'archivos_entrega_verificados':len(m['archivos']),
        'historicos_intactos':len(previo['historicos']),'checkpoint_predeterminado_intacto':True,
        'entrenamiento_habilitado':False,'test_evaluado':False}))


if __name__=='__main__':main()
