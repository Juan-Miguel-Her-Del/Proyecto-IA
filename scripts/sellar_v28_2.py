"""Manifiesto final, externo a los históricos, con hashes de todos los artefactos nuevos."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
RAIZ=Path(__file__).resolve().parents[1];S=RAIZ/'resultados/v28_2_exploratorio'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    destino=S/'manifiesto_integridad.json'
    if destino.exists():raise FileExistsError('Manifiesto ya sellado; no sobrescribir')
    v=leer(S/'verificacion_final.json');t=leer(S/'pruebas_finales.json')
    if not v['aprobada'] or not t['aprobada'] or t['pruebas']!=322:raise ValueError('Cierre no aprobado')
    informe=RAIZ/'documentacion/28_2_entrenamiento_exploratorio.md'
    if 'Pendientes de incorporar tras finalizar' in informe.read_text(encoding='utf-8'):raise ValueError('Informe incompleto')
    for p,h in leer(S/'historicos_protegidos.json').items():
        if sha(RAIZ/p)!=h:raise ValueError('Histórico alterado: '+p)
    paths=[p for p in S.rglob('*') if p.is_file() and p.suffix!='.lock']
    paths += [p for p in (RAIZ/'datos/nuevos/v28_2').rglob('*') if p.is_file()]
    paths += [RAIZ/'configuracion/entrenamiento_exploratorio_v28_2.json',informe]
    paths += list((RAIZ/'ecowaste').glob('*v28_2.py'))+list((RAIZ/'scripts').glob('*v28_2.py'))+[RAIZ/'pruebas/test_experimento_v28_2.py']
    paths += [RAIZ/p for p in v['checkpoints_experimentales']]
    archivos={p.relative_to(RAIZ).as_posix():sha(p) for p in sorted(set(paths))}
    paquetes={}
    for n in ['torch','numpy','pillow','opencv-python','scipy']:
        try:paquetes[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:paquetes[n]=None
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
    m={'version':'integridad_v28_2_1','aprobada':True,'archivos':archivos,'cantidad_archivos':len(archivos),
        'python':sys.version,'ejecutable':sys.executable,'paquetes_entrenamiento_y_evaluacion':paquetes,'gpu_driver':gpu,
        'backend_y_codigo_congelados':'resultados/v28_2_exploratorio/controles_inferencia_cuda_v2/protocolo_congelado.json',
        'pruebas':322,'ramas_pasos':{'A':300,'B':300},'checkpoint_predeterminado_modificado':False,
        'val_test_generalizacion_evaluados':False,'aceptadas_V26':0,'redistribucion_fotos_autorizada_por_este_manifiesto':False,
        'limite':'Hashes comprueban integridad, no exhaustividad de anotación o validez de generalización. No incluye su propio hash; recibo externo.'}
    with destino.open('x',encoding='utf-8') as f:json.dump(m,f,ensure_ascii=False,indent=2)
    with (S/'sha256_manifiesto.txt').open('x',encoding='utf-8') as f:f.write(sha(destino)+'  manifiesto_integridad.json\n')
    print('Integridad sellada:',len(archivos),'archivos;',sha(destino),flush=True)


if __name__=='__main__':main()
