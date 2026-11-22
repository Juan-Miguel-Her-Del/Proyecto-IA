"""Certifica la revision textual V25 sin reemplazar evidencia ni relajar controles."""
import argparse
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.reanudacion_regional import publicar,sha
from scripts.capturar_v25 import SALIDA,leer
from scripts.verificar_v25 import comprobar_archivos,controles

ALCANCE={'scripts/documentar_v25.py','scripts/informar_v25.py','documentacion/25_diagnostico_datos_y_representacion.md'}


def verificar_original():
    original=leer(SALIDA/'manifiesto_integridad.json')
    diferentes={p for p,h in original['codigo_y_documentacion'].items() if sha(RAIZ/p)!=h}
    if not diferentes<=ALCANCE:raise ValueError('Cambios fuera de presentacion: '+str(diferentes))
    for p in diferentes:
        archivo=SALIDA/'presentacion_revision01'/p
        if not archivo.is_file() or sha(archivo)!=original['codigo_y_documentacion'][p]:raise ValueError('Version previa no conservada: '+p)
    comprobar_archivos(RAIZ,{p:h for p,h in original['codigo_y_documentacion'].items() if p not in diferentes})
    comprobar_archivos(SALIDA,original['artefactos']);comprobar_archivos(RAIZ,original['checkpoints_sha256_final'])
    if leer(SALIDA/'verificacion.json')['sha256_manifiesto']!=sha(SALIDA/'manifiesto_integridad.json'):raise ValueError('Primer certificado alterado')
    pruebas=leer(SALIDA/'pruebas_finales.json')
    if not pruebas['aprobadas'] or pruebas['ejecutadas']!=201 or pruebas['omitidas']:raise ValueError('Pruebas no aprobadas')
    return original,diferentes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--publicar',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    original,diferentes=verificar_original();resultado=controles()
    mp=SALIDA/'manifiesto_integridad_revision02.json';vp=SALIDA/'verificacion_revision02.json'
    if a.publicar:
        if mp.exists() or vp.exists():raise FileExistsError('Certificacion existente; verificar sin --publicar')
        compat={'revision':'presentacion_v25_02','sha256_manifiesto_previo':sha(SALIDA/'manifiesto_integridad.json'),
            'diferencias_presentacion':sorted(diferentes),'alcance_permitido':sorted(ALCANCE),
            'versiones_previas_verificadas_por_sha256':True,'inferencia_y_analisis_numerico_intactos':True,
            'pruebas_no_modificadas':201,'motivo':'Interpretacion macro de organico; mensaje stdout del informe; escape Python y documentacion de reproduccion.',
            'nota':'Versiones originales reconstruidas de los cambios de texto y comprobadas contra SHA previo; no reemplazan el codigo vigente.'}
        publicar(SALIDA/'compatibilidad_presentacion_revision02.json',compat)
        codigos={p:sha(RAIZ/p) for p in original['codigo_y_documentacion']}
        codigos[Path(__file__).relative_to(RAIZ).as_posix()]=sha(Path(__file__))
        archivos=[f for f in SALIDA.rglob('*') if f.is_file() and f not in [mp,vp]]
        nuevo={'version':'v25_v1_presentacion_revision02','codigo_y_documentacion':codigos,
            'artefactos':{f.relative_to(SALIDA).as_posix():sha(f) for f in archivos},
            'checkpoints_sha256_final':original['checkpoints_sha256_final'],'sin_entrenamiento':True,'test_reservado':True,
            'sha256_manifiesto_previo':sha(SALIDA/'manifiesto_integridad.json')}
        publicar(mp,nuevo);resultado.update({'sha256_manifiesto':sha(mp),'pruebas':201,'revision':'presentacion_02'})
        publicar(vp,resultado)
    else:
        m=leer(mp);comprobar_archivos(RAIZ,m['codigo_y_documentacion']);comprobar_archivos(SALIDA,m['artefactos']);comprobar_archivos(RAIZ,m['checkpoints_sha256_final'])
        if leer(vp)['sha256_manifiesto']!=sha(mp):raise ValueError('Manifiesto revision02 alterado')
        resultado.update({'sha256_manifiesto':sha(mp),'pruebas':201,'revision':'presentacion_02'})
    print(json.dumps(resultado,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
