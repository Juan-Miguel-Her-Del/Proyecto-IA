"""Integridad y regresion V25; por defecto solo lectura, sin inferencia/entrenamiento."""
import argparse
import io
import json
from pathlib import Path
import re
import sys
import time
import unittest
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import numpy as np
import torch
from ecowaste.datasets import CLASES
from ecowaste.reanudacion_regional import publicar,sha
from scripts.capturar_v25 import SALIDA,CHECKPOINT,SHA,leer
from scripts.auditar_v25 import registros


def comprobar_archivos(raiz,archivos):
    for p,h in archivos.items():
        f=raiz/p
        if not f.is_file() or sha(f)!=h:raise ValueError('Archivo modificado o ausente: '+p)


def controles():
    previo=leer(SALIDA/'protegidos.json');otros={p:h for p,h in previo.items() if p!='README.md'}
    comprobar_archivos(RAIZ,otros)
    if sha(SALIDA/'README_previo.md')!=previo['README.md']:raise ValueError('README historico no conservado')
    stats=leer(SALIDA/'checkpoints_estadisticas.json')
    actuales={p.relative_to(RAIZ).as_posix():[p.stat().st_size,p.stat().st_mtime_ns] for p in (RAIZ/'checkpoints').rglob('*.pt')}
    if stats!=actuales:raise ValueError('Historial de checkpoints alterado')
    if sha(RAIZ/CHECKPOINT)!=SHA:raise ValueError('Referencia cambiada')
    cfg=leer(RAIZ/'configuracion/arquitectura.json')['detector']
    if cfg['checkpoint']!=CHECKPOINT or cfg['modo_inferencia']!='mosaico_cinco_vistas' or cfg['map50_validacion']!=.029122882261380503:
        raise ValueError('Seleccion anterior alterada')
    cap=leer(SALIDA/'captura_train/captura.json');comprobar_archivos(SALIDA,cap['archivos']);comprobar_archivos(RAIZ,cap['codigo'])
    if cap['gt_utilizado'] or cap['entrenamiento'] or cap['test_evaluado'] or cap['regresion_observador_exacta']!=964 or not cap['modelo_antes_despues_identico']:
        raise ValueError('Captura fuera de alcance')
    val=RAIZ/'resultados/v24_v20_evaluacion/referencia_historica';cv=leer(val/'captura.json')
    comprobar_archivos(RAIZ,{x['archivo']:x['sha256'] for x in cv['archivos']})
    entorno=cap['backend']
    if entorno!={**cv['backend_efectivo']}:raise ValueError('Entorno train y val diferente')
    resumen=leer(SALIDA/'analisis/resumen.json');rrs={s:registros(s) for s in ['train','val']}
    for s,carpeta in [('train',SALIDA/'captura_train'),('val',val)]:
        rr=rrs[s];cc=leer(SALIDA/f'censo/instancias_{s}.json');filas=leer(SALIDA/f'analisis/objetos_{s}.json')
        originales={(r['id'],a['id_interno']) for r in rr for a in r['cajas']}
        if len(cc)!=len(originales) or {(f['imagen'],f['id']) for f in cc}!=originales:raise ValueError('Censo incompleto')
        positivos={(r['id'],a['id_interno']) for r in rr for a in r['cajas'] if a['clase']}
        if {(f['imagen'],f['id_gt']) for f in filas}!=positivos:raise ValueError('Analisis positivo incompleto')
        for v in [1,5]:
            ps=leer(carpeta/f'predicciones_{v}v_{s}.json')
            if [x['id'] for x in ps]!=[r['id'] for r in rr]:raise ValueError('Entradas/orden cambiados')
            if v==5:
                for i,(p,r) in enumerate(zip(ps,rr)):
                    tr=torch.load(carpeta/'trazas_5v'/f'{i:03d}.pt',map_location='cpu',weights_only=True)
                    if {k:x.tolist() for k,x in tr['final'].items()}!={k:p[k] for k in ['boxes','labels','scores']}:raise ValueError('Traza no coincide con prediccion')
            if s=='val':
                nombre='detector_mapas_v6_mosaico_predicciones_val.json' if v==5 else 'detector_mapas_v6_auxiliar_ext48_predicciones_val.json'
                if ps!=leer(RAIZ/'resultados'/nombre):raise ValueError('Regresion historica distinta')
            m=resumen['splits'][s]['metricas'][str(v)]['resumen']
            if m['tp50']+m['fn50']!=len(positivos):raise ValueError('Presupuesto GT alterado')
        if sum(f['causa_principal']=='correcto' for f in filas)!=resumen['splits'][s]['metricas']['5']['resumen']['tp50']:
            raise ValueError('TP del analisis incoherentes')
        for imagen in leer(SALIDA/f'censo/imagenes_{s}.json'):
            if sha(RAIZ/imagen['ruta'])!=imagen['archivo_sha256']:raise ValueError('Imagen cambiada tras censo')
        for escena in leer(SALIDA/f'censo/oraculos_{s}.json'):
            esperado={a['id_interno'] for r in rr if r['id']==escena['imagen'] for a in r['cajas'] if a['clase']}
            for g in escena['rejillas']:
                if {f['id'] for f in g['filas']}!=esperado:raise ValueError('Instancia perdida en contrafactual')
    orac=leer(SALIDA/'contrafactuales_resumen.json');historico=leer(RAIZ/'datos/reportes/representacion_instancias_train.json')['totales']
    if orac['train']['48']['total']['fusion_toca']!=historico['instancias_en_componentes_compartidos'] or orac['train']['48']['total']['fragmentacion_componentes']!=historico['instancias_fragmentadas']:
        raise ValueError('Regresion del diagnostico geometrico original')
    org=leer(SALIDA/'organicos_solapes.json')
    if len(org)!=5 or next(f for f in org if f['id']==244)['tokens_mezcla_material']!=0:raise ValueError('Organicos incompletos')
    # Comprobar que la presentacion no enlaza ejemplos inexistentes.
    doc=RAIZ/'documentacion/25_diagnostico_datos_y_representacion.md'
    for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        if not target.startswith('http') and not (doc.parent/target).exists():raise ValueError('Enlace roto: '+target)
    return {'aprobada':True,'archivos_historicos_sha256_intactos':len(otros),'readme_previo_conservado':True,
        'checkpoints_estadisticas_intactas':len(stats),'referencia_original_sha256_intacta':True,
        'regresion_val_exacta':{'una_vista':181,'cinco_vistas':181},'trazas_train_exactas':964,
        'instancias_train_censadas':3267,'positivas_train_analizadas':2505,'test_evaluado':False,
        'entrenamiento':False,'gt_en_inferencia':False,'contrafactual_historico_reproducido':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--publicar',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    inicio=time.perf_counter();resultado=controles();manifest=SALIDA/'manifiesto_integridad.json'
    if a.publicar:
        if manifest.exists():raise FileExistsError('Manifiesto existente; verificar sin --publicar')
        log=io.StringIO();suite=unittest.defaultTestLoader.discover(str(RAIZ/'pruebas'),pattern='test_*.py')
        res=unittest.TextTestRunner(stream=log,verbosity=1).run(suite)
        publicar(SALIDA/'pruebas_finales.json',{'ejecutadas':res.testsRun,'aprobadas':res.wasSuccessful(),'fallos':len(res.failures),
            'errores':len(res.errors),'omitidas':len(res.skipped),'salida':log.getvalue(),'entrenamiento_experimental':False})
        if not res.wasSuccessful() or res.skipped:raise ValueError('Pruebas fallidas')
        code=[*list((RAIZ/'scripts').glob('*v25.py')),RAIZ/'ecowaste/auditoria_datos_representacion.py',RAIZ/'pruebas/test_auditoria_v25.py',
            RAIZ/'configuracion/auditoria_v25.json',RAIZ/'documentacion/25_diagnostico_datos_y_representacion.md',RAIZ/'README.md']
        archivos=[f for f in SALIDA.rglob('*') if f.is_file() and f.name not in ['manifiesto_integridad.json','verificacion.json']]
        publicar(manifest,{'version':'v25_v1','codigo_y_documentacion':{f.relative_to(RAIZ).as_posix():sha(f) for f in code},
            'artefactos':{f.relative_to(SALIDA).as_posix():sha(f) for f in archivos},
            'checkpoints_sha256_final':{f.relative_to(RAIZ).as_posix():sha(f) for f in (RAIZ/'checkpoints').rglob('*.pt')},
            'nota_checkpoints':'SHA inicial/final de referencia; resto estadisticas inicial/final y SHA final. No se afirma doble SHA inicial de todos.',
            'sin_entrenamiento':True,'test_reservado':True})
        resultado.update({'sha256_manifiesto':sha(manifest),'pruebas':res.testsRun,'segundos_verificacion_y_pruebas':time.perf_counter()-inicio})
        publicar(SALIDA/'verificacion.json',resultado)
    else:
        m=leer(manifest);comprobar_archivos(RAIZ,m['codigo_y_documentacion']);comprobar_archivos(SALIDA,m['artefactos']);comprobar_archivos(RAIZ,m['checkpoints_sha256_final'])
        if leer(SALIDA/'verificacion.json')['sha256_manifiesto']!=sha(manifest):raise ValueError('Manifiesto alterado')
        resultado.update({'sha256_manifiesto':sha(manifest),'pruebas':leer(SALIDA/'pruebas_finales.json')['ejecutadas']})
    print(json.dumps(resultado,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
