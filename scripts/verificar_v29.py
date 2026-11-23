"""Cierre de integridad V29: lectura de históricos, sin inferencia/test/adopción."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from datetime import datetime,timezone
from pathlib import Path
import json,sys
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,sha
from ecowaste.reanudacion_regional import publicar
from ecowaste.experimento_v29 import CONFIG,SALIDA,codigo,validar,Diario


def main():
    cfg=leer(CONFIG);validar(cfg);c=leer(SALIDA/'controles/protocolo_congelado.json')
    if c['codigo']!=codigo() or c['config']!=cfg:raise ValueError('Código/protocolo cambió')
    pack=RAIZ/'datos/nuevos/v29/externos/packwise/PackWISE_dataset_v2.zip'
    if sha(pack)!=leer(pack.with_suffix('.descarga.json'))['sha256']:raise ValueError('ZIP PackWISE original modificado')
    previos=leer(SALIDA/'protegidos_antes.json');fallos=[]
    for i,(p,h) in enumerate(previos.items()):
        if not (RAIZ/p).is_file() or sha(RAIZ/p)!=h:fallos.append(p)
        if (i+1)%4000==0:print('Históricos verificados',i+1,'/',len(previos),flush=True)
    if fallos:
        publicar(SALIDA/('integridad_fallida_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),{'archivos_distintos':fallos})
        raise ValueError('Históricos cambiaron: revisar sin sobrescribir')
    ramas={};hist={};archivos={}
    for r in ['A','B']:
        ck,_=Diario(r,c).leer()
        if ck is None or ck['v29']['paso']!=600:raise ValueError('Presupuesto incompleto')
        hist[r]=ck['v29']['historial'];ramas[r]={'pasos':600,'paso_adamw':2899,'checkpoint_final_sha256':sha(RAIZ/leer(SALIDA/f'entrenamiento/{r}/confirmaciones/paso_0600.json')['checkpoint'])}
        for k in cfg['checkpoints']:
            ev=leer(SALIDA/f'entrenamiento/{r}/confirmaciones/paso_{k:04d}.json');archivos[ev['checkpoint']]=ev['sha256_checkpoint']
    iguales=all(a['huella_entradas_comunes']==b['huella_entradas_comunes'] and a['rng_comun_antes']==b['rng_comun_antes'] and
        a['rng_comun_despues']==b['rng_comun_despues'] for a,b in zip(hist['A'],hist['B']))
    if not iguales:raise ValueError('Difieren entradas o RNG históricos')
    met=leer(SALIDA/'metricas.json')
    if set(met['resultados'])!={'Original','A','B'} or any(set(r)!={'historico_train','nuevo_train','nuevo_retenido'} for r in met['resultados'].values()):raise ValueError('Ámbito de evaluación incorrecto')
    if met['test_usado'] or met['generalizacion_desbloqueada'] or met['adopcion'] or met['validacion_historica_usada']:raise ValueError('Ámbito prohibido')
    col=RAIZ/cfg['coleccion'];cm=leer(col/'MANIFIESTO.json')
    for p,h in cm['archivos'].items():
        if sha(col/p)!=h:raise ValueError('Colección alterada')
    for split in ['train','val']:
        coco=leer(col/f'{split}_coco.json')
        for im in coco['images']:
            e=im['eco']
            if sha(RAIZ/e['anotaciones_originales'])!=e['sha256_anotaciones_originales']:raise ValueError('Anotación fuente modificada')
    pruebas=sorted((SALIDA/'pruebas').glob('*/pruebas.json'))[-1];pr=leer(pruebas)
    if not pr['aprobada'] or pr['codigo']!=codigo():raise ValueError('Pruebas incompatibles')
    control=leer(SALIDA/'controles/controles.json')
    if not control['aprobada'] or control['contrato']!=c:raise ValueError('Controles incompatibles')
    exactitud={}
    for r,h in control['ramas'].items():
        p=SALIDA/f'controles/control_{r}.json'
        if sha(p)!=h:raise ValueError('Control alterado')
        rc=leer(p);exactitud[r]=all(x['exacto'] for x in rc['comparaciones'])
        for comp in rc['comparaciones']:
            if sha(RAIZ/comp['archivo'])!=comp['sha256']:raise ValueError('Comparación alterada')
    for nombre,rr in met['resultados'].items():
        for dom,r in rr.items():
            p=RAIZ/r['analisis']
            if sha(p)!=r['sha256_analisis']:raise ValueError('Métricas posthoc alteradas')
            cc=p.parent/'captura.json';cap=leer(cc);archivos[cc.relative_to(RAIZ).as_posix()]=sha(cc)
            for f,h in cap['archivos'].items():
                if sha(cc.parent/f)!=h:raise ValueError('Predicción/logits/traza alterados')
    for p,h in codigo().items():archivos[p]=h
    archivos[CONFIG.relative_to(RAIZ).as_posix()]=sha(CONFIG)
    archivos[(col/'MANIFIESTO.json').relative_to(RAIZ).as_posix()]=sha(col/'MANIFIESTO.json')
    doc=RAIZ/'documentacion/29_entrenamiento_packwise_click.md';archivos[doc.relative_to(RAIZ).as_posix()]=sha(doc)
    for p in SALIDA.rglob('*'):
        if p.is_file() and p.name not in ['integridad_final.json','recibo_integridad_final.json'] and p.suffix not in ['.lock','.pending']:
            archivos[p.relative_to(RAIZ).as_posix()]=sha(p)
    cierre={'version':'integridad_v29_1','aprobada':True,'historicos_verificados':len(previos),
        'checkpoints_historicos_verificados':sum(p.startswith('checkpoints/') for p in previos),
        'ramas':ramas,'comparaciones_diagnosticas_bit_a_bit':exactitud,'entradas_historicas_A_B_identicas':iguales,
        'pruebas':pr['resultados'],'coleccion':cm['resumen'],'aceptadas_V26':0,'revision_humana_simulada':False,
        'predeterminado_modificado':False,'test_evaluado':False,'generalizacion_desbloqueada':False,
        'archivos':archivos,'referencias_anidadas':'Colección y capturas contienen sus manifiestos internos de cada fotografía, logits y predicciones.'}
    publicar(SALIDA/'integridad_final.json',cierre)
    publicar(SALIDA/'recibo_integridad_final.json',{'sha256':sha(SALIDA/'integridad_final.json')})
    print({k:v for k,v in cierre.items() if k!='archivos'},flush=True)


if __name__=='__main__':main()
