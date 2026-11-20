"""Auditoría CPU desde JSON existentes. No carga modelos, no infiere ni entrena."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.datasets import CLASES,objetivo_deteccion
from ecowaste.auditoria_deteccion import auditar_imagen,tamano_objeto,ESTADOS,TAMANOS
from scripts.entrenar_clasificador import escribir_json


def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256((RAIZ/p).read_bytes()).hexdigest()


def agrupar(objetos):
    def resumir(xs):
        counts={s:sum(o['estado']==s for o in xs) for s in ESTADOS};n=len(xs)
        return {'objetos':n,**counts,'fn':n-counts['correcto'],'recall50':counts['correcto']/n if n else None}
    return {'total':resumir(objetos),'por_clase':{c:resumir([o for o in objetos if o['material_real']==c]) for c in CLASES},
        'por_tamano':{t:resumir([o for o in objetos if o['grupo_tamano']==t]) for t in TAMANOS},
        'por_clase_y_tamano':{c:{t:resumir([o for o in objetos if o['material_real']==c and o['grupo_tamano']==t])
                               for t in TAMANOS} for c in CLASES}}


def cobertura_datos(registros):
    resumen={}
    for i,c in enumerate(CLASES):
        escenas=[r for r in registros if any(a['clase']==c for a in r['cajas'])]
        tamanos=[tamano_objeto(b) for r in escenas for b,l in zip(objetivo_deteccion(r)['boxes'],objetivo_deteccion(r)['labels']) if int(l)==i]
        menores=sorted(t['lado_menor_384'] for t in tamanos)
        resumen[c]={'instancias':len(tamanos),'imagenes':len(escenas),
            'grupos_adquisicion':len({r['grupo_adquisicion'] for r in escenas}),
            'por_tamano':{t:sum(s['grupo_tamano']==t for s in tamanos) for t in TAMANOS},
            'lado_menor_minimo_384':min(menores) if menores else None,
            'lado_menor_mediana_384':float(torch.tensor(menores).quantile(.5)) if menores else None,
            'lado_menor_maximo_384':max(menores) if menores else None}
    return resumen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=RAIZ/'configuracion/auditoria_clase_tamano.json')
    args=parser.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if Path(cfg['nombre']).name!=cfg['nombre']:raise ValueError('Nombre inválido')
    if any(cfg[k] for k in ['inferir_nuevamente','entrenar','evaluar_test','cambiar_predeterminado']):raise ValueError('Auditoría fuera de alcance')
    if cfg['iou_caja_razonable']!=.5:raise ValueError('El diagnóstico conserva IoU=0,5 del evaluador')
    salida=RAIZ/f"resultados/{cfg['nombre']}.json";csvpath=RAIZ/f"resultados/{cfg['nombre']}_objetos.csv"
    if salida.exists() or csvpath.exists():raise FileExistsError('Auditoría existente; elegir otro nombre')
    inicio=time.perf_counter();torch.set_num_threads(4)
    seleccion=leer('configuracion/arquitectura.json');hash_sel=sha('configuracion/arquitectura.json')
    hashes_pesos={s:sha(seleccion[s]['checkpoint']) for s in ['detector','clasificador']}
    valid=leer('datos/reportes/validacion.json')
    for p,h in valid['sha256_manifiestos'].items():
        if sha(p)!=h:raise ValueError(f'Manifiesto modificado: {p}')
    splits={s:[json.loads(l) for l in (RAIZ/f'datos/procesados/deteccion/{s}.jsonl').read_text(encoding='utf-8').splitlines()] for s in ['train','val']}
    objetivos=[objetivo_deteccion(r) for r in splits['val']]
    ab=leer('resultados/mapas_v6_ab_propuestas_material.json')
    referencias={1:'resultados/detector_mapas_v6_auxiliar_ext48_predicciones_val.json',5:'resultados/detector_mapas_v6_mosaico_predicciones_val.json'}
    condiciones={};fuentes={};csvfilas=[]
    for vistas in cfg['vistas']:
        referencia=referencias[vistas]
        if sha(referencia)!=ab['regresion_bases'][str(vistas)]['sha256_referencia']:raise ValueError('Predicciones distintas de la base auditada')
        predicciones=leer(referencia);fuentes[referencia]=sha(referencia)
        if len(predicciones)!=len(objetivos):raise ValueError('Predicciones incompletas')
        for umbral in cfg['filtros_puntuacion_existentes']:
            objetos=[];fp=[];totales=Counter()
            for registro,o,p in zip(splits['val'],objetivos,predicciones):
                if p['id']!=registro['id']:raise ValueError('Orden de escenas incorrecto')
                r=auditar_imagen(p,o,umbral,cfg['cobertura_gt_suficiente'])
                totales.update({k:r[k] for k in ['tp','fp','fn','ignoradas','cajas']})
                cajas_gt=[a for a in registro['cajas'] if a['clase']]
                for d in r['objetos']:
                    a=cajas_gt[d['indice_gt']]
                    d.update(id_imagen=registro['id'],id_anotacion=a['id_interno'],material_real=CLASES[d['clase_real']],
                        material_testigo_nombre=CLASES[d['material_testigo']] if d['material_testigo'] is not None else None,
                        ruta_imagen=registro['ruta'],bbox_gt_original_xywh=a['bbox'])
                    objetos.append(d)
                for f in r['falsos_positivos']:f.update(id_imagen=registro['id'],material_predicho=CLASES[f['clase_predicha']]);fp.append(f)
            metricas=ab['resultados'][f'{vistas}_vistas_argmax7']['por_filtro_caja'][str(umbral)]
            esperado=metricas['resumen']
            if (totales['tp'],totales['fp'],totales['fn'],totales['ignoradas'])!=(esperado['tp50'],esperado['fp50'],esperado['fn50'],esperado['predicciones_ignoradas50']):
                raise ValueError('La asignación no reproduce los recuentos existentes')
            stats=agrupar(objetos)
            for c in CLASES:
                mm=metricas['metricas']['por_clase'][c]['por_umbral'][0]
                if stats['por_clase'][c]['correcto']!=mm['tp'] or sum(f['material_predicho']==c for f in fp)!=mm['fp']:
                    raise ValueError(f'Recuento por material incoherente: {c}')
            cm=[[0]*7 for _ in range(6)]
            for d in objetos:
                col=d['clase_real'] if d['estado']=='correcto' else d['material_testigo'] if d['estado']=='material_equivocado' else 6
                cm[d['clase_real']][col]+=1
                csvfilas.append({'vistas':vistas,'umbral_puntuacion':umbral,
                    **{k:d[k] for k in ['id_imagen','id_anotacion','material_real','grupo_tamano','lado_menor_384','estado','subtipo',
                        'material_testigo_nombre','indice_prediccion_testigo','puntuacion_testigo','iou_testigo','cobertura_gt_testigo',
                        'iou_max_cualquier_material','iou_max_material_correcto','cobertura_max_gt','ratio_area_pred_gt']}})
            clave=f'{vistas}_vistas_score_{umbral}'
            condiciones[clave]={'vistas':vistas,'umbral_puntuacion':umbral,'resumen_deteccion_existente':esperado,
                'ap50_por_clase':{c:x['ap50'] for c,x in metricas['metricas']['por_clase'].items()},'recuentos_reproducidos':True,
                'diagnostico_gt':stats,'subtipos_geometria':dict(Counter(d['subtipo'] for d in objetos if d['estado']=='geometria_incorrecta')),
                'tipos_fp':dict(Counter(f['tipo_fp'] for f in fp)),
                'confusion_diagnostica_gt':{'filas':CLASES,'columnas':CLASES+['sin_asignacion_geometrica_univoca'],'matriz':cm},
                'objetos':objetos,'falsos_positivos':fp}
            print(clave,json.dumps(stats['total']),flush=True)
    if sha('configuracion/arquitectura.json')!=hash_sel or any(sha(seleccion[s]['checkpoint'])!=h for s,h in hashes_pesos.items()):
        raise ValueError('Modelo o selección modificados')
    with csvpath.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvfilas[0]));writer.writeheader();writer.writerows(csvfilas)
    informe={'estado':'auditoria_completada','fecha_utc':datetime.now(timezone.utc).isoformat(),'configuracion':cfg,
        'fuentes_predicciones_sha256':fuentes,'fuente_metricas':'resultados/mapas_v6_ab_propuestas_material.json',
        'sha256_fuente_metricas':sha('resultados/mapas_v6_ab_propuestas_material.json'),
        'sha256_manifiestos_invariantes':valid['sha256_manifiestos'],'sha256_seleccion_antes':hash_sel,
        'sha256_seleccion_despues':sha('configuracion/arquitectura.json'),'sha256_pesos':hashes_pesos,
        'condiciones':condiciones,'cobertura_datos':{s:cobertura_datos(rs) for s,rs in splits.items()},
        'csv_objetos':csvpath.relative_to(RAIZ).as_posix(),'sha256_csv_objetos':hashlib.sha256(csvpath.read_bytes()).hexdigest(),
        'inferencias_realizadas':0,'actualizaciones_entrenamiento':0,'gpu_utilizada':False,'dispositivo':'cpu',
        'duracion_segundos':time.perf_counter()-inicio,'test_evaluado':False,'seleccion_modificada':False,
        'limites':['Se auditan cajas finales después de componentes, cortes, NMS y top100; no hay máscaras/logits guardados.',
            'Sin cobertura es una medida sobre cajas finales, no prueba de ausencia de región en el mapa.',
            'La cobertura de una caja puede incluir fondo: no es cobertura de su máscara.',
            'Los tamaños se miden en el lienzo completo 384; las cinco vistas tienen escalas adicionales no identificadas en el JSON.',
            'La atribución de fallos es diagnóstica, no una prueba causal de capas o pérdidas.']}
    escribir_json(salida,informe)
    print(json.dumps({'informe':salida.relative_to(RAIZ).as_posix(),'csv':csvpath.relative_to(RAIZ).as_posix(),
                      'gpu_utilizada':False,'inferencias':0,'test_evaluado':False}))


if __name__=='__main__':main()
