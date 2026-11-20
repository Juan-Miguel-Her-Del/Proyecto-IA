"""Documenta la comparación congelada y diagnostica propuestas y objetos recuperados/perdidos."""
import hashlib
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import numpy as np
import torch
from ecowaste.datasets import CLASES,objetivo_deteccion
from ecowaste.cajas import iou_pares
from scripts.entrenar_clasificador import escribir_json

NOMBRE='mapas_v6_ab_propuestas_material'
DOC='documentacion/12_propuestas_por_material_sin_argmax.md'


def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8'))


def detectados(p,o):
    ious=iou_pares(torch.tensor(p['boxes'],dtype=torch.float32).reshape(-1,4),o['boxes']).tolist()
    orden=sorted(range(len(p['boxes'])),key=lambda i:-p['scores'][i])
    usados=set()
    for j in orden:
        posibles=[k for k,c in enumerate(o['labels'].tolist()) if c==p['labels'][j] and k not in usados and ious[j][k]>=.5]
        if posibles:usados.add(max(posibles,key=lambda k:ious[j][k]))
    return usados


def main():
    torch.set_num_threads(4)
    exp=leer(f'resultados/{NOMBRE}.json');cfg=exp['configuracion']
    if exp['estado']!='comparacion_val_completada':raise ValueError('Comparación incompleta')
    pruebas=leer('resultados/pruebas_propuestas_material.json')
    if not pruebas['correcto']:raise ValueError('Hay pruebas fallidas')
    registros=[json.loads(x) for x in (RAIZ/'datos/procesados/deteccion/val.jsonl').read_text(encoding='utf-8').splitlines()]
    objetivos=[objetivo_deteccion(r) for r in registros]
    preds={k:leer(v['predicciones_val']) for k,v in exp['resultados'].items()}
    geometria={}
    for k,pp in preds.items():
        areas=[];scores=[]
        for r,o,p in zip(registros,objetivos,pp):
            if p['id']!=r['id']:raise ValueError('Orden de escenas distinto')
            g=o['geometria']
            area_contenido=r['ancho']*g['escala_x']/g['lado']*r['alto']*g['escala_y']/g['lado']
            areas.extend((b[2]-b[0])*(b[3]-b[1])/area_contenido for b in p['boxes'])
            scores.extend(p['scores'])
        geometria[k]={'mediana_fraccion_area_contenido':float(np.median(areas)),
            'cajas_con_80porciento_del_contenido':sum(a>=.8 for a in areas),
            'mediana_puntuacion_caja':float(np.median(scores))}
    cambios={}
    for vistas in cfg['vistas']:
        base=f'{vistas}_vistas_argmax7';nueva=f'{vistas}_vistas_por_material'
        antes=set();despues=set()
        for r,o,a,b in zip(registros,objetivos,preds[base],preds[nueva]):
            antes.update((r['id'],i) for i in detectados(a,o))
            despues.update((r['id'],i) for i in detectados(b,o))
        ra=exp['resultados'][base]['por_filtro_caja']['0.05']['resumen']
        rb=exp['resultados'][nueva]['por_filtro_caja']['0.05']['resumen']
        if len(antes)!=ra['tp50'] or len(despues)!=rb['tp50']:raise ValueError('Asignación no coincide con el evaluador')
        cambios[str(vistas)]={'objetos_detectados_por_ambas':len(antes&despues),
            'objetos_recuperados_por_variante':len(despues-antes),'objetos_perdidos_por_variante':len(antes-despues),
            'ids_objetos_recuperados':sorted(despues-antes),'ids_objetos_perdidos':sorted(antes-despues),
            'delta_map50':rb['map50']-ra['map50'],'delta_recall50':rb['recall50']-ra['recall50'],
            'delta_fp50':rb['fp50']-ra['fp50']}
    diag={'geometria_y_puntuacion':geometria,'cambios_objetos_iou050_puntuacion005':cambios,
        'test_evaluado':False,'entrenamiento_realizado':False,
        'sha256_comparacion':hashlib.sha256((RAIZ/f'resultados/{NOMBRE}.json').read_bytes()).hexdigest()}
    escribir_json(RAIZ/'resultados/diagnostico_propuestas_material.json',diag)
    l=['# Propuestas por material sin exigir argmax','',
       '**Resultado: con este checkpoint y el umbral fijado, quitar argmax empeora mAP@50 y recall y multiplica los falsos positivos. Se conserva el postprocesamiento actual.**','',
       '## Diseño de la comparación','',
       f"Checkpoint único: `{exp['checkpoint']}`, época {exp['epoca_checkpoint']}. SHA-256 antes y después: `{exp['sha256_checkpoint_antes']}`. No hay entrenamiento, ajuste de pesos, cambio de pérdidas ni evaluación de test.",'',
       'Se comparan dos políticas usando exactamente los mismos logits de cada imagen/vista:', '',
       '- Actual: una celda se admite para material c si gana el argmax de las siete clases y P(c)≥0,05.',
       '- Variante: una celda se admite para material c si P(c)≥0,05, aunque gane fondo u otro material. Una celda puede pertenecer a varios materiales.','',
       'Se mantiene softmax sobre las siete clases; no se renormaliza excluyendo fondo. Componentes conexos de ocho vecinos, área mínima dos celdas, máscara de padding, puntuación media P(c) del componente, NMS por material IoU 0,5 y máximo 100 cajas no cambian. Los mapas siguen siendo semánticos: pueden unir instancias cercanas del mismo material.','',
       'Se compara por separado una vista y cinco vistas: imagen completa más cuatro zonas del 65 %, con las mismas reglas de corte, transformación y combinación ya registradas. No se comparan políticas con distinto número de vistas. Mismas 181 imágenes val, 361 objetos y seis materiales; evaluador propio de 101 puntos con regiones ignoradas, no COCOeval oficial.','',
       'Los umbrales se fijaron antes de medir en configuracion/experimento_propuestas_material.json. No se hizo un barrido para elegir el resultado más favorable. El filtro de puntuación de caja se aplica después de formar las propuestas; 0,35 no cambia las máscaras de celda ni las cajas.','',
       '## Resultados con puntuación de caja ≥0,05','',
       '| Vistas | Política | mAP@50 | mAP@50:95 | Recall@50 | Precisión@50 | TP | FP | FN | Ignoradas |',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k,r in exp['resultados'].items():
        m=r['por_filtro_caja']['0.05']['resumen']
        l.append(f"| {r['vistas']} | {r['politica']} | {m['map50']:.3%} | {m['map50_95']:.3%} | {m['recall50']:.2%} | {m['precision50']:.2%} | {m['tp50']} | {m['fp50']} | {m['fn50']} | {m['predicciones_ignoradas50']} |")
    l+=['','TP, FP y FN se cuentan a IoU≥0,5, con asignación uno a uno y material correcto. Las ignoradas siguen las regiones fuera de alcance y no se suman a FP. Son recuentos sobre todas las imágenes, no por imagen. Las puntuaciones no están calibradas.','',
        '![Comparación A/B](../resultados/figuras/propuestas_material_ab.png)','',
        '## Filtro de visualización ≥0,35 sobre las mismas cajas','',
        '| Vistas | Política | mAP@50 filtrado | Recall@50 | Precisión@50 | TP | FP | FN |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in exp['resultados'].values():
        m=r['por_filtro_caja']['0.35']['resumen']
        l.append(f"| {r['vistas']} | {r['politica']} | {m['map50']:.3%} | {m['recall50']:.2%} | {m['precision50']:.2%} | {m['tp50']} | {m['fp50']} | {m['fn50']} |")
    l+=['','La caída de FP con 0,35 en la variante no es una mejora suficiente: también desaparecen casi todos los TP. El mAP filtrado describe ese subconjunto, no sustituye el mAP principal calculado con 0,05.','',
        '## Qué se está descartando y qué se recupera','',
        'Admisiones material-celda antes de componentes y NMS, para una vista. Pueden contarse varios materiales en la misma celda; no representan objetos confirmados ni píxeles GT:','',
        '| Material | Admitidas actuales | Admitidas sin argmax | Excluidas porque gana fondo | Excluidas porque gana otro material |',
        '|---|---:|---:|---:|---:|']
    for c in CLASES:
        t=exp['admisiones_celdas']['1'][c]
        l.append(f"| {c} | {t['admisiones_argmax']} | {t['admisiones_sin_argmax']} | {t['excluidas_por_fondo_ganador']} | {t['excluidas_por_otro_material_ganador']} |")
    l+=['','Los conteos completos de cinco vistas están en el JSON; las vistas solapadas repiten contenido. Admitir más celdas no garantiza un conjunto de cajas que contenga las anteriores: los componentes pueden cambiar, crecer o fusionarse y variar su puntuación media.','',
        '| Vistas | GT detectados por ambas | GT recuperados por variante | GT perdidos por variante |',
        '|---|---:|---:|---:|']
    for v,t in cambios.items():l.append(f"| {v} | {t['objetos_detectados_por_ambas']} | {t['objetos_recuperados_por_variante']} | {t['objetos_perdidos_por_variante']} |")
    l+=['','Se reprodujo la misma asignación de GT del evaluador; no se reutiliza una caja para contar dos objetos. Los IDs de casos recuperados y perdidos quedan en resultados/diagnostico_propuestas_material.json.','',
        '| Vistas / política | Mediana de puntuación | Mediana de área / contenido | Cajas con ≥80 % del contenido |',
        '|---|---:|---:|---:|']
    for k,t in geometria.items():l.append(f"| {k} | {t['mediana_puntuacion_caja']:.4f} | {t['mediana_fraccion_area_contenido']:.3%} | {t['cajas_con_80porciento_del_contenido']} |")
    l+=['','El aumento de admisiones de baja probabilidad produce numerosas propuestas de baja puntuación y algunas cajas muy extensas. Estos resultados son compatibles con cambios de geometría y fusiones; no prueban una causa única de cada error. Aunque se recuperan algunos GT, se pierden más y crecen los FP.','',
        '![Primera escena de validación](../resultados/figuras/propuestas_material_ejemplo.png)','',
        'El ejemplo es la primera escena val, sin seleccionarla por el resultado. GT azules, propuestas actuales verdes y propuestas de la variante naranjas; se dibujan como máximo las diez de mayor puntuación de cada condición. Las métricas incluyen todas las propuestas. Las anotaciones solo se usan para dibujar y evaluar, nunca para generarlas.','',
        'Con umbral 0,05 y softmax7, ganar argmax implica probabilidad de al menos 1/7. Al quitarlo entran celdas que antes no podían formar componentes, con puntuaciones bastante menores. Esto explica por qué quitar la condición no equivale a conservar las cajas anteriores y añadir únicamente objetos nuevos.','',
        '## AP por material, filtro 0,05','',
        '| Material | Objetos val | 1 vista actual | 1 vista variante | 5 vistas actual | 5 vistas variante |',
        '|---|---:|---:|---:|---:|---:|']
    for c in CLASES:
        mm=[exp['resultados'][k]['por_filtro_caja']['0.05']['metricas']['por_clase'][c] for k in exp['resultados']]
        l.append(f"| {c} | {mm[0]['objetos']} | "+' | '.join(f"{m['ap50']:.3%}" for m in mm)+' |')
    l+=['','## Verificación','',
        f"Pasaron {pruebas['ejecutadas']} pruebas. Se verifican, entre otros casos, propuestas cuando gana fondo, dos materiales sobre la misma celda, riesgo de propuestas en un mapa uniforme, padding, umbral y área mínima. El comportamiento anterior sigue siendo la opción predeterminada.",'',
        'Ambas bases reproducen exactamente las 181 listas anteriores de cajas, etiquetas y puntuaciones, con delta máximo de puntuación cero. Se comparó el state_dict antes/después, se congelaron los parámetros y se comprobó ausencia de gradientes. Los hashes del checkpoint, seis manifiestos y configuración de selección permanecen intactos. No se modificó el clasificador.','',
        'El resultado completo y hashes están en resultados/mapas_v6_ab_propuestas_material.json; cada condición conserva sus predicciones por ID. La configuración predeterminada sigue en detector_mapas_v6_mosaico con argmax y el mismo checkpoint.','',
        '## Ponderación del fondo: experimento siguiente separado','',
        'La pérdida actual usa CE de material, tres veces CE de fondo y Dice. Esta comparación no identifica el coeficiente de fondo óptimo y no demuestra que deba reducirse. El fondo continúa influyendo en softmax aunque se retire la condición argmax.','',
        'Para estudiarlo con entrenamiento: ramificar el mismo checkpoint propio en un control con coeficiente tres y una variante con coeficiente uno; igualar datos, máscaras, inicialización de optimizador, semillas, orden de lotes, aumentos, pérdida auxiliar, tasa de aprendizaje y número de pasos. Mantener argmax en ambas inferencias. Comparar el mismo presupuesto final además de registrar el mejor checkpoint de cada rama. No combinar simultáneamente el cambio de pérdida con la política sin argmax. Este entrenamiento queda pendiente; no se ejecutó como parte del A/B congelado.','',
        '## Reproducción','', '```powershell',
        "$pythonEco = Join-Path $env:USERPROFILE '.venvs\\EcoWasteFormer\\Scripts\\python.exe'",
        '& $pythonEco -m unittest discover -s pruebas',
        '& $pythonEco scripts/comparar_propuestas_material.py --config configuracion/experimento_propuestas_material.json',
        '& $pythonEco scripts/informar_propuestas_material.py',
        'python scripts/graficar_propuestas_material.py',
        '& $pythonEco scripts/verificar_propuestas_material.py','```','',
        'Para repetir el A/B se debe copiar la configuración y elegir otro nombre; los resultados se protegen contra sobrescritura. La validación se usa reiteradamente durante desarrollo, de modo que no es una prueba independiente final. Test permanece reservado; la detección sigue teniendo calidad insuficiente.','']
    (RAIZ/DOC).write_text('\n'.join(l),encoding='utf-8')
    readme=RAIZ/'README.md';p=readme.read_text(encoding='utf-8').split('\n\n')
    parrafo=f"**Experimento de propuestas con pesos congelados:** retirar argmax empeoró el detector de cinco vistas: mAP@50 2,91 % → 1,15 %, recall 20,50 % → 16,62 % y FP 2.374 → 13.055 a puntuación ≥0,05. Se conserva la selección actual. Pasaron {pruebas['ejecutadas']} pruebas; ver [comparación y diagnóstico]({DOC})."
    anterior=next((i for i,t in enumerate(p) if t.startswith('**Experimento de propuestas con pesos congelados:**')),None)
    if anterior is None:p.insert(next(i for i,t in enumerate(p) if t.startswith('**Detector:**'))+1,parrafo)
    else:p[anterior]=parrafo
    readme.write_text('\n\n'.join(p),encoding='utf-8')
    print(json.dumps({'diagnostico':{v:{k:x for k,x in t.items() if not k.startswith('ids_')} for v,t in cambios.items()},
                      'informe':DOC,'seleccion_modificada':False}))


if __name__=='__main__':main()
