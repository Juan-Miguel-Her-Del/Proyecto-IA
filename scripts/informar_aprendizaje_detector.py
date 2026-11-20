"""Consolida las variantes de aprendizaje y elige según AP50 val sin ocultar fallos."""
import json
from pathlib import Path

RAIZ=Path(__file__).resolve().parents[1]


def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8'))


def main():
    historico=(RAIZ/'documentacion/09_continuacion_entrenamiento_conjunto.md').exists()
    nombres=['detector_mapas_poligonos_v2','detector_mapas_recortes_v3','detector_mapas_recortes_v4_balanceados',
        'detector_mapas_v5_presencia_material','detector_mapas_v6_auxiliar','detector_mapas_v6_auxiliar_refinado']
    resultados={n:leer(f'resultados/{n}.json') for n in nombres}
    if any(d['estado'] not in ['piloto_completado','reevaluacion_val_completada'] for d in resultados.values()):raise ValueError('Pilotos incompletos')
    if any(len({d['hashes_manifiestos'][k] for d in resultados.values()})!=1 for k in ['train','val','anotaciones_poligonos','inventario_procedencia']):
        raise ValueError('Los pilotos no comparten datos de detección')
    elegido=max(nombres,key=lambda n:resultados[n]['val']['map50']);ganador=resultados[elegido]
    nuevo=resultados['detector_mapas_v6_auxiliar'];base=resultados['detector_mapas_poligonos_v2']
    refinado=elegido.endswith('_refinado')
    origen=nuevo if refinado else ganador
    checkpoint=ganador['checkpoint'] if refinado else ganador['checkpoint_mejor']
    config=leer('configuracion/arquitectura.json');det=config['detector']
    det.update(checkpoint=checkpoint,experimento_seleccionado=elegido,map50_validacion=ganador['val']['map50'],
        map50_95_validacion=ganador['val']['map50_95'],parametros_implementados=origen['parametros'],
        formulacion=origen['configuracion'].get('formulacion','softmax7'),
        auxiliar_clasificacion=bool(origen.get('clasificacion_auxiliar')),
        modo_inferencia='regiones_y_clasificacion_auxiliar' if refinado else 'mapas_directos',
        estado='experimental_validacion_insuficiente',informe='documentacion/08_aprendizaje_y_tarea_auxiliar.md',
        informe_ultimo_experimento='documentacion/08_aprendizaje_y_tarea_auxiliar.md',test_evaluado=False)
    if not historico:(RAIZ/'configuracion/arquitectura.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    diag_sep=leer('resultados/diagnostico_presencia_material.json')
    diag_aux=leer('resultados/diagnostico_auxiliar_clasificacion.json')
    aux=nuevo['clasificacion_auxiliar'];ult=nuevo['historial'][-1]
    l=['# Aprendizaje de materiales y clasificación auxiliar','',
        f"**Estado: mejor mAP@50 val {ganador['val']['map50']:.3%}; calidad todavía insuficiente.** Se conserva el test para una versión estabilizada. No se declara detección fiable ni MVP de video completado.",'',
        '## Separación de presencia y material: ensayo descartado','',
        'Los diagnósticos anteriores mostraban errores de material incluso en train y muchas celdas de residuos clasificadas como fondo. Se ensayó separar un logit binario de presencia de seis logits condicionales de material. Se mantienen 438.727 parámetros y el extractor Transformer propio. Es otra formulación, no una corrección de un error demostrado de softmax.','',
        'La salida anterior usa softmax de siete clases y pérdida CE ponderada de materiales + 3×CE de fondo + Dice por material. La alternativa usa BCE media en positivos + BCE media en negativos + CE condicional de seis materiales + Dice binario de presencia. Se ignoran las celdas 255 en todos los términos. La clase se obtiene separadamente cuando la presencia alcanza 0,5, y la puntuación combina presencia y probabilidad condicional.','',
        'Una prueba comprueba que las celdas de fondo no produzcan gradientes directos en los seis logits de material de la alternativa; los positivos sí los producen. También se prueban escenas sin material, todas las celdas ignoradas y carga de checkpoints anteriores.','',
        f"El diagnóstico en las mismas seis escenas train aprobó en {diag_sep['historial'][-1]['paso']} pasos. El piloto V5 obtuvo AP50 val {resultados['detector_mapas_v5_presencia_material']['val']['map50']:.3%}, por debajo del {base['val']['map50']:.3%} anterior. No se adopta. Cambian pérdida e interpretación de la salida; la comparación no identifica un efecto aislado de cada término.",'',
        '## Nueva supervisión de imagen completa','',
        'Se añadió una tarea auxiliar para aprender características de materiales con las etiquetas de imagen de RealWaste. Se comparten los tokens, bloques de atención y fusión con detección; un promedio de características espaciales alimenta una capa Linear de seis clases. La capa añade 582 parámetros: total 439.309. No usa convoluciones ni un modelo importado.','',
        'El nuevo modelo se inicializa aleatoriamente y entrena simultáneamente ambas tareas. No carga el clasificador anterior, el detector anterior ni pesos del diagnóstico. Las etiquetas globales actualizan el extractor y la cabeza auxiliar; no asignan materiales a píxeles ni se transforman en cajas o máscaras. Una prueba verifica ese recorrido de gradientes.','',
        f"La tarea auxiliar usa exclusivamente **{aux['imagenes_train']} imágenes train**. Sus grupos se verifican frente a val/test de clasificación y detección; no hay intersección. Se comprueban los hashes de manifiesto, cada imagen al construir caché y el archivo de caché antes de cargarlo. El checkpoint guarda los hashes de ambos orígenes y de la caché. No se usan imágenes val/test para entrenar la tarea auxiliar.",'',
        '| Material | Fotos auxiliares train |','|---|---:|']
    for c,n in zip(['plastico','papel','carton','vidrio','metal','organico'],aux['conteos_clase_train']):l.append(f'| {c} | {n} |')
    l+=['','La detección sigue teniendo solo cinco instancias orgánicas anotadas en train y una en val. Las fotos auxiliares aportan supervisión global de orgánicos, sin aumentar ese número de instancias con cajas.','',
        '## Protocolo conjunto','',
        'Se conserva el entrenamiento de mapas softmax7 sobre imágenes TACO completas. Por cada lote de ocho escenas de detección se procesa un lote de 16 fotos de clasificación, muestreadas con reemplazo y probabilidad igual por material. El objetivo conjunto es pérdida de detección + 0,5×CE de imagen. Se suman los gradientes y se realiza un solo paso de AdamW por pareja de lotes.','',
        'Se mantiene semilla 42, learning rate 0,0005, calentamiento de dos épocas, AMP, clipping 1, máximo 16 épocas, mínimo ocho y paciencia seis por AP50 val. Se procesan 964 escenas de detección y 1.936 fotos auxiliares muestreadas por época. Hay más datos y cómputo que en el piloto base; no es una comparación a igual coste. Se guardan los estados aleatorios de ambos cargadores, CUDA, optimizador y scaler para reanudar.','',
        f"La prueba conjunta memorizó las seis escenas TACO fijadas y una foto RealWaste por material en {diag_aux['historial'][-1]['paso']} pasos, cumpliendo AP50≥0,8, recall≥0,9, precisión≥0,7 y clasificación auxiliar=1. Sus pesos se descartan. Al final del piloto la accuracy auxiliar de train aumentado fue {ult['accuracy_aux_train']:.1%}; no es accuracy de prueba ni mAP del detector.",'',
        '![Pérdidas y clasificación auxiliar en train](../resultados/figuras/detector_mapas_v6_auxiliar_auxiliar.png)','',
        '## Validación de detección','',
        'Se evalúan las mismas 181 imágenes completas y 361 objetos, incluidas instancias pequeñas. Mismos manifiestos, clases y evaluador propio de AP de 101 puntos, con regiones ignoradas; no se afirma equivalencia exacta con COCOeval. Se selecciona por AP50 val. El uso reiterado de val durante desarrollo impide tratarla como prueba independiente final.','',
        '| Candidato | Método | Mejor época | mAP@50 | mAP@50:95 | Recall@50 |','|---|---|---:|---:|---:|---:|']
    descripciones=['Contornos, base','Recortes uniformes','Recortes balanceados','Presencia/material separados','Clasificación auxiliar conjunta','Mismos pesos V6 y reclasificación de regiones']
    for n,descripcion in zip(nombres,descripciones):
        d=resultados[n];m=d['val'];epoca=d.get('mejor_epoca',d.get('epoca'))
        l.append(f"| {n} | {descripcion} | {epoca} | {m['map50']:.3%} | {m['map50_95']:.3%} | {m['recall50']:.1%} |")
    l+=['',f"El predeterminado elegido es **{elegido}**, usando {checkpoint}. El aumento frente a la base es de {(ganador['val']['map50']-base['val']['map50'])*100:.3f} puntos porcentuales en AP50. Una sola semilla no demuestra significación estadística. El valor absoluto sigue siendo muy bajo.",'',
        '| Material | Objetos val | AP50 base | AP50 elegido |','|---|---:|---:|---:|']
    for c,m in ganador['val']['por_clase'].items():l.append(f"| {c} | {m['objetos']} | {base['val']['por_clase'][c]['ap50']:.3%} | {m['ap50']:.3%} |")
    l+=['','![Curvas del piloto conjunto](../resultados/figuras/detector_mapas_v6_auxiliar.png)','',
        '![Primeras escenas val, sin selección por aciertos](../resultados/figuras/detector_mapas_v6_auxiliar_comparacion_val.png)','',
        '## Reclasificación de regiones: variante descartada','',
        'Se evaluó una segunda inferencia sobre el mismo checkpoint V6: regiones conexas con probabilidad de residuo ≥0,5, mínimo dos celdas, hasta 40 propuestas y padding excluido. Cada propuesta genera un recorte cuadrado con factor de contexto 1,2 desde la foto original. La cabeza auxiliar clasifica ese recorte; la puntuación multiplica presencia media y confianza de material. Se aplica NMS por clase. Las cajas GT solo intervienen al evaluar después, no al generar propuestas o recortes.','',
        f"Esta variante obtuvo AP50 {resultados['detector_mapas_v6_auxiliar_refinado']['val']['map50']:.3%}; no mejora los mapas directos y no se activa por defecto. La opción --refinar queda disponible para reproducir el ensayo, con sus límites. No se presenta como mejora.",'',
        '## Diagnóstico dentro de entrenamiento','',
        'Se conservan las 964 escenas completas train y los mismos 60 recortes estratificados del diagnóstico anterior. Las métricas siguientes cuentan celdas, no objetos. La accuracy condicional usa posiciones GT conocidas para comprobar material aunque la presencia falle; no representa inferencia autónoma.','',
        '| Modelo | Entrada | Accuracy material en GT | Accuracy condicional a GT | GT material confundido con fondo | IoU macro semántico |',
        '|---|---|---:|---:|---:|---:|']
    for path in ['resultados/ajuste_semantico_presencia_material_train.json','resultados/ajuste_semantico_auxiliar_train.json']:
        for nombre,exp in leer(path)['experimentos'].items():
            for entrada,m in exp['mediciones'].items():
                l.append(f"| {nombre} | {entrada} | {m['accuracy_solo_foreground_gt']:.1%} | {m['accuracy_material_condicionada_a_celdas_GT']:.1%} | {m['foreground_gt_predicho_fondo']:.1%} | {m['iou_macro_seis_materiales']:.1%} |")
    l+=['','## Límites y siguiente trabajo','',
        'La supervisión auxiliar mejora este resultado de validación, pero la detección sigue fallando. Varias clases mantienen AP cero y la muestra orgánica val no permite una estimación estable. La clasificación auxiliar global, la segmentación por celdas y el AP de instancias miden tareas distintas; sus porcentajes no se intercambian.','',
        'Conviene continuar con un presupuesto trazable del entrenamiento conjunto y evaluar si el ajuste de las características sigue mejorando, antes de ampliar la arquitectura. Se necesitan más anotaciones de detección orgánica diversas y escenas vacías. No se han activado la webcam ni el test; video, latencia y evaluación independiente siguen pendientes.','',
        '## Reproducción y evidencias','',
        'Pasaron 55 pruebas: pérdidas, gradientes, carga compatible, supervisión auxiliar y propuestas sin usar GT. Se conservan originales, manifiestos y todos los pilotos descartados.','',
        '```powershell',"$pythonEco = Join-Path $env:USERPROFILE '.venvs\\EcoWasteFormer\\Scripts\\python.exe'",
        '& $pythonEco -m unittest discover -s pruebas',
        '& $pythonEco scripts/diagnosticar_presencia_material.py',
        '& $pythonEco scripts/entrenar_detector_mapas.py --config configuracion/entrenamiento_detector_presencia_material.json',
        '& $pythonEco scripts/diagnosticar_auxiliar_clasificacion.py',
        '& $pythonEco scripts/entrenar_detector_mapas.py --config configuracion/entrenamiento_detector_auxiliar.json',
        '& $pythonEco scripts/evaluar_refinamiento.py --checkpoint checkpoints/detector_mapas_v6_auxiliar_mejor.pt --nombre detector_mapas_v6_auxiliar_refinado','```','',
        'Para repetir diagnósticos o experimentos ya existentes se deben usar nombres nuevos; los scripts protegen los resultados contra sobrescritura. Los JSON de resultados y configuración guardan los valores completos sin redondear.','']
    (RAIZ/'documentacion/08_aprendizaje_y_tarea_auxiliar.md').write_text('\n'.join(l),encoding='utf-8')
    readme=RAIZ/'README.md';parrafos=readme.read_text(encoding='utf-8').split('\n\n')
    for i,p in enumerate(parrafos):
        if p.startswith('**Detector:**'):
            valor=f"{ganador['val']['map50']:.2%}".replace('.',',')
            parametros=f"{origen['parametros']:,}".replace(',','.')
            parrafos[i]=f"**Detector:** variante experimental con {parametros} parámetros y entrenamiento desde cero. Mejor mAP@50 val: **{valor}**; calidad insuficiente. Se evaluaron separación de presencia, clasificación auxiliar con 2.714 fotos train y reclasificación de regiones. Pasaron 55 pruebas. Comparación, curvas y límites en [documentacion/08_aprendizaje_y_tarea_auxiliar.md](documentacion/08_aprendizaje_y_tarea_auxiliar.md). Test reservado."
    if not historico:readme.write_text('\n\n'.join(parrafos),encoding='utf-8')
    print(json.dumps({'elegido':elegido,'map50':ganador['val']['map50'],'map50_95':ganador['val']['map50_95'],'test_evaluado':False},indent=2))


if __name__=='__main__':main()
