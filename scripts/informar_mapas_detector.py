"""Consolida resultados de mapas, sin convertir selección val en aprobación de calidad."""
import json
from pathlib import Path

RAIZ=Path(__file__).resolve().parents[1]


def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8'))


def main():
    historico=(RAIZ/'documentacion/07_recortes_multiescala.md').exists()
    nombres=['detector_piloto_v6','detector_mapas_v1','detector_mapas_poligonos_v2']
    resultados={n:leer(f'resultados/{n}.json') for n in nombres}
    if any(d['estado']!='piloto_completado' for d in resultados.values()):
        raise ValueError('Esperar a que terminen los pilotos')
    if any(len({d['hashes_manifiestos'][s] for d in resultados.values()})!=1 for s in ['train','val']):
        raise ValueError('Las particiones de los pilotos son distintas')
    reevaluacion=leer('resultados/detector_mapas_v1_padding_v2.json')
    if reevaluacion['sha256_val']!=resultados['detector_mapas_v1']['hashes_manifiestos']['val']:
        raise ValueError('Reevaluación incompatible')
    # Comparar mapas con el mismo filtrado de padding; conservar el informe histórico.
    metricas={n:d['val'] for n,d in resultados.items()}
    metricas['detector_mapas_v1']=reevaluacion['val']
    elegido=max(nombres,key=lambda n:metricas[n]['map50'])
    d=resultados[elegido];mapas=elegido.startswith('detector_mapas')
    a=leer('configuracion/arquitectura.json')
    det={'imagen':384,'patch_detalle':8,'patch_contexto':16,'embedding':96,
        'bloques_por_rama':2,'cabezas':4,'mlp':192,'ventana_tokens':8,
        'parametros_teoricos_pdf':440182,'parametros_implementados':d['parametros'],
        'estado':'experimental_validacion_insuficiente','checkpoint':d['checkpoint_mejor'],
        'tipo':'mapas_semanticos_y_componentes' if mapas else 'cajas_densas',
        'experimento_seleccionado':elegido,'map50_validacion':metricas[elegido]['map50'],
        'map50_95_validacion':metricas[elegido]['map50_95'],'test_evaluado':False,
        'criterio_seleccion':'mayor_map50_val_entre_candidatos_registrados_no_garantiza_calidad',
        'escala_posicion':.1,'fuente_supervision':d['configuracion'].get('fuente_supervision','cajas'),
        'umbral_visualizacion':d['configuracion']['umbral_visualizacion'],
        'informe':'documentacion/06_contornos_y_mapas_detector.md'}
    if mapas:det.update(rejilla=48,clases_incluido_fondo=7,postprocesamiento='componentes_solo_contenido_real_v2')
    else:det.update(candidatos_por_celda=2,radio_centros=0,normalizacion_clases='global')
    a['detector']=det
    if not historico:(RAIZ/'configuracion/arquitectura.json').write_text(json.dumps(a,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    audit=leer('datos/reportes/poligonos_bordes.json')
    anterior=leer('datos/reportes/poligonos.json')
    perdida_train=anterior['particiones']['train'].get('objetos_sin_celda_positiva',anterior['particiones']['train'].get('objetos_sin_celda_positiva_a_cobertura025'))
    perdida_val=anterior['particiones']['val'].get('objetos_sin_celda_positiva',anterior['particiones']['val'].get('objetos_sin_celda_positiva_a_cobertura025'))
    diag=leer('resultados/diagnostico_detector_mapas_poligonos_padding.json')
    l=['# Contornos originales y detector por mapas de materiales','',
       '**Estado: calidad insuficiente.** Los errores de preparación de objetivos y padding se corrigieron; el piloto sigue sin detectar de forma fiable. La selección siguiente es experimental.','',
       '## Qué cambió','',
       'Se implementó una salida alternativa del Transformer: siete probabilidades por celda (fondo y seis materiales) en una rejilla 48×48. Usa tokenización lineal, atención local/global y fusión propia. Conserva el extractor de características del detector en código, pero cada piloto comienza con pesos aleatorios nuevos. No carga el clasificador ni los pesos de otros pilotos. Cuenta con 438.727 parámetros; difiere de los 440.182 de la cabeza original de cajas. No incorpora convoluciones ni modelos externos.','',
       'El primer piloto de mapas empleó cobertura rectangular débil derivada de cajas. No es segmentación precisa ni una implementación de BoxSup. Después se localizaron los contornos de instancia ya presentes en las anotaciones originales TACO y se usaron para supervisar materiales. Las cajas originales siguen siendo el objetivo de evaluación.','',
       '## Auditoría de contornos','',
       'Se verificaron los 2.505 objetos train y 361 val: índice interno de anotación, ID original, categoría, bbox original, dimensiones y vértices finitos. Los IDs originales repetidos se resuelven por índice interno. Una anotación positiva fusionada procede de una imagen duplicada con hash de píxeles y grupo idénticos. Si falta un contorno o no coincide la procedencia, la carga falla; no se fabrica una máscara rectangular.','',
       'Se aplica la misma geometría letterbox que a la imagen y se rellenan los polígonos a 384×384. Tres vértices train caen fuera del lienzo original; el rasterizador los recorta al lienzo. No se cambian archivos originales ni particiones. Los hashes de anotaciones, inventario y manifiestos se guardan con el entrenamiento.','',
       f"La primera reducción exigía 25 % de cobertura de una celda: {perdida_train} objetos train y {perdida_val} val quedaban sin celda positiva al rasterizarlos individualmente. El ensayo de memorización falló y no se entrenó ese candidato completo.",'',
       'La versión usada considera positivo cualquier píxel del contorno rasterizado dentro de una celda (1/64 del bloque 8×8). Esto conserva al menos una celda al rasterizar individualmente cada objeto auditado. Los materiales distintos superpuestos se ignoran; los de igual material se unen. Los solapes pueden ocultar objetivos semánticos. En la evaluación de cajas se conservan todos los objetos, incluidos los pequeños. Las categorías fuera de alcance mantienen sus regiones ignoradas.','',
       '![Contorno de entrenamiento sobre imagen](../datos/reportes/poligonos_bordes_train.png)','',
       '## Corrección al formar componentes','',
       'El padding se ignoraba en la pérdida, pero podía formar componentes al inferir. Recortar la caja después no impide unir regiones reales con predicciones del relleno. Ahora se excluyen celdas fuera del contenido real antes de calcular componentes; después se recortan las cajas. La decisión usa solo dimensiones y padding, sin consultar cajas o máscaras GT. Una prueba específica cubre falsas regiones y conexiones a través del relleno.','',
       f"El diagnóstico final aprobó en {diag['historial'][-1]['paso']} pasos sobre seis escenas fáciles train y {diag['objetos']} objetos: AP50, recall y precisión de {diag['historial'][-1]['map50']:.1%}, {diag['historial'][-1]['recall50']:.1%} y {diag['historial'][-1]['precision50']:.1%}. No demuestra generalización ni se reutilizan sus pesos. Se conservan los dos diagnósticos rechazados anteriores.",'',
       '## Resultados de validación','',
       'Mismas 181 imágenes, 361 objetos y evaluador propio de AP interpolado a 101 puntos, con IoU de 0,50 a 0,95 y regiones ignoradas. No es el ejecutable oficial COCOeval. Umbral de evaluación 0,05 y tamaño mínimo de componentes de dos celdas para mapas. Test no se evalúa.','',
       '| Candidato | Supervisión | Épocas | Mejor época | mAP@50 | mAP@50:95 |','|---|---|---:|---:|---:|---:|']
    for n in nombres:
        x=resultados[n];m=metricas[n]
        l.append(f"| {n} | {'Contornos' if 'poligonos' in n else 'Cajas'} | {x['epocas_ejecutadas']} | {x['mejor_epoca']} | {m['map50']:.3%} | {m['map50_95']:.3%} |")
    l+=['',f"El checkpoint elegido por mAP@50 es **{elegido}**, con {metricas[elegido]['map50']:.3%}. La diferencia respecto al anterior no demuestra una mejora robusta. La ruta por defecto se actualizó en configuracion/arquitectura.json y sigue marcada como experimental.",'',
        'La tabla usa la reevaluación del checkpoint de mapas V1 con padding corregido, sin entrenamiento nuevo. Su resultado histórico se conserva por separado. Los dos pilotos de mapas comparten arquitectura, presupuesto máximo de 16 épocas, mínimo ocho, paciencia seis, lote ocho y semilla. Se elige la mejor época por AP50 val. Cambian la supervisión y el filtrado de padding; no se puede atribuir la diferencia solo a los polígonos. Los pesos de clase se calculan únicamente con train.','',
        '| Material | Objetos val | AP50 elegido |','|---|---:|---:|']
    for c,m in metricas[elegido]['por_clase'].items():l.append(f"| {c} | {m['objetos']} | {m['ap50']:.3%} |")
    l+=['','![Curvas del piloto con contornos](../resultados/figuras/detector_mapas_poligonos_v2.png)','',
        '![Primeras escenas val y predicciones](../resultados/figuras/detector_mapas_poligonos_v2_comparacion_val.png)','',
        '## Qué queda por resolver','',
        'Las pruebas de componentes y la memorización pasan, pero la detección en escenas nuevas sigue fallando. Cuatro materiales tienen AP50 cero en el candidato seleccionado. Solo hay cinco objetos orgánicos train y uno val: no permite una conclusión fiable sobre esa clase. La cuantización puede perder objetos pequeños y los componentes unen instancias próximas del mismo material.','',
        f"Usar el mapa GT cuantizado como si fuera una predicción da AP50 {audit['oracle_mapas_val']['map50']:.1%} en val. Es un diagnóstico de representación y posproceso con anotaciones conocidas, no una medición del modelo ni un límite absoluto. No se usa como calidad declarada del detector.",'',
        'El siguiente cambio justificado es preparar recortes multiescala a partir de imágenes originales de entrenamiento, conservando todos los contornos visibles y su geometría, para aumentar el tamaño efectivo de objetos pequeños. Debe mantenerse la validación completa y evitar mezclar escenas entre particiones. Se necesitan orgánicos reales más diversos, escenas vacías y videos independientes. No se ha activado la cámara ni se presenta una demo en vivo como fiable.','',
        'La consulta por imagen permite ambos tipos de checkpoint y exporta la advertencia de calidad junto a las detecciones. El asistente conserva esa advertencia al recibir el JSON. Son conexiones técnicas verificables; no acreditan aciertos visuales.','',
        'En la primera imagen val, taco:686, la consulta por defecto produce una caja de plástico que corresponde a la botella y omite la tapa. No se escogió la imagen por el acierto. Con ese JSON y la pregunta «Resume los materiales detectados», el asistente se abstiene con puntuación 0,574, por debajo de su umbral 0,65. La abstención y la advertencia del detector quedan registradas en resultados/integracion_poligonos_val_686.json.','',
        'Se han usado repetidamente los resultados val para desarrollar el método. No equivalen a una evaluación final independiente. El test queda reservado para una versión estabilizada.','',
        '## Evidencias y reproducción','',
        '- datos/reportes/poligonos.json y poligonos_bordes.json: auditorías, hashes y diagnóstico GT.','- resultados/diagnostico_detector_mapas_poligonos*.json: fallos previos y diagnóstico aprobado.','- resultados/detector_mapas_v1.json, detector_mapas_v1_padding_v2.json y detector_mapas_poligonos_v2.json: resultados reales.','- pruebas/test_poligonos.py y test_detector_mapas.py: geometría, procedencia, pérdidas y padding.','',
        '```powershell',"$pythonEco = Join-Path $env:USERPROFILE '.venvs\\EcoWasteFormer\\Scripts\\python.exe'",
        '& $pythonEco -m unittest discover -s pruebas',
        '& $pythonEco scripts/auditar_poligonos.py --cobertura-minima 0.015625 --nombre poligonos_bordes',
        '& $pythonEco scripts/diagnosticar_detector_mapas.py --fuente poligonos --cobertura-minima 0.015625 --nombre diagnostico_detector_mapas_poligonos_padding',
        '& $pythonEco scripts/entrenar_detector_mapas.py --config configuracion/entrenamiento_detector_mapas_poligonos.json','```','',
        'Los diagnósticos y entrenamientos existentes se protegen contra sobrescritura: para repetir, usar nombres nuevos y ajustar la ruta del diagnóstico en la configuración. Los entrenamientos completos conservan optimizador, estados aleatorios y hashes para reanudación.','',
        'Fuente de los datos y contornos: [repositorio de los autores de TACO](https://github.com/pedropro/TACO), [artículo TACO](https://arxiv.org/abs/2003.06975).','']
    (RAIZ/'documentacion/06_contornos_y_mapas_detector.md').write_text('\n'.join(l),encoding='utf-8')
    readme=RAIZ/'README.md'
    texto=readme.read_text(encoding='utf-8')
    parrafos=texto.split('\n\n')
    parametros_texto=f"{d['parametros']:,}".replace(',','.')
    ap_texto=f"{metricas[elegido]['map50']:.2%}".replace('.',',')
    for i,p in enumerate(parrafos):
        if p.startswith('**Detector:**'):
            parrafos[i]=f"**Detector:** variante seleccionada de mapas con {parametros_texto} parámetros, entrenada desde cero con contornos TACO. mAP@50 val: **{ap_texto}**; calidad insuficiente. Pasaron 39 pruebas, auditoría de 2.505 contornos train y 361 val, y memorización de seis escenas. Ver [documentacion/06_contornos_y_mapas_detector.md](documentacion/06_contornos_y_mapas_detector.md) para métricas, correcciones, gráficos y límites. El conjunto de prueba sigue reservado."
    if not historico:readme.write_text('\n\n'.join(parrafos),encoding='utf-8')
    print(json.dumps({'elegido':elegido,'map50_val':metricas[elegido]['map50'],'estado':'calidad_insuficiente'},indent=2))


if __name__=='__main__':main()
