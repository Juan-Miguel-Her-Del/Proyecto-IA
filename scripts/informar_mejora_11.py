"""Consolida nuevos entrenamientos de clasificación y detección conservando el historial."""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
INFORME = 'documentacion/11_mejora_clasificador_y_centros.md'


def leer(p): return json.loads((RAIZ / p).read_text(encoding='utf-8'))


def main():
    historico=(RAIZ/'documentacion/13_ablacion_del_fondo.md').exists()
    nombres_cls = ['clasificador_piloto_v1', 'clasificador_v2_coseno80', 'clasificador_v3_regularizado']
    cs = {n: leer(f'resultados/{n}.json') for n in nombres_cls}
    nombres_det = ['detector_mapas_v6_auxiliar_ext48', 'detector_centros_v11']
    ds = {n: leer(f'resultados/{n}.json') for n in nombres_det}
    variantes = {'detector_mapas_v6_mosaico': leer('resultados/detector_mapas_v6_mosaico.json')}
    if (RAIZ/'resultados/detector_centros_v11_mosaico.json').exists():
        variantes['detector_centros_v11_mosaico'] = leer('resultados/detector_centros_v11_mosaico.json')
    if any(r['estado'] != 'piloto_completado' for r in [*cs.values(), *ds.values()]):
        raise ValueError('Hay entrenamientos pendientes')
    for tabla in [cs, ds]:
        for split in ['train', 'val']:
            if len({r['hashes_manifiestos'][split] for r in tabla.values()}) != 1:
                raise ValueError('Datos distintos entre candidatos')
        if any(r['test_evaluado'] for r in tabla.values()): raise ValueError('Test no reservado')
    c = max(cs, key=lambda n: cs[n]['val']['f1_macro'])
    if any(r['estado']!='reevaluacion_val_completada' for r in variantes.values()):raise ValueError('Inferencia pendiente')
    for nombre,v in variantes.items():
        origen = ds['detector_centros_v11'] if nombre.startswith('detector_centros') else ds[nombres_det[0]]
        if v['test_evaluado'] or v['hashes_manifiestos']!=origen['hashes_manifiestos']:
            raise ValueError('Inferencia de otros datos')
    candidatos = ds | variantes
    d = max(candidatos, key=lambda n: candidatos[n]['val']['map50'])
    clas, det = cs[c], candidatos[d]
    es_mosaico = d in variantes
    fuente = ds['detector_centros_v11'] if d.startswith('detector_centros') else ds[nombres_det[0]]
    checkpoint = det['checkpoint'] if es_mosaico else det['checkpoint_mejor']
    config = leer('configuracion/arquitectura.json')
    config['clasificador'].update(checkpoint=clas['checkpoint_mejor'], experimento_seleccionado=c,
        accuracy_validacion=clas['val']['accuracy'], f1_macro_validacion=clas['val']['f1_macro'],
        estado='entrenado_validacion_test_reservado', test_evaluado=False, informe=INFORME)
    seleccion = config['detector']
    seleccion.update(checkpoint=checkpoint, experimento_seleccionado=d,
        map50_validacion=det['val']['map50'], map50_95_validacion=det['val']['map50_95'],
        parametros_implementados=fuente['parametros'], informe=INFORME, informe_ultimo_experimento=INFORME,
        estado='experimental_validacion_insuficiente', test_evaluado=False)
    if d.startswith('detector_centros'):
        for k in ['clases_incluido_fondo', 'indice_clase_nula', 'consultas', 'capas_decodificador',
                  'centros_atencion', 'posicion_por_capa', 'descartar_clase_nula']:
            seleccion.pop(k, None)
        seleccion.update(tipo='picos_de_centros_y_cajas', modo_inferencia='centros_focales',
            formulacion='centros_gaussianos_focales', fuente_supervision='cajas', rejilla=48,
            candidatos_por_material=2, auxiliar_clasificacion=True,
            postprocesamiento='picos_locales_y_nms_por_clase', ventanas_desplazadas=False)
    else:
        seleccion.update(tipo='mapas_semanticos_y_componentes', modo_inferencia='mapas_directos',
            formulacion='softmax7',fuente_supervision='poligonos',rejilla=48,clases_incluido_fondo=7,
            auxiliar_clasificacion=True,postprocesamiento='componentes_solo_contenido_real_v2',ventanas_desplazadas=False)
    if es_mosaico:
        seleccion.update(modo_inferencia='mosaico_cinco_vistas', vistas_por_imagen=5,
            version_mosaico=det['version'], sha256_checkpoint_inferencia=det['sha256_checkpoint'])
    else:
        for k in ['vistas_por_imagen','version_mosaico','sha256_checkpoint_inferencia']:seleccion.pop(k,None)
    if not historico:(RAIZ / 'configuracion/arquitectura.json').write_text(json.dumps(config, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    base_c, base_d = cs[nombres_cls[0]], ds[nombres_det[0]]
    diag = leer('resultados/diagnostico_centros.json')
    errores = leer('resultados/analisis_clasificadores_11.json')
    nms = leer('resultados/auditoria_postproceso_centros_train.json')
    pruebas = leer('resultados/pruebas_mejora_11.json')
    if not pruebas['correcto']:raise ValueError('Pruebas fallidas')
    l = ['# Mejora del clasificador y detector de centros', '',
         f"**Clasificador elegido: {c}, exactitud val {clas['val']['accuracy']:.2%}, F1 macro {clas['val']['f1_macro']:.3f}.**",
         f"**Detector elegido: {d}, mAP@50 val {det['val']['map50']:.3%}; calidad todavía insuficiente.**", '',
         'Son entrenamientos reales de desarrollo, con pesos guardados. Test continúa reservado y no se declara completado el MVP de video.', '',
         '![Comparación de validación](../resultados/figuras/comparacion_mejora_11.png)', '',
         '## Clasificación', '',
         'El entrenamiento anterior terminó a las 15 épocas, con su mejor F1 en la época 14. V2 conserva la arquitectura propia de 629.382 parámetros y los mismos aumentos de brillo, contraste y espejo; empieza con pesos aleatorios y entrena hasta 80 épocas con warmup cinco, decaimiento coseno hasta el 10 % del LR inicial y weight decay 0,03. Selección por F1 macro val, mínimo 40 épocas y paciencia 18.', '',
         'V3 también empieza desde cero. Añade rotación de hasta 10°, desplazamiento de hasta 5 %, escala 0,9–1,1 y Mixup Beta(0,2;0,2), exclusivamente durante entrenamiento. Cada mezcla conserva los pesos y etiquetas de sus dos imágenes; no se inventa una nueva clase. Se usan hasta 100 épocas, mínimo 55 y paciencia 20. Validación e inferencia conservan letterbox RGB 224 y una sola pasada, sin aumentos ni mezcla.', '',
         'Ambos emplean AdamW, LR inicial 0,0005, lote 32, suavizado de etiquetas 0,05, pesos de clase derivados exclusivamente de train, AMP y recorte de gradiente a uno. Cambian varios aspectos del entrenamiento; estos ensayos no aíslan el efecto de cada componente.', '',
         'Mismas 2.714 fotos train y 610 val de RealWaste, con los grupos de procedencia anteriores. No se entrenó ni evaluó con las 615 fotos test. La exactitud de clasificación describe la imagen completa con un material predominante; no mide detección ni conteo.', '',
         '| Entrenamiento | Épocas / mejor | Exactitud train | Exactitud val | F1 macro val |',
         '|---|---:|---:|---:|---:|']
    for n, r in cs.items():
        l.append(f"| {n} | {r['epocas_ejecutadas']} / {r['mejor_epoca']} | {r['train']['accuracy']:.2%} | {r['val']['accuracy']:.2%} | {r['val']['f1_macro']:.3f} |")
    l += ['', f"Cambio del clasificador seleccionado frente al anterior: {(clas['val']['accuracy']-base_c['val']['accuracy'])*100:+.2f} puntos porcentuales de exactitud y {clas['val']['f1_macro']-base_c['val']['f1_macro']:+.3f} de F1 macro.", '',
          f"Se reprodujeron exactamente las tres matrices de confusión sobre las 610 fotos val. Frente a V1, el elegido corrige {errores['errores_anteriores_corregidos']} errores, introduce {errores['nuevos_errores']} y conserva {errores['errores_compartidos']} errores compartidos. Las predicciones por ID, grupo, clase y puntuación están en resultados/*_predicciones_val.json y el resumen en resultados/analisis_clasificadores_11.json.", '',
          '| Clase | Imágenes val | F1 anterior | F1 elegido |', '|---|---:|---:|---:|']
    for nombre, m in clas['val']['por_clase'].items():
        l.append(f"| {nombre} | {m['soporte']} | {base_c['val']['por_clase'][nombre]['f1']:.3f} | {m['f1']:.3f} |")
    l += ['', '![Clasificador V2](../resultados/figuras/clasificador_v2_coseno80.png)', '',
          '![Clasificador V3](../resultados/figuras/clasificador_v3_regularizado.png)', '',
          '## Detección mediante centros', '',
          'Los mapas semánticos unían instancias del mismo material y las consultas anteriores aprendían mal la geometría. Se implementa una salida distinta sobre nuestro extractor Transformer local/global: seis materiales, dos candidatos de escala por material y celda, y cajas continuas en una cuadrícula de 48×48. Son 453.762 parámetros, con proyecciones lineales, atención y MLP propios; no hay convoluciones ni pesos externos o transferidos desde otro modelo.', '',
          'La idea de representar objetos por centros se relaciona con [Objects as Points](https://arxiv.org/abs/1904.07850). Nuestra implementación usa el extractor Transformer del proyecto y dos escalas por material; no importa CenterNet ni sus pesos.', '',
          'Se auditaron los centros de los 2.505 objetos train. Hay 22 coincidencias de material y celda y un máximo de dos objetos por combinación. Los dos candidatos conservan ambas instancias: se prefiere una escala según la mediana de tamaños train y se usa la libre ante colisión. Una colisión mayor produce error explícito. El objetivo mantiene cajas y etiquetas originales.', '',
          'La salida de calor usa sigmoid por material/escala, con picos uno en los centros asignados y suavizado gaussiano de sigma entre una y cuatro celdas según ancho/alto. La pérdida focal atenúa negativos fáciles y posiciones cercanas a centros. Padding y centros de regiones desconocidas se omiten para negativos; los positivos anotados prevalecen. Regresión: L1 de centro/tamaño y GIoU, con coeficientes cinco y dos. Se normaliza por número de objetos.', '',
          f"Al inyectar centros y cajas GT train perfectos, el postproceso NMS 0,5 conserva {nms['cajas_gt_conservadas_por_iou_nms']['0.5']} de {nms['objetos']} cajas. Hay nueve pérdidas por solapamiento entre anotaciones de igual material; se registran en resultados/auditoria_postproceso_centros_train.json. Es una comprobación geométrica con GT, no una predicción ni AP. Esta pérdida pequeña no explica por sí sola el bajo rendimiento aprendido.", '',
          'Los tamaños iniciales de las dos escalas se calculan con train. La inferencia busca máximos locales 3×3, decodifica cajas, excluye centros en padding, recorta al contenido, aplica NMS por material a IoU 0,5 y conserva como máximo 100 cajas. La evaluación fija umbral 0,001; la visualización usa 0,35. Las puntuaciones no están calibradas.', '',
          f"El diagnóstico previo conserva todos los {diag['objetos_train_asignados']} objetos train y memoriza las mismas seis escenas fáciles más seis fotos auxiliares: AP50={diag['historial'][-1]['map50']:.3f}, recall={diag['historial'][-1]['recall50']:.3f}, precisión={diag['historial'][-1]['precision50']:.3f}, clasificación auxiliar={diag['historial'][-1]['accuracy_aux']:.3f}, a los {diag['historial'][-1]['paso']} pasos. Esos pesos se descartan; no mide generalización.", '',
          'El entrenamiento completo usa las mismas 964 escenas TACO train y clasificación auxiliar RealWaste train con coeficiente 0,5 y muestreo balanceado. No convierte etiquetas globales en cajas. Lotes ocho y 16 respectivamente, AdamW 0,0005, weight decay 0,01, warmup cinco, coseno hasta 10 %, AMP y clipping uno; máximo 80 épocas, mínimo 40 y paciencia 15. La selección es por mAP@50 val.', '',
          'Validación completa sobre las mismas 181 imágenes y 361 objetos. Evaluador propio de AP interpolado de 101 puntos, con regiones ignoradas; no se declara equivalencia exacta con COCOeval. No se elimina GT pequeño ni de clases difíciles.', '',
          '| Detector | Parámetros | Épocas / mejor | mAP@50 val | mAP@50:95 val | Recall@50 val |',
          '|---|---:|---:|---:|---:|---:|']
    for n, r in ds.items():
        l.append(f"| {n} | {r['parametros']} | {r['epocas_ejecutadas']} / {r['mejor_epoca']} | {r['val']['map50']:.3%} | {r['val']['map50_95']:.3%} | {r['val']['recall50']:.2%} |")
    l += ['', '## Inferencia sobre zonas ampliadas', '',
          'Se evalúa la imagen completa y cuatro ventanas que cubren cada una el 65 % del ancho y alto, tomadas en las esquinas. Se redimensiona cada vista desde la imagen original; sus cajas se trasladan al sistema de coordenadas original. Se omiten fragmentos que tocan cortes internos y se combinan mediante NMS por material a IoU 0,5, máximo 100 detecciones. La imagen completa conserva cobertura de bordes. No se consultan anotaciones para proponer o elegir ventanas.', '',
          'Los pesos permanecen congelados y se verifica su SHA-256 antes y después. Esta variante procesa cinco vistas por foto, aumenta el coste y se evalúa aparte del entrenamiento; la mejora no corresponde a entrenar nuevos pesos. Se conserva el umbral de evaluación original de cada modelo.', '',
          'La formación de propuestas conserva ese umbral fijo, también en la consulta por imagen. El umbral visual se aplica después a las puntuaciones de las cajas resultantes. Se corrigió una inconsistencia que recortaba componentes al subir el filtro visual y se añadió una prueba que exige conservar las cajas evaluadas. La consulta real reproduce ahora las coordenadas y materiales guardados en validación, al mismo filtro de puntuación.', '',
          '| Inferencia | Época de los pesos | mAP@50 val | mAP@50:95 val | Recall@50 val |',
          '|---|---:|---:|---:|---:|']
    for n,r in variantes.items():
        l.append(f"| {n} | {r['epoca']} | {r['val']['map50']:.3%} | {r['val']['map50_95']:.3%} | {r['val']['recall50']:.2%} |")
    l += ['', f"Cambio seleccionado frente al detector anterior: {(det['val']['map50']-base_d['val']['map50'])*100:+.3f} puntos porcentuales de mAP@50.", '',
          '| Material | Objetos val | AP50 anterior | AP50 elegido |', '|---|---:|---:|---:|']
    for n, m in det['val']['por_clase'].items():
        l.append(f"| {n} | {m['objetos']} | {base_d['val']['por_clase'][n]['ap50']:.3%} | {m['ap50']:.3%} |")
    l += ['', '![Detector de centros](../resultados/figuras/detector_centros_v11.png)', '',
          '## Selección, uso y límites', '',
          f"Se conservan todos los checkpoints y reportes. Configuración predeterminada: clasificador `{clas['checkpoint_mejor']}` y detector `{checkpoint}`, modo {seleccion['modo_inferencia']}. Las dos consultas por imagen leen la selección de configuracion/arquitectura.json y permiten indicar otro checkpoint explícitamente. El detector permite `--no-mosaico` para comparar una sola vista y `--mosaico` para activarlo explícitamente.", '',
          'La mejora se mide en validación usada reiteradamente para decidir modelos. Una sola semilla y ese uso de val no prueban robustez estadística ni generalización independiente. La foto RealWaste de un objeto predominante difiere de escenas con basura dispersa. Solo hay cinco cajas orgánicas train y una val de detección; la clasificación auxiliar no resuelve esa falta de cobertura.', '',
          'Las puntuaciones y cajas continúan siendo experimentales. No se activó cámara ni se presenta el MVP de video como completado. La integración con el asistente conserva la advertencia de calidad insuficiente.', '',
          '## Verificación y reproducción', '',
          f"Pasaron {pruebas['ejecutadas']} pruebas. Se comprobaron colisiones y conservación de instancias, regiones ignoradas, padding, gradientes, guardado/carga, programación de aprendizaje, aumentos, mezclas y proyección de cajas entre recortes y coordenadas originales. Las consultas reales y su integración con el asistente quedan en resultados/ejemplo_mejora_11_*.json; la verificación de hashes, selección y reproducción de cajas está en resultados/verificacion_mejora_11.json.", '',
          'El ejemplo de clasificación usa la primera imagen val, Cardboard_100.jpg: V3 predice vidrio aunque la etiqueta es cartón. Se conserva ese fallo; no se elige una imagen correcta para la demostración. La exactitud general de validación no garantiza cada respuesta individual.', '',
          '```powershell',
          "$pythonEco = Join-Path $env:USERPROFILE '.venvs\\EcoWasteFormer\\Scripts\\python.exe'",
          '& $pythonEco -m unittest discover -s pruebas',
          '& $pythonEco scripts/entrenar_clasificador.py --config configuracion/entrenamiento_clasificador_coseno.json',
          '& $pythonEco scripts/entrenar_clasificador.py --config configuracion/entrenamiento_clasificador_regularizado.json',
          '& $pythonEco scripts/diagnosticar_centros.py',
          '& $pythonEco scripts/entrenar_detector_centros.py', '```', '',
          'Verificación posterior: `& $pythonEco scripts/verificar_mejora_11.py`. Para regenerar gráficos se usan graficar_clasificador.py con --nombre, graficar_detector.py con --nombre y graficar_mejora_11.py.', '',
          'Para repetir entrenamientos se deben usar nombres nuevos; los resultados previos se protegen contra sobrescritura. La reanudación requiere configuración y hashes coincidentes, y restaura optimizador, scaler y estados aleatorios. Test se mantiene reservado.', '']
    if historico:l[1:1]=['','Informe histórico. La ablación posterior y el estado actual del video están en [13_ablacion_del_fondo.md](13_ablacion_del_fondo.md) y [14_video_y_webcam_experimental.md](14_video_y_webcam_experimental.md).','']
    (RAIZ / INFORME).write_text('\n'.join(l), encoding='utf-8')
    p = (RAIZ / 'README.md').read_text(encoding='utf-8').split('\n\n')
    for i, t in enumerate(p):
        if t.startswith('Fase actual:'):
            p[i] = 'Fase actual: los tres modelos Transformer están implementados y entrenados; RealWaste y TACO están preparados y auditados. Son entrenamientos reales de desarrollo con selección en validación. **El detector mantiene calidad insuficiente y el MVP de video aún no está completado.** Los documentos originales están en `../documentos/`.'
        if t.startswith('**Detector:**'):
            p[i] = f"**Detector:** mejor mAP@50 val **{det['val']['map50']:.2%}**, todavía insuficiente. Selección `{d}`, modo {seleccion['modo_inferencia']}. Se comparó el entrenamiento de centros con inferencia por zonas. Pasaron {pruebas['ejecutadas']} pruebas; test reservado. Ver [informe conjunto]({INFORME})."
        if t.startswith('**Clasificador visual:**'):
            p[i] = f"**Clasificador visual:** 629.382 parámetros, desde cero. Seleccionado `{c}`: **{clas['val']['accuracy']:.2%} de exactitud y F1 macro {clas['val']['f1_macro']:.3f}** en validación, frente a 70,66 % y 0,709 previos. Test reservado. Ver [entrenamientos y límites]({INFORME})."
        if t.startswith('La arquitectura del detector está implementada.'):
            p[i] = f"La arquitectura del detector está implementada. Resultado seleccionado: `resultados/{d}.json`; clasificador: `resultados/{c}.json`. Se conservan todos los ensayos anteriores. Los del asistente están en `resultados/asistente_v1.json` y `resultados/baseline_asistente.json`."
    texto = '\n\n'.join(p).replace('2. Mejorar la representación espacial y la localización: el modelo conjunto mejoró AP50, pero su continuación dejó de mejorar después de la época 19.',
        '2. Mejorar localización y cobertura de detección: se compararon centros por instancia y cinco vistas; el AP continúa siendo insuficiente.')
    if not historico:(RAIZ / 'README.md').write_text(texto, encoding='utf-8')
    print(json.dumps({'clasificador': c, 'accuracy_val': clas['val']['accuracy'], 'f1_macro_val': clas['val']['f1_macro'],
                      'detector': d, 'map50_val': det['val']['map50'], 'test_evaluado': False}, indent=2))


if __name__ == '__main__': main()
