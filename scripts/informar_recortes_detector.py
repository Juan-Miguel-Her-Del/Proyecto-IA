"""Registra el ensayo multiescala y conserva como predeterminado el mejor AP50 val."""
import json
from pathlib import Path

RAIZ=Path(__file__).resolve().parents[1]


def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8'))


def main():
    historico=(RAIZ/'documentacion/08_aprendizaje_y_tarea_auxiliar.md').exists()
    base=leer('resultados/detector_mapas_poligonos_v2.json')
    nuevo=leer('resultados/detector_mapas_recortes_v3.json')
    candidatos=[base,nuevo]
    ruta_balanceado='resultados/detector_mapas_recortes_v4_balanceados.json'
    balanceado=leer(ruta_balanceado) if (RAIZ/ruta_balanceado).exists() else None
    if balanceado:candidatos.append(balanceado)
    if any(d['estado']!='piloto_completado' for d in candidatos):raise ValueError('Esperar al fin del entrenamiento')
    if any(any(base['hashes_manifiestos'][k]!=d['hashes_manifiestos'][k] for k in ['train','val','anotaciones_poligonos','inventario_procedencia']) for d in candidatos):
        raise ValueError('Los experimentos usan orígenes distintos')
    meta=leer('datos/reportes/recortes_multiescala.json')
    diag=leer('resultados/diagnostico_recortes.json')
    solapes=leer('datos/reportes/recortes_solapes.json')
    configuracion=leer('configuracion/arquitectura.json')
    actual=configuracion['detector']
    elegido=max(candidatos,key=lambda d:d['val']['map50'])
    mejora=elegido['val']['map50']>actual['map50_validacion']+1e-8
    if mejora:
        actual.update(checkpoint=elegido['checkpoint_mejor'],experimento_seleccionado=elegido['configuracion']['nombre'],
            map50_validacion=elegido['val']['map50'],map50_95_validacion=elegido['val']['map50_95'],
            informe='documentacion/07_recortes_multiescala.md',recortes_train=True)
    actual['informe_ultimo_experimento']='documentacion/07_recortes_multiescala.md'
    actual['estado']='experimental_validacion_insuficiente'
    if not historico:(RAIZ/'configuracion/arquitectura.json').write_text(json.dumps(configuracion,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    l=['# Recortes multiescala desde los originales de entrenamiento','',
       '**Estado: experimento completado; calidad de detección todavía insuficiente.** Se conservan los pilotos anteriores y el test permanece reservado.','',
       '## Preparación y geometría','',
       'Se generaron dos vistas por objeto positivo de entrenamiento, con ventanas centradas en su bbox y lados de 2 o 3,5 veces su dimensión mayor. La ventana mínima mide 64 píxeles de la imagen original; se limita a los bordes de la foto. Después se aplica letterbox a 384×384. Las vistas se extraen de las imágenes originales o normalizadas, sin partir de la imagen previamente reducida a 384 píxeles. Los hashes de los archivos se comprueban antes de leerlos.','',
       f"La caché contiene **{meta['recortes']:,} recortes**, {meta['ventanas_unicas']:,} ventanas distintas y {meta['anclas_unicas']:,} anotaciones centrales únicas, procedentes de {meta['imagenes_fuente']} escenas train. No son objetos nuevos. Se mantienen las ventanas repetidas para conservar dos vistas por ancla; el muestreo las trata como ejemplos distintos.",'',
       'Se desplazan todos los contornos visibles, se recortan las cajas a la ventana y se conserva su material. Los polígonos se rasterizan dentro del contenido real; no pintan el padding. Los objetos vecinos y las categorías ignoradas se mantienen. No se asigna la clase central a toda la imagen ni se fabrica una máscara a partir de su bbox. Los manifiestos originales no cambian.','',
       'La caché conserva imagen uint8 y máscara uint8 en archivos NumPy accesibles por memoria mapeada, junto a un plan JSONL con imagen, grupo, ancla, contexto, ventana y anotaciones visibles. Esto evita cargar de golpe los aproximadamente 2,2 GB de vistas en RAM. La carga comprueba versión, hashes, formas y etiquetas.','',
       f"El lado menor mediano del ancla a la entrada del modelo pasa de **{meta['tamano_menor_ancla_mediana_antes_px']:.1f} a {meta['tamano_menor_ancla_mediana_despues_px']:.1f} píxeles**. Entre las 5.010 vistas planificadas, las anclas con lado menor inferior a ocho píxeles pasan de 998 a una. Es un cambio geométrico en train; no demuestra mejor detección.",'',
       f"Hay {len(meta['recortes_sin_celdas_del_material_ancla'])} vistas sin celdas del material central en la máscara conjunta. Una auditoría individual encontró {solapes['conteo_causas'].get('solape_materiales',0)} casos debidos a solape con otro material. No se eliminan ni se convierten en negativos: los píxeles ambiguos se ignoran según la política de supervisión. Los casos quedan enumerados en datos/reportes/recortes_solapes.json.",'',
       '![Primeros recortes por material y sus máscaras](../datos/reportes/recortes_multiescala.png)','',
       '## Protocolo del piloto','',
       'Se conserva el Transformer de mapas de 438.727 parámetros, su pérdida, aumentos de color y flip, semilla 42, AdamW, lote ocho, learning rate 0,0005, calentamiento de dos épocas y presupuesto máximo de 16 épocas. El mínimo es ocho y la paciencia seis. La inicialización es aleatoria nueva; no se cargan pesos del piloto anterior ni del diagnóstico.','',
       'Cada muestra tiene probabilidad 0,5 de proceder de una imagen completa y 0,5 de un recorte. Se usa muestreo con reemplazo, con 964 muestras y 121 lotes por época. V3 elige ejemplos uniformemente dentro de cada fuente; V4 aplica el balance descrito abajo. Así el presupuesto máximo de actualizaciones coincide con el piloto anterior. El entrenamiento registra las cantidades realmente muestreadas. El piloto anterior recorría todas las escenas sin reemplazo; la comparación incluye ese cambio de muestreo.','',
       'Los pesos de materiales se calculan a partir de las frecuencias de celdas train esperadas bajo esa mezcla, sin consultar val. V3 no balancea las anclas por clase. V4 balancea la mitad de recortes: cada material recibe probabilidad 1/6 dentro de esa fuente, y elige vistas uniformemente dentro de cada material. La mitad de imágenes completas sigue uniforme. La pérdida y el presupuesto se conservan; sus pesos de clase se recalculan según la distribución esperada de cada variante. Solo hay diez vistas centradas en los cinco orgánicos disponibles: repetirlas no aporta diversidad orgánica nueva.','',
       f"La prueba previa memorizó seis recortes fáciles y {diag['objetos']} objetos en {diag['historial'][-1]['paso']} pasos: AP50 {diag['historial'][-1]['map50']:.1%}, recall {diag['historial'][-1]['recall50']:.1%}, precisión {diag['historial'][-1]['precision50']:.1%}. Se exige AP50≥0,8, recall≥0,9 y precisión≥0,7; los pesos se descartan. No es una evaluación de generalización.",'',
       '## Validación completa','',
       'Se evalúan las mismas 181 imágenes completas y sus 361 objetos. No se evalúa solo el recorte central ni se eliminan objetos pequeños. Se conservan el umbral 0,05, mínimo dos celdas por componente, filtrado del padding y evaluador propio de AP interpolado a 101 puntos. Se selecciona el mejor AP50 val. No se mide test.','',
       '| Piloto | Vistas train | Épocas | Mejor época | mAP@50 | mAP@50:95 | Recall@50 |',
       '|---|---|---:|---:|---:|---:|---:|']
    for d in candidatos:
        vistas='Completas' if not d.get('recortes_train') else ('Recortes con anclas balanceadas' if d['recortes_train']['balanceo_de_anclas_por_clase'] else 'Mezcla completas/recortes')
        l.append(f"| {d['configuracion']['nombre']} | {vistas} | {d['epocas_ejecutadas']} | {d['mejor_epoca']} | {d['val']['map50']:.3%} | {d['val']['map50_95']:.3%} | {d['val']['recall50']:.1%} |")
    l+=['',f"El predeterminado queda en **{actual['experimento_seleccionado']}**, AP50 val {actual['map50_validacion']:.3%}. {'Se actualizó porque superó el AP50 registrado.' if mejora else 'Se conserva el anterior porque este piloto no superó su AP50.'} La diferencia entre ejecuciones con una sola semilla no prueba significación estadística.",'',
        '| Material | Objetos val | AP50 base | AP50 recortes |'+(' AP50 balanceado |' if balanceado else ''),
        '|---|---:|---:|---:|'+('---:|' if balanceado else '')]
    for clase,m in nuevo['val']['por_clase'].items():
        l.append(f"| {clase} | {m['objetos']} | {base['val']['por_clase'][clase]['ap50']:.3%} | {m['ap50']:.3%} |"+(f" {balanceado['val']['por_clase'][clase]['ap50']:.3%} |" if balanceado else ''))
    l+=['','![Curvas del entrenamiento con recortes](../resultados/figuras/detector_mapas_recortes_v3.png)','',
        '![Primeras escenas completas val](../resultados/figuras/detector_mapas_recortes_v3_comparacion_val.png)','',
        '## Diagnóstico de aprendizaje semántico','',
        'Se evaluaron las 964 imágenes completas train y 60 recortes elegidos con semilla 42 y diez anclas por clase. Se usa el checkpoint elegido por AP50 val de cada piloto. Las matrices excluyen celdas ignoradas. Son mediciones dentro de train; los recortes estratificados tampoco representan la distribución natural. Las siguientes métricas describen materiales por celda y no equivalen a AP de cajas.','',
        '| Piloto | Entrada train | Aciertos en celdas GT de material | Material GT confundido con fondo | IoU macro semántico |',
        '|---|---|---:|---:|---:|']
    for path in ['resultados/ajuste_semantico_mapas_train.json','resultados/ajuste_semantico_mapas_balanceados_train.json']:
        if not (RAIZ/path).exists():continue
        for nombre,exp in leer(path)['experimentos'].items():
            for entrada,m in exp['mediciones'].items():
                l.append(f"| {nombre} | {entrada} | {m['accuracy_solo_foreground_gt']:.1%} | {m['foreground_gt_predicho_fondo']:.1%} | {m['iou_macro_seis_materiales']:.1%} |")
    l+=['','Los recortes y el balance mejoran algunas mediciones semánticas en train, pero el AP50 sobre imágenes completas no mejora. La baja calidad ya aparece al reconocer materiales de ejemplos usados en entrenamiento; no puede atribuirse todo el fallo a agrupar componentes o a escenas nuevas. Estas mediciones no identifican una causa única.','',
        'Las matrices y mediciones completas están en resultados/ajuste_semantico_mapas_train.json y resultados/ajuste_semantico_mapas_balanceados_train.json. El control balanceado está en resultados/detector_mapas_recortes_v4_balanceados.json.','',
        '## Límites y trabajo pendiente','',
        'Pasar las pruebas geométricas y memorizar ejemplos no hace que el detector generalice. La ausencia de suficientes orgánicos, la unión de instancias del mismo material y la cuantización de objetos pequeños siguen limitando el método. Las decisiones tomadas repetidamente con val impiden usarla como evaluación final independiente.','',
        'El siguiente trabajo debe revisar el aprendizaje de características y la separación fondo/material mediante diagnósticos controlados, antes de seguir ampliando vistas o aplicar ventanas de inferencia. La mejora sobre recortes no basta para cambiar el modo de evaluación de estos pilotos. Siguen pendientes datos orgánicos diversos, escenas vacías, videos independientes y el MVP de video.','',
        '## Evidencias y reproducción','',
        '- datos/reportes/recortes_multiescala.json: auditoría, tamaños y hashes.','- datos/cache/recortes_multiescala_v1/plan.jsonl: procedencia de cada vista.','- resultados/diagnostico_recortes.json: prueba previa sin reutilización de pesos.','- resultados/detector_mapas_recortes_v3.json: resultados, conteos por época y configuración.','- configuracion/entrenamiento_detector_recortes.json: protocolo reproducible.','',
        '```powershell',"$pythonEco = Join-Path $env:USERPROFILE '.venvs\\EcoWasteFormer\\Scripts\\python.exe'",
        '& $pythonEco -m unittest discover -s pruebas','& $pythonEco scripts/preparar_recortes.py',
        '& $pythonEco scripts/diagnosticar_recortes.py',
        '& $pythonEco scripts/entrenar_detector_mapas.py --config configuracion/entrenamiento_detector_recortes.json',
        '& $pythonEco scripts/entrenar_detector_mapas.py --config configuracion/entrenamiento_detector_recortes_balanceados.json',
        '& $pythonEco scripts/diagnosticar_ajuste_mapas.py',
        '& $pythonEco scripts/diagnosticar_ajuste_mapas.py --nombre ajuste_semantico_mapas_balanceados_train --experimentos detector_mapas_recortes_v4_balanceados','```','',
        'La caché, el diagnóstico y el experimento existentes no se sobrescriben. Para repetirlos se deben usar nombres nuevos y ajustar las rutas en la configuración. Los checkpoints guardan optimizador, scaler, estados aleatorios y hashes para reanudar el mismo experimento.','']
    (RAIZ/'documentacion/07_recortes_multiescala.md').write_text('\n'.join(l),encoding='utf-8')
    readme=RAIZ/'README.md';texto=readme.read_text(encoding='utf-8')
    parrafos=texto.split('\n\n')
    for i,p in enumerate(parrafos):
        if p.startswith('**Detector:**'):
            valor=f"{actual['map50_validacion']:.2%}".replace('.',',')
            parrafos[i]=f"**Detector:** variante experimental de mapas, 438.727 parámetros y entrenamiento desde cero con contornos TACO. Mejor mAP@50 val: **{valor}**; calidad insuficiente. Se prepararon 5.010 recortes exclusivos de train y pasaron 46 pruebas. Resultados comparados y límites en [documentacion/07_recortes_multiescala.md](documentacion/07_recortes_multiescala.md). Test reservado."
    if not historico:readme.write_text('\n\n'.join(parrafos),encoding='utf-8')
    print(json.dumps({'seleccionado':actual['experimento_seleccionado'],'map50_val':actual['map50_validacion'],
                      'map50_piloto_recortes':nuevo['val']['map50'],'test_evaluado':False},indent=2))


if __name__=='__main__':main()
