"""Informe de los dos pilotos y sus limitaciones, sin modificar métricas."""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
def leer(path):
    return json.loads((RAIZ / path).read_text(encoding="utf-8"))

experimentos = {n: leer(f"resultados/{n}.json") for n in ["detector_piloto_v1", "detector_piloto_v2"]}
diag = leer("resultados/diagnostico_detector.json")
localizacion = leer("resultados/diagnostico_localizacion.json")
filas = ["# Detector Transformer: implementación y diagnóstico", "",
    "Fecha: 19/11/2026. Estado: código implementado, pruebas y memorización aprobadas, dos pilotos completados; **calidad insuficiente en validación**. No está listo para una demostración fiable ni para interpretar sus cajas como residuos reales confirmados. Test permanece reservado.", "",
    "## Componentes implementados", "",
    "| Componente | Configuración | Parámetros |", "|---|---|---:|",
    "| Tokenizador de detalle | Patches 8×8, 2.304 tokens de dimensión 96 | 18.528 |",
    "| Tokenizador de contexto | Patches 16×16, 576 tokens de dimensión 96 | 73.824 |",
    "| Dos bloques por rama | Atención local en ventanas de 8×8 tokens y atención global; cuatro cabezas | 299.136 |",
    "| Fusión | Contexto interpolado, concatenación, proyecciones, compuerta y LayerNorm | 37.248 |",
    "| Cabezal | MLP 96→96→22 | 11.446 |",
    "| **Total comprobado** | | **440.182** |", "",
    "La entrada es RGB 384×384 con letterbox y normalización fija a [-1,1]. La salida [B,48,48,2,11] contiene presencia, seis logits de material y cuatro valores de caja por candidato: 4.608 candidatos antes de filtrar. Las posiciones son sinusoidales fijas. El código usa operaciones básicas y bloques propios; no importa detectores completos ni carga pesos externos.", "",
    "El núcleo está en ecowaste/detector.py; geometría, asignación, pérdidas, evaluación e inferencia están separados. Se conserva el mismo clasificador y asistente previamente entrenados.", "",
    "## Asignación, pérdidas y regiones ignoradas", "",
    "Cada anotación se asigna a la celda de su centro y a uno de dos candidatos según tamaño. Los tamaños iniciales y el corte de escala se calculan solo con entrenamiento. Se reserva el candidato libre cuando dos objetos comparten centro; cualquier objeto sin candidato se registra como colisión y su región no se usa como fondo. No se alteran las cajas originales para forzar una asignación.", "",
    "La presencia usa BCE con positivos y negativos normalizados por separado. Se seleccionan negativos difíciles y aleatorios sin reemplazo: hasta diez por positivo, con mínimo 32 por imagen cuando están disponibles. Clases y cajas solo se supervisan en positivos. Los coeficientes son L_obj + L_cls + 5·L_SmoothL1 + 2·L_GIoU.", "",
    "Las regiones ambiguas de TACO y el relleno de letterbox se excluyen del aprendizaje de fondo. Una anotación positiva válida conserva la supervisión cuando se solapa con una región ignorada. Las pérdidas se calculan en FP32 para evitar problemas con objetos de área pequeña; el forward del piloto utiliza AMP FP16.", "",
    "GIoU procede de [Rezatofighi y colaboradores, 2019](https://arxiv.org/abs/1902.09630). NMS elimina cajas muy solapadas de la misma clase; estas operaciones conocidas no se presentan como invenciones del proyecto.", "",
    "En la segunda variante se omiten también los negativos cuyos centros están dentro de cajas GT válidas. La celda central sigue siendo el único positivo por objeto: no se fabrican etiquetas ni cajas nuevas. Se registró ese cambio en una configuración separada.", "",
    "## Pruebas y capacidad de ajuste", "",
    "Pasaron **21 pruebas** en pruebas/: orden/inversión de ventanas, parámetros y formas, IoU/GIoU con casos conocidos, colisiones, regiones ignoradas, escenas sin objetos, gradientes de fondo, NMS, AP con predicciones perfectas, errores que no deben ignorarse, cambios de orientación, letterbox, flip con relleno asimétrico y guardado/carga. Incluyen las pruebas del clasificador y del asistente.", "",
    f"El diagnóstico eligió seis escenas fáciles de train con {diag['objetos']} objetos en total. Tras {diag['historial'][-1]['paso']} pasos, el recall sobre esos mismos objetos fue {diag['historial'][-1]['recall50']:.1%}, la precisión {diag['historial'][-1]['precision50']:.1%} y mAP@50 {diag['historial'][-1]['map50']:.1%}. La AP puede ser perfecta aunque haya falsos positivos de puntuación menor, por lo que se conserva también la precisión. Este resultado solo demuestra ajuste a esas escenas, no generalización. Sus pesos no se reutilizaron en los pilotos.", "",
    "## Experimentos controlados", "",
    "Ambos pilotos comenzaron desde cero con semilla 42, 964 imágenes de train y 181 de val. Batch 4, máximo ocho épocas, AdamW 0,0005, warmup de dos épocas, clipping 1, precisión mixta y los mismos aumentos de brillo, contraste y flip exclusivamente en train. La selección se hizo por mAP@50 en validación y la paciencia fue de cuatro épocas sin mejora.", "",
    "| Experimento | Cambio | Épocas ejecutadas | Época elegida | mAP@50 val | mAP@50:95 val |", "|---|---|---:|---:|---:|---:|"]
for nombre, d in experimentos.items():
    cambio = "Supervisión original" if nombre.endswith("v1") else "Omitir negativos dentro de GT"
    filas.append(f"| {nombre} | {cambio} | {d['epocas_ejecutadas']} | {d['mejor_epoca']} | {d['val']['map50']:.3%} | {d['val']['map50_95']:.3%} |")
filas += ["", "Los dos pilotos se detuvieron por falta de mejora. La variante no mejora la métrica principal. La reducción de pérdida de entrenamiento no acredita una detección útil y no justifica seguir añadiendo épocas sin revisar el problema.", "",
    "**Estos valores son AP de detección, no porcentajes de acierto de clasificación.** La evaluación usa interpolación de precisión a 101 puntos de recall y regiones ignoradas propias. No es COCOeval oficial y no debe compararse directamente con cifras de artículos que utilicen otro protocolo.", "",
    "Se evaluó con umbral mínimo de puntuación 0,01, NMS IoU 0,5 y máximo 100 cajas por imagen. La visualización usa 0,25: ese umbral distinto permite inspección, pero no elimina el problema de falsos positivos. El conjunto val contiene 361 objetos, de los cuales solo uno es orgánico.", "",
    "| Material | Objetos val | AP@50 v1 | AP@50 v2 |", "|---|---:|---:|---:|"]
for c, v in experimentos["detector_piloto_v1"]["val"]["por_clase"].items():
    v2 = experimentos["detector_piloto_v2"]["val"]["por_clase"][c]
    filas.append(f"| {c} | {v['objetos']} | {v['ap50']:.3%} | {v2['ap50']:.3%} |")
filas += ["", "## Diagnóstico adicional de localización", "",
    "Se volvieron a evaluar las mismas predicciones guardadas colapsando todas las clases en una sola categoría residuo. No se entrenó otro modelo ni se cambiaron las cajas o puntuaciones:", "",
    "| Experimento | AP@50 sin exigir material | Recall@50 sin exigir material |", "|---|---:|---:|"]
for nombre, d in localizacion["experimentos"].items():
    filas.append(f"| {nombre} | {d['map50_sin_material']:.3%} | {d['recall50_sin_material']:.2%} |")
filas += ["", "Este diagnóstico sugiere que cambiar únicamente el clasificador de materiales no basta: la localización y el orden de confianza de las cajas también son débiles. La prueba de memorización aprobada no descarta subajuste sobre el dataset completo ni demuestra que la representación elegida sea adecuada para escenas nuevas.", "",
    "## Recursos y trazabilidad", ""]
for nombre, d in experimentos.items():
    filas.append(f"- {nombre}: {d['duracion_segundos']:.1f} s de entrenamiento, guardado y evaluación, sin carga inicial de imágenes; pico asignado por PyTorch {d['vram_pico_asignada_bytes']/1024**2:.1f} MiB; colisiones en train sin aumentos: {d['colisiones_train_sin_aumentos']}.")
filas += ["", "El pico asignado no incluye contexto CUDA ni toda la memoria reservada del dispositivo. La caché de imágenes 384×384 uint8 ocupa unos 406,7 MiB en train y 76,4 MiB en val; la segunda ejecución usa lectura concurrente y deja una caché persistente vinculada al hash del manifiesto. Ese cambio no modifica los tensores de entrenamiento ni las anotaciones.", "",
    "Los checkpoints guardan optimizador, scaler, RNG, historial, configuración y hashes para reanudar una ejecución interrumpida. Los resultados de v1 se conservan separados de v2. No se evaluó el detector sobre las 284 imágenes de prueba.", "",
    "## Uso técnico por imagen y conexión al asistente", "",
    "```powershell", "$pythonEco = Join-Path $env:USERPROFILE '.venvs\\EcoWasteFormer\\Scripts\\python.exe'",
    "& $pythonEco scripts/detectar_imagen.py 'C:\\ruta\\foto.jpg' --imagen-salida resultados/detecciones.jpg --json-salida resultados/detecciones.json",
    "& $pythonEco scripts/consultar_asistente.py '¿Qué residuos detectaste?' --detecciones resultados/detecciones.json", "```", "",
    "La salida usa coordenadas xyxy en píxeles de la imagen orientada. Se deshace letterbox y se conserva la imagen original. El JSON incluye fuente, fecha, puntuación y limitación del modelo. Esta conexión transmite resultados del detector al asistente, pero no verifica que las detecciones sean correctas.", "",
    "La ejecución de ejemplo usa la primera imagen del manifiesto de validación y se conserva en resultados/ejemplo_deteccion_val.json y .jpg. La respuesta del asistente está en resultados/integracion_detector_asistente.json. No se eligió un ejemplo favorable ni se sustituyeron predicciones por anotaciones.", "",
    "## Figuras y reproducción", "",
    "![Piloto original](../resultados/figuras/detector_piloto_v1.png)", "",
    "![Variante de supervisión](../resultados/figuras/detector_piloto_v2.png)", "",
    "![Anotaciones y predicciones](../resultados/figuras/detector_piloto_v1_comparacion_val.png)", "",
    "Se muestran las primeras escenas de grupos diferentes de validación, sin seleccionarlas por aciertos. La comparación dibuja todas las cajas que superan 0,25 y etiqueta las cinco primeras para conservar legibilidad. No oculta las cajas falsas restantes.", "",
    "```powershell", "& $pythonEco -m unittest discover -s pruebas -v",
    "& $pythonEco scripts/diagnosticar_detector.py",
    "& $pythonEco scripts/entrenar_detector.py",
    "& $pythonEco scripts/entrenar_detector.py --config configuracion/entrenamiento_detector_v2.json",
    "& $pythonEco scripts/diagnosticar_localizacion.py", "```", "",
    "Los scripts protegen los experimentos existentes de sobrescritura; usar un nombre nuevo para repetirlos. Se puede reanudar con --reanudar checkpoints/detector_piloto_v1_ultimo.pt y su configuración correspondiente. Las gráficas se recrean con el Python base y Matplotlib opcional: python scripts/graficar_detector.py --nombre detector_piloto_v1, y lo mismo para v2.", "",
    "## Qué corresponde corregir a continuación", "",
    "La prioridad es mejorar la detección antes de presentarla como funcional en video. Revisar con experimentos separados la supervisión espacial de centros/tamaños, la selección de negativos y el balance entre materiales; ampliar las anotaciones y reservar nuevas escenas para evaluación externa. Un cambio de asignación espacial debe poder regresar a la caja real y tener pruebas de geometría antes de entrenar.", "",
    "Se mantienen cinco objetos orgánicos en train y uno en val: se necesitan más ejemplos reales y diversos, no copias ni cajas inventadas. También faltan videos independientes y escenas vacías verificadas. El clasificador visual y el asistente conservan sus resultados previos; la debilidad del detector impide dar por completado el MVP integrado.", ""]
(RAIZ / "documentacion/04_detector_transformer.md").write_text("\n".join(filas), encoding="utf-8")
print("Informe guardado en documentacion/04_detector_transformer.md")
