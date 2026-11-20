"""Consolida correcciones, pilotos y continuidad sin sobrescribir los informes anteriores."""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
def leer(path):
    return json.loads((RAIZ / path).read_text(encoding="utf-8"))

nombres = ["detector_piloto_v1", "detector_piloto_v2", "detector_piloto_v4", "detector_piloto_v5", "detector_piloto_v5_ext24", "detector_piloto_v6"]
resultados = {n: leer(f"resultados/{n}.json") for n in nombres}
if any(d["estado"] != "piloto_completado" for d in resultados.values()):
    raise RuntimeError("Esperar a que concluyan los experimentos antes de generar el informe final")
auditoria = leer("resultados/auditoria_supervision_detector.json")
rechazado = leer("resultados/diagnostico_detector_v3.json")
ajuste = leer("resultados/detector_piloto_v5_ajuste_train.json")
sensibilidad = leer("resultados/sensibilidad_detector.json")["experimentos"]["detector_piloto_v1"]
mejor_nombre = max(resultados, key=lambda n: resultados[n]["val"]["map50"])
mejor = resultados[mejor_nombre]
config = leer("configuracion/arquitectura.json")
if (not (RAIZ/'documentacion/06_contornos_y_mapas_detector.md').exists()
        and config['detector'].get('tipo') not in ['mapas_semanticos_y_componentes','consultas_espaciales']):
    config["detector"].update(checkpoint=mejor["checkpoint_mejor"], estado="experimental_validacion_insuficiente",
        radio_centros=mejor["arquitectura"].get("radio_centros", 0),
        experimento_seleccionado=mejor_nombre,
        map50_validacion=mejor["val"]["map50"], map50_95_validacion=mejor["val"]["map50_95"],
        normalizacion_clases=mejor["configuracion"].get("normalizacion_clases", "por_imagen"),
        criterio_seleccion="mayor_map50_val_evaluador_propio_entre_experimentos_registrados",
        test_evaluado=False)
    (RAIZ / "configuracion/arquitectura.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lineas = ["# Correcciones de supervisión y ponderación del detector", "",
    "Este informe corresponde a la familia de cajas densas. La selección consolidada posterior y la variante de contornos se describen en [06_contornos_y_mapas_detector.md](06_contornos_y_mapas_detector.md).", "",
    "Estado: correcciones implementadas y evaluadas, con prueba final reservada. El detector sigue siendo experimental y los resultados de validación no completan el MVP de video.", "",
    "## Qué se encontró y corrigió", "",
    "### Supervisión espacial", "",
    f"La versión inicial asignaba {auditoria['modos']['0']['positivos']:,} candidatos positivos entre {auditoria['candidatos_totales_train']:,} candidatos de entrenamiento: {auditoria['modos']['0']['porcentaje_positivos_sobre_candidatos']:.4f} %. Una variante añade vecinos a distancia de una celda del centro si sus puntos caen dentro de la caja anotada. Los centros de todos los objetos se reservan primero para evitar que los vecinos desplacen otra anotación.", "",
    f"La auditoría completa de train obtiene {auditoria['modos']['1']['positivos']:,} asignaciones con radio 1, conservando los mismos 2.505 objetos y cinco objetos orgánicos. No son imágenes ni objetos nuevos. No hay colisiones sin representar y todas las cajas quedan dentro del rango de regresión de su candidato.", "",
    "El decodificador del centro se amplía de sigmoid(t) a 3·sigmoid(t)−1 para regresar a la caja verdadera desde una celda vecina. Cada objeto conserva peso total uno aunque supervise varios candidatos; se prueba explícitamente que una y nueve predicciones equivalentes produzcan la misma pérdida. Las anotaciones no se modifican y no se crean cajas ficticias.", "",
    "Esta supervisión por varias posiciones se relaciona con ideas de detectores densos como [FCOS](https://arxiv.org/abs/1904.01355). Se utilizan la arquitectura Transformer y las operaciones propias del proyecto; no se importa FCOS ni se presenta el principio como una invención nueva.", "",
    "### Ponderación de clases", "",
    "Se encontró una limitación concreta de la implementación anterior: la entropía cruzada ponderada se normalizaba por imagen. Cuando una imagen tenía una sola clase, el peso se cancelaba: w·L/w = L. Por tanto, los pesos no cambiaban la contribución relativa de esas imágenes como se pretendía.", "",
    "Se añadió la reducción conjunta de las pérdidas positivas del lote, manteniendo los pesos calculados solo con train. Una prueba con pesos 0,1 y 1,0 verifica que la razón entre contribuciones sea 10 en lugar de 1. La reducción anterior se conserva explícitamente para reproducir los pilotos históricos.", "",
    "La corrección del comportamiento matemático no garantiza una mejora de detección. Se evaluó separada del cambio espacial para distinguir sus efectos.", "",
    "## Pruebas y filtros antes de entrenar", "",
    "Pasaron **31 pruebas** del proyecto, incluidas pruebas de la alternativa de mapas. Para el detector de cajas se comprueban regresión a cajas reales desde vecinos, prioridad de centros, ausencia de supervisión de fondo en objetos pequeños, peso total por objeto, efecto de pesos de clase, escala posicional y carga compatible de configuraciones anteriores. Este detector de cajas conserva 440.182 parámetros.", "",
    f"La combinación de vecinos con interiores ignorados (V3) se rechazó en el diagnóstico: tras {rechazado['historial'][-1]['paso']} pasos alcanzó recall {rechazado['historial'][-1]['recall50']:.1%} en escenas memorizadas, pero precisión de solo {rechazado['historial'][-1]['precision50']:.1%}. No se lanzó un entrenamiento completo con esa variante. No se relajó el criterio para declararla aprobada.", "",
    "Las variantes V4 (vecinos conservando negativos interiores) y V5 (reducción global de clases, asignación original) sí aprobaron su diagnóstico previo, con umbral y criterio de precisión iguales a los anteriores. Los pesos de diagnóstico no se reutilizaron en los pilotos.", "",
    "## Resultados comparables de validación", "",
    "Mismos datos, mismo evaluador propio de AP de 101 puntos con regiones ignoradas y test sin evaluar. La referencia es la versión original. V5_ext24 es una continuación trazable de V5, no una repetición independiente.", "",
    "| Experimento | Cambio principal | Épocas acumuladas | Mejor época | mAP@50 | mAP@50:95 |", "|---|---|---:|---:|---:|---:|"]
cambios = {"detector_piloto_v1": "Asignación original, CE por imagen", "detector_piloto_v2": "Ignorar interiores GT",
    "detector_piloto_v4": "Vecinos + negativos interiores", "detector_piloto_v5": "CE ponderada conjunta del lote",
    "detector_piloto_v5_ext24": "Continuación V5, presupuesto máximo 24, mínimo 12",
    "detector_piloto_v6": "V5 desde cero con posiciones a escala 0,1; mismo límite 24"}
for n, d in resultados.items():
    lineas.append(f"| {n} | {cambios[n]} | {d['epocas_ejecutadas']} | {d['mejor_epoca']} | {d['val']['map50']:.3%} | {d['val']['map50_95']:.3%} |")
lineas += ["", f"El candidato elegido entre las variantes de cajas densas por mAP@50 es **{mejor_nombre}**, época {mejor['mejor_epoca']}. La selección consolidada con variantes posteriores está en el informe 06. Sigue marcado como experimental: escoger el mejor de estos candidatos no lo convierte en un detector fiable.", "",
    "La variante V6 conserva capas y parámetros y reduce a 0,1 la amplitud de las posiciones sinusoidales. Se mantiene la reducción global corregida de clases. Su tope, mínimo y paciencia coinciden con la continuación V5; las ejecuciones pueden durar distintas épocas al depender de la parada por validación. El resultado no demuestra por sí solo una causa única de mejora.", "",
    "## Por qué se amplió el presupuesto", "",
    f"Un diagnóstico sobre {ajuste['escenas']} escenas de entrenamiento elegidas por presencia de clases midió accuracy de material {ajuste['solo_celdas_supervisadas']['accuracy_material']:.1%} e IoU medio {ajuste['solo_celdas_supervisadas']['iou_medio']:.3f} en las celdas supervisadas del último estado de V5. Son mediciones con índices GT conocidos, no resultados de inferencia. Indican que el problema aparece dentro del entrenamiento y no solo al pasar a escenas nuevas.", "",
    "La continuación conserva los pesos propios, optimizador, scaler y estados aleatorios de V5. Solo cambia el presupuesto/parada, con límite total 24, mínimo 12 y paciencia 8. El archivo anterior no se sobrescribe. Se guarda el hash del checkpoint de origen y se comprueba que las restantes opciones y los hashes de datos coincidan. Esta continuación pertenece al mismo aprendizaje iniciado desde cero y no utiliza modelos externos.", "",
    "## Dependencia de la imagen: diagnóstico limitado", "",
    f"En un sondeo de seis imágenes val frente a una entrada gris uniforme, el modelo original mostró correlación media {sensibilidad['correlacion_media_presencia_vs_gris']:.3f} entre mapas de presencia, cambio medio de probabilidad {sensibilidad['cambio_medio_probabilidad_presencia']:.3f} y coincidencia de material {sensibilidad['material_coincide_con_gris_media']:.1%}. Emitió {sensibilidad['cajas_sobre_gris_con_puntuacion_025']} cajas sobre la entrada uniforme con umbral 0,25.", "",
    "Esto es compatible con dependencia excesiva de patrones espaciales y una respuesta visual débil, pero es un muestreo pequeño y no prueba una causa única. La imagen gris es sintética: no representa una evaluación de escenas vacías reales.", "",
    "## Límites y evidencias", "",
    "Se han tomado varias decisiones usando validación. Los valores de esa partición no deben publicarse como evaluación final independiente. Siguen pendientes datos orgánicos reales diversos, escenas vacías y videos de sesiones nuevas. Las mejoras se deben confirmar antes de presentar detección en vivo como fiable.", "",
    "- resultados/auditoria_supervision_detector.json: verificación de todas las asignaciones train.",
    "- resultados/diagnostico_detector_v3.json, v4.json y v5.json: diagnósticos, incluido el ensayo rechazado.",
    "- resultados/detector_piloto_v4.json, v5.json y v5_ext24.json: métricas, historial y trazabilidad.",
    "- resultados/detector_piloto_v5_ajuste_train.json: ajuste del modelo en escenas de entrenamiento.",
    "- resultados/sensibilidad_detector.json: sondeo frente a entrada uniforme.", "",
    "## Reproducción", "", "```powershell",
    "$pythonEco = Join-Path $env:USERPROFILE '.venvs\\EcoWasteFormer\\Scripts\\python.exe'",
    "& $pythonEco -m unittest discover -s pruebas -v",
    "& $pythonEco scripts/auditar_supervision_detector.py",
    "& $pythonEco scripts/diagnosticar_detector.py --radio-centros 1 --no-ignorar-interiores-gt --nombre diagnostico_detector_v4",
    "& $pythonEco scripts/entrenar_detector.py --config configuracion/entrenamiento_detector_v4.json",
    "& $pythonEco scripts/diagnosticar_detector.py --nombre diagnostico_detector_v5 --normalizacion-clases global",
    "& $pythonEco scripts/entrenar_detector.py --config configuracion/entrenamiento_detector_v5.json",
    "& $pythonEco scripts/entrenar_detector.py --config configuracion/entrenamiento_detector_v5_ext24.json --continuar-desde checkpoints/detector_piloto_v5_ultimo.pt",
    "```", "", "Los experimentos ya existentes están protegidos contra sobrescritura. Para repetirlos, usar identificadores nuevos conservando la configuración. Las gráficas se generan con scripts/graficar_detector.py --nombre identificador y el Python base con Matplotlib opcional.", ""]
(RAIZ / "documentacion/05_supervision_y_ponderacion_detector.md").write_text("\n".join(lineas), encoding="utf-8")
print(json.dumps({"experimento_elegido": mejor_nombre, "map50_val": mejor["val"]["map50"], "test_evaluado": False}, indent=2))
