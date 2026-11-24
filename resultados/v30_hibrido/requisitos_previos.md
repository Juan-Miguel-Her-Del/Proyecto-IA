# Cierre de requisitos y diseño revisado — 19 de noviembre de 2026

Este documento registra la fase inicial de diseño y entorno. El estado posterior de implementación del asistente y preparación de datos está en [02_datasets_y_asistente.md](02_datasets_y_asistente.md); las afirmaciones sobre tareas pendientes en las secciones siguientes corresponden al cierre de aquella fase inicial.

## 1. Decisión sobre el asistente

El enunciado, página 1, solicita un Transformer que brinde retroalimentación sobre los resultados. El registro anterior proponía Transformers visuales y un asistente exclusivamente de reglas. Corregimos esa diferencia incorporando **EcoWasteFormer-Assistant**, un Transformer de lenguaje pequeño implementado por el equipo y entrenado desde inicialización aleatoria.

Flujo acordado para la implementación:

```text
Imagen / webcam / archivo de video
  -> detector Transformer propio
  -> cajas + materiales + puntuaciones + fotograma/instante
  -> resultados estructurados

Pregunta + contexto de resultados
  -> tokenizador de palabras propio
  -> encoder Transformer propio de lenguaje
  -> intención de la pregunta
  -> consulta de resultados y reglas de separación revisadas
  -> respuesta mediante plantilla, con condiciones e incertidumbre
```

El clasificador visual se conserva como módulo independiente y posible refinamiento de recortes, sujeto a comprobar su utilidad. Hay tres modelos pequeños, no tres modelos generativos grandes. El asistente no se ejecuta en cada fotograma: responde cuando el usuario consulta y usa una instantánea identificada de las detecciones.

**Función concreta del Transformer del asistente:** procesar la pregunta y el contexto serializado de detecciones para seleccionar entre seis intenciones: resumen, material, separación, preparación del residuo, incertidumbre y fuera de alcance. La salida neuronal determina qué información y explicación se presentan. Las plantillas convierten esa decisión en texto apoyado en resultados y fuentes.

Ejemplo futuro, no resultado ejecutado: ante una detección de plástico y la pregunta «¿Dónde lo deposito?», la intención es separación. La respuesta menciona la detección, pregunta por limpieza/contaminación si falta ese dato y explica la recomendación de forma condicional. No afirma haber reconocido suciedad, peligrosidad o composición interna si el modelo no evalúa esas propiedades.

Esta solución incluye un Transformer en la retroalimentación sin depender de pesos externos. Es un asistente de dominio limitado con respuestas fundamentadas. **Decisión cerrada por el usuario el 19/11/2026: proceder con este diseño, sin requerir aceptación adicional del profesor.** La implementación y los datos avanzan con esta definición.

## 2. Tamaño y entrenamiento del asistente

Propuesta inicial: longitud máxima de 64 tokens, embedding 64, dos bloques, cuatro cabezas, MLP 128, promedio enmascarado y seis logits. Atención Q/K/V, normalizaciones y conexiones residuales implementadas por el equipo con operaciones básicas de PyTorch. Vocabulario construido exclusivamente con entrenamiento; tokens PAD, UNK, separación de pregunta/contexto y marcadores de material y estado. El padding se excluye de la atención y de la agregación.

El corpus será elaborado y revisado por el equipo. Contendrá preguntas naturales, negaciones, ausencia de detecciones, varias clases simultáneas y preguntas ajenas al tema. Las variantes de una misma plantilla permanecerán en la misma partición; el test tendrá preguntas redactadas independientemente. Los contextos sintéticos estarán identificados como tales y no se presentarán como detecciones reales.

Evaluación: F1 macro y por intención, matriz de confusión, ejemplos con lenguaje nuevo y contraste con un método simple de palabras clave. El umbral de rechazo se calibrará en validación; una puntuación alta no prueba por sí sola que una pregunta esté dentro del dominio. Se revisará además que cada afirmación de la respuesta pueda rastrearse a una detección, una regla o una condición explícita. La calidad del asistente y la del detector se medirán por separado y en integración.

Sin checkpoint entrenado y evaluado, la aplicación no habilitará respuestas neuronales como si fueran fiables. El modo de prueba de reglas deberá identificarse como tal.

## 3. Restricciones de implementación

- Mantener modelos visuales propios con núcleo Transformer y entrenar los tres modelos desde cero.
- No descargar pesos, importar detectores/clasificadores completos, usar fine-tuning externo ni reutilizar checkpoints anteriores.
- Se pueden usar Linear, LayerNorm, operaciones de atención, funciones de pérdida y optimizadores. Usar matemáticas publicadas no equivale a inventarlas; citar sus fuentes.
- Los límites académicos proceden del registro de la indicación verbal del profesor. Adoptamos su interpretación conservadora; no afirmamos que todas esas prohibiciones estén escritas en el enunciado.
- No introducir CNN como sustituto del Transformer visual para este diseño. Una recomendación del artículo de referencia no modifica estas restricciones.

## 4. Matriz de trazabilidad y aceptación

Todas las funciones siguientes están pendientes de implementación y evaluación. Las propuestas y los documentos no se contabilizan como funciones terminadas.

| Origen | Requisito | Responsable | Evidencia necesaria |
|---|---|---|---|
| Enunciado, proyecto 4 | Detectar residuos en video | Detector + motor de video | MP4 y webcam continuos, con varios objetos y fin/desconexión gestionados |
| Enunciado, proyecto 4 | Cinco materiales | Detector y clasificador | Métricas por clase sobre prueba independiente, con cajas correctas para detección |
| Enunciado, proyecto 4 | Orgánico/inorgánico | Taxonomía + modelos | Ejemplos de ambas categorías y convención documentada |
| Enunciado, proyecto 4 | Mostrar clasificación | Interfaz | Cajas, material y puntuación proyectados a la imagen original |
| Enunciado, página 1 y proyecto 4 | Transformer que dé retroalimentación y asistente de separación | Transformer de lenguaje + resultados + reglas | Checkpoint propio, evaluación de intenciones y respuestas rastreables a resultados |
| Registro de diseño | Fotografías individuales | Inferencia | Imágenes independientes con detecciones visualizadas |
| Registro de indicación verbal | Arquitecturas propias y entrenamiento desde cero | Código + experimentos | Capas explícitas, semillas, configuraciones y trazabilidad de inicialización |
| Enunciado, entregables | Código, informe, exposición y demo | Proyecto completo | Instalación reproducible, informe con resultados reales y demostración de 10 minutos |

La clase orgánico representa restos alimentarios/vegetales. Agrupar papel y cartón como inorgánicos es una convención operativa del proyecto, no una afirmación de química. Material, aprovechabilidad y recipiente recomendado son campos distintos.

## 5. Plan de datos corregido

### Clasificación

Fuente inicial: [RealWaste en UCI](https://archive.ics.uci.edu/dataset/908/realwaste), licencia CC BY 4.0 y atribución a Single, Iranmanesh y Raad (2023). La ficha declara 4.752 imágenes, nueve categorías y resolución 524×524.

Mapeo propuesto de las categorías publicadas:

| Categoría original | Clase del proyecto | Imágenes declaradas |
|---|---|---:|
| Plastic | plástico | 921 |
| Paper | papel | 500 |
| Cardboard | cartón | 461 |
| Glass | vidrio | 420 |
| Metal | metal | 790 |
| Food Organics + Vegetation | orgánico | 411 + 436 = 847 |

Esto explica las **3.939 imágenes** mencionadas en el trabajo previo: coinciden con sumar las categorías seleccionadas, no con el total de RealWaste. Es una reconciliación de las cifras publicadas, todavía no un conteo del archivo descargado. Miscellaneous Trash y Textile Trash no se reclasificarán arbitrariamente; quedarán fuera del entrenamiento de seis clases y podrán servir en pruebas de residuos fuera de alcance.

### Detección

Fuente inicial: [TACO, repositorio de los autores](https://github.com/pedropro/TACO), con anotaciones COCO. Importar datos y anotaciones, sin utilizar su detector ni pesos. Auditar el mapeo de categorías por material; excluir o revisar envases ambiguos y compuestos. El repositorio indica licencia MIT para el código; no asumimos que esa licencia cubra todas las fotografías: registrar procedencia y licencia de cada imagen antes de su uso.

La cobertura de orgánicos y las cajas válidas deben comprobarse en las anotaciones. Si falta cobertura, añadir fotos propias con cajas revisadas y escenas variadas. Las imágenes de RealWaste no se convertirán en ejemplos de detección asignando la imagen completa como caja.

SALSA, Mostafa Abla y Synthetic Outdoor Waste quedan como fuentes candidatas, no como dependencias del primer entrenamiento: incorporarlas solo después de verificar origen, licencia, categorías y solapamientos.

### Control antes del primer entrenamiento

1. Inventario con URL, licencia, versión/hash, identificador original y mapeo de etiquetas.
2. Verificación de imágenes, cajas, objetos pequeños y distribución por clase/fuente.
3. Duplicados exactos y perceptuales; los recortes heredan la partición de su imagen original.
4. Partición orientativa 70/15/15 por grupos de escena/sesión, conservando clases; no separar fotogramas vecinos entre entrenamiento y prueba.
5. Videos propios independientes para prueba final y escenas sin residuos para medir falsos positivos.

La preparación, descarga y validación de estos datos es el siguiente entregable. Las fuentes verificadas no equivalen a datasets ya preparados.

## 6. Referencia científica y reglas de separación

Se localizó el [artículo indicado en el enunciado](https://doi.org/10.1007/s00521-024-10855-2). La sección 3.3 declara que su clasificador combina DenseNet-201 y MaxViT preentrenados; la sección 3.6 utiliza YOLOv9/GELAN-E para detección. Sirve como trabajo relacionado, no como implementación que podamos importar bajo nuestras restricciones. La comparación futura debe explicar diferencias de datos, clases, inicialización y hardware; no equiparar porcentajes obtenidos en tareas distintas. La revisión bibliográfica completa queda pendiente.

Para la base de recomendaciones, la [información oficial de MinAmbiente](https://www.minambiente.gov.co/preguntas-frecuentes/) recoge verde para orgánicos aprovechables, blanco para aprovechables y negro para no aprovechables. Las reglas aplicarán condiciones de estado del residuo y disponibilidad local; no basta la clase visual para decidir siempre un recipiente. Cada regla deberá conservar fuente y fecha de revisión.

## 7. Entorno y límites de las mediciones

Se encontró Python 3.12.6 de 64 bits en `AppData\Local\Programs\Python\Python312\python.exe`. El lanzador `py -3.12` falla al intentar abrir una instalación de Microsoft Store; el proyecto usa el ejecutable funcional sin modificar ni desinstalar otros Python.

`nvidia-smi` detectó RTX 4060 Laptop, 8.188 MiB y controlador 616.56. Su dato de compatibilidad CUDA no acredita que PyTorch ejecute operaciones en la GPU: el diagnóstico debe probarlo.

La [tabla oficial de versiones de PyTorch](https://pytorch.org/get-started/previous-versions/) publica `torch==2.14.1` con CUDA 12.6 para Windows/Linux. Se selecciona esa combinación reproducible del diseño anterior después de comprobar su publicación. No se declara que sea la versión más reciente. El índice oficial contiene la rueda CPython 3.12 para Windows; la operación CUDA real se verifica con `scripts/verificar_entorno.py`.

El entorno aislado se ubica fuera de OneDrive en `.venvs\EcoWasteFormer`. Las versiones realmente instaladas se guardarán en `requirements-lock.txt` y el diagnóstico en `resultados/entorno.json`.

Los 629.382 parámetros del clasificador y 440.182 del detector siguen siendo cuentas teóricas del PDF. La arquitectura no se ha ejecutado. La memoria del diagnóstico solo corresponde a operaciones pequeñas: no valida lotes, memoria de entrenamiento, precisión, mAP ni FPS de esos modelos. Las metas de 90% y 10 FPS no son garantías ni criterios inventados del enunciado.

### Resultado ejecutado del diagnóstico

Informe: `resultados/entorno.json`, generado el 19/11/2026 a las 20:59 UTC (15:59 Colombia), con código de salida 0 y `correcto: true`.

- Python 3.12.6; PyTorch 2.14.1+cu126; runtime CUDA de PyTorch 12.6.
- NVIDIA GeForce RTX 4060 Laptop GPU; 8.585.216.000 bytes de VRAM reportados por PyTorch.
- Multiplicación GPU contrastada numéricamente contra CPU: correcta.
- Atención de tamaño `[2, 4, 64, 16]` y retropropagación: correctas; gradientes presentes, finitos y no nulos.
- NumPy 2.5.5, Pillow 13.0.0, OpenCV 4.14.0.94 y PyYAML 6.0.3: importaciones correctas.
- `pip check`: sin incompatibilidades declaradas. Versiones exactas conservadas en `requirements-lock.txt`.

La preparación del entorno queda completada. No se descargaron datasets, no se implementaron los tres modelos ni se inició entrenamiento en esta fase.

## 8. Condiciones para avanzar

- Entorno: diagnóstico CUDA aprobado y versiones registradas.
- Datos: manifiesto, particiones y anotaciones verificadas antes de entrenamientos reales.
- Componentes: formas, pérdidas, gradientes y guardado/carga correctos; después, prueba de memorización pequeña para diagnosticar aprendizaje.
- Evaluación: separar clasificación (F1/accuracy), detección (mAP/recall), asistente (intenciones y fundamento de respuestas) y video (latencia/FPS medidos).
- Alcance: completar primero imagen, webcam, MP4 y asistente; seguimiento avanzado e históricos quedan para una etapa posterior.
