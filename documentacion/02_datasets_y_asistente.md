# Preparación y validación de datos — 19/11/2026

El diseño del asistente quedó decidido por el usuario y no requiere aprobación adicional del profesor. Se implementó y entrenó un prototipo Transformer propio. Los modelos visuales aún no se han implementado ni entrenado.

## Resultado de la preparación visual

Se descargaron e inspeccionaron 6,252 imágenes: 4.752 de RealWaste y 1.500 de TACO. Tras normalizar orientación, 6,252 se decodifican y coinciden con la geometría esperada.

| Dataset preparado | Entrenamiento | Validación | Prueba | Total |
|---|---:|---:|---:|---:|
| RealWaste, clasificación | 2714 | 610 | 615 | 3939 |
| TACO, detección | 964 | 181 | 268 | 1413 |

Las rutas de los manifiestos son relativas a la raíz del proyecto. No se duplican las fotografías en carpetas train/val/test. Los COCO usan categorías 1..6; el lector PyTorch devuelve etiquetas 0..5 y cajas xyxy normalizadas.

## Correcciones y controles

- 153 imágenes normalizadas con EXIF a PNG, preservando los archivos originales. Las 152 discrepancias de ancho/alto se explicaban por orientación; se incluyó también una rotación de 180 grados.
- 7 cajas con desbordes menores de dos píxeles recortadas por intersección geométrica; se conserva al menos el 98 % del área de cada caja y se registran las coordenadas originales.
- 2 identificadores de anotación repetidos en la fuente (309, 4040), sustituidos por identificadores internos únicos con trazabilidad.
- Exclusiones de los manifiestos: {"categoria_fuera_de_alcance": 813, "sin_cajas_de_las_seis_clases_no_usar_como_negativo": 86, "duplicado_exacto": 1}. Los originales se conservan.
- Validación estructural: correcta. Fugas detectadas por grupos, adquisición, píxeles, archivos y corpus: {"grupos": 0, "adquisicion": 0, "pixeles": 0, "archivos": 0, "grupos_texto": 0, "preguntas_texto": 0}.
- Lotes TACO completos y bloques conservadores de diez nombres consecutivos por categoría RealWaste se mantienen juntos. Se añaden coincidencias exactas y perceptuales. Esta agrupación reduce correlaciones, pero no sustituye una prueba externa con nuevas sesiones.
- Las categorías ambiguas/compuestas o ajenas al alcance de TACO se conservan como regiones ignoradas. El lector ecowaste/datasets.py entrega ignore_boxes; la futura pérdida del detector debe enmascarar esas regiones para no contarlas como negativos.
- Se cargó una imagen real por clase para cada tarea y se comprobaron tensores, rangos, clases y geometría; evidencia en datos/reportes/carga_tensores.json.
- Revisión visual por el asistente de paneles de muestras de entrenamiento y de ejemplos con orientación corregida; no es una revisión humana exhaustiva de las etiquetas.

## Cobertura por clase

| Clase | Clasificación: imágenes | Detección: cajas |
|---|---:|---:|
| plastico | 921 | 2206 |
| papel | 500 | 179 |
| carton | 461 | 206 |
| vidrio | 420 | 254 |
| metal | 790 | 534 |
| organico | 847 | 8 |

La clasificación permite iniciar experimentos con las seis clases. **La detección dispone de solo ocho cajas orgánicas en total**: queda preparada técnicamente para pruebas, pero no permite una evaluación representativa de esa clase. Harán falta imágenes orgánicas reales con cajas revisadas y videos independientes. No se fabricaron cajas a partir de imágenes de clasificación ni se sustituyó diversidad por duplicación de muestras.

## Asistente implementado

Transformer propio: 88,710 parámetros, vocabulario de 332 tokens construido solo con entrenamiento, dos bloques y cuatro cabezas. Se entrenó desde cero en CUDA con semilla 42. Checkpoint: checkpoints/asistente_v1.pt.

Corpus inicial: 144 preguntas base de entrenamiento con tres variantes por grupo (432 registros), 36 de validación y 36 de prueba. Los contextos son sintéticos y están marcados como tales; las variantes no se reparten entre particiones. Su autoría asistida por IA y la revisión humana pendiente constan en datos/asistente/procedencia.json.

Resultados en la prueba inicial: accuracy 58.3%, F1 macro 0.570. Con abstención: cobertura 66.7% y accuracy entre respuestas aceptadas 66.7%. El umbral 0.65 se seleccionó con validación.

La referencia simple Naive Bayes obtuvo accuracy 63.9% y F1 macro 0.632 en la misma prueba. El Transformer no supera todavía esa referencia. Es un prototipo funcional con comprensión limitada; requiere ampliar y revisar el corpus antes de utilizarlo como asistente fiable. No son métricas de detección visual ni de usuarios independientes.

Las respuestas se construyen a partir de detecciones suministradas y reglas documentadas. Se probó la consulta de separación con datos sintéticos explícitamente identificados. El motor no analiza por sí solo una cámara y no declara conocer limpieza o composición interna no suministradas.

## Reproducción

Desde la carpeta EcoWasteFormer en PowerShell:

```powershell
$pythonEco = Join-Path $env:USERPROFILE '.venvs\EcoWasteFormer\Scripts\python.exe'
& $pythonEco scripts/descargar_datos.py --fuente todo
& $pythonEco scripts/preparar_datos.py --reusar-inventario
& $pythonEco scripts/validar_datos.py
& $pythonEco scripts/generar_informe_datos.py
& $pythonEco -m unittest discover -s pruebas -v
& $pythonEco scripts/consultar_asistente.py '¿Dónde pongo el plástico?' --detecciones datos/ejemplos/detecciones_sinteticas.json
```

Para reconstruir el asistente: preparar_corpus_asistente.py, entrenar_asistente.py y comparar_baseline_asistente.py, en ese orden. Mantener la prueba fija; los experimentos futuros deben seleccionar cambios con validación y reservar una nueva prueba externa para la evaluación final.

## Archivos de evidencia

- datos/reportes/validacion.json: controles, conteos y hashes de manifiestos.
- datos/reportes/preparacion.json: integridad, exclusiones, tamaño de objetos y límites.
- datos/reportes/inventario_imagenes.jsonl: procedencia y hashes por imagen.
- datos/reportes/correcciones_cajas.json e ids_origen_repetidos.json: cambios trazables.
- datos/reportes/muestras_realwaste.jpg, muestras_taco.jpg y orientacion_y_cajas.jpg: paneles de inspección.
- resultados/asistente_v1.json y baseline_asistente.json: métricas completas y configuración.
- pruebas/test_componentes.py: siete pruebas superadas de geometría, atención, checkpoint y fundamento de respuestas.

## Fuentes y atribución

[RealWaste, UCI](https://archive.ics.uci.edu/dataset/908/realwaste), Single, Iranmanesh y Raad (2023), DOI 10.24432/C5SS4G, CC BY 4.0. ZIP de origen y hash conservados.

[TACO, repositorio original](https://github.com/pedropro/TACO), Proença y Simões (2020), [artículo](https://arxiv.org/abs/2003.06975). Las anotaciones descargadas no especifican licencias por imagen y los autores declaran imágenes bajo licencias públicas. Se conservan URLs e identificadores; no se atribuye a las fotos la licencia MIT del código. No se han republicado las imágenes.

[MinAmbiente](https://www.minambiente.gov.co/preguntas-frecuentes/): fuente de las reglas de separación. La recomendación depende también del estado del objeto y del servicio local.
