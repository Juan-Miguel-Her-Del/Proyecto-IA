# V27: adquisición de datasets reales y preparación de nuevo entrenamiento

**Se obtuvieron 60 fotografías reales nuevas para este repositorio y 225 anotaciones espaciales originales, todas pendientes. No hay instancias nuevas aceptadas, colección congelada ni autorización de entrenamiento.** El checkpoint original de época 19, arquitectura, pérdidas, históricos, generalización bloqueada y test permanecen intactos.

Las cifras principales distinguen registros de anotación de objetos físicos únicos. No se puede convertir el número de cajas de una cinta industrial, fotogramas de un video o fotos de un plato en esa cantidad de residuos distintos.

## Resultado práctico

| Fuente original | Fotos descargadas y verificadas | Anotaciones espaciales originales en esas fotos | Estado |
|---|---:|---:|---|
| ZeroWaste-f, train | 24 | 128 polígonos: 80 cartón, 40 plástico, 8 metal propuestos | Geometría validada; revisar material, cobertura completa y metadatos |
| WaRP-D, train | 24 | 97 cajas: 58 plástico, 12 vidrio y 27 con material pendiente | Sin polígonos; anotación visible manual necesaria |
| Food Waste HPI, train | 12 | 0 | Fotografías de sobras reales; anotación completa necesaria |
| **Total** | **60** | **225** | **0 aceptadas por revisión V26** |

La elección de HPI fue una foto por cada uno de los 12 recibos/comidas distintos de train. Se conservaron los 215 registros train consultados, pero **no se descargaron 215 fotografías**. Tampoco se asignó una caja a cada ingrediente o al plato completo. Las fotos proceden del servicio de assets del proveedor: se preservan sus JPEG exactos y sus URLs/revisiones, sin afirmar que se haya descargado el Parquet completo.

Se obtuvieron asimismo las anotaciones train completas de ZeroWaste y las 2.452 etiquetas YOLO train de WaRP. Ello permite verificar frecuencias y preparar futuras selecciones sin presentar esas referencias de imagen como fotos descargadas o instancias ya útiles.

## Fuentes investigadas y evidencia de licencia

La [matriz JSON](../resultados/v27_adquisicion/matriz_datasets.json) y su [CSV](../resultados/v27_adquisicion/matriz_datasets.csv) contienen 13 entradas de fuentes/espejos: enlaces, cifras declaradas, cantidades comprobadas en archivo, categorías, formato, condiciones académicas/comerciales, acceso, calidad, diversidad, duplicados y potencial por material. `null` significa no comprobado/no disponible; no debe interpretarse como cero.

| Fuente | Cifras declaradas por autores/ficha | Formato y condiciones comprobadas | Decisión V27 |
|---|---|---|---|
| [ZeroWaste](https://ai.bu.edu/zerowaste/) | 4.503 imágenes en ZeroWaste-f; 4 materiales | COCO por instancia. [Autores](https://github.com/dbash/zerowaste) indican CC BY-NC 4.0, mientras [Zenodo](https://zenodo.org/records/6412647) anuncia CC BY 4.0 | Piloto local académico bajo alcance no comercial de los autores; contradicción preservada, ningún permiso comercial presumido. Solo ZeroWaste-f, nunca ZeroWasteAug |
| [WaRP, AIRI](https://github.com/AIRI-Institute/WaRP) → [Kaggle del autor](https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset) | 28 clases; 2.452 imágenes train y 522 de evaluación declaradas | Detección YOLO. API oficial del distribuidor enlazado confirma `CC0: Public Domain`, versión 3 | Piloto obtenido. Permite uso académico/comercial según ese otorgamiento; conservar procedencia y cita |
| [SpectralWaste original/HF](https://huggingface.co/datasets/ferpb/spectralwaste-segmentation) | 852 etiquetadas; repo original 7.655 muestras frente a 7.653 en HF; HF train 514 | CC BY 4.0 en LICENSE/HF y registro Zenodo. Original con instancias; HF transforma a **semántica** 256×256 | No descargar 226 GB ni reconstruir instancias desde componentes. Interés secundario para cartón/plásticos; no resuelve vidrio u orgánicos |
| [Food Waste HPI](https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset) | 375 fotos; train 215, 12 comidas; múltiples vistas por recibo | CC BY 4.0 **explícita para fotografías y registros**, incluso uso comercial con atribución. Etiquetas de ingredientes/pesos, no detección | 12 fotos reales de sobras obtenidas, 0 cajas/máscaras originales |
| [FoodInsSeg](https://github.com/jamesjg/FoodInsSeg) | 7.118 fotos, 119.048 máscaras; train 4.983 / 82.716 | COCO de 103 ingredientes. CC BY-NC-SA 4.0 y declaración adicional de uso académico, sin fines comerciales | Enlace Drive publicado; archivo no materializado ni cifras verificadas localmente. Comida servida no implica residuo ni identidad física de cada ingrediente |
| [UC Food Waste](https://github.com/CollaborativeRoboticsLab/UC-Food-Waste) | 542 fotos multimodales | Polígonos CSV; plástico, vidrio, metal y **background organic**. No se confirmó licencia de dataset suficiente en lo consultado | No descargar fotos. Fondo orgánico no equivale a instancias orgánicas |
| [TrashCan](https://irvlab.cs.umn.edu/resources/trashcan) | 7.212 fotos, variantes material/instancia | Máscaras. El [laboratorio](https://irvlab.cs.umn.edu/resources) limita sus datasets a investigación académica; no asumir CC BY de un espejo | Acceso original devolvió 403; sin fotos descargadas. Dominio submarino y fauna viva ajenos a residuos orgánicos de cocina |
| [FLIC](https://www.mdpi.com/2076-3417/16/11/5465) | 401 bandejas antes/después: 802 fotos, 22 días | Semántica de comida refinada tras SAM. La licencia del artículo no demuestra por sí sola licencia de las fotografías del Drive | Candidato de sobras reales pendiente de confirmar licencia/archivo; no convertir componentes semánticos en instancias |
| [Roboflow CEP, GARBAGE CLASSIFICATION 3](https://universe.roboflow.com/cep-owycw/garbage-classification-3-i5yna) | 10.464 fotos, 7 clases, 6 versiones declaradas | Bbox anunciadas, CC BY 4.0 en ficha; procedencia de fotos y anotaciones no corroboradas | No obtenido: acceso directo 403, exportación no verificada. `BIODEGRADABLE` no se mapea automáticamente a comida |
| [Roboflow wast/myfast](https://universe.roboflow.com/myfast/wast-6srsc) | 425 fotos, **0 versiones de dataset** en ficha | Instancia/CC BY 4.0 anunciadas, material y procedencia sin verificación en archivo | No considerar el preview como una exportación disponible |
| [Espejo WaRP-D Roboflow](https://universe.roboflow.com/trash-recycling-data/warp-d-jsdws) | 2.965 fotos declaradas | Misma taxonomía de WaRP; ficha anuncia CC BY 4.0 | Preferir original; no sumar espejo como datos independientes ni confundir cambio de etiqueta de licencia con nuevos derechos |
| [Kaggle Garbage Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/versions/8) | 19.762 fotos, 10 clases | Clasificación; MIT anunciada, procedencia de fotos sin corroborar | No descargar sin corroboración; no fabricar cajas desde clases globales |
| [Voxel51 Food Waste](https://huggingface.co/datasets/Voxel51/food-waste-dataset) | Ficha 375 fotos frente a preview train de 430 filas | Deriva de HPI; máscaras propuestas con YOLO-E. MIT anunciada en conversión no elimina CC BY de fotografías originales | No usar propuestas automáticas como GT ni contar fotos derivadas otra vez |

Estas condiciones son las de las fuentes indicadas, no una transferencia de derechos de sus fotografías a nuestro código. Los términos [CC BY](https://creativecommons.org/licenses/by/4.0/deed.es), [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/deed.es) y [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.es) se conservaron como evidencia. En ZeroWaste el uso académico local respeta la condición NC de los autores; la entrada Zenodo no se usa para declarar eliminada esa restricción. Antes de una utilización incompatible o redistribución comercial debe resolverse la contradicción, sin entrenar o publicar ahora por ese motivo.

Las consultas web y los accesos directos pueden diferir por caché/bloqueos: los 403 y el error de certificado de una consulta de metadatos GitHub se conservan. No se desactivó TLS, no se eludió autenticación y no se creó una cuenta/token de Roboflow o Kaggle. No se contactó a autores ni se aceptaron términos privados en nombre del usuario.

## Qué se verificó realmente en archivos

| Fuente/split | Referencias de imagen train comprobadas en metadatos | Registros espaciales train realmente leídos | Fotos del piloto abiertas y verificadas |
|---|---:|---:|---:|
| ZeroWaste-f/train | 3.002 en COCO original | 18.002 | 24 de 1.920×1.080 |
| WaRP-D/train | 2.452 en índice ZIP y etiquetas | 8.823 | 17 de 1.920×1.080 y 7 de 960×540 |
| HPI/train | 215 filas del servicio de datos | 0 | 12 de 1.688×1.126 |

No se certifica la integridad de todas las fotos de un dataset a partir de sus nombres. No se descargaron fotos ni se abrieron anotaciones de sus splits test. El test original EcoWasteFormer tampoco se utilizó: su SHA se comprueba únicamente para preservar bytes.

Las 60 fotografías ocupan **47.784.176 bytes**. Hay cinco montajes de inspección derivados y separados; no se cuentan como imágenes nuevas. Los pilotos provienen de fotos reales: el de ZeroWaste usa el procesamiento de desenfoque publicado por sus autores, sin componer escenas ni usar la variante aumentada. WaRP usa escenas completas `Warp-D/train/images`, no los recortes de clasificación `Warp-C`.

### Frecuencias train obtenidas, no cuotas acreditadas

| Material/propuesta | ZeroWaste train | WaRP-D train | Observación |
|---|---:|---:|---|
| plástico | 5.802 | 6.071 | Etiquetas de fuente; hace falta comprobar instancia/material y diversidad física |
| cartón | 11.941 | **0 mapeadas automáticamente** | Los 650 envases `milk-cardboard`/`juice-cardboard` quedan como multicapa por revisar |
| vidrio | 0 | 448 | 177 transparentes, 136 oscuros y 135 verdes; bbox sin polígono |
| metal | 259 | **0 mapeadas automáticamente** | 562 `cans` disponibles para confirmar composición; no se promueven por nombre solamente |
| papel | 0 | 0 | No hay GT de papel suficientemente establecido en estos archivos para nuestra tarea |
| orgánico | 0 | 0 | HPI contiene fotografías de sobras, pero **ninguna anotación espacial** |

Los valores cero describen **mapeos/anotaciones disponibles**, no ausencia visual de esos materiales. No es válido concluir que una foto de ZeroWaste carece de papel porque su taxonomía no lo anote. En la inspección se ve abundante papel sin etiqueta: integrar esos fotogramas sin completar/ignorar justificadamente esos objetos introduciría falsos negativos de supervisión para nuestras seis clases.

Los 18.002 y 8.823 registros no están deduplicados por objeto físico entre fotogramas. El número de máscaras/cajas no acredita esa cantidad de residuos nuevos. Los recortes WarP-C proceden de los mismos objetos de WaRP-D y no se suman como otra colección. Las cifras declaradas de FoodInsSeg, SpectralWaste y los previews no se sustituyeron por estimaciones inventadas de instancias útiles.

## Descarga acotada, hashes y procedencia

Los originales están en `datos/nuevos/v27/externos/`, ignorados por Git. Se preservan:

- snapshots de fichas/licencias/API y sus recibos fechados, con URL, respuesta, tamaño y SHA256;
- índices originales de los ZIP, CRC y tamaños por miembro;
- `zerowaste/anotaciones_muestra/labels.json`: COCO train original de 9.174.096 bytes;
- los TXT train y catálogo original de WaRP, más TXT exactos del piloto;
- filas originales HPI, metadatos nutricionales y JPEG servidos por la fuente;
- fotos originales, propuestas pendientes y tareas humanas sin propuestas;
- atribución de autores/licencias en `ATRIBUCION.txt` y contactos locales de inspección.

Se leyeron miembros train mediante HTTP Range y `zipfile`, sin extraer archivos de varios GB completos. El tamaño remoto observado fue 7.518.242.799 bytes para `zerowaste-f-final.zip` y 844.977.310 para el ZIP WaRP. Los recibos distinguen esos tamaños de los bytes transferidos. Los CRC de miembros extraídos fueron verificados por `zipfile` y los archivos locales tienen SHA256; **no se afirma haber verificado el MD5 del ZIP completo** de ZeroWaste, porque no se descargó completo. La contraseña ZeroWaste utilizada es la publicada en su registro oficial, no una credencial privada.

Se conservó un intento inicial fallido de ZeroWaste por normalización de la cabecera HTTP `Content-Range`; se corrigió y se publicó un recibo nuevo. Se verificó que un servidor que ignore rangos sea rechazado antes de leer su cuerpo completo. El presupuesto de cada lector limita transferencias y tamaños de miembros. El filtrado no abre miembros test/val ni rutas fuera de train; los catálogos de clases son metadatos compartidos necesarios.

La revisión actual HPI era `f008d0c7…` y los assets servidos pertenecían a `d125987d…`. Se descargaron fichas y árboles de ambas revisiones: **el Parquet train conserva el mismo LFS SHA256** `1e7d20a1378cecc42090044d8760dd97155cdfd689940b4a8ab080bf75d71460`, y ambas fichas otorgan CC BY 4.0 a las fotos. No se asumió que una caché fuera automáticamente la revisión actual. El tamaño publicado es resolución de las fotos descargadas, no una resolución supuesta de previews.

La [configuración V27](../configuracion/adquisicion_publica_v27.json) registra revisión HPI, referencia WaRP versión 3, registro ZeroWaste fijo, tamaños y límites. Los recibos exactos registran cómo se obtuvo este piloto. Cambios futuros en servidores deben compararse con esos hashes, no reemplazar silenciosamente archivos locales.

## Control geométrico y puente V26

El adaptador nuevo conserva categorías y geometría originales y produce **propuestas**, no `IMPORTADO.json` ni `CONGELADA.json`. Reutiliza las funciones de geometría, letterbox, máscaras y duplicados de V26 sin modificar el código protegido.

**ZeroWaste:** las 128 anotaciones del piloto contienen polígonos que pasan finitud, límites, área y ausencia de cruces de V26. El cierre redundante se normaliza en una copia. Las 128 cajas COCO originales difieren, a la tolerancia 1e-6, de los extremos de sus polígonos. Se conserva `bbox_original` y se calcula `bbox_v26_derivada` desde el polígono original, como diagnóstico/propuesta identificable; no se reescribe el archivo de origen ni se inventa una región desde una etiqueta global. Un polígono geométricamente válido todavía requiere confirmar visibilidad, material y completitud de la escena.

**WaRP:** sus 97 cajas piloto pasan la comprobación YOLO→píxeles, límites y tamaños. No hay polígonos originales en ese subconjunto. **No se convirtió ningún rectángulo en máscara GT**. Las 12 cajas de vidrio son oportunidades para anotación manual visible, no doce regiones válidas de supervisión semántica ya obtenidas. Veintisiete cajas de otras categorías quedan pendientes de material; las otras 58 tienen propuesta de plástico según la definición de autores para los prefijos `bottle-`.

**HPI:** cero cajas/máscaras espaciales y cero regiones orgánicas verificadas en 48×48. Una lista nutricional de ingredientes no da ubicación ni número de instancias visibles. Deben anotarse manualmente los restos y decidir una granularidad coherente, sin contar cada vista o fragmentación de un ingrediente como nueva identidad para inflar la cuota.

No se calculó una supuesta cantidad de regiones orgánicas útiles sustituyendo máscaras ausentes por rectángulos o componentes de otra etiqueta. Cuando falta un material/polígono, el diagnóstico regional informa que no puede calcularse, en lugar de ignorar esa falta y afirmar ausencia de ambigüedad.

### El bloqueo real encontrado en V26

Los datasets públicos no proporcionan identidades físicas verificadas por objeto, dispositivo, sitio/sesión y fecha de captura normalizada al nivel exigido por V26. Se preservan como **null**, no como nombres ficticios. Los IDs de anotación y los prefijos de archivos no se renombran a identidades físicas. La restricción conservadora propuesta de mantener toda una fuente junta es una regla de partición, no una afirmación sobre sesiones u objetos reales.

Se realizó un [preflight real V26](../resultados/v27_adquisicion/preflight_v26.json) con una foto descargada y campos desconocidos: el importador rechazó correctamente el paquete y no publicó `IMPORTADO.json`. **Por tanto hay 0 paquetes externos nativos V26 importados, 0 revisiones humanas ejecutadas y 0 instancias aceptadas.** No se simularon anotadores ni se registraron aprobaciones sobre estas fotografías reales.

Hay 60 imágenes preparadas para adquisición/revisión y 128 propuestas poligonales que podrían adaptarse geométricamente. Las 97 cajas WaRP necesitan polígonos; las 12 fotos HPI, anotación desde cero. La importación completa sigue bloqueada por metadatos/revisión, no por un supuesto fallo del entrenamiento.

Un equipo humano debe verificar la identidad/agrupación que sea recuperable, inspeccionar la imagen completa con las seis clases, anotar y adjudicar. Si los metadatos históricos de captura no pueden obtenerse, **no se rellenarán con fechas o IDs inventados para satisfacer V26**. Para usar una fuente pública como conjunto auxiliar con campos explícitamente desconocidos haría falta aprobar una especificación de datos versionada que limite su uso a train y excluya esos registros de cuotas de identidad/diversidad. No se cambió ese contrato ni se habilitó entrenamiento en esta fase.

## Duplicados y diversidad

Se compararon SHA de archivo/píxeles y pHash con distancia ≤4 frente a los manifiestos **train/val de detección y clasificación** y entre los 60 candidatos. No se consultaron anotaciones de test para esa comparación.

| Resultado del piloto | Conteo |
|---|---:|
| Duplicados exactos contra train/val histórico | 0 |
| Parecidos pHash≤4 contra histórico | 0 |
| Parecidos pHash≤4 entre los 60 candidatos | 0 |
| Identidades físicas distintas acreditadas por V26 | 0 |

Esto no demuestra independencia. La inspección del asistente de los cinco contactos detecta escenas muy parecidas en ZeroWaste IDs 2706, 2709 y 2710, pese a que el pHash no las señaló. No se cambia el umbral para presentar éxito; se deja la advertencia de correlación y se conserva la agrupación por fuente/prefijos disponible. Esa inspección automatizada **no equivale a revisión humana ni a adjudicación**.

ZeroWaste aporta cartón deteriorado y deformado, pero una cinta y fondos muy parecidos. WaRP añade vidrio con reflejos, oclusión y diferentes tamaños, también industrial. HPI aporta restos reales, con poca diversidad de plato, fondo y preparación. Las fuentes no reemplazan los escenarios de calle, mesa, cámara física y video independiente que debe cubrir el proyecto.

## Plan para cumplir cuotas sin inflarlas

| Material train | Meta V26 | Propuestas del piloto con material identificable | Aceptadas | Pendientes oficiales |
|---|---:|---:|---:|---:|
| plástico | 150 | 98 | 0 | 150 |
| papel | 300 | 0 | 0 | 300 |
| cartón | 300 | 80 | 0 | 300 |
| vidrio | 250 | 12 cajas sin polígonos | 0 | 250 |
| metal | 150 | 8 | 0 | 150 |
| orgánico | 400 | 0 espaciales; 12 fotos para anotar | 0 | 400 |
| **Total** | **1.550** | **198 con material propuesto + 27 pendientes** | **0** | **1.550** |

La columna de propuestas no descuenta cuotas: falta revisión y no están acreditadas identidades físicas. Las 11.941 anotaciones `cardboard`, 448 `glass-*`, 5.802+6.071 plásticos y 259 metales de metadatos train muestran dónde buscar candidatos; **no garantizan** suficiente diversidad ni esa cantidad de GT finalmente usable.

Prioridades de adquisición concretas:

1. **Cartón:** revisar primero los 80 polígonos piloto y la coherencia de “cardboard” con nuestra separación papel/cartón; auditar objetos no anotados de otras clases. Después seleccionar de ZeroWaste por secuencias separadas y completar con corrugado, impreso, deteriorado, parcial y fondos nuevos. No contar 300 fotogramas correlacionados como 300 objetos.
2. **Vidrio:** anotar polígonos visibles de las 12 cajas piloto, comprobar las tres categorías originales y buscar más entre las 448 cajas train disponibles si la calidad lo respalda. Capturar además vasos/tarros, transparencia, reflejos y fondos fuera de la planta, con ≥100 escenas/objetos físicos prioritarios según V26. Los envases multicapa de WaRP no llenan la cuota de cartón.
3. **Orgánicos:** comenzar con anotación independiente de los 12 platos HPI y comparar criterios de pieza/resto/ingrediente. No multiplicar 12 comidas usando sus 215 vistas. La adquisición propia sigue siendo necesaria para cubrir 400 instancias y ≥300 regiones útiles de dos tokens C8 en 48×48, ≥100 objetos/escenas y diversidad de condiciones. No hay hoy una cantidad espacial pública orgánica verificable que permita descontar la meta; ni fondo UC ni máscaras automáticas del espejo sirven como atajo.
4. **Papel:** planificar las 300 anotaciones reales nuevas, idealmente ≥100 escenas/objetos. En los candidatos descargados no hay etiquetas de papel suficientemente verificadas. El papel visible en ZeroWaste se puede anotar manualmente solo tras decidir protocolo y permiso, pero no se convierte en GT por su apariencia en esta auditoría.
5. **Plástico y metal:** aprovechar los candidatos geométricos útiles de ZeroWaste/WaRP tras revisión y completar diversidad con capturas propias. Los 259 registros metal de ZeroWaste pueden repetir objetos; no prometer 150 identidades diferentes basándose en esa cuenta.
6. **Generalización:** permanecen pendientes **600 instancias nuevas independientes**, capturadas/agrupadas por sitio, objeto, sesión y secuencia bajo el protocolo V26. No reservar unas imágenes parecidas de estas mismas cintas como supuesta prueba externa ni desbloquear la colección. Su suficiencia se comprobará con el inventario real y revisión, no con cifras de proveedores.

El [plan JSON de cuotas pendientes](../resultados/v27_adquisicion/plan_cuotas_pendientes.json) conserva todas las metas V26. Una futura colección controlada sería: núcleo propio con identidad física y metadatos verificables, más fuentes externas revisadas bajo un contrato explícito y compatible, y generalización propia separada. Hoy no existe una colección suficiente, revisada y congelada: **V28 de entrenamiento permanece deshabilitada**.

## Archivos entregados y reproducción

| Archivo/carpeta | Contenido |
|---|---|
| `resultados/v27_adquisicion/matriz_datasets.json` y `.csv` | Fuentes, licencias, cifras y limitaciones por dataset |
| `piloto_zerowaste.json`, `piloto_warp.json`, `piloto_hpi.json` | Recibos, miembros/URLs, cantidades y procedencia real |
| `piloto_preparado.json` | QA resumida; 60 fotos, 225 propuestas, 0 aceptadas |
| `duplicados_train_val_y_pilotos.json` | Comparación real de hashes/pHash, solo train/val histórico |
| `preflight_v26.json` | Rechazo correcto sin inventar metadata, 0 revisiones simuladas |
| `fuentes_adicionales.json` | Licencias, errores HTTP y prueba de igualdad HPI entre revisiones |
| `datos/nuevos/v27/externos/propuestas_pendientes.json` | Geometría original, propuestas derivadas trazables y unknowns explícitos |
| `datos/nuevos/v27/externos/tareas_humanas_sin_propuestas.json` | Formularios vacíos de anotación/adquisición; no aprobación humana |
| `datos/nuevos/v27/externos/contactos/` | Cinco contactos locales, **no publicados en Git** |

Los hashes locales y el manifiesto final permiten verificar los archivos conservados sin repetir descargas. Las URLs temporales de assets pueden caducar: se conservan sus bytes y revisión. Una repetición de adquisición requiere rutas nuevas/una copia de trabajo separada, las mismas revisiones, índices y selección; no se reemplazan los recibos de esta ejecución.

```powershell
$pyEco = Join-Path $env:USERPROFILE '.venvs\EcoWasteFormer\Scripts\python.exe'
& $pyEco -m unittest discover -s pruebas
& $pyEco scripts/verificar_adquisicion_v27.py
```

Para adquisición desde una salida vacía: `investigar_v27.py --preparar`, `adquirir_v27.py --fase indices`, `adquirir_v27.py --fase hpi` y `adquirir_v27.py --fase pilotos`; después `completar_fuentes_v27.py`, `preparar_piloto_v27.py` e `informar_adquisicion_v27.py`. No ejecutar inicializaciones sobre esta salida ya existente. Las operaciones de red son públicas y acotadas; en el entorno actual requieren permiso fuera del sandbox porque los sockets internos están bloqueados.

## Pruebas, integridad y límites

Se conservan las **246 pruebas previas** y se añaden **15** sobre propuestas, mapeos conservadores, cajas sin máscaras, no invención de identidades, geometría, lectura acotada y rechazo de respuestas HTTP incorrectas. La certificación registra el resultado real de la suite completa de **261 pruebas** y el manifiesto de código, resultados y archivos descargados.

Los archivos históricos se comprueban por SHA, el seleccionado de época 19 conserva su SHA original y los 105 checkpoints conservan estadísticas. La verificación V26 sigue aprobada. Las fotografías originales, máscaras y montajes están ignorados por Git y no se enviaron a ningún repositorio. No se entrenó, no se ejecutó un detector, no se evaluó test y no se desbloqueó generalización.

Los metadatos y la inspección de un piloto no establecen una tasa de error de etiquetas de todo el dataset ni equivalencia de dominios. Un mapeo de taxonomía sigue siendo una propuesta a revisar. La ausencia de duplicados por hash no prueba objetos distintos; un polígono válido no demuestra material ni completitud de seis clases. No se confunden grants de código, ficha, fotografías y anotaciones automáticas. Los permisos comerciales de datos indicados no son una conclusión jurídica sobre pesos futuros.

La siguiente acción de datos es revisar/adjudicar las muestras reales y completar la adquisición propia y la procedencia pendiente. **No iniciar otro entrenamiento con los mismos datos insuficientes ni avanzar a V28 antes de disponer de una colección suficiente y congelada.**
