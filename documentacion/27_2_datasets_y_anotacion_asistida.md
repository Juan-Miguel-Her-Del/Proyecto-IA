# V27.2: datasets compatibles y anotación asistida por IA

Se entrega un editor local funcional con **72 imágenes y 256 propuestas pendientes de corrección**. Hay 245 propuestas con contornos preliminares y 11 que conservan solo caja. SAM 2.1 Tiny generó 86 contornos utilizables geométricamente a partir de las 97 cajas WaRP; los otros 11 fallaron controles y no se ofrecieron como polígonos válidos. Ninguna propuesta está aceptada en V26.

No se entrenó EcoWasteFormer, no se modificaron sus pérdidas o arquitectura y no se calculó un nuevo mAP. El seleccionado original de época 19, los históricos, test y la colección de generalización bloqueada se conservan. SAM es un asistente externo de anotación; sus pesos no sustituyen los del detector.

## Comenzar a corregir

1. Ejecutar [ABRIR_ANOTADOR_V27_2.cmd](../scripts/ABRIR_ANOTADOR_V27_2.cmd). Abre el editor en **http://127.0.0.1:8767**. Si ya está ejecutándose con este proyecto, el lanzador abre esa instancia.
2. Escribir un identificador personal estable, por ejemplo `juan_miguel`. Seleccionar fuente, imagen y objeto. Las cajas discontinuas todavía requieren contorno; las máscaras visibles siguen siendo propuestas.
3. Revisar el material. Para mejorar el contorno, usar una caja alrededor del objeto o añadir **clics positivos para incluir y negativos para excluir**; pulsar «Proponer / refinar con SAM». No hace falta dibujar todo el polígono. Los clics no cambian el material por sí solos.
4. Para un objeto omitido, pulsar «Otro objeto», dibujar su caja, usar SAM y escoger material. En HPI no hay cajas originales: indicar un resto visible, no asignar una máscara a cada ingrediente de la lista nutricional ni al plato por su etiqueta global.
5. Si hace falta, mover vértices; Shift+clic inserta y Alt+clic elimina un vértice. Hay polígono manual opcional y botón de deshacer. Inspeccionar fragmentos y huecos; un score alto de SAM no demuestra material ni calidad suficiente.
6. Guardar sin la casilla de cobertura completa conserva un **borrador**. Marcar la casilla únicamente después de revisar toda la escena, materiales y objetos omitidos. Para ambigüedad, compuesto, fuera de alcance o descarte se exige motivo. Los controles rechazan materiales sin asignar, cajas sin máscara y geometría inválida al declarar revisión completa.
7. «Exportar COCO pendiente» descarga polígonos/máscaras y sus estados, referencias y huellas. **No produce aceptación V26 ni un dataset autorizado para entrenar.** Conservar la carpeta de proyecto y los diarios junto a la exportación.

Los controles y botones inferiores están en la barra lateral, que se desplaza de forma independiente. Para liberar RAM después de segmentar se puede pulsar «Liberar SAM». El editor usa **CPU**: no carga CUDA ni reserva memoria GPU. Al arrancar no carga siquiera los pesos SAM; lo hace al solicitar una nueva propuesta. Las propuestas iniciales ya están guardadas.

El entorno es `%USERPROFILE%\.venvs\EcoWasteAnotacion`, separado de EcoWasteFormer, con PyTorch `2.5.1+cpu` y torchvision `0.20.1+cpu`. No se cambió PyTorch del proyecto histórico. El servidor escucha exclusivamente en loopback, no requiere cuenta y no envía fotos a servicios de anotación externos. Cerrar el proceso del lanzador detiene esa instancia; una instancia iniciada separadamente sigue disponible en su URL local.

## Fuentes verificadas y decisión

La [comparación JSON](../resultados/v27_2_anotacion/comparacion_fuentes.json) y el [CSV](../resultados/v27_2_anotacion/comparacion_fuentes.csv) separan las cifras publicadas de las verificadas en archivos. El alcance de licencia procede de las fuentes citadas; se conserva atribución, restricciones y procedencia, sin presumir derechos adicionales sobre contenido de terceros.

| Fuente | Declaración de sus autores | Verificado localmente | Decisión |
|---|---|---|---|
| [WasteSegCluster v4](https://data.mendeley.com/datasets/3wnf3n242s/4) | 1.947 imágenes, aproximadamente 3.900 instancias; cuatro categorías; CC BY 4.0 | ZIP oficial íntegro; 1.947 JPG en índice. COCO train: 1.795 imágenes y 3.559 anotaciones; **359 nombres fuente con exactamente cinco variantes cada uno**. Valid externo: 103 imágenes y 249 anotaciones | Excluir las variantes aumentadas train del piloto. Usar 12 imágenes publicadas de valid externo, con sus 31 polígonos originales, como candidatas auxiliares train; no son validación de EcoWasteFormer |
| [DWSD v1](https://data.mendeley.com/datasets/gr99ny6b8p/1) | 784 imágenes, 2.350 segmentos, 14 categorías; CC BY 4.0 | ZIP íntegro con 784 pares PNG, 640 Train y 144 Test en índice. Se leyeron las 640 máscaras Train: todas modo L. **No contiene LabelMe JSON, tabla de valores a materiales ni IDs de instancia verificables** | Ocho pares extraídos quedan en cuarentena. No convertir sus componentes en instancias GT. No forman parte de las 72 tareas activas |
| [RealWaste UCI](https://archive.ics.uci.edu/dataset/908/realwaste) | 4.752 fotografías de clasificación y nueve categorías; CC BY 4.0 | El ZIP histórico ya contiene 4.752 referencias de fotografía. Se comprobó su índice y SHA actual, sin abrir imágenes test. No existen anotaciones espaciales originales | Cero fotografías nuevas acreditadas. No descargar otra copia ni crear máscaras desde clases globales. Una futura anotación de train histórico sería reuso explícito |
| [ZeroWaste-f](https://github.com/dbash/zerowaste) | Fuente ya investigada en V27 | Reutilizadas 24 fotos y 128 polígonos | Conservar alcance académico no comercial de los autores y discrepancia CC BY del registro; completar omisiones de nuestras seis clases |
| [WaRP-D](https://github.com/AIRI-Institute/WaRP) | Fuente ya investigada en V27; CC0 declarada en distribuidor original enlazado | Reutilizadas 24 fotos y 97 cajas. SAM propone 86 contornos que pasan el control exterior de polígonos; 11 incidencias conservadas | Corregir material y contornos, no aceptar automáticamente. Los cartones multicapa no se convierten en cartón simple |
| [HPI Food Waste](https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset) | CC BY 4.0 explícita para fotos y registros | Reutilizadas 12 fotos, una por comida train; **cero máscaras originales** | Prioridad de anotación asistida para restos reales. Las 215 vistas train no acreditan 215 comidas ni residuos distintos |

Los permisos CC BY permiten uso académico/comercial bajo sus condiciones de atribución; ello no elimina la revisión de procedencia o derechos de terceros. ZeroWaste conserva su restricción NC según autores: este piloto no constituye una colección homogénea con permiso comercial irrestricto.

### WasteSegCluster: qué significa realmente Organic

El archivo corresponde a una exportación Roboflow `Extened_Dataset of Waste Classification` v3, de marzo de 2026, publicada como Mendeley v4. Se conservaron ambos README. Declaran auto-orientación y resize fit-within 640; el train contiene cinco versiones por nombre fuente, con volteos, giros y ruido. **No se presentan esas 1.795 variantes como fotografías independientes ni originales de cámara.**

Se eligió la partición pública `valid`, que tiene un registro por nombre fuente y no las cinco variantes train. No se evaluó ningún modelo sobre ella. Las fotos publicadas siguen estando redimensionadas; no se afirma disponer del RAW o JPEG original de cámara. No se abrieron anotaciones ni ejemplos `test` de esa fuente. El ZIP completo, descargado para verificar su SHA, contiene también esas carpetas del proveedor.

Las frecuencias train leídas son: Plastic 1.698, Organic 681, Paper 665 y Composite_Rubber_Textile 515. La categoría 0 es supercategoría y no aparece como anotación train. No son recuentos de objetos físicos nuevos.

En los ejemplos Organic seleccionados se ven tallos, troncos o material vegetal en agua y no evidencia suficiente para equipararlos a sobras de comida. Por tanto sus **cuatro propuestas piloto Organic quedan sin asignar**. La taxonomía histórica del clasificador incluye Vegetation en orgánico; eso no demuestra que una planta viva, un objeto sumergido ambiguo o cada etiqueta Organic del proveedor cumpla el criterio de nuevos residuos alimentarios. Se requiere decidir caso por caso, conservando la distinción. El piloto no llena cuotas orgánicas con esa etiqueta.

Las otras propuestas WSC son 17 Plastic→plástico, 5 Paper→papel y 5 Composite_Rubber_Textile→compuesto, todas pendientes. Material compuesto no se fuerza a una de las seis clases. Las fechas COCO coinciden con exportación; **no se usaron como fechas reales de captura**.

### DWSD: límites del archivo publicado

La ficha menciona LabelMe y categorías como papel, vidrio, botellas metálicas, aluminio y diferentes plásticos. El archivo recibido es `DSWD.zip`, con directorios Image/Mask y sin sus JSON de formas. Las máscaras Train contienen valores observados `0,1,2,3,4,5,6,8,9,10,11,12,13,14,15`; sin clave no se puede demostrar qué valor corresponde a vidrio, papel, plástico o metal, ni reconstruir instancias tocándose.

Operativamente se conservan como **referencia semántica sin identidad verificable de instancia**, no como 2.350 instancias listas. No se interpreta cada componente desconectado como objeto físico. La inspección del asistente de ocho muestras mostró giros con bordes negros y variaciones fuertes de color, además de una escena sin residuo claro. El archivo no documenta una correspondencia recuperable con originales sin transformar. Esto justifica la cuarentena, sin afirmar que todo el dataset sea sintético ni inventar cuántos originales independientes contiene.

### RealWaste: fuente ya histórica

Se verificó el índice del ZIP existente: Cardboard 461, Food Organics 411, Glass 420, Metal 790, Miscellaneous Trash 495, Paper 500, Plastic 921, Textile Trash 318 y Vegetation 436. Los conteos train/val históricos pertinentes están en [realwaste_historico_verificado.json](../resultados/v27_2_anotacion/realwaste_historico_verificado.json) y el recibo inicial del piloto. **No se consultó el manifiesto test para seleccionar muestras ni se abrieron imágenes test.**

Su procedencia ya fue usada por el clasificador: reanotarla no aporta por sí sola nueva diversidad visual. Food Organics y Paper podrían servir para un ensayo explícito de anotación sobre train histórico, pero no se añaden aquí a las cuotas ni al piloto como adquisición nueva. Ninguna etiqueta global se convirtió en caja o máscara.

## Piloto activo y anotaciones pendientes

Se preservó la preparación inicial de 80 imágenes y después se creó **otra selección de 72**, excluyendo DWSD. No se sobrescribió el proyecto inicial ni su evidencia. El proyecto activo está en `datos/nuevos/v27_2/proyecto/seleccion_72/`; las correcciones futuras vivirán allí, fuera de Git.

| Fuente | Imágenes activas | Propuestas originales pendientes | Trabajo preparado |
|---|---:|---:|---|
| ZeroWaste | 24 | 128 | Polígonos existentes visibles; revisar materiales, cobertura y geometría |
| WaRP | 24 | 97 | 86 contornos SAM y 11 cajas por refinar |
| HPI | 12 | 0 | Restos reales para indicar mediante caja/clics y SAM |
| WasteSegCluster | 12 | 31 | Polígonos existentes; Organic sin asignar |
| **Total activo** | **72** | **256** | **245 con contornos preliminares, 11 solo caja** |

La distribución propuesta es: plástico 115, cartón 80, vidrio 12, metal 8, papel 5, compuesto 5 y sin asignar 31. **No es distribución de GT aceptado.** Se ofrecen fotografías de comida para orgánico, pero todavía no hay máscaras orgánicas corregidas ni regiones orgánicas elegibles a 48×48. La cantidad de nuevos objetos omitidos que el usuario añadirá es desconocida; no se inventa un conteo.

De los contornos iniciales, 24 contienen huecos. Se preservan y señalan; **245 contornos no equivalen a 245 anotaciones importables nativamente en V26**. También hay que comprobar fragmentación, fusión, visibilidad y material. Las geometrías SAM rechazadas permanecen en evidencia automática, sin corregirse en nombre de una persona.

Las 20 muestras nuevas extraídas —12 WSC activas y 8 DWSD en cuarentena— se compararon por SHA de archivo/píxeles y pHash≤4 frente a detección/clasificación train/val históricos y V27. No aparecieron candidatos. Las 60 imágenes V27 reutilizadas se declaran como tales, no como nuevas. Los nombres fuente WSC agrupan variantes; bonid HPI agrupa comidas; ninguno se convierte en identidad física o sesión demostrada. Un pHash negativo no acredita independencia y persisten escenas correlacionadas, incluidas las ya observadas en V27.

## Elección de herramienta y coste

Se revisaron la [instalación oficial de SAM 2](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md), su [licencia y pesos](https://github.com/facebookresearch/sam2) y el [despliegue de anotación automática de CVAT](https://docs.cvat.ai/docs/administration/community/advanced/installation_automatic_annotation/). SAM recomienda WSL/Linux para Windows; CVAT y sus funciones requieren servicios adicionales y Docker/Nuclio según modalidad. Docker está instalado, pero el daemon no estaba disponible en esta máquina.

| Opción | Aporte | Coste/limitación local | Decisión |
|---|---|---|---|
| CVAT + servicio SAM | Herramienta estable con proyectos y revisión multiusuario | Requiere desplegar y mantener servicios; Docker no activo. No se verificó una instalación CVAT funcional ni se creó cuenta cloud | Alternativa futura si aumenta el equipo; no se presenta como instalada |
| Editor local + SAM 2.1 Tiny oficial | Cajas, clics +/−, contornos existentes, vértices, material, omisiones, motivos, historial y COCO | Un flujo local pequeño; no sustituye todas las funciones de CVAT, la autenticación ni adjudicación V26 | Implementado y probado en Edge |

Código SAM fijado a `2b90b9f5ceec907a1c18123530e92e794ad901a4`, pesos `sam2.1_hiera_tiny.pt`, licencia Apache-2.0, CPU y cuatro hilos. Se conserva el ZIP de código, LICENSE, SHA de pesos y censo de archivos extraídos; se comprueban antes de cargar el modelo. Los pesos externos están en `datos/nuevos/v27_2/modelos/`, no en `checkpoints/`.

La selección multimáscara se prefijó como máximo score SAM, sin GT. Se desactivó postprocesamiento opcional SAM, sin entrenamiento o ajuste al proyecto. SAM propone regiones, **no predice nuestro material**. Reutiliza el embedding de la imagen durante sus correcciones. La primera prueba real necesitó aproximadamente 4,57 s, y las 97 cajas del lote se procesaron en **46,98 s**, con **0 bytes de memoria GPU**. Son tiempos observados de esta CPU, no garantías de latencia.

La máscara binaria automática original se conserva como RLE; los contornos editables son una representación derivada. La conversión/simplificación puede cambiar píxeles pequeños y hay que inspeccionarla. No se afirma identidad píxel a píxel entre el RLE automático y un polígono corregido, ni se asignan nuevos IDs físicos a fragmentos de una máscara.

## Contrato auxiliar versionado y puente V26

Se añadieron [configuración auxiliar](../configuracion/auxiliares_publicos_v27_2.json) y [esquema independiente](../esquemas/auxiliares_publicos_v27_2.json). No se editó `incorporacion.py`, `revision_datos.py`, políticas, manifiestos o contratos V26. Los desconocidos físicos siguen siendo `null`: objeto, sitio, sesión, dispositivo y captura. IDs auxiliares identifican registros, no objetos reales.

| Estado auxiliar | Significado | Lo que no acredita |
|---|---|---|
| `propuestas_pendientes` | Anotación pública o SAM, aún sin corrección registrada | No revisión humana |
| `borrador_usuario` | Cambios guardados sin declaración de cobertura completa | No imagen completamente revisada |
| `corregida_usuario_unico` | Un usuario declaró revisar la imagen completa y guardó geometría/materiales válidos | No independencia ni adjudicación V26 |
| `con_revision_independiente_pendiente_adjudicacion` | Otra persona autodeclarada anotó sin las propuestas anteriores y sobre la revisión vigente | No identidad personal verificada ni aceptación V26 |
| Aceptada formalmente V26 | Exclusivamente su flujo original con evidencia física, revisores y adjudicación requeridos | **No se genera en este editor** |

La vista independiente comienza vacía. El servidor liga el guardado a la vista emitida, exige otro identificador y rechaza bases antiguas; cambiar el cuerpo JSON no transforma una vista asistida en independiente. Una modificación posterior de la primera revisión invalida el estado de revisión independiente anterior. El identificador personal es autodeclarado: el software no demuestra que dos nombres sean personas diferentes. Debe documentarse esa limitación y no presentarse como doble revisión certificada.

Cada guardado crea un evento nuevo con imagen/proyecto SHA, persona, momento, modo, cobertura, instantánea de anotaciones, cambios y enlace hash al anterior. Un conflicto entre pestañas rechaza la escritura. La propuesta SAM tiene su propio recibo, indicaciones, versión y pesos, sin generar un evento humano. Los originales permanecen inmutables y toda propuesta o corrección conserva `aceptada_V26=false`.

La exportación es **COCO por instancia con metadatos auxiliares**: polígonos, cajas derivadas, área de máscara, materiales/ignore, hashes y referencias a revisiones. Si hay huecos, usa RLE COCO sin comprimir para conservarlos. La geometría simple reutiliza los controles V26; aun así no es un `IMPORTADO.json` aceptado. El contrato V26 debe seguir rechazando datos sin evidencia física y no admite huecos bajo su esquema actual. No se rellenan huecos, fechas, sesiones o identidades ficticias para eludirlo.

No se congeló una colección auxiliar ni se añadió un loader de entrenamiento. El COCO pendiente no debe entregarse a un entrenador como GT verificado sin seleccionar revisión, resolver máscaras/ambigüedades y autorizar un protocolo separado.

## Estudio exploratorio propuesto, sin ejecución

La [propuesta JSON](../configuracion/exploratorio_auxiliar_v27_2.json) plantea, después de recibir correcciones, seleccionar **40–60 imágenes** revisadas y al menos ocho polígonos por material, con restos de comida para orgánico y fuentes agrupadas. Hoy no se cumplen esos mínimos revisados. Ni esa escala ni el piloto descuentan cuotas de identidades físicas V26.

Una comparación futura tendría control con datos originales y variante con auxiliares revisados, desde el mismo origen y con presupuesto prefijado. **Ese presupuesto todavía no está aprobado**: no se heredan las 1.452 actualizaciones ni las hipótesis V20. Sin control independiente por origen se limitaría a capacidad/diagnóstico train, sin concluir mejora de generalización ni seguir seleccionando variantes sobre la misma validación de desarrollo. La evaluación final exige una colección nueva separada y autorizada; test original y generalización permanecen reservados.

Siguen pendientes formalmente **1.550 instancias train, 600 de generalización y 300 regiones orgánicas útiles a 48×48**. Anotar con IA puede ahorrar tiempo, pero no convierte variantes de una misma foto en objetos nuevos, ni sustituye diversidad y trazabilidad de captura. V28 continúa deshabilitada.

## Archivos, pruebas e integridad

| Entregable | Ubicación |
|---|---|
| Catálogo/licencias y recibos de acceso | `resultados/v27_2_anotacion/consultas/` y snapshots privados |
| ZIP WSC oficial | 100.414.243 bytes, SHA `2f49ef8a3736ee44fbd451143c9b21e6d3745f6507bc624245c4a85e64197925` |
| ZIP DWSD oficial | 252.719.363 bytes, SHA `fb9229c330ed253f4fb7c18bd46dddebe4df6f8d3f602747f505ff7601a11737` |
| Auditorías de archivos y taxonomías | `auditoria_wsc.json`, `auditoria_dwsd.json`, `comparacion_fuentes.json/.csv` |
| Estado numérico del piloto | [estado_entrega_piloto.json](../resultados/v27_2_anotacion/estado_entrega_piloto.json) |
| 86 propuestas SAM y 11 incidencias | [piloto_sam2_cpu.json](../resultados/v27_2_anotacion/piloto_sam2_cpu.json) |
| Proyecto activo y recibos SAM | `datos/nuevos/v27_2/proyecto/seleccion_72/` — privado, ignorado por Git |
| Prueba visual y funcional | `prueba_interfaz.json` y revisión 02; capturas de fotos reales locales, no publicadas |
| Dependencias aisladas y versiones | `dependencias_sam_cpu.json`, `dependencias_pruebas_ui.json` |
| Verificación reproducible | `scripts/verificar_anotacion_v27_2.py`, manifiesto y certificado V27.2 |

Se descargaron completos los dos ZIP oficiales y sus SHA coinciden con el catálogo público del proveedor. No se verificaron todas las imágenes decodificadas por solo contar nombres; las del piloto sí se abrieron, midieron y hashearon. Se conservaron anotaciones originales, máscaras de referencia, licencias, atribución y transformaciones de propuestas por separado. Fotos, modelos, montajes, proyectos y correcciones están ignorados por Git; no se subieron a ningún repositorio.

Se añaden **31 pruebas unitarias** a las 261 existentes: contrato auxiliar, desconocidos, RLE/huecos, geometría, material, descarte, diarios, reanudación, conflictos, revisión independiente y separación de propuestas entre imágenes, además de la API local. La suite completa es de **292 pruebas**; su resultado real y log quedan certificados. El test visual en Edge prueba SAM CPU, cambio de material, guardado, independencia, rechazo del mismo actor, exportación, arrastre y deshacer únicamente en un fixture eliminado. Las 72 fotos reales se visitan sin guardar revisiones simuladas. No hubo errores JS en el control aprobado.

Las incidencias de preparación no se ocultaron: sockets bloqueados dentro del sandbox, endpoints Mendeley que devolvieron 401/404 y un error de cadena TLS en GitHub. Se usó el catálogo público servido por la página; no se eludió autenticación. Agregar el bundle público actualizado de certifi permitió verificar TLS de los archivos SAM, sin desactivarlo. El primer test del navegador intentó una espera basada en evaluación dinámica, que la CSP rechazó; se cambió el test a locators, conservando la CSP. Las 11 geometrías SAM rechazadas y la selección inicial de 80 imágenes se conservan.

La primera redirección PowerShell etiquetó la salida stderr habitual de unittest como `NativeCommandError`, aunque el log terminaba en 292 pruebas y `OK`. Se conserva ese log; la certificación ejecuta Python mediante subprocess y registra su código de salida directamente, sin esa redirección.

La certificación compara los SHA históricos, estadísticas de 105 checkpoints y el SHA del seleccionado. También vuelve a verificar V27 en lectura. Los diarios futuros del usuario son mutables por adición y se validan por su cadena; no se confunden con el estado inicial certificado de cero revisiones.

```powershell
$pyEco = Join-Path $env:USERPROFILE '.venvs\EcoWasteFormer\Scripts\python.exe'
& $pyEco scripts/verificar_anotacion_v27_2.py

$pyAnota = Join-Path $env:USERPROFILE '.venvs\EcoWasteAnotacion\Scripts\python.exe'
& $pyAnota scripts/anotar_v27_2.py --abrir
```

La siguiente intervención es humana y concreta: corregir propuestas, completar las omisiones y registrar las decisiones. **Preparar máscaras con IA no equivale a haberlas revisado ni demuestra una mejora del detector.**
