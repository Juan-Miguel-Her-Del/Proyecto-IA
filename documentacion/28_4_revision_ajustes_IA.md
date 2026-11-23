# V28.4 — revisión humana de ajustes IA

Se entrega un editor de revisión en **http://127.0.0.1:8771/**. Utiliza la publicación `correcciones_IA_3` de V28.3, sin volver a ejecutar SAM sobre las imágenes. El editor anterior de 8768 y el comparador de 8770 se conservan.

## Uso rápido

1. Comprueba tu identificador, normalmente `juan_miguel`, y abre una de las 44 imágenes con propuestas. A la izquierda aparece la base original del grupo; a la derecha, la propuesta IA o tu edición guardada. Zoom, opacidad y **Acercar al objeto** ayudan con piezas pequeñas. El contexto de otros objetos se muestra atenuado.
2. Revisa el resumen de objetos nuevos, cambios de material, contorno e ignore. Marca **He inspeccionado esta propuesta o grupo** sólo después de compararla. Puedes visitar otros grupos de la misma imagen y seleccionar varios ya inspeccionados.
3. Pulsa **Aceptar IA seleccionada**, **Rechazar seleccionada y conservar base** o **Marcar seleccionada para revisión posterior**. El rechazo y la revisión posterior requieren explicar el motivo. Ver una imagen o marcar una casilla no guarda una aprobación. La aceptación por lotes se limita a las propuestas seleccionadas que hayas inspeccionado.
4. Si hace falta modificar algo, elige el objeto del grupo y cambia su material o justificación. Para refinar el contorno, dibuja una caja o añade clics positivos/negativos y pulsa **Refinar objeto con SAM**. También hay pincel para añadir/quitar máscara, abrir huecos y deshacer. SAM se ejecuta en CPU únicamente al pulsarlo; su resultado es un borrador. Pulsa **Guardar mi edición del grupo** para registrar tu corrección. No es necesario redibujar los contornos que ya sirven.
5. Cuando no queden propuestas pendientes, comprueba las advertencias y la cobertura completa de la fotografía, incluidos residuos omitidos e ignore. Marca la casilla de inspección de toda la imagen, escribe qué verificaste y pulsa **Confirmar cobertura de esta imagen**. Aceptar propuestas no marca una imagen completa. Posponer sigue bloqueando su candidatura. Una nueva decisión posterior invalida la cobertura anterior.

Las partes rojas/rosas son ignore: no son positivos. Los envases multicapa, la vegetación, la comida pequeña y los solapes mantienen advertencias explícitas. Si no se puede confirmar el material, conserva el ignore y su motivo. No clasifiques como fondo un residuo omitido: puedes añadir un objeto y delimitarlo, o posponer la revisión.

Los cambios dependientes se revisan juntos. Por ejemplo, reducir dos máscaras por un solape y crear su región ignore constituye un grupo vinculado; aceptar sólo una parte produciría una combinación incoherente. Puedes editar objetos dentro del grupo, pero su decisión se guarda completa. Hay **92 propuestas/grupos**, no sólo 41: incluyen las nuevas regiones, cambios existentes, ignore y confirmaciones de base que todavía no tenía revisión completa.

En las cuatro imágenes sin guardado humano previo y en el borrador, rechazar una modificación no convierte las propuestas originales en anotaciones revisadas. Cuando el rechazo restaura una base sin revisar, debes confirmarla explícitamente mediante aceptación o edición/descarte justificado antes de confirmar la cobertura.

## Abrir y recuperar trabajo

Se puede usar [ABRIR_REVISION_IA_V28_4.cmd](../scripts/ABRIR_REVISION_IA_V28_4.cmd), o desde la raíz:

```powershell
& "$env:USERPROFILE\.venvs\EcoWasteAnotacion\Scripts\python.exe" scripts/iniciar_editor_v28_4.py
```

El lanzador usa una ventana oculta y comprueba que el puerto corresponde a esta versión/proyecto. No sustituye automáticamente otro servicio. El backend registra la huella del código que cargó y rechaza decisiones nuevas si el código cambia sin reiniciar.

El borrador se conserva en IndexedDB por proyecto e imagen. **Descargar copia de emergencia** funciona sin red. **Recuperar borrador local** e importar una copia requieren tu acción y comprueban imagen, proyecto y estado base. Una copia antigua no se mezcla sobre nuevas decisiones. Si hay conflicto, descarga la copia y usa **Abrir último guardado conservando copia** para comparar.

Si una respuesta se pierde, **Reintentar solicitud sin confirmar** consulta el mismo identificador y no duplica el evento. **Actualizar sesión conservando borrador** renueva el contexto; si el recibo ya confirma la escritura, muestra el guardado existente. Los HTTP400 identifican la causa y conservan el borrador, permitiendo corregir un objeto inválido antes de volver a guardar. Cada solicitud autorizada de guardado tiene además una copia cruda en el servidor, incluso si falla su validación. Las copias de emergencia no son revisiones humanas ni GT aprobada.

Se bloquea otro clic de guardado antes de esperar IndexedDB. No se prometen borradores persistentes si se borra deliberadamente el almacenamiento; ante errores de almacenamiento se comunica el fallo y queda la descarga de emergencia.

## Qué se conserva y qué se registra

- Los **197 eventos humanos originales** y sus cadenas permanecen en `datos/nuevos/v27_2/proyecto/seleccion_72/`. V28.4 no llama a su función de guardado. Se creó un respaldo nuevo verificable de **904 archivos**, sin sustituir los anteriores, incluido el histórico de 731 archivos.
- Las propuestas V28.3, sus **41 positivos añadidos**, **24 ignore nuevos**, máscaras originales, decisiones IA y manifiestos permanecen intactos. El catálogo V28.4 referencia sus versiones y SHA; los cambios en fuentes o en la revisión humana base bloquean aplicar una copia obsoleta.
- El nuevo registro vive en `datos/nuevos/v28_4/proyecto/eventos/`. Cada archivo inmutable contiene número, hash anterior, identificador de solicitud, persona declarada, imagen, versión de IA, huella de la publicación y de cada propuesta, acción, anotaciones RLE y nota. Una decisión múltiple conserva las decisiones de todos los grupos incluidos.
- Aceptación, rechazo, edición, revisión posterior y confirmación de cobertura son acciones distintas. Los GET de navegación/inspección no crean eventos humanos. No hay aprobación global automática de imágenes. La concurrencia se comprueba por estado de imagen y los reintentos son idempotentes, incluso después de reiniciar.
- La procedencia original de una máscara se conserva. `revision_V284` indica la nueva decisión real del usuario; aceptar una propuesta IA no transforma su origen en dibujo manual. Rechazar no equivale a aceptar el ajuste IA. Una región nueva no se presenta como parte de una revisión humana anterior.
- La revisión es individual. No se infiere revisión independiente porque aparezca otro identificador ni se acredita aceptación V26. Los identificadores de instancia tampoco demuestran objetos físicos distintos; procedencias, licencias y agrupaciones desconocidas se mantienen.

Las fotografías, máscaras privadas, borradores, respaldos y capturas permanecen fuera de Git mediante el `.gitignore` de V28.4. No se han redistribuido imágenes de terceros.

## Auditoría, conteos y exportación

Después de cada decisión confirmada se solicita una auditoría RLE en segundo plano. El guardado tiene su propio recibo: un fallo de auditoría no convierte una escritura confirmada en una escritura perdida. El panel distingue auditoría vigente, pendiente y error; la exportación exige una auditoría vigente.

**Actualizar auditoría incremental** también puede invocarse sin nuevas decisiones:

```powershell
& "$env:USERPROFILE\.venvs\EcoWasteAnotacion\Scripts\python.exe" scripts/auditar_revision_v28_4.py
```

Los snapshots de `proyecto/auditorias/` se encadenan, tienen huellas por archivo y no se sobrescriben. La caché es por estado de imagen, código QA y vigencia de origen. Se distinguen propuestas aceptadas/editadas, positivas confirmadas dentro de propuestas —aunque todavía falte cobertura— e imágenes/objetos utilizables. Las rechazadas y pospuestas no se cuentan como ajustes aceptados. Un error técnico no se elimina marcando cobertura.

Estado inicial, **antes de tus decisiones V28.4**:

| Concepto | Cantidad |
|---|---:|
| Imágenes con propuestas/grupos | 44 |
| Propuestas/grupos pendientes | 92 |
| Decisiones humanas nuevas creadas por el agente | 0 |
| Imágenes nuevas con cobertura confirmada V28.4 | 0 |
| Candidatas heredadas de V28.3 | 28 |
| Positivos de esas 28 candidatas | 113 |
| Aceptaciones formales V26 | 0 |

Distribución de los 113 positivos heredados: plástico 55, papel 19, cartón 16, vidrio 8, metal 6 y orgánico 9. No se suman los 41 positivos propuestos por IA a la colección revisada antes de tus decisiones y la confirmación de cobertura. Todavía no se cumplen los mínimos exploratorios previos: 40 imágenes, ocho ejemplos adecuados por material y ocho regiones alimentarias, sin cambiar las cuotas oficiales V26.

El QA RLE comprueba máscaras no vacías, dimensiones, etiquetas, caja/área derivadas, duplicados, solapes, fragmentación y ocupación de 48×48. Excluye padding, ignore y mezclas de material para contar regiones. Usa RLE nativo reescalado por nearest-neighbor a letterbox384; este rasterizador versionado **puede diferir del antiguo rasterizado directo de polígonos**. En las 28 heredadas se obtienen regiones48: plástico 50, papel 17, cartón 15, vidrio 6, metal 6 y orgánico 9; tres son alimentarias HPI. No se afirma identidad de tokens con los resultados históricos ni se modifican sus métricas.

**Exportar COCO de imágenes utilizables** incluye sólo candidatas heredadas vigentes y nuevas imágenes cuya cobertura fue confirmada sin pendientes ni errores. **Exportar estado con pendientes** es una descarga de diagnóstico bloqueada, no GT aprobado: las propuestas sin decisión siguen separadas en el catálogo y no se incorporan como aceptadas por haberlas mostrado. La salida conserva RLE COCO sin comprimir en orden Fortran, incluidos huecos y componentes; `bbox` y `area` se derivan de la máscara. Las regiones ignoradas conservan su máscara, categoría y `ignore=1`; no se transforman en fondo. Los descartes se mantienen en el diario aunque no sean instancias activas de COCO.

Cada exportación crea otra carpeta con COCO y manifiesto. Conservar una máscara no significa aprobarla: los campos de revisión individual, base humana previa, cobertura, ignore y aceptación V26 son distintos. La salida sigue marcada **NO_ENTRENAR_AUXILIAR_USUARIO_UNICO**. Ningún adaptador histórico que sólo lea polígonos queda habilitado automáticamente para este contrato RLE.

## Verificación e integridad

La suite conserva las 371 pruebas anteriores y añade 38 pruebas del flujo: **409 pruebas unitarias/HTTP**. Comprueban grupos dependientes, cobertura separada, advertencias, personas y hashes, RLE inválido, huecos, ignore, edición repetida de objetos añadidos, compatibilidad de origen, conflictos, reinicio e idempotencia, caché, snapshots y exportaciones inmutables. Los recibos versionados están en `resultados/v28_4_revision_IA/pruebas/`.

Las pruebas Edge usan únicamente fixtures descartables para escribir decisiones. Se prueban 13 escenarios: aceptación múltiple/cobertura/descarga COCO, pincel y material, HTTP400 recuperable, IndexedDB, copia offline, respuesta perdida, doble clic, dos pestañas, SAM simulado a demanda, rechazo, revisión posterior, sesión caducada y navegación real de sólo lectura. El SAM de las pruebas es explícitamente simulado; **no se volvió a segmentar la colección real**. En las fotografías reales no se envían POST de decisiones ni se marcan casillas de aprobación.

El cierre verifica respaldo nuevo e históricos, 197 eventos originales, cero eventos nuevos simulados, fuentes V28.3, exportación real inicial y huellas de código/evidencias. Los recibos incluyen las limitaciones: aprobación humana individual no demuestra exhaustividad, calidad taxonómica o independencia física ni constituye aceptación V26. Se conserva el checkpoint original de época 19; test y generalización siguen reservados.

## Experimento posterior, todavía deshabilitado

Después de tus confirmaciones se deberá congelar otra colección auxiliar y fijar su manifiesto, agrupaciones, exposición, presupuesto y evaluación antes de autorizar un entrenamiento. No se reutilizan automáticamente 300 ni 1.452 actualizaciones.

La comparación propuesta conservará **los mismos lotes históricos completos en A y B**, con idéntico orden, aumentos y número de exposiciones históricas. B incorporaría los auxiliares en micro-lotes adicionales/acumulación, con proporción, normalización y cadencia fijadas antes de medir; se documentará su coste adicional. No se repetirá por defecto la sustitución de dos de ocho imágenes históricas que redujo la exposición histórica de B en V28.2. El peso relativo y presupuesto quedan pendientes de la colección realmente revisada, sin nueva pérdida ni cambio arquitectónico en esta fase.

Se necesitarán datos no utilizados para ajustar pesos y grupos separados para evaluar generalización. Si no existe esa colección independiente, las conclusiones del siguiente piloto se limitarán a aprendizaje/ajuste y retención histórica. No se abrirán test o generalización originales, ni se adoptará automáticamente un checkpoint.
