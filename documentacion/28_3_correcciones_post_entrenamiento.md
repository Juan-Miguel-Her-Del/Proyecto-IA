# V28.3 — Correcciones prioritarias después del piloto

**El editor está preparado para que Juan Miguel haga las correcciones visuales. No se ha ejecutado otro entrenamiento ni se ha cambiado ninguna anotación humana.** Se conserva el detector original de época19, los checkpoints A/B de V28.2 y todos los resultados anteriores.

## Empezar

Abrir **[Editor V28.3](http://127.0.0.1:8768/?imagen=hpi%3A0)**. Usa el mismo proyecto activo `datos/nuevos/v27_2/proyecto/seleccion_72/`; no crea otras72 tareas ni reconstruye propuestas. El identificador `juan_miguel` se recupera de los eventos existentes. SAM2 Tiny sigue funcionando en **CPU**, bajo el entorno `EcoWasteAnotacion`; no se carga hasta solicitarlo.

Si el servicio no está activo, abrir [ABRIR_ANOTADOR_V28_3.cmd](../scripts/ABRIR_ANOTADOR_V28_3.cmd). El editor anterior de8767 permanece intacto. Para continuar, usar8768; evitar editar la misma imagen en dos pestañas/versiones simultáneamente. Los conflictos se rechazan conservando ambas copias. Este cambio no recupera memoria que ya se hubiera perdido en sesiones anteriores.

1. Elegir una fila de **Correcciones prioritarias** y pulsar **Abrir imagen y objeto seleccionados**. También funcionan la selección directa y las flechas de pendientes.
2. Leer el motivo, el objeto/material actual, la posible omisión y el estado del guardado. Los contornos actuales se muestran sobre la fotografía. Para observaciones sin objeto específico se abre la imagen completa.
3. Corregir únicamente lo necesario: refinar con caja/clics SAM, mover vértices, cambiar material, añadir el residuo omitido o conservar una región ambigua con explicación. SAM propone geometría y no decide el material.
4. Guardar. Mantener o confirmar la casilla de cobertura completa solo cuando corresponda. Sin ella se registra un borrador. Un guardado válido no convierte automáticamente una incidencia visual en resuelta.
5. Escribir qué se corrigió o decidió y registrar el estado de **esa incidencia**: pendiente, corregida o requiere decisión. Para cerrar se exige una revisión completa nueva del mismo usuario, vinculada por huella, y ausencia de errores técnicos.
6. Pulsar **Actualizar auditoría incremental**. Consultar los mínimos exploratorios y los bloqueos restantes de cada imagen. Las cifras antiguas se identifican como desactualizadas si cambia un guardado, una resolución o el código QA.

No es necesario volver a recorrer todas las72. Las28 candidatas anteriores se conservan mientras sus datos sigan siendo coherentes. Las pistas opcionales de V28.2 no bloquean por sí mismas una imagen: un fallo del modelo no demuestra que su anotación sea incorrecta.

## Orden de corrección

| Prioridad | Trabajo | Alcance |
|---|---|---|
| 1 | Cobertura alimentaria HPI | **0,132,155,78,96,192**, en ese orden; añadir los restos realmente omitidos y resolver fragmentos/oclusiones señalados |
| 2 | Escenas WaRP con vidrio excluidas | POSAD_1_13-Sep_02-24-53; POSAD_1_13-Sep_03-20-46; Robo_25-Mar_14-39-46; prepared_data_all_MGS_20-Oct_01-03-59 |
| 3 | Nueve envases multicapa | Comprobar composición de milk-cardboard/juice-cardboard; confirmar material con justificación o marcar compuesto/ambiguo y su región ignore |
| 4 | Papel omitido ZeroWaste | 1001,1076,1709,2670,2688,2706,2709,2710,2843,331,588,896 |
| 5 | Guardados pendientes | ZeroWaste1823,1896,219,41 sin revisión; WaRP SPB_25-Mar_11-23-03 como borrador |
| 6 | Otras observaciones | Diferencias de relleno, fragmentos, solapes y descartes que todavía requieren decisión |
| 7 | Pistas opcionales V28.2 | Vidrio/metal sin AP auxiliar y separación alimentaria; observaciones de diagnóstico, sin marcar etiquetas como erróneas |

Una de las nueve incidencias multicapa está dentro de una escena de vidrio y aparece primero en prioridad2; las otras ocho aparecen en3. No se duplica el objeto para inflar cuotas. Las escenas ZeroWaste2706/2709/2710 siguen relacionadas: no tratarlas como objetos físicos independientes demostrados ni separarlas arbitrariamente en train/evaluación.

La escena **WaRP prepared_data_all_MGS_20-Oct_01-03-59** contiene, según el guardado actual, dos regiones de vidrio y tres de metal potencialmente utilizables48×48. Sus dos incidencias deben resolverse antes de acreditarlas; incluye una pieza orgánica/espuma dudosa y fragmentación. Puede ayudar a los déficits sin redibujar todo. Las escenas Monitoring_photo1_04-Mar_03-24-33 y Monitoring_photo1_04-Mar_03-52-24 tienen, cada una, una región de metal potencial y una incidencia multicapa. Esto es un inventario de posibilidades, **no datos nuevos aceptados**.

No dividir residuos artificialmente para llegar a ocho. No sustituir “no se distingue” por fondo si sigue existiendo residuo visible: conservar la ambigüedad y una región ignore cuando corresponda. Los materiales físicos no se pueden confirmar solo porque la fuente o SAM los sugiera. Las discrepancias de relleno de contornos requieren decisión visual; el agente no cambió automáticamente los anillos ni los huecos.

## Estado de partida real

Antes de que el usuario haga nuevas correcciones:

| Material | Polígonos válidos en candidatas | Regiones utilizables48×48 | Faltan hasta ocho regiones |
|---|---:|---:|---:|
| Plástico | 55 | 51 | 0 |
| Papel | 19 | 18 | 0 |
| Cartón | 16 | 15 | 0 |
| Vidrio | 8 | **7** | **1** |
| Metal | 6 | **6** | **2** |
| Orgánico | 9 | 9 | 0 por material |

Hay **28 imágenes candidatas y113 polígonos positivos válidos**, además de tres compuestos ignorados en ese subconjunto. Faltan12 imágenes para40. Los nueve orgánicos incluyen solo **tres regiones alimentarias HPI** utilizables; faltan cinco para ocho alimentarias. Vegetación WSC no se acredita como diversidad alimentaria.

Las regiones48×48 requieren al menos dos celdas conectadas con soporte no ambiguo, excluyendo padding y mezcla de materiales según QA V28. Una geometría válida no garantiza resolución suficiente. Por eso ocho polígonos de vidrio no se presentan como ocho regiones adecuadas. Se conservan ambos conteos, sin cambiar la pérdida, la resolución o los targets históricos del detector.

Estos mínimos son exploratorios: no sustituyen las cuotas oficiales V26 de1.550 instancias train,600 de generalización y≥300 orgánicos utilizables. La aceptación formal V26 sigue en **cero**. No se inventan identidades físicas, sesiones ni revisores independientes.

## Guardados y borradores

La auditoría histórica registró **256 respuestas400 y134 respuestas200**. Los logs anteriores contienen método/ruta/estado y contexto cercano, pero no cuerpo de la solicitud ni excepción del servidor. No se puede determinar retrospectivamente la causa exacta de cada400 ni deducir que256 eventos humanos se perdieran. Las134 revisiones válidas están conservadas con sus hashes.

Se investigaron y reprodujeron los mecanismos de rechazo existentes:

| Mecanismo | Respuesta nueva |
|---|---|
| Sin contorno o material, polígono cruzado, hueco inválido | Identifica número, ID y material del objeto y el motivo de validación; no relaja geometría |
| Ambigüedad/descarte sin motivo | Solicita justificación; no transforma el objeto en fondo automáticamente |
| Otra pestaña ya guardó | Rechaza versión obsoleta, conserva el borrador y permite comparar con la versión del servidor |
| Reinicio/token de vista o CSRF caducado | Permite recargar contexto sin reemplazar contornos locales; no rebasea sobre una revisión ajena |
| Respuesta perdida después del commit | Recibo persistente por solicitud: reintentar el mismo contenido confirma el evento existente sin duplicarlo |
| Mismo ID de solicitud con contenido distinto | Rechaza la reutilización; evita mezclar operaciones |
| Fallo de red antes del commit | No declara éxito; conserva el borrador y deja disponible la descarga sin red |

Cada nuevo rechazo de guardado genera un ID de error y un diagnóstico privado con la excepción exacta. Cuando la solicitud está autorizada y pertenece al proyecto, el servidor conserva antes de validar una copia cruda —incluso de geometría inválida— en `datos/nuevos/v28_3/gestion/.../borradores/`. No se transforma en revisión ni en GT. Rechazos de seguridad no autorizan escribir revisiones o copias arbitrarias.

El navegador conserva por proyecto/imagen/modalidad el borrador y hasta ocho estados locales previos en IndexedDB, incluidos vértices, objetos nuevos, indicaciones SAM y polígonos todavía sin cerrar. Las indicaciones de un objeto se conservan al seleccionar otro. La imagen activa queda en la URL y se recupera al recargar. La casilla de cobertura refleja el último guardado real; ya no se reinicia silenciosamente a falso.

**Descargar copia de emergencia** funciona sin servidor/red y conserva el contexto de revisión. **Recuperar borrador local**, importar una copia o recuperar una copia del servidor requieren acción del usuario. Se comprueba proyecto, imagen, modalidad y revisión base; una copia antigua no sobrescribe una revisión nueva. Si hay conflicto, se conserva el borrador y se puede descargar la versión guardada para comparar. No se fusionan contornos automáticamente.

Durante las pruebas se detectó y corrigió que una recarga abría la primera imagen en lugar de la imagen cuyo borrador se estaba trabajando. También se evitó que escribir una nota de incidencia ensuciara las anotaciones y bloqueara su resolución. Estas incidencias de implementación no se atribuyen a los256 rechazos históricos.

No se garantiza persistencia si se borra el almacenamiento del navegador o se cierran deliberadamente sesiones sin guardar ante un fallo de almacenamiento. El editor comunica los fallos de persistencia y ofrece copia del servidor o descarga local. Recuperar un borrador nunca significa que el usuario haya completado una revisión.

## Auditoría incremental y versiones

El botón del editor ejecuta QA en CPU en segundo plano. También se puede abrir [AUDITAR_CORRECCIONES_V28_3.cmd](../scripts/AUDITAR_CORRECCIONES_V28_3.cmd), o usar desde el repositorio:

```powershell
& "$env:USERPROFILE\.venvs\EcoWasteAnotacion\Scripts\python.exe" scripts/auditar_v28_3.py
```

Cada ejecución publica una carpeta nueva bajo `resultados/v28_3_correcciones/auditorias/` con:

- `imagenes.json`: guardado vigente, errores técnicos, incidencias abiertas, QA por objeto y candidatura exploratoria.
- `incidencias.json`: estado y fundamento de cada observación, vinculados a la revisión o al ámbito del objeto sin cambios.
- `auditoria.json`: conteos, déficits, caché utilizada, hashes y enlace SHA a la auditoría anterior.

El suplemento `plan_prioridades.json` identifica escenas ya anotadas cuyo desbloqueo podría aportar materiales deficitarios. Se genera con `scripts/informar_v28_3.py` y conserva explícitamente que el potencial todavía no se acredita. No se sobrescribe otro plan.

La caché se identifica por proyecto, imagen, última revisión y huellas del código QA y sus dependencias. Su contenido se valida con digest. Si cambia una revisión se recalcula esa imagen; una resolución de incidencia puede actualizar la candidatura reutilizando la geometría ya comprobada. Una segunda ejecución con los mismos guardados reutilizó **72/72** entradas y recalculó **cero**. Los cambios de código durante la preparación generaron otras versiones, que permanecen conservadas.

Las resoluciones son un diario separado, encadenado y ligado a la huella humana. Una incidencia de objeto sigue vigente si solo cambia otro objeto y su ámbito permanece idéntico; se reabre si cambia su material/geometría/estado o la cobertura relevante. Una observación de imagen completa se reabre cuando cambia ese ámbito. Los avisos nuevos se convierten en incidencias navegables. Los errores técnicos actuales siempre bloquean la candidatura, aunque se pulse “corregida”.

No se exportó ni congeló otra colección COCO como lista para entrenamiento. Una candidatura indica que no quedan incidencias identificadas según estos controles y las decisiones de un usuario; **no certifica exhaustividad visual, verdad del material, independencia física ni aceptación V26**. La auditoría no puede descubrir por sí sola todos los residuos omitidos.

## Pruebas e integridad

- **351 pruebas unitarias/HTTP aprobadas:**322 anteriores y29 nuevas. Incluyen idempotencia, reinicio, conflictos, contexto preciso, copias inválidas, aislamiento de proyectos, rutas, firmas de incidencias, caché alterada y versiones de auditoría.
- **12 casos de navegador aprobados** con Edge sin GPU: navegación por pendientes, HTTP400 con objeto/causa, emergencia, recuperación tras recargar, respuesta perdida tras commit, bloqueo inmediato de guardados simultáneos, resolución explícita, polígono sin terminar, token caducado, descarga offline y dos pestañas en conflicto. Las escrituras se hicieron únicamente en fixtures descartables.
- Las72 fotografías reales se abrieron solo para navegación/lectura en la prueba; no hubo POST de anotación sobre el proyecto real.
- Respaldo nuevo íntegro de **731 archivos**, con **134 eventos humanos** y cadenas válidas. Los anteriores permanecen intactos. No se registró ninguna resolución humana real por parte del agente.
- Código anterior, datasets históricos, resultados y **113 checkpoints** conservados por SHA. El código/UI V27.2 también queda intacto: V28.3 se entrega en rutas nuevas.

Evidencia en [resultados V28.3](../resultados/v28_3_correcciones/), [respaldo verificado](../resultados/v28_3_correcciones/respaldo.json) y [certificación final del guardado](../resultados/v28_3_correcciones/verificacion_guardado_final.json). El [manifiesto vigente](../resultados/v28_3_correcciones/manifiesto_integridad_guardado_final.json) cubre código, documentación, pruebas, auditorías y datos auxiliares de gestión. Se conserva también el primer cierre: el suplemento registra únicamente el bloqueo de guardado antes de esperar IndexedDB y su prueba adicional; no cambia backend, QA, datos o revisiones. Fotos, capturas, copias y borradores privados están ignorados por Git; no se redistribuyen fotografías.

El servicio preparado escucha únicamente en `127.0.0.1:8768`. El lanzador de fondo es `scripts/iniciar_editor_v28_3.py`: no modifica políticas de ejecución del sistema. No se detuvo el editor anterior ni se ejecutó entrenamiento, inferencia de evaluación o selección de checkpoints.

## Antes del siguiente entrenamiento

Primero corregir las prioridades y volver a auditar. Debe verificarse la colección resultante, respaldar sus nuevos eventos y congelar otro manifiesto antes de diseñar y autorizar un experimento específico. Alcanzar40/ocho/ocho no activa nada automáticamente.

El diseño posterior debe atender a retención del aprendizaje histórico y a una evaluación de datos que no hayan ajustado los pesos, manteniendo juntos los grupos relacionados. Las28 imágenes ya entrenadas por B no pueden demostrar generalización. Los datos públicos con identidad física desconocida no garantizan por sí solos independencia; cuando no pueda demostrarse, deberá limitarse el alcance de las conclusiones o recopilar una colección nueva con grupos/sesiones conocidos. Test y generalización originales permanecen reservados.

**Pendiente del usuario: realizar las correcciones visuales.** Hasta entonces las cifras siguen siendo28 candidatas,113 positivos, tres regiones alimentarias HPI utilizables y cero aceptaciones V26.

Actualización posterior: el usuario autorizó al asistente a corregir las imágenes. Los ajustes IA, separados de los diarios humanos y sin entrenamiento, se documentan en [28_3_ajustes_asistente_IA.md](28_3_ajustes_asistente_IA.md). No se convierten automáticamente en revisiones humanas, nuevas candidaturas verificadas ni aceptación V26.
