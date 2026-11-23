# V28.3: ajustes de anotaciones realizados por el asistente IA

El usuario autorizó que el asistente terminara de ajustar las imágenes. Esta autorización permite crear correcciones identificadas como IA; no convierte al asistente en `juan_miguel`, revisor independiente o adjudicador V26. No se ha entrenado ni evaluado ningún detector durante estos ajustes.

La publicación vigente es `datos/nuevos/v28_3/gestion/ajustes_IA_20261123T045058Z_7d39442c/correcciones_IA_3/`. Las versiones 1 y 2 se conservan como publicaciones intermedias. La evidencia reproducible está en `resultados/v28_3_correcciones/ajustes_asistente/20261123T045058Z_7d39442c/`.

## Resultado y alcance

Se inspeccionaron las **44 imágenes con incidencias bloqueantes**. La copia IA contiene ajustes en las 44, **41 regiones positivas añadidas** y **97 objetos con ajustes**, incluidos los añadidos. Esto no significa que las 44 escenas tengan cobertura exhaustiva ni que sus anotaciones hayan recibido otra revisión humana.

Las 41 regiones añadidas son **23 de papel y 18 orgánicas**. Además se crearon **24 regiones ignore**, veinte por intersecciones y cuatro por componentes secundarios dudosos; no son nuevas instancias positivas. En las 44 fotografías de esta copia quedan los siguientes positivos provisionales, sumando anotaciones de base y ajustes IA:

| Material | Regiones positivas provisionales |
|---|---:|
| Plástico | 90 |
| Papel | 49 |
| Cartón | 77 |
| Vidrio | 5 |
| Metal | 11 |
| Orgánico | 102 |
| Total | 334 |

Estas cifras no son las del proyecto completo de 72 imágenes ni una declaración de aceptación. El COCO contiene 373 máscaras: 334 positivas provisionales y 39 ignore, incluidas regiones preexistentes y grupos reclasificados. No se suman a cuotas V26 ni se interpreta que más regiones equivalgan a más objetos físicos independientes.

- Se añadieron restos alimentarios y hojas de papel omitidos, tras comprobar visualmente las propuestas SAM. Las máscaras que unían objetos distintos, incluían plato/fondo o sólo cubrían una parte del objeto se descartaron o refinaron. Se conservan indicaciones, resultados originales y decisiones de selección.
- En `hpi:23` se separaron cuatro rodajas con límites visibles. El resto del grupo queda como ignore; no se inventaron superficies ocultas ni identidades físicas.
- En `zerowaste:219` SAM devolvía sólo parte de una hoja. Su contorno fue trazado visualmente por IA y registrado como tal, conservando la propuesta SAM original. No se presenta ese trazado como dibujo humano ni salida íntegra de SAM.
- Se conservaron como ignore los nueve envases tipo brick de composición no demostrable, la pieza dudosa alimento/espuma, una región vegetal de pertenencia incierta a la taxonomía y una máscara sobre plástico transparente cuya identidad no se podía separar del fondo. «Ambiguo» no afirma que se haya probado una composición multicapa.
- Los contactos de identidad incierta se trasladaron localmente a ignore, conservando la unión de las regiones. En `wsc:63`, la zapatilla visible delante del plástico conserva su máscara; sólo se retira su intersección del objeto trasero.
- Los componentes secundarios dudosos de cuatro máscaras inspeccionadas a mayor escala pasan a ignore. No se los acredita como nuevos objetos ni se dibujan puentes entre partes separadas.
- Tres discrepancias entre relleno por paridad del editor y unión del exportador quedan representadas explícitamente mediante RLE, conservando el relleno mostrado y los polígonos originales como procedencia.

Se utilizó SAM 2.1 Tiny **en CPU**, sin entrenamiento. Sus puntuaciones no son confianza del material. El material de los objetos añadidos se propuso por inspección visual del asistente, nunca mediante etiquetas globales convertidas en cajas GT.

## Cómo ver los cambios

El visor local de sólo lectura está preparado en **http://127.0.0.1:8770/**. El editor de siempre, **http://127.0.0.1:8768/**, contiene un enlace al visor. No se fuerza una recarga del editor ni se descartan borradores abiertos.

El visor muestra cada fotografía antes y después, el número de regiones añadidas, observaciones pendientes y sus anotaciones JSON. Colores: plástico cian, papel amarillo, cartón naranja, vidrio verde agua, metal violeta, orgánico verde y ambiguo/ignore rojo. Las cuatro fotografías sin guardado humano se identifican como propuestas originales, no como revisiones del usuario.

No es necesario volver a dibujar las regiones añadidas por IA para examinarlas. Conviene revisar primero los sectores rojos y las observaciones. El visor no tiene botones que simulen revisiones ni aceptación. El contador del editor sigue correspondiendo a los guardados humanos originales.

Para volver a iniciar el visor, desde la raíz del repositorio:

```powershell
& "$env:USERPROFILE\.venvs\EcoWasteAnotacion\Scripts\python.exe" scripts/ver_ajustes_ia_v28_3.py --manifiesto resultados/v28_3_correcciones/ajustes_asistente/20261123T045058Z_7d39442c/correcciones_IA_3/manifiesto_integridad.json
```

## Qué sigue sin certificarse

Las pilas densas de ZeroWaste y WaRP pueden contener residuos adicionales sin delimitar. En HPI persisten alimentos pequeños y mezclas cuya granularidad de instancia no se puede demostrar con estos ajustes. Las partes rojas de `hpi:23` tampoco se cuentan como nuevas instancias. No se asegura exhaustividad visual porque haya desaparecido un solape técnico o porque una máscara sea binariamente válida.

La copia COCO es **auxiliar, corregida por IA, pendiente de confirmación y bloqueada para entrenamiento**. No se acreditan cuotas V26, objetos físicos, sesiones o independencia de escenas desconocidas. Conserva procedencias, licencias y agrupaciones originales, incluidos los fotogramas relacionados. No se redistribuyen las fotografías en Git.

El archivo `anotaciones_coco_IA_NO_ENTRENAR.json` usa **RLE COCO canónico por instancia**, con área y caja derivadas de la máscara. Los polígonos históricos quedan como procedencia; en objetos nuevos pueden estar vacíos. No debe cargarse mediante adaptadores antiguos que sólo consulten polígonos. La compatibilidad con una futura revisión/exportación y un entrenamiento se debe comprobar explícitamente, sin alterar el contrato V26 ni los bloqueos de V28. No se ha habilitado esa integración ni otro entrenamiento.

## Protección y comprobaciones

Antes de corregir se respaldaron **904 archivos actuales**, incluidos los **197 eventos humanos** existentes. El respaldo histórico de 731 archivos y los eventos anteriores permanecen intactos. No se creó ningún evento de revisión humana o resolución de incidencia en nombre del usuario.

La publicación comprueba por SHA los **15.140 archivos históricos protegidos**, incluidos los checkpoints. La verificación adicional compara máscaras RLE, geometría, COCO, imágenes, historial y referencia original. Comprueba que los píxeles de las regiones originales permanecen representados en la copia, como positivos o ignore: no se confunde preservación de información con conservación de todas las etiquetas como positivas.

**371 pruebas unitarias/HTTP aprobadas**, incluidas las 355 previas y 16 pruebas del contrato IA: RLE exacto con huecos, cajas derivadas, máscara vacía, paridad, preservación de uniones, oclusión, fragmentos dudosos y separación explícita de IA frente a revisión humana. `control_calidad_real_v2.json` registra la comprobación sobre las fotografías y anotaciones reales; no certifica exhaustividad ni materiales ambiguos. Conserva el primer control y precisa sus nombres de campos: las 24 regiones ignore nuevas incluyen veinte solapes y cuatro componentes; las regiones procedentes de SAM incluyen correcciones IA y un trazado visual. La prueba del visor incluye **nueve comprobaciones aprobadas**, sin peticiones de escritura: 44 comparaciones, 88 vistas, cuatro casos sin guardado humano correctamente identificados, máscaras visibles, descarga COCO y enlace desde el editor.

Los manifiestos y recibos de cierre anteriores se conservan. Esta fase agrega evidencia propia; no sustituye resultados anteriores. El checkpoint seleccionado de época 19 mantiene su SHA `eb8fd00ee536ead21e3a2a0ecd5b06baad04acb2e437b22e0eac3686a7bf36b9`. Test y generalización permanecen reservados, y el modelo predeterminado no cambia.

La decisión sigue siendo **no entrenar todavía**. Estos ajustes reducen el trabajo visual pendiente; no prueban mejora del detector ni constituyen aceptación V26.
