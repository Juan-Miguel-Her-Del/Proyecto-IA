# EcoWasteFormer

Proyecto académico de detección y clasificación de residuos con modelos propios entrenados desde cero.

## Estado

Fase actual: los tres modelos Transformer están implementados y entrenados; RealWaste y TACO están preparados y auditados. MP4 y webcam tienen inferencia experimental, exportación y conexión al asistente. **El detector mantiene calidad insuficiente; la cámara física y los videos independientes siguen pendientes de validación.** Los documentos originales están en `../documentos/`.

**Detector:** se conserva `detector_mapas_v6_mosaico`, checkpoint de época 19, argmax y cinco vistas; mAP@50 val **2,91 %**, todavía insuficiente. Pasaron 109 pruebas. Las ramas nuevas no sustituyen al predeterminado; ver [ablación](documentacion/13_ablacion_del_fondo.md).

**Ablación de fondo y video:** completadas dos ramas de 1.452 actualizaciones desde el mismo estado de época 19; fondo=1 no mejora suficientemente y se conserva el detector actual. MP4/webcam implementados con advertencias y 109 pruebas aprobadas. Ver [ablación](documentacion/13_ablacion_del_fondo.md) y [uso de video/cámara](documentacion/14_video_y_webcam_experimental.md).

**Experimento de propuestas con pesos congelados:** retirar argmax empeoró el detector de cinco vistas: mAP@50 2,91 % → 1,15 %, recall 20,50 % → 16,62 % y FP 2.374 → 13.055 a puntuación ≥0,05. Se conserva la selección actual. Pasaron 98 pruebas; ver [comparación y diagnóstico](documentacion/12_propuestas_por_material_sin_argmax.md).

**Clasificador visual:** 629.382 parámetros, desde cero. Seleccionado `clasificador_v3_regularizado`: **85.74% de exactitud y F1 macro 0.857** en validación, frente a 70,66 % y 0,709 previos. Test reservado. Ver [entrenamientos y límites](documentacion/11_mejora_clasificador_y_centros.md).

Ver el estado, los conteos, resultados y límites en [documentacion/02_datasets_y_asistente.md](documentacion/02_datasets_y_asistente.md). El asistente es un prototipo con comprensión limitada: obtuvo 58,3 % de aciertos sobre 36 preguntas iniciales de prueba y aún no supera la referencia simple. La detección tiene cobertura insuficiente de orgánicos en TACO.

**Entorno verificado el 19/11/2026:** Python 3.12.6, PyTorch 2.14.1+cu126 y RTX 4060 Laptop. Pasaron comparación numérica CPU/GPU, atención con retropropagación, comprobación de gradientes y carga de dependencias. Evidencia en [resultados/entorno.json](resultados/entorno.json); versiones exactas en `requirements-lock.txt`.

La decisión actualizada y la matriz de aceptación están en [documentacion/01_decisiones_y_requisitos.md](documentacion/01_decisiones_y_requisitos.md). Este documento sustituye la propuesta del PDF de utilizar únicamente reglas para el asistente.

El usuario decidió proceder con este diseño sin aprobación adicional del profesor.

## Entorno Windows

El entorno virtual está fuera de OneDrive para evitar sincronizar las dependencias:

`C:\Users\JUAN M\.venvs\EcoWasteFormer`

Desde esta carpeta, en PowerShell:

```powershell
$pythonEco = Join-Path $env:USERPROFILE '.venvs\EcoWasteFormer\Scripts\python.exe'
& $pythonEco scripts/verificar_entorno.py --salida resultados/entorno.json
```

Para reproducir todas las versiones registradas en otro entorno compatible, instalar con `& $pythonEco -m pip install -r requirements-lock.txt` en lugar de los dos archivos de requisitos anteriores.

El diagnóstico realiza una multiplicación y un paso de atención con retropropagación en GPU; no entrena un modelo ni mide la calidad del proyecto. Devuelve código 1 si CUDA o alguna prueba numérica falla.

Para reconstruir el entorno se usa `scripts/preparar_entorno.ps1`. No es necesario activar el entorno ni cambiar la política de ejecución de Windows. Seleccionar el ejecutable anterior como intérprete en VS Code.

Si Windows impide ejecutar archivos `.ps1`, usar directamente estos comandos en PowerShell, desde esta carpeta:

```powershell
python -m venv "$env:USERPROFILE\.venvs\EcoWasteFormer"
$pythonEco = Join-Path $env:USERPROFILE '.venvs\EcoWasteFormer\Scripts\python.exe'
& $pythonEco -m pip install -r requirements-cuda.txt
& $pythonEco -m pip install -r requirements.txt
& $pythonEco scripts/verificar_entorno.py --salida resultados/entorno.json
```

## Asistente y datos

El checkpoint propio está en `checkpoints/asistente_v1.pt`. Consulta de ejemplo con detecciones explícitamente sintéticas:

```powershell
& $pythonEco scripts/consultar_asistente.py '¿Dónde pongo el plástico?' --detecciones datos/ejemplos/detecciones_sinteticas.json
```

Los scripts de descarga, preparación y validación son `scripts/descargar_datos.py`, `scripts/preparar_datos.py` y `scripts/validar_datos.py`. Evidencia en `datos/reportes/validacion.json`. Los manifiestos están en `datos/procesados/{clasificacion,deteccion}/{train,val,test}.jsonl`; las imágenes siguen en sus carpetas originales o normalizadas. `ecowaste/datasets.py` carga tensores y conserva las regiones de detección que deben ignorarse.

## Próximos entregables

1. Ampliar el corpus del asistente y los ejemplos reales de detección orgánica; reservar nuevos videos para prueba externa.
2. Mejorar localización y cobertura de detección: se compararon centros por instancia y cinco vistas; el AP continúa siendo insuficiente.
3. Mejorar y validar los modelos con escenas independientes antes de presentar detecciones de video como fiables.
4. Validar la webcam física y videos independientes; medir latencia sostenida y calidad. La integración experimental ya está implementada.

La arquitectura del detector está implementada. Resultado seleccionado: `resultados/detector_mapas_v6_mosaico.json`; clasificador: `resultados/clasificador_v3_regularizado.json`. Se conservan todos los ensayos anteriores. Los del asistente están en `resultados/asistente_v1.json` y `resultados/baseline_asistente.json`.

Para clasificar una imagen con un residuo predominante:

```powershell
& $pythonEco scripts/clasificar_imagen.py 'C:\ruta\mi_residuo.jpg'
```

Esta función clasifica la imagen completa. Para inspeccionar técnicamente el detector experimental:

```powershell
& $pythonEco scripts/detectar_imagen.py 'C:\ruta\mi_foto.jpg' --imagen-salida resultados/detecciones.jpg --json-salida resultados/detecciones.json
```

Las cajas de este piloto contienen numerosos errores; la conexión al asistente es una prueba de integración y no acredita la validez de esas detecciones.


## MP4 y webcam

```powershell
& $pythonEco scripts/detectar_video.py --video 'C:\ruta\mi_video.mp4' --salida resultados/mi_video
& $pythonEco scripts/detectar_video.py --camara 0 --salida resultados/camara_sesion1 --mostrar
```

La carpeta de salida debe ser nueva. Se conserva argmax y cinco vistas; las cajas son experimentales. Ver [salidas, controles y límites](documentacion/14_video_y_webcam_experimental.md).
