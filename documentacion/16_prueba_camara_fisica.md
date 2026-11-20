# Prueba de cámara física

**Captura física verificada: Integrated Camera, 12 frames, 640×480, argmax y cinco vistas.** La prueba funcional no establece precisión ni fiabilidad de detección.

## Ejecución y acceso al dispositivo

El primer intento dentro del aislamiento no pudo abrir DirectShow. La consulta de dispositivos también recibió «Acceso denegado». Tras la autorización para ejecutar fuera del aislamiento, Windows enumeró Integrated Camera con estado OK y el índice cero se abrió correctamente. No se necesitó cambiar umbrales, backend ni pesos. Se conservan ambos intentos.

El intento exitoso terminó con código 0 y releyó 12 frames del MP4 exportado. El proceso de captura finalizó y la implementación libera VideoCapture/VideoWriter en finally. No se abrió ventana ni se capturó audio.

Mediana de inferencia: 48.1 ms; p95: 238.5 ms; procesamiento: 11.6 FPS. Solo son doce frames y hay coste de arranque: no equivalen a una prueba sostenida de webcam.

El MP4 y JSONL incluyen la advertencia experimental, puntuaciones no calibradas y recuentos por frame, sin seguimiento de identidades. Los frames capturados no se añadieron a los datasets y no tienen anotaciones para calcular mAP. No se incrusta ninguna foto de cámara en los informes.

## Evidencia local

- resultados/prueba_camara_fisica_01.json: intento aislado fallido.
- resultados/dispositivos_camara_fisica_fuera_aislamiento.json: enumeración del dispositivo físico.
- resultados/prueba_camara_fisica_02_fuera_aislamiento.json: verificación física y hashes.
- resultados/prueba_camara_fisica_02_fuera_aislamiento/: MP4 anotado, frames.jsonl y resumen.

## Uso posterior

```powershell
$pythonEco = Join-Path $env:USERPROFILE '.venvs\EcoWasteFormer\Scripts\python.exe'
& $pythonEco scripts/detectar_video.py --camara 0 --salida resultados/camara_sesion_nueva --mostrar
& $pythonEco scripts/probar_camara_fisica.py --nombre prueba_camara_fisica_nueva --max-frames 12 --timeout 30
```

La salida debe ser una carpeta nueva. La prueba física requiere acceso al dispositivo; en este entorno el aislamiento lo bloquea, mientras que la ejecución autorizada fuera de él funciona. El script de prueba limita frames y tiempo, registra fallos y nunca los presenta como una cámara validada.

## Prioridades actualizadas

| Área | Estado | Siguiente paso |
|---|---|---|
| Clasificador | 85,74 % accuracy val; conservado | Validación independiente |
| Detector | 2,91 % mAP@50 val; calidad insuficiente | Revisión de los casos y preparación de supervisión regional de material |
| Postprocesamiento | Argmax y cinco vistas | Mantener |
| Pérdida de fondo | Ablación terminada | No repetir por ahora |
| MP4 | Implementado y probado técnicamente | Video independiente con anotaciones |
| Webcam | Prueba física breve completada | Prueba sostenida y escenas variadas, con latencia y calidad |
| Asistente | Integrado; comprensión limitada | Ampliar y evaluar corpus |

La cámara física deja de estar pendiente como prueba de apertura/captura. Permanecen pendientes la validación prolongada y la evaluación de calidad sobre videos independientes. El detector sigue sin ser fiable.
