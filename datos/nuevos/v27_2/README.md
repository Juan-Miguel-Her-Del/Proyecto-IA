# Anotación auxiliar V27.2

Piloto activo: **72 imágenes, 256 propuestas pendientes, 0 aceptadas V26** al preparar la entrega.

Abrir [scripts/ABRIR_ANOTADOR_V27_2.cmd](../../../scripts/ABRIR_ANOTADOR_V27_2.cmd) o consultar [las instrucciones V27.2](../../../documentacion/27_2_datasets_y_anotacion_asistida.md).

- `proyecto/seleccion_72/PROYECTO.json`: base inmutable del piloto activo.
- `proyecto/seleccion_72/propuestas_ia/`: máscaras SAM, indicaciones y procedencia; no revisiones humanas.
- `proyecto/seleccion_72/revisiones/`: se crea cuando el usuario guarda; diarios por adición, no borrar ni sobrescribir.
- `proyecto/PROYECTO.json`: preparación inicial de 80 fotos, conservada. DWSD está excluido del piloto activo.
- `externos/`: fuentes originales, licencias, muestras y contactos privados; DWSD queda en cuarentena.
- `modelos/`: SAM 2 externo, separado de los checkpoints del detector.

La exportación COCO sigue pendiente de aceptación formal. Los metadatos físicos desconocidos son `null`; los IDs auxiliares no acreditan identidad física. No entrenar, no usar test ni desbloquear generalización. No publicar fotos de terceros o las correcciones privadas en Git.
