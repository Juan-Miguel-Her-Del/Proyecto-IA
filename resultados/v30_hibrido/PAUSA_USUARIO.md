# V30 — pausa solicitada por el usuario

No reanudar hasta que el usuario lo solicite. No hay entrenamiento ampliado iniciado.

## Estado guardado

- Implementados en archivos nuevos: backbone CNN multiescala residual, control Transformer y detector híbrido con cabeza de mapas original y máscara geométrica de padding.
- Preparados entrenador, evaluación, visualización, informe, verificación, demostración condicionada a métricas y configuración V30. Todavía requieren revisión conjunta y certificación antes de producción.
- Pasaron las primeras 22 pruebas de arquitectura en CPU. Se añadieron después tres pruebas de plan/reanudación; falta ejecutarlas y ejecutar la suite histórica completa.
- Smoke CUDA completado: una actualización AdamW real por rama, pérdidas finitas, gradientes no nulos CNN/Transformer/cabeza. Misma huella de entradas A/B. Evidencia: `smoke_cuda.json`. Pesos de smoke descartados.
- No se han congelado protocolo/hashes ni ejecutado C1/C2/R, memorización o entrenamiento ampliado. No existen checkpoints productivos V30.
- Configuración propuesta: A y B desde cero, semilla 3001, 8 históricas + 2 PackWISE + 2 CLICK y 16 globales por paso; misma pérdida original, LR 0,0005, backend V23 completo. Presupuesto fijo 2400 por rama, condicionado a diagnóstico train-only y puerta de 200 pasos. No aumentar por métricas bajas.
- Requisito CNN autorizado por el usuario, pero falta actualizar `documentacion/01_decisiones_y_requisitos.md` después de conservar sus bytes previos. No hay autorización de pesos externos.
- Checkpoint original, archivos anteriores, revisiones humanas y configuración predeterminada intactos. Test no evaluado; generalización permanece bloqueada.
- Al pausar: sesión de smoke terminada (código 0), GPU 0 % y 0 MiB. No se lanzó ningún otro trabajo CUDA.

## Próximo paso cuando el usuario pida continuar

1. Revisar los nuevos archivos y ejecutar pruebas completas; corregir fallos antes de congelar.
2. Congelar el protocolo y respaldar/verificar históricos; actualizar únicamente la autorización CNN en requisitos preservando el original.
3. Comparar C1/C2/R, cuatro pasos reales por rama, rtol=1e-4/atol=1e-5 y restauración exacta; descartar pesos.
4. Memorización descartable de 240 pasos en ocho escenas train con seis clases y escenas multinstancia. Detener si falla el criterio prefijado.
5. Solo tras controles aprobados, entrenar A y B a 200 pasos, comprobar puertas y completar como máximo 2400. Checkpoints nuevos en 0/200/800/1600/2400.
6. Evaluar finales y original separando train histórico/nuevo, desarrollo y retenido externo (solo tres clases; independencia física no demostrada). No usar test ni reajustar protocolo.
7. Generar informe V30, ejemplos prefijados, gráficas y cierre de integridad. Habilitar selección del híbrido únicamente si supera el criterio documentado; no sustituir automáticamente el original.

Scripts nuevos usan publicaciones inmutables: no volver a ejecutar el smoke sobre su salida existente. No iniciar otra hipótesis si falla este experimento.
