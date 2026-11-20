"""Consolida ablación pareada y entrega de MP4/webcam sin cambiar el detector elegido."""
import hashlib
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.ablacion_fondo import resumir_deteccion
from scripts.entrenar_clasificador import escribir_json


def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8'))


def main():
    r=leer('resultados/ablacion_fondo_ep19.json');video=leer('resultados/verificacion_video_real.json')
    imagen=leer('resultados/verificacion_video_imagen.json');pruebas=leer('resultados/pruebas_ablacion_video.json')
    if r['estado']!='ablacion_entrenamiento_completada' or not video['correcto'] or not imagen['correcto'] or not pruebas['correcto']:
        raise ValueError('Faltan verificaciones')
    cinco={f'{b}_{s}':leer(f'resultados/ablacion_fondo_ep19_{b}_{s}_cinco_vistas.json') for b in ['A','B'] for s in ['final','mejor']}
    base=leer('resultados/detector_mapas_v6_auxiliar_ext48.json')['val']
    base5=leer('resultados/detector_mapas_v6_mosaico.json')['val']
    etapas=[('Origen época 19',r['semantica_train_origen'])]
    for b in ['A','B']:
        for s in ['final','mejor']:etapas.append((f'{b} {s}',r['ramas'][b][f'semantica_train_{s}']))
    l=['# Ablación controlada del coeficiente de fondo','',
       '**Decisión: no adoptar fondo=1 como sustituto actual. Se conserva el checkpoint original de época 19, argmax y cinco vistas.**','',
       '## Protocolo ejecutado','',
       f"Dos ramas parten de `{r['checkpoint_origen']}` (SHA-256 `{r['sha256_origen']}`). A mantiene coeficiente tres y B usa uno. Se restauraron y compararon por huella los pesos, todo el estado AdamW, scaler y los generadores CPU, CUDA, DataLoader de detección y muestreo auxiliar. Las cuatro huellas iniciales coinciden con el archivo original en ambas ramas.",'',
       'Se mantienen extractor y cabezales, imágenes, máscaras TACO de polígonos, cobertura 1/64, ponderación por material, Dice y clasificación auxiliar RealWaste (coeficiente 0,5 y lote balanceado 16). Lote de detección ocho, AdamW y LR 0,0005 guardados. El scaler inicial es 2.048 con growth_tracker 299. No se reinicia warmup ni se crea un optimizador sin su estado aprendido.','',
       f"Presupuesto prefijado: {r['configuracion']['epocas_adicionales']} épocas (20–31), {r['actualizaciones_por_rama']} actualizaciones por rama, sin parada temprana. AdamW pasa de 2.299 a 3.751 en todos sus parámetros. Se comprueba cada época que no se omitió un paso. Ambas ramas usan los mismos 11.568 ejemplos de detección y 23.232 ejemplos auxiliares muestreados durante este presupuesto.",'',
       'Las trazas registran IDs/índices de ambas fuentes y parámetros de espejo, brillo y contraste de cada lote. Las huellas acumuladas y las de las doce épocas coinciden entre A y B. Evaluación semántica de train sin aumentos; val conserva argmax, máscara de padding, umbral de propuestas 0,05, componentes de dos celdas y NMS 0,5. No se utiliza la variante de propuestas sin argmax.','',
       'Se registra el checkpoint final al mismo número de pasos y el mejor de las épocas adicionales por **mAP@50 val de una vista**, criterio del entrenamiento original. Ambos mejores ocurrieron en época 24 (605 actualizaciones adicionales). También se guarda el mejor incluyendo el punto común: A conserva época 19 y B elige época 24 por ese criterio de una vista. Las cinco vistas se evalúan por separado sobre esos cuatro checkpoints; no se denomina al ganador de una vista «mejor de cinco vistas».','',
       '## Aprendizaje semántico sobre train','',
       '| Estado | mIoU siete clases | mIoU seis materiales | Material → fondo | Fondo → material |',
       '|---|---:|---:|---:|---:|']
    for n,m in etapas:l.append(f"| {n} | {m['miou_siete_clases']:.2%} | {m['miou_materiales']:.2%} | {m['material_predicho_fondo']} | {m['fondo_predicho_material']} |")
    l+=['','Son celdas de mapas 48×48, no objetos. Se excluyen etiquetas 255; las matrices completas 7×7, confusión binaria y IoU de cada clase se guardan en el JSON. mIoU es la media de IoU de clases con unión no vacía. La comparación usa las mismas máscaras, sin recortes nuevos ni cambios de anotaciones.','',
        'Al mismo presupuesto B reduce material perdido como fondo (37.909 → 22.496 frente a A), pero casi triplica el fondo predicho como material (29.603 → 88.523). El mIoU de materiales de B queda por debajo de A (26,88 % frente a 31,81 %). La mejora de cobertura de material no basta para mejorar detección.','',
        '![Matrices semánticas normalizadas](../resultados/figuras/ablacion_fondo_confusiones.png)','',
        '## Validación: una vista','',
        '| Estado | Época | Pasos adicionales | mAP@50 | mAP@50:95 | Recall@50 | Precisión@50 | FP |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    bb=resumir_deteccion(base)
    l.append(f"| Origen | 19 | 0 | {bb['map50']:.3%} | {bb['map50_95']:.3%} | {bb['recall50']:.2%} | {bb['precision50']:.2%} | {bb['fp50']} |")
    for b in ['A','B']:
        for s in ['final','mejor']:
            m=r['ramas'][b][f'resumen_{s}'];e=31 if s=='final' else r['ramas'][b]['mejor_epoca_durante_actualizaciones']
            l.append(f"| {b} {s} | {e} | {(e-19)*121} | {m['map50']:.3%} | {m['map50_95']:.3%} | {m['recall50']:.2%} | {m['precision50']:.2%} | {m['fp50']} |")
    l+=['','![Aprendizaje y confusión fondo/material](../resultados/figuras/ablacion_fondo_aprendizaje.png)','',
        '## Validación: cinco vistas, mismo modo del detector predeterminado','',
        '| Estado | Época | mAP@50 | mAP@50:95 | Recall@50 | Precisión@50 | FP |',
        '|---|---:|---:|---:|---:|---:|---:|']
    bb5=resumir_deteccion(base5)
    l.append(f"| Origen predeterminado | 19 | {bb5['map50']:.3%} | {bb5['map50_95']:.3%} | {bb5['recall50']:.2%} | {bb5['precision50']:.2%} | {bb5['fp50']} |")
    for k,d in cinco.items():
        m=resumir_deteccion(d['val'])
        l.append(f"| {k.replace('_',' ')} | {d['epoca']} | {m['map50']:.3%} | {m['map50_95']:.3%} | {m['recall50']:.2%} | {m['precision50']:.2%} | {m['fp50']} |")
    l+=['','A igual presupuesto el mAP@50 de cinco vistas apenas cambia entre A y B (1,6445 % y 1,6552 %), ambos inferiores al 2,9123 % de origen. B obtiene cinco TP más, pero FP pasa de 2.090 a 4.424 y la precisión baja de 3,37 % a 1,73 %. Su mejor checkpoint por una vista tampoco mejora el modo de cinco vistas.','',
        '![Comparación de cinco vistas](../resultados/figuras/ablacion_fondo_cinco_vistas.png)','',
        '## AP individual de los seis materiales','',
        'Filtro de propuestas 0,05; mismos 361 objetos val.','',
        '| Material | Objetos | Origen, 5 vistas | A final | A mejor | B final | B mejor |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c,m in base5['por_clase'].items():
        vals=[cinco[k]['val']['por_clase'][c]['ap50'] for k in ['A_final','A_mejor','B_final','B_mejor']]
        l.append(f"| {c} | {m['objetos']} | {m['ap50']:.3%} | "+' | '.join(f'{v:.3%}' for v in vals)+' |')
    l+=['','| Material | Origen, 1 vista | A final | A mejor | B final | B mejor |','|---|---:|---:|---:|---:|---:|']
    for c,m in base['por_clase'].items():
        vals=[r['ramas'][b][f'resumen_{s}']['ap50_por_material'][c] for b,s in [('A','final'),('A','mejor'),('B','final'),('B','mejor')]]
        l.append(f"| {c} | {m['ap50']:.3%} | "+' | '.join(f'{v:.3%}' for v in vals)+' |')
    max_ap=max(abs(h['delta_map50']) for h in r['ramas']['A']['regresion_control_historico'])
    max_loss=max(abs(h['delta_loss']) for h in r['ramas']['A']['regresion_control_historico'])
    l+=['','## Reproducibilidad y límites','',
        'Se verifican estados iniciales completos, orden, parámetros de aumentos, LR y número real de actualizaciones. La repetición numérica **no es idéntica bit a bit a la continuación histórica en AMP**: el control registra diferencias frente al historial anterior, sin ocultarlas. No se activó un modo de algoritmos deterministas estricto; las huellas de entradas no equivalen a garantizar una trayectoria numérica idéntica entre ejecuciones GPU.', '',
        f"La mayor diferencia del control frente al historial previo fue {max_ap*100:.3f} puntos porcentuales de mAP@50 y {max_loss:.6f} de pérdida. Estas diferencias quedan por época en regresion_control_historico. Esta es una ablación pareada de una sola ejecución por rama, no una estimación de variabilidad con varias semillas.",'',
        'La validación ya se ha usado reiteradamente durante desarrollo. Solo hay cinco objetos orgánicos train y uno val. No se afirma significación estadística ni calidad suficiente para uso fiable. El escaso aumento final de mAP de B en cinco vistas no justifica desplegarla con más del doble de FP.','',
        '## Decisiones y reproducción','',
        '- Argmax y cinco vistas: conservar.',
        '- Propuestas sin argmax: descartar como configuración actual, conservando el A/B anterior.',
        '- Fondo=1: no adoptar con esta evidencia; se conservan las dos ramas y todos sus checkpoints.',
        '- Detector predeterminado y clasificador independiente: intactos; no se evaluó test.','',
        '```powershell',"$pythonEco = Join-Path $env:USERPROFILE '.venvs\\EcoWasteFormer\\Scripts\\python.exe'",
        '& $pythonEco scripts/ablacion_fondo.py --config configuracion/ablacion_fondo.json',
        '& $pythonEco scripts/informar_ablacion_video.py',
        'python scripts/graficar_ablacion_fondo.py',
        '& $pythonEco scripts/verificar_ablacion_video.py','```','',
        'La ejecución original se protege contra sobrescritura; para repetir se debe usar otro nombre de experimento. Los JSON por rama guardan los checkpoints finales, mejores durante actualizaciones y mejores incluyendo origen, además de matrices y trazas. Las evaluaciones de cinco vistas están en resultados/ablacion_fondo_ep19_{A,B}_{final,mejor}_cinco_vistas.json.','']
    (RAIZ/'documentacion/13_ablacion_del_fondo.md').write_text('\n'.join(l),encoding='utf-8')
    v=['# MP4 y webcam: integración experimental','',
       '**Implementado: lectura de MP4, captura de cámara explícita, detector persistente, anotación, exportación MP4/JSONL y consulta al asistente sobre el último frame.** La cámara física no se activó en estas pruebas.','',
       '## Uso','', '```powershell',"$pythonEco = Join-Path $env:USERPROFILE '.venvs\\EcoWasteFormer\\Scripts\\python.exe'",
       "& $pythonEco scripts/detectar_video.py --video 'C:\\ruta\\mi_video.mp4' --salida resultados/mi_video --pregunta '¿Cómo separo estos residuos?'",
       '& $pythonEco scripts/detectar_video.py --camara 0 --salida resultados/camara_sesion1 --mostrar','```','',
       'La carpeta de salida debe ser nueva. En la ventana, q o Escape detienen la lectura; Ctrl+C también libera los recursos. La cámara solo se abre con --camara; un MP4 inexistente o ilegible nunca hace fallback a la cámara. --mostrar es explícito y sin esa opción no se abre ninguna ventana.','',
       'Opciones: --max-frames N limita la sesión; --sin-video conserva JSONL y resumen sin exportar MP4; --vistas 1 permite comparar una sola vista. El valor predeterminado sigue siendo cinco vistas y argmax, con el mismo checkpoint de época 19 y umbral de propuestas 0,05. --umbral filtra puntuaciones de cajas ya formadas; no recorta sus componentes.','',
       '## Archivos de salida','',
       '- anotado.mp4: cajas en coordenadas originales y banner visible «EXPERIMENTAL - DETECCION NO FIABLE» en cada frame; salida sin audio.',
       '- frames.jsonl: cabecera con modelo, huella del checkpoint y parámetros; una línea por frame con tiempo, materiales, puntuaciones, cajas, latencia y advertencia; cierre con resumen.',
       '- resumen.json: frames procesados, motivo de fin, latencia mediana/p95, FPS de procesamiento y último frame.',
       '- asistente_ultimo_frame.json: solo cuando se especifica --pregunta; usa una instantánea identificada del último frame y conserva la advertencia del detector.','',
       'El modelo se carga una vez por sesión. No hay seguimiento de identidades ni conteo de objetos únicos acumulados. La misma botella en varios frames puede producir varias predicciones; las cantidades son por frame. El asistente no se invoca automáticamente en cada fotograma.','',
       '## Pruebas ejecutadas','',
       f"Se generó un MP4 técnico de ocho frames, a partir de cuatro fotos de validación repetidas. Se procesaron los ocho con el detector real y cinco vistas; se releyeron los ocho frames del MP4 exportado, se verificaron registros y advertencias, y se ejecutó el asistente sobre el frame 7. Las cajas, materiales y puntuaciones del frame 0 coinciden exactamente con la consulta por imagen sobre ese mismo frame decodificado ({imagen['detecciones_identicas']} detecciones).",'',
       f"Medición en ese montaje 320×240: mediana de inferencia {video['mediana_inferencia_ms']:.1f} ms, p95 {video['p95_inferencia_ms']:.1f} ms y {video['fps_procesamiento']:.1f} FPS de procesamiento total. Son ocho frames en entorno de desarrollo; incluyen arranque y no constituyen un benchmark de cámara ni garantía de tiempo real.",'',
       '![Primer frame de la prueba técnica](../resultados/demo_video_experimental/primer_frame.jpg)','',
       'Evidencia: resultados/verificacion_video_real.json, resultados/verificacion_video_imagen.json y resultados/demo_video_experimental/. El montaje de fotos **no es un video independiente** y no se usa para afirmar precisión de detección en video.','',
       f"Pasaron {pruebas['ejecutadas']} pruebas del proyecto. Las de video cubren lectura/exportación MP4, límite de frames, modo sin ventana, cámara simulada, resolución impar, FPS ausente, rechazo de salidas existentes, archivo ausente, lectura truncada y liberación de recursos ante error de inferencia. Las pruebas de cámara reemplazan VideoCapture por una fuente simulada: no activan hardware.",'',
       '## Alcance y pendientes','',
       'La integración funcional experimental de MP4/webcam queda implementada. Siguen pendientes la comprobación con una cámara física y videos independientes, calidad por clase, latencia sostenida y robustez frente a escenas nuevas. El detector mantiene mAP@50 val 2,91 %: esta integración no lo convierte en fiable.','',
       'El video exportado usa FPS nominal de la fuente (30 si no está disponible). El JSONL conserva los tiempos de captura; si la inferencia de cámara es más lenta que la adquisición, la duración/cadencia del MP4 puede diferir del tiempo de captura. No se conserva audio.','']
    (RAIZ/'documentacion/14_video_y_webcam_experimental.md').write_text('\n'.join(v),encoding='utf-8')
    resumen={'decision':'conservar_argmax_cinco_vistas_y_checkpoint_origen','adoptar_fondo_uno':False,
        'val_cinco_vistas':{k:resumir_deteccion(d['val']) for k,d in cinco.items()},
        'estado_video':'implementado_experimental_con_camara_simulada','verificacion_mp4_real':video,
        'pruebas':pruebas,'test_evaluado':False,'seleccion_modificada':False,
        'sha256_ablacion':hashlib.sha256((RAIZ/'resultados/ablacion_fondo_ep19.json').read_bytes()).hexdigest()}
    escribir_json(RAIZ/'resultados/resumen_ablacion_video.json',resumen)
    readme=RAIZ/'README.md';p=readme.read_text(encoding='utf-8').split('\n\n')
    for i,t in enumerate(p):
        if t.startswith('Fase actual:'):
            p[i]='Fase actual: los tres modelos Transformer están implementados y entrenados; RealWaste y TACO están preparados y auditados. MP4 y webcam tienen inferencia experimental, exportación y conexión al asistente. **El detector mantiene calidad insuficiente; la cámara física y los videos independientes siguen pendientes de validación.** Los documentos originales están en `../documentos/`.'
        if t.startswith('**Detector:**'):
            p[i]=f"**Detector:** se conserva `detector_mapas_v6_mosaico`, checkpoint de época 19, argmax y cinco vistas; mAP@50 val **2,91 %**, todavía insuficiente. Pasaron {pruebas['ejecutadas']} pruebas. Las ramas nuevas no sustituyen al predeterminado; ver [ablación](documentacion/13_ablacion_del_fondo.md)."
        if t.startswith('**Ablación de fondo y video:**'):
            p[i]=''
    nuevo=f"**Ablación de fondo y video:** completadas dos ramas de 1.452 actualizaciones desde el mismo estado de época 19; fondo=1 no mejora suficientemente y se conserva el detector actual. MP4/webcam implementados con advertencias y {pruebas['ejecutadas']} pruebas aprobadas. Ver [ablación](documentacion/13_ablacion_del_fondo.md) y [uso de video/cámara](documentacion/14_video_y_webcam_experimental.md)."
    p.insert(next(i for i,t in enumerate(p) if t.startswith('**Detector:**'))+1,nuevo)
    texto='\n\n'.join(t for t in p if t!='').replace('4. Integrar imagen, webcam, video y asistente; medir latencia y calidad con datos independientes.',
        '4. Validar la webcam física y videos independientes; medir latencia sostenida y calidad. La integración experimental ya está implementada.')
    if '## MP4 y webcam' not in texto:
        texto+='\n\n## MP4 y webcam\n\n```powershell\n& $pythonEco scripts/detectar_video.py --video \'C:\\ruta\\mi_video.mp4\' --salida resultados/mi_video\n& $pythonEco scripts/detectar_video.py --camara 0 --salida resultados/camara_sesion1 --mostrar\n```\n\nLa carpeta de salida debe ser nueva. Se conserva argmax y cinco vistas; las cajas son experimentales. Ver [salidas, controles y límites](documentacion/14_video_y_webcam_experimental.md).\n'
    readme.write_text(texto,encoding='utf-8')
    print(json.dumps({'ablacion':'documentacion/13_ablacion_del_fondo.md','video':'documentacion/14_video_y_webcam_experimental.md',
                      'adoptar_fondo_uno':False,'pruebas':pruebas['ejecutadas'],'predeterminado_inalterado':True}))


if __name__=='__main__':main()
