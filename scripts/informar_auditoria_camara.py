"""Informe de fallos por clase/tamaño y prueba física, sin cambiar modelos o datos."""
import csv
import hashlib
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.datasets import CLASES
from ecowaste.auditoria_deteccion import TAMANOS
from scripts.entrenar_clasificador import escribir_json


def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8-sig'))
def porcentaje(v):return f'{v:.2%}' if v is not None else '—'


def main():
    r=leer('resultados/auditoria_clase_tamano.json')
    cam=leer('resultados/prueba_camara_fisica_02_fuera_aislamiento.json')
    dispositivos=leer('resultados/dispositivos_camara_fisica_fuera_aislamiento.json')
    pruebas=leer('resultados/pruebas_auditoria_camara.json')
    if r['estado']!='auditoria_completada' or not pruebas['correcto']:raise ValueError('Auditoría o pruebas incompletas')
    c=r['condiciones']['5_vistas_score_0.05'];datos=r['cobertura_datos']
    prioridades=[o for o in c['objetos'] if o['material_real'] in ['carton','vidrio','organico'] and o['estado']!='correcto']
    cola=RAIZ/'datos/reportes/revision_clases_sin_ap.csv'
    with cola.open('w',newline='',encoding='utf-8-sig') as f:
        campos=['id_imagen','id_anotacion','material_real','estado','subtipo','grupo_tamano','lado_menor_384',
                'material_testigo_nombre','iou_testigo','cobertura_gt_testigo','ruta_imagen','bbox_gt_original_xywh']
        w=csv.DictWriter(f,fieldnames=campos);w.writeheader()
        for o in prioridades:w.writerow({k:o[k] for k in campos})
    l=['# Auditoría del detector por clase y tamaño','',
       '**No se reentrenó ni se ejecutó nuevamente el detector. Se reutilizaron exclusivamente predicciones y anotaciones guardadas, en CPU.** Se conserva el clasificador, argmax, cinco vistas, pesos y coeficiente de fondo actuales.','',
       '## Alcance y reglas prefijadas','',
       'Mismas 181 imágenes y 361 GT de validación. Se auditan los JSON existentes de una y cinco vistas del checkpoint propio de época 19. Los filtros 0,05 y 0,35 ya existían; no se probaron nuevos umbrales. La asignación reproduce exactamente TP, FP, FN e ignoradas anteriores, incluidos los recuentos de cada clase.','',
       'Primero se conserva la asignación uno a uno del evaluador, por puntuación, material e IoU≥0,5. Un GT detectado se marca correcto. Para cada GT restante:', '',
       '1. **Material equivocado:** existe una caja evaluable con IoU≥0,5 y otro material, sin una caja correcta disponible.',
       '2. **Geometría incorrecta:** no hay caja razonable disponible, pero alguna caja cubre al menos 50 % del área del GT; su IoU queda por debajo de 0,5. Se identifica si es demasiado grande, pequeña o presenta forma/desplazamiento. Una caja correcta compartida por varios GT, ya asignada a otro, se registra también aquí como conflicto de asignación.',
       '3. **Sin cobertura suficiente:** ninguna caja final evaluable cubre 50 % del GT. Puede haber cajas parciales, pequeñas o alejadas; no significa necesariamente que el mapa interno no generara región.','',
       'Las predicciones ignoradas por las regiones fuera de alcance, según la misma regla del evaluador a IoU 0,5, no se usan para inventar cobertura. Las duplicadas y los errores de material sobre GT válido siguen contando como FP.','',
       'Es una partición operacional que prioriza geometría suficiente antes de atribuir el material. Los fallos pueden coexistir: una caja parcial también puede tener etiqueta equivocada, pero no se considera «caja razonable con material equivocado» hasta alcanzar IoU 0,5. Cada fila conserva IoU, cobertura y etiqueta del testigo para revisar esos casos combinados.','',
       '**Limitación principal:** solo están guardadas cajas finales, después de componentes, cortes de vistas, NMS y top100. La cobertura de su rectángulo no es cobertura de la máscara. No podemos separar causalmente un fallo del mapa, del filtro o de NMS sin las trazas previas, y no se hizo nueva inferencia para obtenerlas.','',
       '## Resultado del detector predeterminado: cinco vistas, puntuación ≥0,05','',
       '| Material | GT val | AP@50 existente | Correctos | Sin cobertura suficiente | Geometría incorrecta | Material equivocado |',
       '|---|---:|---:|---:|---:|---:|---:|']
    for nombre,m in c['diagnostico_gt']['por_clase'].items():
        l.append(f"| {nombre} | {m['objetos']} | {c['ap50_por_clase'][nombre]:.3%} | {m['correcto']} | {m['sin_cobertura_suficiente']} | {m['geometria_incorrecta']} | {m['material_equivocado']} |")
    t=c['diagnostico_gt']['total']
    l+=['',f"Total: {t['correcto']} GT detectados, {t['sin_cobertura_suficiente']} sin cobertura suficiente, {t['geometria_incorrecta']} con geometría incorrecta y {t['material_equivocado']} con material equivocado. Los últimos tres grupos suman los {t['fn']} FN existentes; no son una nueva estimación optimista de recall.",'',
        '![Tipos de fallo por clase](../resultados/figuras/auditoria_fallos_clase.png)','',
        '## Tamaño en la entrada del modelo','',
        'Se usa el **lado menor del bbox al redimensionar la imagen completa al lienzo de 384**: <8, [8,16), [16,32) y ≥32 píxeles. La referencia de ocho píxeles es el patch de detalle del modelo; no es un límite absoluto de detección. Estas categorías no son las de AP small/medium/large de COCO. En cinco vistas existen ampliaciones adicionales, cuyo origen no está registrado en las cajas combinadas.','',
        '| Lado menor en 384 | GT | TP | Recall@50 por GT | Sin cobertura | Geometría | Material |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for tam,m in c['diagnostico_gt']['por_tamano'].items():
        l.append(f"| {tam} | {m['objetos']} | {m['correcto']} | {porcentaje(m['recall50'])} | {m['sin_cobertura_suficiente']} | {m['geometria_incorrecta']} | {m['material_equivocado']} |")
    l+=['','Los TP por tamaño conservan la asignación global. No se eliminan GT de otras tallas ni se recalcula un AP por tamaño que contabilice incorrectamente sus detecciones como FP. El JSON cruza clase × tamaño para todos los estados y conserva cada ID de anotación.','',
        '![Recall y errores por tamaño](../resultados/figuras/auditoria_tamanos.png)','',
        '## Interpretación de las tres clases sin AP','',
        '**Cartón:** 14 de 25 GT tienen una caja geométricamente razonable con otro material: once como plástico y tres como metal. Son objetos de al menos 16 píxeles de lado menor. Otros seis fallan por geometría y cinco por falta de cobertura. El AP cero no se explica solamente por objetos minúsculos o ausencia total de propuestas.','',
        '**Vidrio:** 16 de 22 GT carecen de cobertura suficiente en cajas finales, tres tienen geometría incorrecta y tres una caja razonable con otro material. En train, 107/203 vidrios quedan por debajo de ocho píxeles y 133/203 por debajo de 16. Sin embargo, también faltan cajas para vidrios más grandes en val: el tamaño por sí solo no explica todos los fallos.','',
        '**Orgánico:** solo cinco GT train y uno val. El único GT val tiene lado menor 6,12 píxeles y no tiene cobertura suficiente. Un solo objeto val no permite caracterizar la clase ni esperar robustez; las fotos auxiliares RealWaste con etiqueta global no son cajas adicionales.','',
        '![Confusión diagnóstica por GT](../resultados/figuras/auditoria_materiales.png)','',
        'Esta matriz ilustra el material de la caja testigo elegida con ayuda del GT. No es una nueva métrica de clasificación autónoma: para un GT correcto se usa su pareja real; para un error de material la caja razonable de mayor IoU; los otros casos se agrupan en la última columna.','',
        '## Casos para revisión visual','',
        f"Se generó datos/reportes/revision_clases_sin_ap.csv con los {len(prioridades)} GT de cartón, vidrio y orgánico no detectados, sus imágenes, cajas originales, diagnóstico y testigo. No son anotaciones nuevas y no se incorporan a train.",'',
        '![Ejemplos fijados por orden de validación](../resultados/figuras/auditoria_ejemplos.png)','',
        'La galería usa los dos primeros cartones con error de material, los dos primeros vidrios sin cobertura y el único orgánico, en el orden del manifiesto. Se muestran imagen completa y acercamiento para inspección: azul=GT, naranja=caja testigo si existe. Las imágenes y acercamientos no se usan para entrenar ni para producir nuevas predicciones.','',
        '## Cobertura de anotaciones train/val','',
        '| Material | Instancias train | Imágenes train | Grupos de adquisición train | Train <8 px | Instancias val | Val <8 px |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for nombre in CLASES:
        a,b=datos['train'][nombre],datos['val'][nombre]
        l.append(f"| {nombre} | {a['instancias']} | {a['imagenes']} | {a['grupos_adquisicion']} | {a['por_tamano']['menos_8_px']} | {b['instancias']} | {b['por_tamano']['menos_8_px']} |")
    l+=['','Los grupos son los del manifiesto de procedencia, no una garantía de independencia por sesión real. Las máscaras y cajas existentes permanecen intactas.','',
        '## Cambios de entrenamiento que sí merece preparar','',
        '- **Cartón/material:** el candidato más directo es supervisar material sobre regiones GT de TACO usando la cabeza auxiliar y características regionales, con un control sin esa pérdida. Mantener fondo=3, cajas y postprocesamiento. Es una hipótesis de entrenamiento respaldada por el tipo de error, no una mejora ya demostrada; no se ejecutó ahora.',
        '- **Vidrio/cobertura:** revisar primero los casos identificados y su escala. En la siguiente inferencia que ya sea necesaria, conservar propuestas antes de NMS y origen de cada vista para localizar dónde se pierden. Los recortes balanceados ya se probaron sin mejorar; esta auditoría no justifica repetirlos automáticamente ni cambiar resolución a ciegas.',
        '- **Orgánicos/datos reales:** recopilar sesiones diversas con objetos y tamaños variados; dibujar cajas reales, revisar clase y geometría, conservar procedencia y dividir por sesión antes de entrenar. Reservar nuevas escenas independientes. No asignar la imagen completa como bbox por tener una etiqueta global y no convertir predicciones automáticas en GT sin revisión.',
        '- **Conservar:** clasificador 85,74 % accuracy val; argmax y cinco vistas. No repetir por ahora ablaciones de fondo ni barridos de umbrales.','',
        '## Verificación y reproducción','',
        f"Auditoría completada en {r['duracion_segundos']:.2f} s de ejecución local en CPU: cero cargas de modelo, cero inferencias nuevas, cero actualizaciones y cero uso de GPU por el proceso de auditoría. Pasaron {pruebas['ejecutadas']} pruebas del proyecto. Los recuentos reproducen las cuatro condiciones existentes (una/cinco vistas y puntuaciones 0,05/0,35), incluidas TP/FP por clase.",'',
        '```powershell',"$pythonEco = Join-Path $env:USERPROFILE '.venvs\\EcoWasteFormer\\Scripts\\python.exe'",
        '& $pythonEco scripts/auditar_clase_tamano.py',
        '& $pythonEco scripts/informar_auditoria_camara.py',
        'python scripts/graficar_auditoria_clase_tamano.py',
        '& $pythonEco scripts/verificar_auditoria_camara.py','```','',
        'La ejecución existente se protege contra sobrescritura: para repetir se debe copiar la configuración con otro nombre. Evidencia completa en resultados/auditoria_clase_tamano.json y CSV por objeto. Test no se usó para seleccionar modificaciones.','']
    (RAIZ/'documentacion/15_auditoria_clase_y_tamano.md').write_text('\n'.join(l),encoding='utf-8')
    v=['# Prueba de cámara física','',
       '**Captura física verificada: Integrated Camera, 12 frames, 640×480, argmax y cinco vistas.** La prueba funcional no establece precisión ni fiabilidad de detección.','',
       '## Ejecución y acceso al dispositivo','',
       'El primer intento dentro del aislamiento no pudo abrir DirectShow. La consulta de dispositivos también recibió «Acceso denegado». Tras la autorización para ejecutar fuera del aislamiento, Windows enumeró Integrated Camera con estado OK y el índice cero se abrió correctamente. No se necesitó cambiar umbrales, backend ni pesos. Se conservan ambos intentos.','',
       f"El intento exitoso terminó con código {cam.get('codigo_salida')} y releyó {cam.get('frames_mp4_releidos')} frames del MP4 exportado. El proceso de captura finalizó y la implementación libera VideoCapture/VideoWriter en finally. No se abrió ventana ni se capturó audio.",'',
       f"Mediana de inferencia: {cam.get('mediana_inferencia_ms',0):.1f} ms; p95: {cam.get('p95_inferencia_ms',0):.1f} ms; procesamiento: {cam.get('fps_procesamiento',0):.1f} FPS. Solo son doce frames y hay coste de arranque: no equivalen a una prueba sostenida de webcam.",'',
       'El MP4 y JSONL incluyen la advertencia experimental, puntuaciones no calibradas y recuentos por frame, sin seguimiento de identidades. Los frames capturados no se añadieron a los datasets y no tienen anotaciones para calcular mAP. No se incrusta ninguna foto de cámara en los informes.','',
       '## Evidencia local','',
       '- resultados/prueba_camara_fisica_01.json: intento aislado fallido.',
       '- resultados/dispositivos_camara_fisica_fuera_aislamiento.json: enumeración del dispositivo físico.',
       '- resultados/prueba_camara_fisica_02_fuera_aislamiento.json: verificación física y hashes.',
       '- resultados/prueba_camara_fisica_02_fuera_aislamiento/: MP4 anotado, frames.jsonl y resumen.', '',
       '## Uso posterior','', '```powershell',"$pythonEco = Join-Path $env:USERPROFILE '.venvs\\EcoWasteFormer\\Scripts\\python.exe'",
       '& $pythonEco scripts/detectar_video.py --camara 0 --salida resultados/camara_sesion_nueva --mostrar',
       '& $pythonEco scripts/probar_camara_fisica.py --nombre prueba_camara_fisica_nueva --max-frames 12 --timeout 30','```','',
       'La salida debe ser una carpeta nueva. La prueba física requiere acceso al dispositivo; en este entorno el aislamiento lo bloquea, mientras que la ejecución autorizada fuera de él funciona. El script de prueba limita frames y tiempo, registra fallos y nunca los presenta como una cámara validada.','',
       '## Prioridades actualizadas','',
       '| Área | Estado | Siguiente paso |','|---|---|---|',
       '| Clasificador | 85,74 % accuracy val; conservado | Validación independiente |',
       '| Detector | 2,91 % mAP@50 val; calidad insuficiente | Revisión de los casos y preparación de supervisión regional de material |',
       '| Postprocesamiento | Argmax y cinco vistas | Mantener |',
       '| Pérdida de fondo | Ablación terminada | No repetir por ahora |',
       '| MP4 | Implementado y probado técnicamente | Video independiente con anotaciones |',
       '| Webcam | Prueba física breve completada | Prueba sostenida y escenas variadas, con latencia y calidad |',
       '| Asistente | Integrado; comprensión limitada | Ampliar y evaluar corpus |','',
       'La cámara física deja de estar pendiente como prueba de apertura/captura. Permanecen pendientes la validación prolongada y la evaluación de calidad sobre videos independientes. El detector sigue sin ser fiable.','']
    if not cam.get('captura_real_verificada') or not dispositivos.get('consulta_correcta'):raise ValueError('La prueba física no está completada')
    (RAIZ/'documentacion/16_prueba_camara_fisica.md').write_text('\n'.join(v),encoding='utf-8')
    resumen={'auditoria_cpu_completada':True,'diagnostico_actual':c['diagnostico_gt'],
        'camara_fisica_completada':True,'frames_camara':cam['frames_capturados'],'resolucion_camara':cam['resolucion'],
        'camara_es_evaluacion_calidad':False,'instancias_train_organicos':datos['train']['organico']['instancias'],
        'cola_revision':cola.relative_to(RAIZ).as_posix(),'gt_revision':len(prioridades),
        'inferencias_auditoria':0,'entrenamientos':0,'test_evaluado':False,'pruebas':pruebas['ejecutadas']}
    escribir_json(RAIZ/'resultados/resumen_auditoria_camara.json',resumen)
    readme=RAIZ/'README.md';p=readme.read_text(encoding='utf-8').split('\n\n')
    for i,t in enumerate(p):
        if t.startswith('Fase actual:'):
            p[i]='Fase actual: modelos propios entrenados, datasets auditados y flujo MP4/webcam/asistente implementado. La cámara física capturó y procesó 12 frames de 640×480. **El detector mantiene calidad insuficiente; faltan videos independientes anotados y una prueba sostenida.** Los documentos originales están en `../documentos/`.'
        if t.startswith('**Auditoría por clase y cámara física:**'):p[i]=''
    nuevo=f"**Auditoría por clase y cámara física:** completadas sin reentrenar. Cartón: 14/25 GT tienen caja razonable con material equivocado; vidrio: 16/22 sin cobertura suficiente en cajas finales; orgánico: cinco GT train y uno val de 6,12 px. Cámara real: 12 frames de 640×480 verificados. Pasaron {pruebas['ejecutadas']} pruebas; ver [auditoría](documentacion/15_auditoria_clase_y_tamano.md) y [prueba física](documentacion/16_prueba_camara_fisica.md)."
    p.insert(next(i for i,t in enumerate(p) if t.startswith('**Detector:**'))+1,nuevo)
    texto='\n\n'.join(t for t in p if t!='').replace('4. Validar la webcam física y videos independientes; medir latencia sostenida y calidad. La integración experimental ya está implementada.',
        '4. Probar video independiente anotado y webcam de forma sostenida. La lectura física de cámara ya se verificó; falta calidad y rendimiento en sesiones largas.')
    readme.write_text(texto,encoding='utf-8')
    print(json.dumps({'auditoria':'documentacion/15_auditoria_clase_y_tamano.md','camara':'documentacion/16_prueba_camara_fisica.md',
                      'pruebas':pruebas['ejecutadas'],'nuevos_entrenamientos':0}))


if __name__=='__main__':main()
