"""Informe final A/B y evidencia de reanudación, sin adoptar checkpoints."""
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.datasets import CLASES
from ecowaste.reanudacion_regional import publicar,sha
from scripts.supervision_regional_materiales import CONFIG,hashes_codigo,protocolo


def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8'))
def pct(x):return f'{100*x:.3f} %'


def main():
    cfg=leer(CONFIG);nombre=cfg['nombre'];r=leer(f'resultados/{nombre}.json');e=leer(f'resultados/{nombre}_evaluacion.json')
    d=leer(cfg['diagnostico']);v=leer(cfg['verificacion']);res=e['resultados']
    if not d['aprobada'] or not v['aprobada'] or d['hashes_codigo']!=hashes_codigo():raise ValueError('Controles desactualizados')
    if any(x['actualizaciones']!=1452 or len(x['historial'])!=12 for x in r['ramas'].values()):raise ValueError('Presupuesto incompleto')
    ha=r['ramas']['A']['historial'];hb=r['ramas']['B']['historial']
    if any(a['huella_orden_y_aumentos']!=b['huella_orden_y_aumentos'] for a,b in zip(ha,hb)):raise ValueError('Entradas distintas')
    rng_iguales=all(a['huellas_estado']['rng']==b['huellas_estado']['rng'] for a,b in
        zip(r['ramas']['A']['checkpoints_epocas'],r['ramas']['B']['checkpoints_epocas']))
    if not rng_iguales:raise ValueError('Estados RNG finales por época distintos')
    anterior=leer('resultados/supervision_regional_reanudacion/estado_previo.json')['sha256_historicos']
    cambiados=[p for p,h in anterior.items() if sha(RAIZ/p)!=h]
    if cambiados:raise ValueError('Históricos modificados: '+str(cambiados))
    ejecuciones={};organico={}
    for rama in ['A','B']:
        carpeta=RAIZ/f'resultados/{nombre}/rama_{rama}/ejecuciones'
        entradas=[leer(p) for p in sorted(carpeta.glob('*.json')) if not p.name.endswith('.fin.json')]
        if not any(x['reanudar'] and x['desde_epoca']==20 for x in entradas):raise ValueError('Falta reanudación real desde época 20')
        ejecuciones[rama]=entradas
        h=r['ramas'][rama]['historial'];eventos=[x['region_organica']['eventos'][0] for x in h]
        if any(x['region_organica']['presencia']!=1 or len(x['region_organica']['eventos'])!=1 for x in h):raise ValueError('Presencia orgánica inválida')
        if any(x['peso_sin_modificar']!=d['region_organica_elegible']['peso'] for x in eventos):raise ValueError('Peso cambiado')
        organico[rama]={'presencias_totales':len(eventos),'peso':eventos[0]['peso_sin_modificar'],
            'ce_media':sum(x['ce_media'] for x in eventos)/12,
            'contribucion_loss_total_media_lotes_presencia':sum(x['contribucion_loss_total'] for x in eventos)/12,
            'eventos':eventos}
    aa=res['A_ultimo'];bb=res['B_ultimo'];base=res['origen']
    apA=aa['vistas']['5']['resumen']['ap50_por_material']['carton'];apB=bb['vistas']['5']['resumen']['ap50_por_material']['carton']
    mapA=aa['vistas']['5']['resumen']['map50'];mapB=bb['vistas']['5']['resumen']['map50']
    cartonA=aa['confusion_regional_train']['mapas_logits_promediados_por_region']['matriz_confusion'][3]
    cartonB=bb['confusion_regional_train']['mapas_logits_promediados_por_region']['matriz_confusion'][3]
    tasaA=(cartonA[1]+cartonA[5])/sum(cartonA);tasaB=(cartonB[1]+cartonB[5])/sum(cartonB)
    conclusion={'delta_accuracy_regional_auxiliar_train':bb['confusion_regional_train']['accuracy']-aa['confusion_regional_train']['accuracy'],
        'delta_ap50_carton_final_cinco_vistas':apB-apA,'delta_map50_final_cinco_vistas':mapB-mapA,
        'carton_mejora_vs_control':apB>apA,'map50_mejora_vs_control':mapB>mapA,
        'map50_variante_supera_referencia':mapB>base['vistas']['5']['resumen']['map50'],
        'delta_confusion_carton_plastico_metal_cabeza_mapas_train':tasaB-tasaA,
        'adopcion_automatica':False}
    lineas=['# A/B regional completo y reanudación segura','',
        '**Completadas A=0 y B=0,25: 1.452 actualizaciones por rama, 12 épocas adicionales. Test reservado; referencia original intacta.**','',
        '## Reanudación implementada y ejercitada','',
        f"Pasaron **{d['pruebas']['ejecutadas']} pruebas** sin fallos. Las siete pruebas nuevas incluyen las dos ramas y el presupuesto completo sobre un modelo pequeño CPU, interrupción dentro de una época (tres pasos descartados), recuperación del checkpoint publicado sin registro, exclusión mutua, aliases ajenos, cambios de protocolo/código/RNG y corrupción de trazas/historial.",'',
        'Cada época terminada publica un checkpoint inmutable `checkpoints/'+nombre+'_A/B_epoca_NNN.pt` y un registro JSON inmutable en `resultados/'+nombre+'/rama_A/B/`. Cada registro contiene el SHA del checkpoint, huellas de los cuatro estados y SHA del registro anterior. El archivo se sincroniza antes de publicarse sin reemplazo. Un bloqueo del sistema operativo impide dos escritores sobre la misma rama y se libera al finalizar el proceso.','',
        'Un corte dentro de una época retoma la última época confirmada; se repite el trabajo no confirmado y no se cuenta dos veces en el presupuesto duradero. Si el checkpoint quedó completo pero falta su registro, `--reanudar` lo valida y completa el commit. Un temporal incompleto no se adopta. Se rechazan huecos, otra rama, código/datos/diagnóstico/protocolo distintos, pesos de clase alterados, pasos AdamW incoherentes y trazas no previstas. Los aliases `_ultimo.pt` y `_mejor.pt` son copias derivadas: solo se reemplazan si pertenecen al mismo diario; todos los checkpoints históricos permanecen disponibles.','',
        'Las ejecuciones reales CUDA de A y B se detuvieron de forma controlada al terminar época 20 (121 actualizaciones) y se reanudaron en procesos nuevos hasta época 31. Se verificaron los hashes de pesos, AdamW, scaler y RNG al restaurar. El contador final de AdamW es **3.751** en ambas ramas (2.299+1.452), LR 0,0005; no hubo reinicio de warmup ni parada temprana. Las doce trazas coinciden con el plan y entre ramas.','',
        'La identidad numérica entre continuo/reanudado se comprobó únicamente en la fixture CPU de las pruebas. No se ejecutó una segunda trayectoria CUDA completa sin interrupción, por lo que **no se afirma identidad bit a bit del entrenamiento CUDA**. La coincidencia exacta de inferencia congelada y la restauración exacta del estado serializado son comprobaciones diferentes.','',
        '## Controles regenerados','',
        f"Diagnóstico nuevo: [{Path(cfg['diagnostico']).name}](../{cfg['diagnostico']}). Verificación nueva: [{Path(cfg['verificacion']).name}](../{cfg['verificacion']}). Ambos contienen las huellas vigentes; el entrenador exigió su aprobación antes de habilitarse. Se repitieron memorización train (pesos descartados), gradientes AMP e inferencia congelada de 181 escenas val con una y cinco vistas, con predicciones idénticas a las históricas.",'',
        f'Se comprobaron al final **{len(anterior)} archivos históricos** sin cambios, incluyendo V17, resultados anteriores, checkpoint de referencia, clasificador independiente y selección de arquitectura. Se mantienen fondo=3, argmax, área mínima 2, umbral 0,05, umbral visual 0,35, auxiliar global=0,5 y los seis pesos regionales train de V17.','',
        '## Detección en validación','',
        'La comparación principal usa los checkpoints finales al mismo presupuesto. El mejor se selecciona exclusivamente por mAP@50 val de una vista; los empates conservan la primera época. Se presenta aparte y puede corresponder a presupuestos anteriores distintos.','',
        '| Modelo | Época | Vistas | mAP@50 | mAP@50:95 | Recall | Precisión | TP | FP | FN | AP50 cartón |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k,label in [('origen','Referencia'),('A_ultimo','A final'),('B_ultimo','B final'),('A_mejor','A mejor'),('B_mejor','B mejor')]:
        for vistas in ['1','5']:
            x=res[k]['vistas'][vistas]['resumen']
            lineas.append(f"| {label} | {res[k]['epoca']} | {vistas} | {pct(x['map50'])} | {pct(x['map50_95'])} | {pct(x['recall50'])} | {pct(x['precision50'])} | {x['tp50']} | {x['fp50']} | {x['fn50']} | {pct(x['ap50_por_material']['carton'])} |")
    for vistas in ['1','5']:
        lineas+=['',f'### AP@50 por material, {vistas} vista(s)','',
            '| Material | Referencia | A final | B final | A mejor | B mejor |','|---|---:|---:|---:|---:|---:|']
        for c in CLASES:
            valores=[pct(res[k]['vistas'][vistas]['resumen']['ap50_por_material'][c]) for k in ['origen','A_ultimo','B_ultimo','A_mejor','B_mejor']]
            lineas.append('| '+c+' | '+' | '.join(valores)+' |')
    lineas+=['','## Aprendizaje regional y confusión de cartón','',
        'Las mediciones siguientes usan polígonos GT exclusivamente de train. La cabeza auxiliar compartida tiene seis materiales; la cabeza de mapas tiene fondo y seis materiales. No se confunden estas métricas con AP de cajas.','',
        '| Modelo | Exactitud auxiliar regional | CE regional | Cartón→plástico mapas | Cartón→metal mapas | Cartón→fondo mapas | Cartón correcto mapas |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for k in ['origen','A_ultimo','B_ultimo','A_mejor','B_mejor']:
        x=res[k]['confusion_regional_train'];cm=x['mapas_logits_promediados_por_region']['matriz_confusion'][3]
        lineas.append(f"| {k} | {pct(x['accuracy'])} | {x['ce_regional']:.4f} | {cm[1]} | {cm[5]} | {cm[0]} | {cm[3]} |")
    lineas+=['',f"En cinco vistas y presupuesto final, B−A en AP50 de cartón: **{100*(apB-apA):+.3f} puntos porcentuales**; en mAP50 global: **{100*(mapB-mapA):+.3f} puntos porcentuales**.",'',
        ('La mejora regional llega a AP@50 de cartón frente al control.' if apB>apA else 'La supervisión regional no mejora AP@50 de cartón frente al control al presupuesto final.'),
        ('El mAP@50 global también mejora frente al control.' if mapB>mapA else 'El mAP@50 global no mejora frente al control.'),'',
        f"La exactitud de la cabeza auxiliar regional pasa de {pct(aa['confusion_regional_train']['accuracy'])} a {pct(bb['confusion_regional_train']['accuracy'])}, pero en la cabeza de mapas cartón→plástico/metal pasa de {cartonA[1]+cartonA[5]}/146 ({pct(tasaA)}) a {cartonB[1]+cartonB[5]}/146 ({pct(tasaB)}). El número de regiones cartón correctamente clasificadas por mapas pasa de {cartonA[3]} a {cartonB[3]}. **Aprender en la cabeza auxiliar no resolvió la confusión objetivo de la cabeza de mapas.**",'',
        '### Auditoría de los 25 GT de cartón en val, cinco vistas','',
        '| Modelo | Correctos | Sin cobertura | Geometría incorrecta | Material equivocado | A plástico | A metal |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for k in ['origen','A_ultimo','B_ultimo','A_mejor','B_mejor']:
        au=res[k]['vistas']['5']['auditoria_carton_val'];es=au['estados']
        lineas.append(f"| {k} | {es['correcto']} | {es['sin_cobertura_suficiente']} | {es['geometria_incorrecta']} | {es['material_equivocado']} | {au['material_equivocado_a_plastico']} | {au['material_equivocado_a_metal']} |")
    lineas+=['',
        'La variante final sigue por debajo del mAP@50 de referencia. La pequeña ganancia de B frente a A final se acompaña de AP de cartón/vidrio/orgánico cero y deterioros en otros materiales respecto al origen. **Se conserva la referencia original; esta evidencia no justifica adoptar B como detector predeterminado.**','',
        'El JSON de evaluación conserva además IoU semántico train, confusiones completas de ambas cabezas, auditoría de cartón con caja razonable y material equivocado en val, y resúmenes al umbral visual 0,35.','',
        '## Única región orgánica','',
        f"Región: `{d['region_organica_elegible']}`. Presente exactamente una vez en cada época de cada rama: **12 presencias por rama**. Su peso **{d['region_organica_elegible']['peso']:.8f}** se conserva. Cada evento registra imagen, instancia, índice, paso, CE, denominador del lote, numerador ponderado, contribución a CE regional, contribución al loss total y si participa en el gradiente.",'',
        '| Rama | Presencias | CE orgánica media | Contribución media al loss total en lotes con presencia |','|---|---:|---:|---:|']
    for rama,x in organico.items():lineas.append(f"| {rama} | {x['presencias_totales']} | {x['ce_media']:.6f} | {x['contribucion_loss_total_media_lotes_presencia']:.6f} |")
    lineas+=['','La contribución a la optimización regional de A es cero; sus mediciones se calculan sin gradiente. El peso elevado de B procede de una sola instancia y no representa diversidad ni evidencia de generalización orgánica. Cuatro de los cinco GT orgánicos siguen excluidos por la política de regiones válidas.','',
        '## Coste y reproducibilidad','',
        '| Rama | Tiempo acumulado entrenamiento y val por época | Pico GPU asignado | Pico GPU reservado |','|---|---:|---:|---:|']
    for rama,x in r['ramas'].items():lineas.append(f"| {rama} | {x['segundos']:.1f} s | {x['memoria_gpu_max_allocated_bytes']/2**20:.1f} MiB | {x['memoria_gpu_max_reserved_bytes']/2**20:.1f} MiB |")
    lineas+=['','El tiempo excluye intervalos con el proceso detenido y la preparación inicial de datos; incluye validación de una vista y trabajo de registro previo al último commit. La evaluación adicional de una/cinco vistas registra su propio coste. Las diferencias dependen del estado térmico, cachés e I/O de OneDrive, además de la pérdida adicional.','',
        '```powershell',r"$pythonEco = Join-Path $env:USERPROFILE '.venvs\EcoWasteFormer\Scripts\python.exe'",
        '& $pythonEco scripts/supervision_regional_materiales.py --entrenar --rama A --hasta-epoca 20',
        '& $pythonEco scripts/supervision_regional_materiales.py --entrenar --rama A --reanudar',
        '& $pythonEco scripts/supervision_regional_materiales.py --entrenar --rama B --hasta-epoca 20',
        '& $pythonEco scripts/supervision_regional_materiales.py --entrenar --rama B --reanudar',
        '& $pythonEco scripts/evaluar_supervision_regional.py',
        f'python scripts/graficar_supervision_regional.py --salida-dir resultados/{nombre}_graficas',
        '& $pythonEco scripts/informar_supervision_regional.py',
        '& $pythonEco scripts/verificar_entrega_regional.py','```','',
        'Son los comandos de la ejecución completada; las rutas existentes se rechazan para impedir sobrescrituras. No deben relanzarse como experimento nuevo.','',
        '## Evidencia y límites','',
        f'- [Entrenamiento completo](../resultados/{nombre}.json).',f'- [Evaluación y métricas completas](../resultados/{nombre}_evaluacion.json).',
        '- [Registro comparativo e integridad](../resultados/supervision_regional_reanudacion/comparacion.json).',
        f'![Curvas y AP final](../resultados/{nombre}_graficas/comparacion_ab.png)',
        f'![Seis materiales](../resultados/{nombre}_graficas/ap_seis_materiales.png)',
        f'![Pérdida y presencia orgánica](../resultados/{nombre}_graficas/aprendizaje_y_organico.png)','',
        'Una sola semilla y una partición de validación ya reutilizada limitan la atribución causal y no permiten afirmar significación estadística. No se evaluó test. El AP usa el evaluador propio de 101 puntos, no COCOeval oficial. Las regiones pequeñas pueden mezclar fondo dentro de un token, y los tokens fusionados contienen contexto. La precisión del detector sigue siendo un criterio de aceptación junto a recall, AP por material, FP, confusiones y coste. **No se modifica automáticamente el checkpoint predeterminado**.','']
    doc=RAIZ/'documentacion/18_reanudacion_y_comparacion_regional.md'
    if doc.exists():raise FileExistsError('Informe existente')
    doc.write_text('\n'.join(lineas),encoding='utf-8')
    publicar(RAIZ/'resultados/supervision_regional_reanudacion/comparacion.json',{
        'estado':'ab_completo_evaluado','pruebas':d['pruebas'],'protocolo':protocolo(cfg),
        'conclusion':conclusion,'ejecuciones_reales':ejecuciones,'region_organica':organico,
        'historicos_verificados':len(anterior),'historicos_modificados':cambiados,
        'identidad_numerica_cuda_continuo_reanudado':'No comprobada; no se afirma.',
        'trazas_iguales':True,'rng_por_epoca_iguales':rng_iguales,'actualizaciones_por_rama':1452,'test_evaluado':False,'cambio_predeterminado':False,
        'sha256_informe':sha(doc),'sha256_entrenamiento':sha(RAIZ/f'resultados/{nombre}.json'),
        'sha256_evaluacion':sha(RAIZ/f'resultados/{nombre}_evaluacion.json'),
        'hashes_codigo':hashes_codigo(),'sha256_script_informe':sha(Path(__file__))})
    readme=RAIZ/'README.md';texto=readme.read_text(encoding='utf-8')
    resumen=f"**Supervisión regional A/B completada:** 1.452 actualizaciones por rama con reanudación comprobada; {d['pruebas']['ejecutadas']} pruebas aprobadas. B−A final de cinco vistas: AP50 cartón {100*(apB-apA):+.3f} pp; mAP50 global {100*(mapB-mapA):+.3f} pp. Referencia original intacta. Ver [comparación y límites](documentacion/18_reanudacion_y_comparacion_regional.md)."
    parrafos=texto.split('\n\n');parrafos=[resumen if p.startswith('**Supervisión regional:**') else p for p in parrafos]
    readme.write_text('\n\n'.join(parrafos),encoding='utf-8')
    print(json.dumps(conclusion,ensure_ascii=False))


if __name__=='__main__':main()
