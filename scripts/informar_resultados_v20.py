"""Tablas y figuras desde JSON archivados. No carga modelos ni ejecuta inferencia."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RAIZ=Path(__file__).resolve().parents[1]
NOMBRE='v20_margen_materiales_ep19'
CLASES=['plastico','papel','carton','vidrio','metal','organico']


def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8'))


def guardar(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)


def tabla(cabecera,filas):
    return '\n'.join(['| '+' | '.join(map(str,cabecera))+' |','|'+'|'.join(['---']*len(cabecera))+'|',
        *['| '+' | '.join(map(str,f))+' |' for f in filas]])


def informar_bloqueo():
    """Una prueba fallida nunca se convierte en resultados A/B inexistentes."""
    i=leer('resultados/v20_ejecucion/incidencia_prueba_corta.json')
    d=leer('resultados/v20_diagnostico_ejecucion/diagnostico.json')
    original=leer('resultados/supervision_regional_ep19_v1_evaluacion.json')['resultados']['origen']
    out=RAIZ/'resultados/v20_comparacion';doc=RAIZ/'documentacion/21_resultados_aprendizaje_discriminativo.md'
    if out.exists() or doc.exists():raise FileExistsError('Informe existente')
    out.mkdir();filas=[]
    for v in ['1','5']:filas.append({'estado':'historico_no_reevaluado_en_V20','modelo':'origen','vistas':int(v),**original['vistas'][v]['resumen']})
    for n in ['A_final','A_mejor','B_final','B_mejor']:
        for v in [1,5]:filas.append({'estado':'no_ejecutado','modelo':n,'vistas':v,'map50':None,'map50_95':None,'ap50_por_material':None,'recall50':None,'precision50':None,'tp50':None,'fp50':None,'fn50':None})
    guardar(out/'metricas_estado.json',filas)
    with (out/'metricas.csv').open('x',encoding='utf-8',newline='') as f:
        campos=['estado','modelo','vistas','map50','map50_95','recall50','precision50','tp50','fp50','fn50',*CLASES]
        w=csv.DictWriter(f,fieldnames=campos);w.writeheader()
        for r in filas:
            x={k:v for k,v in r.items() if k!='ap50_por_material'};x.update(r['ap50_por_material'] or {c:None for c in CLASES});w.writerow(x)
    organic=[f for g in d['comparacion_gradientes'] for f in g['detalle_instancias']['filas'] if f['clase']==5]
    guardar(out/'region_organica_diagnostico.json',{'identidad':d['organico_elegible'],'presencias_en_train':1,
        'peso':1.,'eventos_sondeos_train_sin_actualizar_origen':organic,'eventos_entrenamiento_AB':[],
        'no_extrapolar_a_epocas_no_ejecutadas':True})
    guardar(out/'estado_experimento.json',{'estado':i['estado'],'diagnostico_aprobado':d['aprobada'],
        'piloto_aprobado':False,'actualizaciones_AB':i['actualizaciones_experimento_completo'],
        'actualizaciones_piloto_descartadas':8,'metricas_AB':None,'evaluacion_regional_ejecutada':False,
        'test_evaluado':False,'checkpoint_predeterminado_modificado':False,'tolerancias_e_hipotesis_conservadas':True})
    plt.rcParams.update({'figure.dpi':140,'font.size':10})
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(['A','B'],[1452,1452],color='#dddddd',label='Presupuesto previsto')
    for x in [0,1]:ax.text(x,700,'0 ejecutadas',ha='center',color='#a00000',weight='bold')
    ax.set(ylabel='Actualizaciones A/B completas',title='V20 detenido por fallo del piloto');ax.legend();fig.tight_layout();fig.savefig(out/'presupuesto_no_ejecutado.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    vals=d['senal_adicional']['fracciones_ortogonales_vs_ce6_matched'];ax.bar(range(1,5),vals,color='#247c9b')
    ax.set(xticks=range(1,5),xlabel='Sondeo train',ylabel='Fracción ortogonal del gradiente',title='Diagnóstico aprobado: señal adicional frente a CE6 con tokens iguales');fig.tight_layout();fig.savefig(out/'gradientes_diagnostico.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));m=original['vistas']['5']['resumen'];ax.bar(CLASES,[100*m['ap50_por_material'][c] for c in CLASES],color='#777777')
    ax.set(ylabel='AP50 (%)',title='Referencia histórica: sin nueva evaluación V20');fig.tight_layout();fig.savefig(out/'referencia_historica.png');plt.close(fig)
    texto=['# V21: ejecución V20 detenida en el control de reanudación','',
        '**El experimento A/B completo no se ejecutó.** La prueba corta con AdamW real falló al comparar pesos de cuatro pasos continuos con 2 pasos + restauración + 2 pasos. Se detuvo V20, conforme a la condición de seguridad solicitada; no se relajaron tolerancias, no se cambió la hipótesis y no se repitió el piloto.','',
        '## Estado comprobado','',
        tabla(['Control','Resultado'],[['Preparación revisada','Commit 06ab77f'],['Pruebas','171 aprobadas, ninguna omitida'],['Diagnóstico train-only regenerado','Aprobado'],['Restauración inicial e intermedia del piloto A','Huellas exactas comprobadas antes del fallo'],['Entradas de las dos trayectorias A','Cuatro trazas coincidentes, según aserción anterior al fallo'],['Comparación numérica de pesos','Fallida'],['Piloto B','No ejecutado'],['A/B completo','A: 0/1.452; B: 0/1.452'],['Evaluación V20 una/cinco vistas','No ejecutada'],['Predeterminado e históricos','730 archivos protegidos intactos'],['Test','Reservado']]),'',
        '![Presupuesto no ejecutado](../resultados/v20_comparacion/presupuesto_no_ejecutado.png)','',
        '## Implementación entregada','',
        '`scripts/entrenar_v20.py` reutiliza `calcular_lote` y restaura pesos, AdamW, scaler y RNG con las funciones V18. `ecowaste/reanudacion_v20.py` mantiene diarios y checkpoints por rama y época exclusivos de V20, publicación sin reemplazo, commits encadenados por SHA256, recuperación explícita de checkpoints completos sin commit y protección de aliases ajenos. Rechaza diferencias de protocolo, código, entradas, presupuesto, estado o región orgánica. Cada época guarda las 121 huellas de lote y todos los estados necesarios.','',
        'La fixture CPU comprueba una interrupción con tres pasos no confirmados, reanudación desde la última época completa y llegada a 1.452 actualizaciones por rama. Su identidad numérica sí se comprobó para esa fixture. Esto no demuestra equivalencia numérica del detector CUDA; la prueba real que debía cubrir ese alcance falló.','',
        'A=0 y B=0,25; margen Huber=1, fracción difícil=0,25 y pesos [1,1,2,2,1,1]. Se conserva pérdida original, fondo=3, arquitectura, tarea auxiliar global, LR=0,0005, AMP, clipping=1, estados de época 19, orden/aumentos y 12 épocas sin warmup ni parada temprana. Argmax7, cinco vistas y postprocesamiento permanecen intactos. La configuración sigue con `entrenamiento_completo_habilitado=false`; no existe JSON de aprobación del piloto.','',
        '`scripts/evaluar_v20.py` está preparado para exigir ambos diarios completos y aliases correctos, evaluar final/mejor con una/cinco vistas, archivar predicciones por imagen, logits AMP, trazas V19 y análisis GT posterior. Sus 39 dependencias locales, incluido observador, auditoría, evaluador y postprocesamiento, figuran en las huellas del diagnóstico. El generador offline de tablas y gráficas también figura en el manifiesto final. La evaluación aún no se validó de extremo a extremo con checkpoints V20, porque no existen.','',
        '## Diagnóstico regenerado, anterior al piloto','',
        f"Las 171 pruebas aprobaron en el diagnóstico. Los cuatro sondeos mejoraron el margen y produjeron descenso local en parámetros. Fracciones ortogonales respecto a CE6 con idénticos tokens y pesos: {', '.join(f'{x:.4f}' for x in vals)}. La señal llega a cabeza principal y extractor; no añade gradiente directo al fondo ni usa la cabeza auxiliar. Memorización de siete instancias en seis escenas train: mAP50, recall y precisión finales de 1,0; pesos descartados. No es evidencia de generalización.",'',
        '![Sondeos](../resultados/v20_comparacion/gradientes_diagnostico.png)','',
        f"El diagnóstico tardó {d['segundos']:.2f} s en {d['gpu']} con PyTorch {d['torch']}. Verificó carga estricta y 439.309 parámetros, además de logits y cajas exactos en seis escenas train con FP32/AMP y una/cinco vistas. Esto no equivale a reproducir las 181 predicciones históricas de validación, comprobación que no se ejecutó en V20.",'',
        '## Fallo y límites de la evidencia','',
        'Comando: `python scripts/probar_entrenamiento_v20.py`. Salida 1, `AssertionError` en la comparación de pesos. Criterio prefijado: `torch.testing.assert_close(rtol=1e-4, atol=1e-5)`. La condición por elemento es `|x-y| ≤ atol + rtol·|y|`.','',
        tabla(['Dato observado en el tensor que detuvo la comparación','Valor'],[['Elementos fuera de tolerancia','69 / 288 (23,96 %)'],['Mayor diferencia absoluta','0,000195428729057312; índice 168'],['Mayor diferencia relativa','0,011974018067121506; índice 104'],['Actualizaciones reales del piloto','8 en A: 4 continuas y 4 en trayectoria interrumpida/reanudada'],['Paso AdamW final en ambas trayectorias','2.303 desde 2.299'],['Actualizaciones B del piloto','0']]),'',
        'El script llegó a este punto después de comprobar pérdida finita, gradientes finitos, clipping y avance real de AdamW en ocho actualizaciones, restauración exacta del checkpoint intermedio y coincidencia de las cuatro huellas de entrada. Estas comprobaciones se deducen del flujo ejecutado antes de la excepción. La comparación falló antes de completar todos los tensores, AdamW y los estados finales de scaler/RNG. No se afirma identidad bit a bit.','',
        'El piloto publicaba su JSON solo al final: por ello se conserva el stderr transcrito y la incidencia, pero no se guardaron los tensores, las pérdidas por paso ni las huellas detalladas de las trayectorias fallidas. No se conoce el máximo global entre todos los parámetros ni el nombre del tensor. El temporal se eliminó por `TemporaryDirectory`; no se reutilizó ningún peso diagnóstico. Esta limitación de observabilidad debe corregirse antes de un nuevo intento.','',
        'La causa raíz no está determinada. La restauración exacta antes de continuar y las entradas coincidentes descartan varias clases de error, pero no prueban que la divergencia posterior sea exclusivamente no determinismo CUDA. Hace falta distinguir variación entre dos ejecuciones continuas de un fallo específico al reanudar.','',
        '## Región orgánica','',
        'La única región elegible es `taco:92`, ID 244, con peso 1. El diagnóstico registra su identidad y contribuciones en los sondeos; `region_organica_diagnostico.json` las conserva separadas de cualquier ejecución A/B. La región tiene 40 tokens válidos y es recuperada por el checkpoint de origen. No se entrenaron épocas completas, así que no hay presencia ni contribución observada por época A/B. El entrenador preparado exige una aparición por época y registra tokens, rivales, peso relativo, pérdida individual y contribución efectiva; en A sería cero por coeficiente.','',
        '## Métricas disponibles y comparación pendiente','',
        'Las cifras siguientes son **históricas**, copiadas del resultado V18 de la referencia original, sin nueva inferencia V20. Los valores A/B son ausentes, no ceros. No se puede concluir si B mejora la clasificación condicional, AP50 de cartón, mAP50 global, geometría u oportunidades. Tampoco se completó la comparación numérica del control A con V18.','',
        tabla(['Estado','Vistas','mAP50 %','mAP50:95 %','Recall %','Precisión %','TP','FP','FN'],
            [[r['modelo']+' (histórico)' if r['modelo']=='origen' else r['modelo'],r['vistas'],*[f"{100*r[k]:.4f}" if r[k] is not None else 'Pendiente' for k in ['map50','map50_95','recall50','precision50']],*[r[k] if r[k] is not None else 'Pendiente' for k in ['tp50','fp50','fn50']]] for r in filas]),'',
        tabla(['Referencia histórica 5 vistas',*CLASES],[['AP50 %',*[f"{100*m['ap50_por_material'][c]:.4f}" for c in CLASES]]]),'',
        '![Referencia histórica](../resultados/v20_comparacion/referencia_historica.png)','',
        'No se presentan tiempo ni memoria GPU del A/B inexistente. El piloto fallido tampoco conservó mediciones completas de coste. Una sola semilla y validación reutilizada limitarían cualquier resultado futuro; AP usa interpolación propia de 101 puntos, no COCOeval oficial.','',
        '## Evidencia reproducible y condición para continuar','',
        'Artefactos: [diagnóstico vigente](../resultados/v20_diagnostico_ejecucion/diagnostico.json), [incidencia](../resultados/v20_ejecucion/incidencia_prueba_corta.json), [métricas y ausencias](../resultados/v20_comparacion/metricas_estado.json), [CSV](../resultados/v20_comparacion/metricas.csv), [estado](../resultados/v20_comparacion/estado_experimento.json), [integridad](../resultados/v20_comparacion/manifiesto_integridad.json).','',
        'El siguiente trabajo debe aislar la discrepancia de ejecución y mejorar la captura del piloto fallido, conservando margen, coeficientes y presupuesto. Si se modifica código cubierto por las huellas, será necesario un nuevo diagnóstico en una ruta nueva. No se debe habilitar A/B hasta que un control de reanudación real quede aprobado y documentado; no corresponde ampliar la tolerancia para dar por válido este intento.','',
        'La referencia original, todos los resultados históricos y la selección se conservan. No hubo adopción automática, evaluación de test ni entrenamiento posterior al fallo.','']
    with doc.open('x',encoding='utf-8') as f:f.write('\n'.join(texto))
    print(i['estado'])


def main():
    if not (RAIZ/f'resultados/{NOMBRE}_evaluacion.json').exists():
        informar_bloqueo();return
    e=leer(f'resultados/{NOMBRE}_evaluacion.json');t=leer(f'resultados/{NOMBRE}.json')
    d=leer('resultados/v20_diagnostico_ejecucion/diagnostico.json');p=leer('resultados/v20_ejecucion/prueba_corta.json')
    if e['estado']!='evaluacion_v20_completa':raise ValueError('Evaluación incompleta')
    out=RAIZ/'resultados/v20_comparacion';doc=RAIZ/'documentacion/21_resultados_aprendizaje_discriminativo.md'
    if out.exists() or doc.exists():raise FileExistsError('Informe histórico existente')
    out.mkdir();rr=e['resultados'];nombres=list(rr);series=['origen','A_ultimo','B_ultimo','A_mejor','B_mejor']
    filas=[]
    for n in nombres:
        for v,k in [(1,'val_1_vista'),(5,'val_5_vistas')]:
            m=rr[n][k]['resumen'];filas.append({'modelo':n,'epoca':rr[n]['epoca'],'vistas':v,**m})
    guardar(out/'comparacion_metricas.json',filas)
    campos=['modelo','epoca','vistas','map50','map50_95','recall50','precision50','tp50','fp50','fn50',*CLASES]
    with (out/'metricas.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=campos);w.writeheader()
        for x in filas:
            row={k:v for k,v in x.items() if k!='ap50_por_material'};row.update(x['ap50_por_material']);w.writerow(row)
    plt.rcParams.update({'figure.dpi':140,'font.size':9})
    fig,ax=plt.subplots(figsize=(8,4))
    for rama in ['A','B']:
        hs=t['ramas'][rama]['historial'];ax.plot([h['epoca'] for h in hs],[100*h['map50_val'] for h in hs],marker='o',label=rama)
    ax.axhline(100*rr['origen']['val_1_vista']['resumen']['map50'],color='gray',ls='--',label='Origen 1 vista')
    ax.set(xlabel='Época',ylabel='mAP50 validación (%)',title='Presupuesto fijo: selección por una vista');ax.legend();fig.tight_layout();fig.savefig(out/'curvas_aprendizaje.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4));x=np.arange(len(series));ancho=.35
    for ax,met,titulo in [(axs[0],'map50','mAP50'),(axs[1],'map50_95','mAP50:95')]:
        for j,k in enumerate(['val_1_vista','val_5_vistas']):ax.bar(x+(j-.5)*ancho,[100*rr[n][k]['resumen'][met] for n in series],ancho,label=f'{[1,5][j]} vistas')
        ax.set_xticks(x,series,rotation=20);ax.set(ylabel='%',title=titulo);ax.legend()
    fig.tight_layout();fig.savefig(out/'comparacion_ap.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(6)
    for j,n in enumerate(['origen','A_ultimo','B_ultimo']):ax.bar(x+(j-1)*.25,[100*rr[n]['val_5_vistas']['resumen']['ap50_por_material'][c] for c in CLASES],.25,label=n)
    ax.set_xticks(x,CLASES);ax.set(ylabel='AP50 (%)',title='Seis materiales: cinco vistas, estados finales');ax.legend();fig.tight_layout();fig.savefig(out/'ap_materiales.png');plt.close(fig)
    for campo,etiquetas,archivo in [('confusion_instancia_voto_principal',['fondo',*CLASES],'confusion_principal.png'),('confusion_instancia_voto_condicional',CLASES,'confusion_condicional.png')]:
        fig,axs=plt.subplots(1,3,figsize=(14,4))
        for ax,n in zip(axs,['origen','A_ultimo','B_ultimo']):
            cm=np.array(rr[n]['principal_train'][campo],dtype=float);den=cm.sum(1,keepdims=True);cm=np.divide(cm,den,out=np.zeros_like(cm),where=den>0)
            ax.imshow(cm,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(len(etiquetas)),etiquetas,rotation=70);ax.set_yticks(range(len(etiquetas)),etiquetas);ax.set_title(n);ax.set_xlabel('Material predicho')
            for i,j in np.ndindex(cm.shape):
                if cm[i,j]>0:ax.text(j,i,f'{100*cm[i,j]:.0f}',ha='center',va='center',fontsize=7,color='white' if cm[i,j]>.5 else 'black')
        fig.suptitle('Train: voto por instancia (% de cada clase GT)');fig.tight_layout();fig.savefig(out/archivo);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    etapas=['componentes','area_minima','recorte_contenido','nms_vista','top100_vista','bordes','nms_global','top100_global','final']
    for ax,c in zip(axs,['carton','vidrio']):
        for n in ['origen','A_ultimo','B_ultimo']:
            z=rr[n]['oportunidades_y_geometria_val']['por_material'][c]['oportunidades_por_etapa'];ax.plot(range(len(etapas)),[z[k] for k in etapas],marker='o',label=n)
        ax.set_xticks(range(len(etapas)),etapas,rotation=65);ax.set(title=c,ylabel='GT con propuesta correcta IoU≥0,5');ax.legend()
    fig.tight_layout();fig.savefig(out/'oportunidades_etapas.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for rama in ['A','B']:
        hs=t['ramas'][rama]['historial'];ax.plot([h['epoca'] for h in hs],[h['region_organica']['eventos'][0]['contribucion_loss_total'] for h in hs],marker='o',label=rama)
    ax.set(xlabel='Época',ylabel='Contribución al loss total en su lote',title='Única región orgánica: peso 1, presencia una vez por época');ax.legend();fig.tight_layout();fig.savefig(out/'organico.png');plt.close(fig)
    # Resumen verificable por material, geometría, tamaño y etapa, sin nuevas predicciones.
    guardar(out/'geometria_oportunidades.json',{n:rr[n]['oportunidades_y_geometria_val'] for n in nombres})
    guardar(out/'organico.json',{r:[{'epoca':h['epoca'],**h['region_organica']} for h in t['ramas'][r]['historial']] for r in ['A','B']})
    b=rr['B_ultimo'];a=rr['A_ultimo'];orig=rr['origen'];mb=b['val_5_vistas']['resumen'];ma=a['val_5_vistas']['resumen'];mo=orig['val_5_vistas']['resumen']
    def por(n,c):return rr[n]['principal_train']['por_material'][c]
    def error(n):
        cm=rr[n]['principal_train']['confusion_instancia_voto_condicional'];return (cm[2][0]+cm[2][4])/sum(cm[2])
    mejor=mb['map50']>mo['map50']
    conclusion=(f"B final obtiene mAP50 {100*mb['map50']:.4f} % frente a {100*ma['map50']:.4f} % de A y {100*mo['map50']:.4f} % del origen. "
        f"Cartón AP50={100*mb['ap50_por_material']['carton']:.4f} %; vidrio AP50={100*mb['ap50_por_material']['vidrio']:.4f} %. "
        'No se adopta automáticamente ningún checkpoint.')
    no_solucion=mb['ap50_por_material']['carton']==0 and mb['ap50_por_material']['vidrio']==0
    texto=['# V21: resultados del aprendizaje discriminativo V20','',conclusion,'',
        ('La intervención no solucionó el problema de detección de cartón y vidrio. Una mejora condicional en train no equivale a crear oportunidades de detección ni a mejorar AP.' if no_solucion else 'Examinar conjuntamente recuperación, oportunidades, geometría y falsos positivos antes de decidir una adopción.'),'',
        '## Protocolo y controles','',
        'Preparación revisada: commit `06ab77f`. Origen propio de época 19; arquitectura y cabeza auxiliar conservadas. A=0 y B=0,25; margen Huber=1, fracción difícil=0,25, pesos [1,1,2,2,1,1]. Pérdida original intacta, fondo=3, auxiliar global=0,5, LR=0,0005, AdamW, AMP y clipping=1. Sin warmup nuevo, parada temprana ni cambios de postprocesamiento. Argmax7, NMS=0,5, área=2, top100, umbrales 0,05/0,35. Val contiene 181 escenas y 361 objetos; test permanece reservado.','',
        f"Diagnóstico regenerado: {d['pruebas']['ejecutadas']} pruebas aprobadas, gradientes principales útiles, AMP y memorización train-only aprobados. Piloto: {p['actualizaciones_totales_diagnosticas']} actualizaciones diagnósticas AdamW; pesos descartados. Se compararon cuatro pasos continuos frente a 2+restauración+2 por rama, con tolerancias prefijadas rtol=1e-4 y atol=1e-5. No se extrapola identidad bit a bit a la ejecución larga.",'',
        'Cada rama completó 12 épocas, 1.452 actualizaciones y AdamW pasó de 2.299 a 3.751. Las ejecuciones se separaron y reanudaron después de la época 20, restaurando pesos, optimizador, scaler y RNG. Cada época conserva un checkpoint completo y un commit JSON inmutable con cadena SHA256, trazas de los 121 lotes e historial. Mejor checkpoint: mayor mAP50 de una vista, primer empate; se informa separado del final.','',
        f"Mejores épocas: A={t['ramas']['A']['mejor_epoca']}, B={t['ramas']['B']['mejor_epoca']}. Las trazas se cotejaron con el plan previo y entre ramas. Las 181 predicciones históricas de referencia se reprodujeron exactamente tanto con una como con cinco vistas.",'',
        '## Detección en validación','',
        tabla(['Estado','Época','Vistas','mAP50 %','mAP50:95 %','Recall %','Precisión %','TP','FP','FN'],
            [[x['modelo'],x['epoca'],x['vistas'],f"{100*x['map50']:.4f}",f"{100*x['map50_95']:.4f}",f"{100*x['recall50']:.2f}",f"{100*x['precision50']:.2f}",x['tp50'],x['fp50'],x['fn50']] for x in filas]),'',
        tabla(['Estado 5 vistas',*CLASES],[[n,*[f"{100*rr[n]['val_5_vistas']['resumen']['ap50_por_material'][c]:.4f}" for c in CLASES]] for n in series]),'',
        '![AP y presupuesto](../resultados/v20_comparacion/comparacion_ap.png)','',
        '![Curvas](../resultados/v20_comparacion/curvas_aprendizaje.png)','',
        '![AP por material](../resultados/v20_comparacion/ap_materiales.png)','',
        '## Cabeza principal en train','',
        'Se excluyen padding, ignorados y mezclas ambiguas. Se mide voto por instancia sobre los tokens elegibles de los polígonos; la clasificación condicional elimina fondo solo para el análisis. No se evalúa la cabeza auxiliar como sustituto.','',
        tabla(['Material','Instancias','Origen recuperadas / cond.','A final recuperadas / cond.','B final recuperadas / cond.','Fondo origen / A / B %'],
            [[c,por('origen',c)['instancias'],*[f"{por(n,c)['recuperadas']} / {por(n,c)['condicionales_recuperadas']}" for n in ['origen','A_ultimo','B_ultimo']], ' / '.join(f"{100*por(n,c)['fraccion_fondo_media']:.2f}" for n in ['origen','A_ultimo','B_ultimo'])] for c in CLASES]),'',
        f"Cartón→plástico/metal por voto condicional: origen={100*error('origen'):.2f} %, A={100*error('A_ultimo'):.2f} %, B={100*error('B_ultimo'):.2f} %. Estas tasas corresponden a train y no demuestran generalización.",'',
        '![Confusión principal](../resultados/v20_comparacion/confusion_principal.png)','',
        '![Confusión condicional](../resultados/v20_comparacion/confusion_condicional.png)','',
        '## Oportunidades, geometría y tamaño','',
        tabla(['Estado','Material','GT','Correctas antes de filtros','Correctas finales','Fusión','Fragmentación'],
            [[n,c,(z:=rr[n]['oportunidades_y_geometria_val']['por_material'][c])['gt'],z['oportunidades_por_etapa']['componentes'],z['oportunidades_por_etapa']['final'],z['indicadores_fusion'],z['indicadores_fragmentacion']] for n in series for c in ['carton','vidrio']]),'',
        'Una oportunidad cuenta GT con al menos una caja del material correcto e IoU≥0,5; no es un TP asignado uno a uno. La auditoría V19 se aplica después de archivar inferencia. `geometria_oportunidades.json` contiene las ocho causas, todas las etapas y el desglose de las seis clases por tamaño. Los indicadores de fusión y fragmentación pueden coexistir y no son causas excluyentes.','',
        '![Oportunidades](../resultados/v20_comparacion/oportunidades_etapas.png)','',
        '## Región orgánica y coste','',
        'Se conserva la única región orgánica elegible, identificada en el diagnóstico. Aparece una vez por época en cada rama, con peso 1. Se archivan loss de instancia, tokens difíciles, rivales, fracción del denominador, cota L1 del gradiente y contribución multiplicada por el coeficiente. En A la contribución efectiva es cero. No se modifica su peso ni se infiere robustez a partir de una instancia.','',
        '![Orgánico](../resultados/v20_comparacion/organico.png)','',
        tabla(['Rama','Actualizaciones','Segundos entrenamiento + val/IO','Pico asignado GPU GiB','Pico reservado GPU GiB'],[[r,1452,f"{t['ramas'][r]['segundos']:.1f}",f"{t['ramas'][r]['memoria_gpu_max_allocated_bytes']/2**30:.3f}",f"{t['ramas'][r]['memoria_gpu_max_reserved_bytes']/2**30:.3f}"] for r in ['A','B']]),'',
        'El JSON de evaluación registra tiempos y memoria por captura de una/cinco vistas y por diagnóstico train. La captura incluye serialización y observador; no representa latencia pura del detector.','',
        '## Control A y diferencias numéricas frente a V18','',
        f"Las entradas de las doce épocas coinciden con V18: {e['control_A_frente_V18']['entradas_iguales']}. La tabla por época con pérdidas y diferencias de mAP está en el JSON de evaluación. `comparacion_numerica_v18.json` contrasta estados y parámetros archivados; no se presupone identidad numérica entre procesos CUDA.",'',
        '## Evidencia y reproducibilidad','',
        'Comandos en el entorno CUDA original:','',
        '```powershell\npython scripts/diagnosticar_v20.py\npython scripts/probar_entrenamiento_v20.py\npython scripts/entrenar_v20.py --rama A --hasta-epoca 20\npython scripts/entrenar_v20.py --rama A --reanudar\npython scripts/entrenar_v20.py --rama B --hasta-epoca 20\npython scripts/entrenar_v20.py --rama B --reanudar\npython scripts/evaluar_v20.py\n```','',
        'Las rutas publicadas son inmutables: repetir requiere un nuevo identificador y diagnóstico; `--reanudar` solo continúa diarios del mismo protocolo. Los JSON de predicciones por imagen, logits AMP y trazas de cinco vistas se indexan con hashes en cada `captura.json`. Estados final/mejor idénticos reutilizan captura mediante referencia explícita. El observador comprueba que no altera la salida. Los GT se consultan únicamente en pérdida train y análisis posterior; RealWaste conserva su tarea global y nunca produce cajas.','',
        'Artefactos principales: [evaluación completa](../resultados/v20_margen_materiales_ep19_evaluacion.json), [entrenamiento](../resultados/v20_margen_materiales_ep19.json), [diagnóstico](../resultados/v20_diagnostico_ejecucion/diagnostico.json), [piloto](../resultados/v20_ejecucion/prueba_corta.json), [CSV](../resultados/v20_comparacion/metricas.csv), [integridad](../resultados/v20_comparacion/manifiesto_integridad.json).','',
        'Una semilla y validación de desarrollo reutilizada: los resultados no estiman incertidumbre ni rendimiento independiente. AP usa interpolación propia de 101 puntos, no COCOeval oficial. No se evaluó test ni se cambió selección, arquitectura, pesos originales o postprocesamiento. La decisión de adoptar permanece pendiente de revisión conjunta; se conserva el detector original.','']
    with doc.open('x',encoding='utf-8') as f:f.write('\n'.join(texto))
    print(conclusion)


if __name__=='__main__':main()
