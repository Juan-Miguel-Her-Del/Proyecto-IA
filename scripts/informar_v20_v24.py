"""Resultados finales desde JSON: distingue backend histórico y comparación controlada."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
RAIZ=Path(__file__).resolve().parents[1]
ROOT=RAIZ/'resultados/v24_v20_evaluacion'
CLASES=['plastico','papel','carton','vidrio','metal','organico']


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def tabla(cab,filas):return '\n'.join(['| '+' | '.join(cab)+' |','|'+'|'.join(['---']*len(cab))+'|',*['| '+' | '.join(map(str,f))+' |' for f in filas]])


def guardar(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)


def main():
    cfg=leer(RAIZ/'configuracion/v20_backend_v23_1_completo_eval24_ep19.json');nombre=cfg['nombre']
    e=leer(RAIZ/f'resultados/{nombre}_evaluacion.json');t=leer(RAIZ/f'resultados/{nombre}.json');rr=e['resultados']
    h=e['referencia_historica_backend_original'];et=leer(RAIZ/cfg['control_etapa_20']);d=leer(RAIZ/cfg['diagnostico'])
    revision=leer(ROOT/'revision_cpu_diagnostico/diagnostico.json')
    out=ROOT/'informe';doc=RAIZ/'documentacion/24_resultados_v20_backend_determinista.md'
    if out.exists() or doc.exists():raise FileExistsError('Informe existente')
    if e['estado']!='evaluacion_v20_completa' or any(t['ramas'][r]['actualizaciones']!=1452 for r in ['A','B']):raise ValueError('Experimento incompleto')
    out.mkdir();filas=[]
    for v in ['1','5']:filas.append({'estado':'historico_original','epoca':19,'vistas':int(v),**h['metricas'][v]})
    for n,x in rr.items():
        for v,k in [(1,'val_1_vista'),(5,'val_5_vistas')]:filas.append({'estado':n,'epoca':x['epoca'],'vistas':v,**x[k]['resumen']})
    guardar(out/'metricas.json',filas)
    with (out/'metricas.csv').open('x',encoding='utf-8',newline='') as f:
        campos=['estado','epoca','vistas','map50','map50_95','recall50','precision50','tp50','fp50','fn50',*CLASES]
        w=csv.DictWriter(f,fieldnames=campos);w.writeheader()
        for x in filas:w.writerow({**{k:v for k,v in x.items() if k!='ap50_por_material'},**x['ap50_por_material']})
    plt.rcParams.update({'figure.dpi':140,'font.size':9})
    fig,ax=plt.subplots(figsize=(8,4))
    for r in ['A','B']:
        hs=t['ramas'][r]['historial'];ax.plot([s['epoca'] for s in hs],[100*s['map50_val'] for s in hs],marker='o',label=r)
    ax.axhline(100*rr['origen']['val_1_vista']['resumen']['map50'],ls='--',color='gray',label='Origen, mismo backend')
    ax.set(xlabel='Época',ylabel='mAP50 validación (%)',title='Presupuesto fijo, criterio de mejor: una vista');ax.legend();fig.tight_layout();fig.savefig(out/'curvas.png');plt.close(fig)
    ns=['origen','A_ultimo','A_mejor','B_ultimo','B_mejor'];fig,ax=plt.subplots(figsize=(9,4));xx=np.arange(len(ns))
    for j,k in enumerate(['val_1_vista','val_5_vistas']):ax.bar(xx+(j-.5)*.35,[100*rr[n][k]['resumen']['map50'] for n in ns],.35,label=f'{[1,5][j]} vistas')
    ax.axhline(100*h['metricas']['5']['map50'],ls='--',color='gray',label='Histórico cinco vistas')
    ax.set_xticks(xx,ns,rotation=20);ax.set(ylabel='mAP50 (%)',title='Referencia y A/B con backend determinista');ax.legend();fig.tight_layout();fig.savefig(out/'map50.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));xx=np.arange(6)
    for j,n in enumerate(['origen','A_ultimo','B_ultimo']):ax.bar(xx+(j-1)*.25,[100*rr[n]['val_5_vistas']['resumen']['ap50_por_material'][c] for c in CLASES],.25,label=n)
    ax.set_xticks(xx,CLASES);ax.set(ylabel='AP50 (%)',title='Materiales, estados finales, cinco vistas');ax.legend();fig.tight_layout();fig.savefig(out/'ap_materiales.png');plt.close(fig)
    for campo,labels,archivo in [('confusion_instancia_voto_principal',['fondo',*CLASES],'principal'),('confusion_instancia_voto_condicional',CLASES,'condicional')]:
        fig,axs=plt.subplots(1,3,figsize=(14,4))
        for ax,n in zip(axs,['origen','A_ultimo','B_ultimo']):
            cm=np.array(rr[n]['principal_train'][campo],dtype=float);den=cm.sum(1,keepdims=True);cm=np.divide(cm,den,out=np.zeros_like(cm),where=den>0)
            ax.imshow(cm,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(len(labels)),labels,rotation=70);ax.set_yticks(range(len(labels)),labels);ax.set_title(n)
            for i,j in np.ndindex(cm.shape):
                if cm[i,j]>0:ax.text(j,i,f'{100*cm[i,j]:.0f}',ha='center',va='center',fontsize=7,color='white' if cm[i,j]>.5 else 'black')
        fig.suptitle('Cabeza principal, train: porcentaje por clase GT');fig.tight_layout();fig.savefig(out/f'confusion_{archivo}.png');plt.close(fig)
    etapas=['componentes','area_minima','recorte_contenido','nms_vista','top100_vista','bordes','nms_global','top100_global','final']
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,c in zip(axs,['carton','vidrio']):
        for n in ['origen','A_ultimo','B_ultimo']:
            z=rr[n]['oportunidades_y_geometria_val']['por_material'][c]['oportunidades_por_etapa'];ax.plot(range(len(etapas)),[z[k] for k in etapas],marker='o',label=n)
        ax.set_xticks(range(len(etapas)),etapas,rotation=65);ax.set(title=c,ylabel='GT con caja correcta e IoU≥0,5');ax.legend()
    fig.tight_layout();fig.savefig(out/'oportunidades.png');plt.close(fig)
    guardar(out/'geometria_y_tamanos.json',{n:x['oportunidades_y_geometria_val'] for n,x in rr.items()})
    registros={x['id']:x for x in [json.loads(s) for s in (RAIZ/'datos/procesados/deteccion/val.jsonl').read_text(encoding='utf-8').splitlines()]}
    objetos=leer((RAIZ/rr['origen']['captura']).parent/'objetos_val_5v.json')
    for material in ['carton','vidrio']:
        gt=next(x for x in objetos if x['material']==material);reg=registros[gt['imagen']]
        with Image.open(RAIZ/reg['ruta']) as im:im=im.convert('RGB');imagen=np.array(im)
        alto,ancho=imagen.shape[:2];escala=384/max(ancho,alto);nw=max(1,round(ancho*escala));nh=max(1,round(alto*escala))
        px,py=(384-nw)//2,(384-nh)//2;sx,sy=nw/ancho,nh/alto;gx,gy,gw,gh=gt['bbox_original']
        fig,axs=plt.subplots(1,3,figsize=(12,4))
        for ax,n in zip(axs,['origen','A_ultimo','B_ultimo']):
            ax.imshow(imagen);ax.add_patch(Rectangle((gx,gy),gw,gh,fill=False,edgecolor='lime',lw=2));ax.text(gx,gy,'GT '+material,color='lime',fontsize=8)
            pp=next(x for x in leer((RAIZ/rr[n]['captura']).parent/'predicciones_5v_val.json') if x['id']==gt['imagen'])
            for b,s,c in list(zip(pp['boxes'],pp['scores'],pp['labels']))[:20]:
                if s<cfg['umbral_visual']:continue
                x0,y0,x1,y1=(b[0]*384-px)/sx,(b[1]*384-py)/sy,(b[2]*384-px)/sx,(b[3]*384-py)/sy
                ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='orange',lw=1))
                if gx-2*gw<=x0<=gx+3*gw and gy-2*gh<=y0<=gy+3*gh:ax.text(x0,y0,f'{CLASES[c]} {s:.2f}',color='orange',fontsize=7)
            ax.set_xlim(max(0,gx-2*gw),min(ancho,gx+3*gw));ax.set_ylim(min(alto,gy+3*gh),max(0,gy-2*gh));ax.set_title(n);ax.axis('off')
        fig.suptitle(f"Primer GT {material} por orden de val: {gt['imagen']}; verde=GT, naranja=predicción ≥0,35");fig.tight_layout();fig.savefig(out/f'ejemplo_{material}.png');plt.close(fig)
    org={r:[{'epoca':s['epoca'],**s['region_organica']} for s in t['ramas'][r]['historial']] for r in ['A','B']};guardar(out/'organico.json',org)
    fig,ax=plt.subplots(figsize=(8,4))
    for r in ['A','B']:ax.plot([s['epoca'] for s in org[r]],[s['eventos'][0]['contribucion_loss_total'] for s in org[r]],marker='o',label=r)
    ax.set(xlabel='Época',ylabel='Contribución efectiva en su lote',title='Región orgánica única, peso 1');ax.legend();fig.tight_layout();fig.savefig(out/'organico.png');plt.close(fig)
    def mat(n,c):return rr[n]['principal_train']['por_material'][c]
    def error(n):
        cm=rr[n]['principal_train']['confusion_instancia_voto_condicional'];return (cm[2][0]+cm[2][4])/sum(cm[2])
    ma=rr['A_ultimo']['val_5_vistas']['resumen'];mb=rr['B_ultimo']['val_5_vistas']['resumen'];mo=rr['origen']['val_5_vistas']['resumen']
    cond_a=mat('A_ultimo','carton')['condicionales_recuperadas'];cond_b=mat('B_ultimo','carton')['condicionales_recuperadas']
    pre_a=rr['A_ultimo']['oportunidades_y_geometria_val']['por_material']['carton']['oportunidades_por_etapa']['componentes']
    pre_b=rr['B_ultimo']['oportunidades_y_geometria_val']['por_material']['carton']['oportunidades_por_etapa']['componentes']
    no_soluciona=cond_b>cond_a and mb['ap50_por_material']['carton']<=ma['ap50_por_material']['carton'] and pre_b<=pre_a
    conclusion=(f"B final: mAP50 de cinco vistas {100*mb['map50']:.4f} %, A {100*ma['map50']:.4f} %, origen con el mismo backend {100*mo['map50']:.4f} % e histórico {100*h['metricas']['5']['map50']:.4f} %. "
        f"AP50 de cartón B={100*mb['ap50_por_material']['carton']:.4f} %; vidrio B={100*mb['ap50_por_material']['vidrio']:.4f} %. No se adopta automáticamente ningún checkpoint.")
    extra='B mejora la clasificación condicional de cartón en train, pero no su AP50 ni sus oportunidades antes de filtros: la intervención no solucionó ese problema de detección.' if no_soluciona else 'La decisión debe considerar conjuntamente AP por material, mAP, recall, precisión, FP, geometría y oportunidades; la recuperación condicional en train no demuestra generalización.'
    if mb['ap50_por_material']['carton']==0 and mb['ap50_por_material']['vidrio']==0 and max(mb['map50'],rr['B_mejor']['val_5_vistas']['resumen']['map50'])<mo['map50']:
        extra+=' No se recomienda adoptar B: cartón y vidrio siguen sin AP50 y ninguno de sus dos estados evaluados supera el mAP50 de la referencia. La intervención no solucionó la detección de esos materiales.'
    guardar(out/'conclusion.json',{'conclusion':conclusion,'interpretacion':extra,'condicionales_carton_A_B':[cond_a,cond_b],
        'oportunidades_carton_pre_A_B':[pre_a,pre_b],'mejora_condicional_sin_resolver_deteccion':no_soluciona,'adopcion_automatica':False,'test_evaluado':False})
    texto=['# V24: resultados V20 con evaluación numéricamente coherente','',conclusion,'',extra,'',
        '## Integración y verificaciones','',
        '`evaluar_v20.py` acepta `--config`, aplica el contexto numérico declarado durante inferencia/diagnóstico principal y verifica protocolo, hashes, origen, backend, entorno y opciones del optimizador de cada checkpoint experimental. La referencia de época 19 sin etiqueta de backend solo se admite como origen verificado por SHA256. No se cambió el modelo, la pérdida ni el postprocesamiento.','',
        f"Pasan {d['pruebas']['ejecutadas']} pruebas. El diagnóstico train-only se regeneró en una ruta nueva. Los certificados V23 se revalidaron por alcance: entrenador, paso, modelo, pérdida, backend y reanudación conservan sus SHA256; el preparador solo amplió su lista de pruebas, comprobado por AST. La evidencia de 125 pasos no se repitió ni se reescribió. La revalidación es explícita y enlaza el manifiesto y controles originales.",'',
        'Además, los checkpoints reales de época 20 de A y B reproducen exactamente las huellas de pesos, AdamW, scaler y RNG de la época 20 de los controles V23 C1. La comparación se conserva en `comparacion_epoca20_vs_control125.json`; proporciona una comprobación adicional de que la integración no alteró el aprendizaje previo a esa validación.','',
        'Hubo un fallo de codificación cp1252 en el lanzador al reenviar la salida del primer intento histórico. No había comenzado el entrenamiento. Se conservaron log y captura parcial, y se corrigió únicamente la entrada/salida del lanzador a UTF-8. La nueva comprobación histórica terminó correctamente; no se cambiaron tolerancias ni hipótesis para resolver ese incidente.','',
        f"Después de completar el entrenamiento, la primera evaluación se detuvo en `bincount` ponderado CUDA del resumen regional train, operación no admitida con determinismo estricto. Se conservó su captura completa del origen. La revisión 2 realiza esa reducción GT en CPU, manteniendo el forward GPU V23 y el determinismo; no cambió modelo, pérdida ni checkpoints. Tiene diagnóstico propio con {revision['pruebas']['ejecutadas']} pruebas aprobadas y certificado de compatibilidad explícito. Los hashes originales de entrenamiento se conservan y se distinguen de los hashes de evaluación; el certificado solo admite el cambio del evaluador y demuestra por AST que las funciones de inferencia no cambiaron. La captura del origen se reutilizó tras verificar todos sus archivos y reproducir exactamente un lote de ocho logits con el checkpoint/backend esperado. No se autoriza nuevo entrenamiento bajo los hashes de esta revisión.",'',
        '## Histórica exacta y referencia bajo SDPA MATH','',
        'La regresión histórica se ejecutó en un intérprete separado con las opciones originales, incluyendo cuBLAS no configurado y SDPA histórico. Reprodujo exactamente las predicciones de las 181 imágenes en una y cinco vistas. La evaluación nueva usa el modo determinista completo V23 para origen, A y B. Las diferencias de la referencia entre backends se registran y no se tratan como fallo de integridad ni como mejora aprendida.','',
        tabla(['Vistas','Histórico mAP50 %','Origen backend nuevo mAP50 %','Imágenes idénticas entre backends','Diferentes'],[[v,f"{100*h['metricas'][v]['map50']:.6f}",f"{100*rr['origen'][['val_1_vista','val_5_vistas'][v=='5']]['resumen']['map50']:.6f}",rr['origen']['regresion_historica'][v]['imagenes_exactas'],len(rr['origen']['regresion_historica'][v]['imagenes_diferentes'])] for v in ['1','5']]),'',
        '## Ejecución por etapas','',
        'A=0 y B=0,25 comenzaron desde el mismo checkpoint propio de época 19, con pesos, AdamW, scaler y RNG originales. Se conserva LR=0,0005, AMP, clipping, pérdida de fondo=3, auxiliar global=0,5, margen Huber=1, fracción difícil=0,25 y pesos [1,1,2,2,1,1]. Sin warmup nuevo, parada temprana, cambios arquitectónicos ni cambios de umbral.','',
        tabla(['Rama, época 20','Pasos','Validación repetida exacta','mAP50 una vista %'],[[r,et['ramas'][r]['actualizaciones'],et['ramas'][r]['validacion_reproducida_exactamente'],f"{100*et['ramas'][r]['resumen']['map50']:.4f}"] for r in ['A','B']]),'',
        'Tras comprobar ambos diarios, estados, trazas y validación de época 20 se reanudaron las dos ramas hasta la 31. Cada rama completó 1.452 actualizaciones (AdamW 2.299→3.751); 12 checkpoints/commits de época inmutables y trazas cotejadas con el plan previo. El mejor checkpoint se seleccionó por mAP50 de una vista, primer empate, y se informa separado del final.','',
        f"Mejores épocas: A={t['ramas']['A']['mejor_epoca']}, B={t['ramas']['B']['mejor_epoca']}. La ejecución nueva tiene identificador `{nombre}` y conserva el backend numérico V23; las configuraciones previas y checkpoints diagnósticos no se reutilizan como estados entrenados.",'',
        '## Métricas de validación','',
        tabla(['Estado','Época','Vistas','mAP50 %','mAP50:95 %','Recall %','Precisión %','TP','FP','FN'],[[x['estado'],x['epoca'],x['vistas'],*[f"{100*x[k]:.4f}" for k in ['map50','map50_95','recall50','precision50']],*[x[k] for k in ['tp50','fp50','fn50']]] for x in filas]),'',
        tabla(['Estado cinco vistas',*CLASES],[[n,*[f"{100*rr[n]['val_5_vistas']['resumen']['ap50_por_material'][c]:.4f}" for c in CLASES]] for n in ns]),'',
        '![Curvas](../resultados/v24_v20_evaluacion/informe/curvas.png)','',
        '![mAP](../resultados/v24_v20_evaluacion/informe/map50.png)','',
        '![Materiales](../resultados/v24_v20_evaluacion/informe/ap_materiales.png)','',
        '## Cabeza principal y confusiones train','',
        'Se mide exclusivamente la cabeza de mapas sobre polígonos elegibles train, excluyendo padding, ignorados y mezclas ambiguas. La recuperación condicional quita fondo solo para el análisis; no es la cabeza auxiliar ni un cambio de inferencia.','',
        tabla(['Material','Instancias','Origen recuperadas/cond.','A final recuperadas/cond.','B final recuperadas/cond.','Fondo medio origen/A/B %'],[[c,mat('origen',c)['instancias'],*[f"{mat(n,c)['recuperadas']} / {mat(n,c)['condicionales_recuperadas']}" for n in ['origen','A_ultimo','B_ultimo']], ' / '.join(f"{100*mat(n,c)['fraccion_fondo_media']:.2f}" for n in ['origen','A_ultimo','B_ultimo'])] for c in CLASES]),'',
        f"Cartón→plástico/metal por voto condicional: origen {100*error('origen'):.2f} %, A {100*error('A_ultimo'):.2f} %, B {100*error('B_ultimo'):.2f} %. Son mediciones en train, no una estimación independiente de generalización.",'',
        '![Principal](../resultados/v24_v20_evaluacion/informe/confusion_principal.png)','',
        '![Condicional](../resultados/v24_v20_evaluacion/informe/confusion_condicional.png)','',
        '## Oportunidades, geometría y tamaño','',
        tabla(['Estado','Material','GT','Antes de filtros','Final','Fusión','Fragmentación'],[[n,c,(z:=rr[n]['oportunidades_y_geometria_val']['por_material'][c])['gt'],z['oportunidades_por_etapa']['componentes'],z['oportunidades_por_etapa']['final'],z['indicadores_fusion'],z['indicadores_fragmentacion']] for n in ns for c in ['carton','vidrio']]),'',
        'Una oportunidad cuenta GT con al menos una propuesta del material correcto e IoU≥0,5; no equivale a un TP asignado uno a uno. El observador V19 exige igualdad exacta con el postprocesamiento de producción aplicado a los mismos logits. GT se consulta después de terminar la captura. El JSON de geometría incluye las ocho causas, todas las etapas y el desglose de seis materiales por tamaño.','',
        '![Oportunidades](../resultados/v24_v20_evaluacion/informe/oportunidades.png)','',
        'Ejemplos ilustrativos elegidos como el primer GT de cada material por orden de val, sin seleccionar por resultado. Se muestra el mismo recorte ampliado del objeto en los tres modelos y el umbral visual existente 0,35; las métricas usan 0,05.','',
        '![Ejemplo cartón](../resultados/v24_v20_evaluacion/informe/ejemplo_carton.png)','',
        '![Ejemplo vidrio](../resultados/v24_v20_evaluacion/informe/ejemplo_vidrio.png)','',
        '## Orgánico, coste y control V18','',
        'La única región orgánica elegible, taco:92/244, aparece una vez por época en cada rama y conserva peso 1. Se registra pérdida individual, tokens, rivales, fracción de peso y contribución efectiva; en A es cero por coeficiente. No se espera robustez a partir de una instancia.','',
        '![Orgánico](../resultados/v24_v20_evaluacion/informe/organico.png)','',
        tabla(['Rama','Pasos','Segundos entrenamiento/val/IO','Pico GPU asignado GiB','Pico GPU reservado GiB'],[[r,1452,f"{t['ramas'][r]['segundos']:.2f}",f"{t['ramas'][r]['memoria_gpu_max_allocated_bytes']/2**30:.3f}",f"{t['ramas'][r]['memoria_gpu_max_reserved_bytes']/2**30:.3f}"] for r in ['A','B']]),'',
        f"Las entradas de A coinciden con las de V18 en las doce épocas: {e['control_A_frente_V18']['entradas_iguales']}. El JSON de evaluación contiene pérdidas y diferencias de mAP por época. La ejecución histórica usaba otros kernels/opciones numéricas; no se presupone identidad de parámetros ni se atribuyen estas diferencias a la pérdida regional.",'',
        'Los tiempos de captura incluyen archivos, observador y memoria GPU por una/cinco vistas; no son latencia pura de producción. Se guardan predicciones finales por imagen, logits AMP y trazas, con hashes, para futuras auditorías sin repetir inferencia. Estados final/mejor con el mismo SHA reutilizan la captura y lo declaran.','',
        '## Integridad y límites','',
        'Test permaneció reservado y el checkpoint predeterminado no se modificó. Los resultados históricos y la evidencia de 125 pasos se conservan; los dos scripts modificados tienen copia exacta previa. El manifiesto final registra código, configuración autorizada, certificados, diarios, resultados, cachés de logits/trazas y figuras. Una semilla, validación de desarrollo reutilizada y AP propio de 101 puntos limitan las conclusiones; no es COCOeval oficial ni una prueba independiente.','',
        f"Artefactos: [entrenamiento](../resultados/{nombre}.json), [evaluación](../resultados/{nombre}_evaluacion.json), "+
        '[referencia histórica](../resultados/v24_v20_evaluacion/referencia_historica.json), [etapa 20](../resultados/v24_v20_evaluacion/etapa20/verificacion.json), [revalidación](../resultados/v24_v20_evaluacion/revalidacion_por_alcance.json), [CSV](../resultados/v24_v20_evaluacion/informe/metricas.csv), [conclusión](../resultados/v24_v20_evaluacion/informe/conclusion.json), [integridad](../resultados/v24_v20_evaluacion/manifiesto_integridad.json).','']
    with doc.open('x',encoding='utf-8') as f:f.write('\n'.join(texto))
    print(conclusion);print(extra)


if __name__=='__main__':main()
