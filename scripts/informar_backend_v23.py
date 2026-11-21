"""Informe V23 desde controles archivados; no ejecuta redes ni habilita entrenamiento."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RAIZ=Path(__file__).resolve().parents[1]
ROOT=RAIZ/'resultados/v23_backend_v20'


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def tabla(cab,filas):return '\n'.join(['| '+' | '.join(cab)+' |','|'+'|'.join(['---']*len(cab))+'|',*['| '+' | '.join(map(str,f))+' |' for f in filas]])


def main():
    decision=leer(ROOT/'decision.json');prep=leer(ROOT/'preparacion_completa.json');modo=decision['candidato']
    corto=leer(ROOT/f'corto_{modo}.json');minimo=leer(ROOT/'corto_minimo.json');largo=leer(ROOT/f'largo_{modo}.json')
    cfg=leer(RAIZ/prep['configuracion']);diag=leer(RAIZ/cfg['diagnostico']);out=ROOT/'informe'
    doc=RAIZ/'documentacion/23_validacion_backend_determinista_v20.md'
    if out.exists() or doc.exists():raise FileExistsError('Informe existente')
    out.mkdir();filas=[]
    for etiqueta,r in [('minimo',minimo),*([(modo,corto)] if modo!='minimo' else [])]:
        for rama,x in r['ramas'].items():
            for k,s in enumerate(x['pasos'],1):
                for par in ['C2','R']:
                    y=s[par];w=y['etapas']['post.pesos'];filas.append({'modo':etiqueta,'rama':rama,'paso':k,'par':'C1_'+par,
                        'restauracion_exacta':x['restauracion_exacta'],'todas_etapas_tolerancia':y['dentro_tolerancia'],
                        'elementos_fuera_todas_etapas':y['elementos_fuera_tolerancia'],'delta_max_pesos':w['max_delta'],
                        'fuera_pesos':w['fuera_tolerancia'],'max_error_normalizado_pesos':w['max_error_normalizado']})
    with (out/'comparacion_corta.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(filas[0]));w.writeheader();w.writerows(filas)
    plt.rcParams.update({'figure.dpi':140,'font.size':9})
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,rama in zip(axs,['A','B']):
        for etiqueta,r in [('minimo',minimo),*([(modo,corto)] if modo!='minimo' else [])]:
            for par in ['C2','R']:
                ax.plot(range(1,5),[s[par]['etapas']['post.pesos']['max_error_normalizado'] for s in r['ramas'][rama]['pasos']],marker='o',label=etiqueta+' C1-'+par)
        ax.axhline(1,color='gray',ls='--',label='Límite por elemento');ax.axvline(2.5,color='gray',ls=':');ax.set(xlabel='Paso',ylabel='Máximo error normalizado en pesos',title='Rama '+rama);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'criterio_corto.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4));largas=[]
    for ax,rama in zip(axs,['A','B']):
        root=ROOT/'largo'/modo/'trazas'/rama
        for t in ['C1','C2','R']:
            ss=[leer(root/t/f'paso_{k:04d}.json') for k in range(1,126)]
            ax.plot(range(1,126),[s['loss_original'] for s in ss],label=t,alpha=.75)
            fin=leer(root/t/'fin.json');largas.append([rama,t,125,fin['paso_adamw'],*fin['huellas_estado'].values()])
        ax.axvline(121.5,color='red',ls='--');ax.set(xlabel='Actualización',ylabel='Pérdida original',title='Rama '+rama);ax.legend()
    fig.tight_layout();fig.savefig(out/'control_largo.png');plt.close(fig)
    with (out/'huellas_finales.json').open('x',encoding='utf-8') as f:json.dump(largas,f,ensure_ascii=False,indent=2)
    ultimo=[f for f in filas if f['paso']==4];rej=leer(ROOT/'rechazos_reanudacion.json')
    texto=['# V23: validación del backend determinista para V20','',
        f"**Decisión: candidato {modo}.** SDPA MATH + cuDNN determinista cumple el control completo: **{minimo['aprobada']}**. El candidato seleccionado cumple el control corto: **{corto['aprobada']}** y el control de 125 actualizaciones con límite real de época: **{largo['aprobada']}**. El A/B completo sigue deshabilitado.",'',
        '## Comparación de la combinación mínima','',
        'Se activaron exclusivamente SDPA MATH, `torch.backends.cudnn.deterministic=True` y `cudnn.benchmark=False`. Autocast AMP, GradScaler restaurado, AdamW con foreach/fused históricos, clipping y demás opciones se conservaron. Cada C1 y C2 empieza desde el checkpoint de época 19 en un proceso independiente; R ejecuta dos pasos, guarda el estado completo y continúa otros dos en otro proceso. Se repite con A=0 y B=0,25.','',
        'Criterio fijado antes de ejecutar: recarga e inputs/RNG exactos y ambos pares C1–C2/C1–R dentro de **rtol=1e-4, atol=1e-5 en todos los tensores y etapas**. El error normalizado es `abs(a-b)/(atol+rtol*abs(b))`; superar 1 viola la tolerancia. No se aprueba por un delta absoluto pequeño.','',
        tabla(['Modo','Rama','Par, paso 4','Máx. delta pesos','Elementos fuera en pesos','Máx. error normalizado pesos','Todas etapas cumplen'],[[x['modo'],x['rama'],x['par'],f"{x['delta_max_pesos']:.9g}",x['fuera_pesos'],f"{x['max_error_normalizado_pesos']:.6g}",x['todas_etapas_tolerancia']] for x in ultimo]),'',
        '![Criterio por paso](../resultados/v23_backend_v20/informe/criterio_corto.png)','',
        'La comparación completa incluye entradas, logits de mapas y auxiliar, pérdidas, gradientes escalados, sin escala y después del clipping, norma, pesos, momentos/contador AdamW, scaler y RNG. Los JSON de `corto_comparaciones` conservan todos los nombres, dtypes, máximos y elementos fuera de tolerancia; las cuatro actualizaciones no se interrumpen ante una diferencia numérica.','',
        '## Corrección al informe 22 y alcance causal','',
        'El informe 22 llamó convolucional a un residuo del extractor; esa caracterización era incorrecta. `ecowaste/detector.py` define ambos tokenizadores como `nn.Linear` sobre parches y también usa `F.interpolate(..., mode="bilinear")`. No se cambió esa arquitectura. La bandera cuDNN no garantiza determinismo de todas estas operaciones. La variación residual sigue localizada en el backward; este ensayo de selección no aísla un único kernel nativo como causa exclusiva. El informe histórico se conserva intacto y esta corrección se registra aquí.','',
        '## Configuración seleccionada y contrato de ejecución','',
        '```json\n'+json.dumps(decision['backend_numerico'],ensure_ascii=False,indent=2)+'\n```','',
        'El candidato completo reproduce las opciones diagnósticas de V22; no altera la hipótesis de aprendizaje. cuBLAS se configura en el entorno del proceso antes de iniciar Python. El contexto numérico aplica SDPA/cuDNN/determinismo/TF32 y las opciones de AdamW/clipping declaradas, registra sus valores efectivos y restaura las opciones globales al salir. La modalidad mínima conserva los defaults históricos ajenos a sus tres cambios; los rechaza si no coinciden.','',
        'El entrenador guarda `backend_numerico` y `backend_efectivo` en cada checkpoint versionado. El diario rechaza políticas ausentes/distintas u opciones incompatibles; al reanudar se comprueba también igualdad del entorno efectivo, incluidas versiones PyTorch/CUDA/cuDNN, GPU e hilos. Cada actualización verifica las opciones vigentes. No se mezclan los diarios V23 con V18 ni con los checkpoints experimentales V20 anteriores.','',
        '## Control prolongado con mecanismos reales del entrenador','',
        'Se ejecutaron C1, C2 y R durante **125 actualizaciones reales por trayectoria y rama**: los 121 lotes de la época 20 (964 escenas, último lote de cuatro) y cuatro lotes de la época 21. C1 y C2 guardan checkpoint al final de época y continúan; R termina el proceso tras el paso 121 y restaura en otro proceso. AdamW avanza de 2.299 a 2.424. Sin warmup nuevo, sin parada temprana y sin cargar validación o test.','',
        'El control llama al mismo `ejecutar_epoca_train` y `paso_adamw` del entrenador, usa sus loaders/orden/aumentos y la restauración de pesos, AdamW, scaler y RNG. Publica la época completa con `DiarioV20`: checkpoint inmutable, registro JSON encadenado y validaciones de protocolo/datos/estados. La variante train-only del diario prohíbe métricas de validación y selección del mejor modelo. La época 21 parcial queda como evidencia diagnóstica, sin commit de época completa.','',
        tabla(['Rama','Recarga R exacta','125 pasos C1–C2/C1–R aprobados','Final C1–C2 exacto','Final C1–R exacto','Huellas por etapa exactas en todos los pasos'],[[b,x['restauracion_exacta'],x['aprobada'],x['finales']['C2']['exacto'],x['finales']['R']['exacto'],all(s['pares'][t]['huellas_exactas'] for s in x['pasos'] for t in ['C2','R'])] for b,x in largo['ramas'].items()]),'',
        '![Control prolongado](../resultados/v23_backend_v20/informe/control_largo.png)','',
        'Se conservan huellas SHA256 de todas las etapas de cada actualización. C1 guarda instantáneas completas de los 125 pasos; C2/R conservan los pasos 1, 2, 120, 121, 122 y 125 y cualquier paso con una huella diferente. Una discrepancia obliga a guardar y comparar todos sus tensores con la tolerancia original. Los estados finales se comparan exhaustivamente. Esta estrategia conserva evidencia sin duplicar instantáneas idénticas.','',
        'La única región orgánica elegible mantiene peso 1 y aparece una vez en la época 20 completa. Cada checkpoint conserva su contribución individual y multiplicada por el coeficiente. No se usa un valor mAP ficticio ni se selecciona un modelo en el control.','',
        '## Pruebas y rechazo de incompatibilidades','',
        f"Pasan {diag['pruebas']['ejecutadas']} pruebas, incluidas las 177 existentes. Se regeneró el diagnóstico de código antes de los ensayos y el del protocolo nuevo después de los controles. Ambos verifican la pérdida y regresión de inferencia histórica; los controles C1/C2/R validan separadamente el backend de entrenamiento. Las pruebas de memorización son train-only y sus pesos se descartan.",'',
        tabla(['Caso probado contra estado real/guardia','Rechazado','Motivo'],[[k,v['rechazado'],v['motivo']] for k,v in rej.items()]),'',
        '## Protocolo A/B preparado, sin habilitar','',
        f"Configuración nueva: `{prep['configuracion']}`. Versión `{cfg['version_protocolo']}`, identificador `{cfg['nombre']}` y rutas exclusivas de checkpoints/diarios. Ambas ramas volverán a iniciar desde el origen de época 19, nunca desde los pesos diagnósticos. Se conservan 12 épocas y 1.452 pasos por rama, LR=0,0005, AMP/scaler, margen Huber=1, fracción difícil=0,25, pesos [1,1,2,2,1,1], A=0/B=0,25, fondo=3 y auxiliar global=0,5. Arquitectura, argmax, cinco vistas, umbrales y postprocesamiento no se cambian.",'',
        'Se publican certificados nuevos de corto y largo, vinculados por hashes al diagnóstico y al protocolo. La guardia exige esos controles y el backend coincidente. Tanto la configuración original como la nueva mantienen `entrenamiento_completo_habilitado=false`; una invocación del entrenador se rechaza antes de entrenar. Actualizaciones del A/B completo: A=0 y B=0. Los pesos del control no son candidatos de adopción.','',
        'Comandos previstos para una futura ejecución autorizada, con la configuración todavía bloqueada:','',
        '```powershell\n'+("$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'\n" if modo=='completo' else '')+
        f"python scripts/entrenar_v20.py --config {prep['configuracion']} --rama A\npython scripts/entrenar_v20.py --config {prep['configuracion']} --rama B\n# Para continuar únicamente un diario del mismo protocolo:\npython scripts/entrenar_v20.py --config {prep['configuracion']} --rama A --reanudar\n```",'',
        '## Límites y evidencia','',
        'La reproducibilidad se comprobó en cuatro y 125 pasos, en esta GPU y estas versiones; no se extrapola identidad numérica a las 1.452 actualizaciones. El control no cubre el paso por validación ni la evaluación final, que no se ejecutaron en esta tarea. No aporta evidencia de mejora de AP/mAP ni de la pérdida discriminativa. La referencia original y todos los resultados históricos permanecen intactos; los cinco scripts autorizados conservan copias exactas de sus bytes anteriores.','',
        'Artefactos: [protocolo prefijado](../resultados/v23_backend_v20/protocolo_ejecutado.json), [mínimo](../resultados/v23_backend_v20/corto_minimo.json), [decisión](../resultados/v23_backend_v20/decision.json), '+
        f"[candidato corto](../resultados/v23_backend_v20/corto_{modo}.json), [control largo](../resultados/v23_backend_v20/largo_{modo}.json), "+
        '[preparación](../resultados/v23_backend_v20/preparacion_completa.json), [rechazos](../resultados/v23_backend_v20/rechazos_reanudacion.json), [CSV](../resultados/v23_backend_v20/informe/comparacion_corta.csv), [integridad](../resultados/v23_backend_v20/manifiesto_integridad.json).','']
    with doc.open('x',encoding='utf-8') as f:f.write('\n'.join(texto))
    print('Informe 23 generado; candidato',modo,'A/B deshabilitado')


if __name__=='__main__':main()
