"""Informe y figuras V22 desde evidencia archivada; no usa PyTorch ni inferencia."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
RAIZ=Path(__file__).resolve().parents[1]
ROOT=RAIZ/'resultados/v22_reproducibilidad_v20'


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def tabla(cab,filas):return '\n'.join(['| '+' | '.join(cab)+' |','|'+'|'.join(['---']*len(cab))+'|',*['| '+' | '.join(map(str,f))+' |' for f in filas]])


def main():
    r=leer(ROOT/'resumen.json');e=leer(ROOT/'evidencia_etapas.json');q=leer(ROOT/'qkv_por_particion.json')
    protocolo=leer(ROOT/'protocolo_ejecutado.json')['protocolo'];proc=leer(ROOT/'procesos.json');d=leer(ROOT/'preparacion/diagnostico.json')
    out=ROOT/'informe';doc=RAIZ/'documentacion/22_diagnostico_reproducibilidad_v20.md'
    if out.exists() or doc.exists():raise FileExistsError('Informe existente')
    out.mkdir();plt.rcParams.update({'figure.dpi':140,'font.size':9})
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,rama in zip(axs,['A','B']):
        for modo in protocolo['modos']:
            ps=r['modos'][modo][rama]['pasos'];ax.plot(range(1,5),[s['C1_R']['etapas']['post.pesos']['max_delta'] for s in ps],marker='o',label=modo)
        ax.axvline(2.5,ls='--',color='gray');ax.set(xlabel='Actualización',ylabel='Máximo |peso C1 − peso R|',title=f'Rama {rama}: restauración entre pasos 2 y 3');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'diferencias_por_paso.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4));x=np.arange(len(protocolo['modos']))
    for ax,rama in zip(axs,['A','B']):
        for j,par in enumerate(['C1_C2','C1_R']):ax.bar(x+(j-.5)*.35,[r['modos'][m][rama]['pasos'][-1][par]['etapas']['post.pesos']['max_delta'] for m in protocolo['modos']],.35,label=par)
        ax.set_xticks(x,protocolo['modos'],rotation=45);ax.set(ylabel='Máxima diferencia absoluta de pesos',title=f'Paso 4, rama {rama}');ax.legend()
    fig.tight_layout();fig.savefig(out/'continuos_frente_reanudado.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,rama in zip(axs,['A','B']):
        for par in ['C1_C2','C1_R']:
            ax.plot(range(1,5),[max(f['max_delta_peso'] for f in q if f['rama']==rama and f['paso']==k and f['comparacion']==par and f['parte']=='K') for k in range(1,5)],marker='o',label=par)
        ax.set(xlabel='Actualización',ylabel='Máximo delta en sesgos K',title=f'Partición K de QKV, rama {rama}');ax.legend()
    fig.tight_layout();fig.savefig(out/'sensibilidad_sesgos_k.png');plt.close(fig)
    comparativas=[]
    for modo in protocolo['modos']:
        for rama in ['A','B']:
            z=r['modos'][modo][rama];s=z['pasos'][-1]
            comparativas.append([modo,rama,*[f"{s[c]['etapas']['post.pesos']['max_delta']:.9g}" for c in ['C1_C2','C1_R']],
                *[s[c]['etapas']['post.pesos']['fuera_tolerancia'] for c in ['C1_C2','C1_R']],
                z['restauracion_exacta'],z['todos_dentro_tolerancia_original'],z['sin_divergencia_adicional_segun_envolvente']])
    with (out/'resumen_modalidades.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['modo','rama','max_delta_C1_C2','max_delta_C1_R','fuera_C1_C2','fuera_C1_R','restauracion_exacta','todos_tolerancia','envolvente']);w.writerows(comparativas)
    etapas=['pre.pesos','pre.adamw','pre.entradas','pre.logits_taco','pre.gradientes_taco_escalados','pre.logits_auxiliar',
        'pre.gradientes_total_escalados','pre.gradientes_sin_escala','pre.norma_antes_clip','pre.gradientes_clip','post.pesos','post.adamw']
    primeras=[]
    for rama in ['A','B']:
        for etapa in etapas:
            z=r['modos']['historico'][rama]['pasos'][0]['C1_C2']['etapas'][etapa]
            primeras.append([rama,etapa,z['exacta'],f"{z['max_delta']:.9g}",z['fuera_tolerancia']])
    peores=sorted((x for x in q if x['paso']==4),key=lambda x:-x['max_delta_peso'])[:8]
    det=all(r['modos']['determinista'][b]['todos_dentro_tolerancia_original'] for b in ['A','B'])
    exact_det=all(r['modos']['determinista'][b]['pasos'][k][c]['exacto'] for b in ['A','B'] for k in range(4) for c in ['C1_C2','C1_R'])
    hist=all(r['modos']['historico'][b]['sin_divergencia_adicional_segun_envolvente'] for b in ['A','B'])
    replay=all(r['replay_adam'][b][n]['exacto'] for b in ['A','B'] for n in ['dos_replays_default','replay_frente_actual'])
    identificacion=('Se localiza variabilidad numérica del backward CUDA anterior a cualquier restauración, que llega a los gradientes y es amplificada por AdamW en sesgos QKV. La recarga de los estados es exacta. '
        'No se encontró evidencia de corrupción del checkpoint ni de RNG desalineado. La identificación de un único kernel nativo como causa exclusiva requiere más aislamiento: SDPA y reducciones del extractor pueden coexistir.')
    resultado={'causa_localizada':identificacion,'criterio_historico_satisfecho':hist,'criterio_determinista_satisfecho':det,
        'identidad_exacta_determinista_comprobada_solo_estas_trayectorias':exact_det,'replay_AdamW_exacto':replay,
        'AB_habilitado':False,'piloto_historico_original_no_reclasificado_como_aprobado':True,'hipotesis_perdida_y_presupuesto_intactos':True}
    with (out/'conclusion.json').open('x',encoding='utf-8') as f:json.dump(resultado,f,ensure_ascii=False,indent=2)
    texto=['# V22: diagnóstico de reproducibilidad del piloto CUDA V20','',identificacion,'',
        '**A/B completo permanece deshabilitado.** No se modificaron pérdida, coeficientes, arquitectura, presupuesto ni entrenamiento histórico. No se usaron validación ni test. No se reinterpreta el piloto fallido como aprobado.','',
        '## Protocolo fijado antes de ejecutar','',
        'Desde el mismo checkpoint propio de época 19 y sus estados AdamW/scaler/RNG se compararon C1 (cuatro pasos continuos), C2 (otra ejecución continua independiente) y R (dos pasos, guardar, nuevo proceso, restaurar y dos pasos). Se conservaron los mismos cuatro lotes train y auxiliar global, aumentos, LR y clipping. Se ejecutaron ambas ramas, A=0 y B=0,25, para verificar también la trayectoria de la pérdida discriminativa sin modificarla.','',
        'Cada trayectoria empieza en un proceso independiente; R utiliza otro proceso después del paso 2. Todos conservan checkpoint completo al terminar el paso 2, incluido el plan restante. Los artefactos son exclusivos de `resultados/v22_reproducibilidad_v20`; no se sobrescribió la incidencia ni ningún resultado histórico.','',
        'Se mantiene **rtol=1e-4, atol=1e-5**. Por elemento, el error normalizado es `|a-b|/(atol+rtol·|b|)`; superar 1 falla la tolerancia original. Se registran todos los tensores, nombres, dtypes, formas, diferencias máximas y elementos fuera de tolerancia. Los máximos relativos con referencia cero se señalan como infinitos, sin insertar NaN/Infinity en los JSON.','',
        'Criterio histórico prefijado: restauración e inputs/RNG exactos y, por tensor, etapa y actualización, error máximo normalizado C1–R ≤ max(1, error C1–C2), con cantidad de elementos fuera de tolerancia C1–R ≤ C1–C2. No se multiplican tolerancias. Es una envolvente observada conservadora, no una estimación estadística a partir de un único par continuo. Criterio determinista: ambos pares dentro de la tolerancia original en todos los pasos y estados, con restauración e inputs exactos. Pasar únicamente en modo determinista no valida el protocolo histórico.','',
        '## Controles, alcance y conservación','',
        f"Diagnóstico regenerado: {d['pruebas']['ejecutadas']} pruebas aprobadas; gradientes y memorización train-only aprobados. Las 42 dependencias de código están registradas en las nuevas huellas. El diagnóstico anterior y su manifiesto se conservan como históricos; ya no describen el código instrumentado actual. El índice previo registró 834 archivos y autoriza cambios solo en los dos scripts instrumentados. El manifiesto nuevo identifica los hashes antes/después.",'',
        f"Se completaron {r['actualizaciones_trayectorias']} actualizaciones diagnósticas de red y {r['actualizaciones_replay_adam']} replays aislados de AdamW. La regeneración incluyó por separado la memorización train-only de 200 pasos con pesos descartados. Ninguna actualización corresponde al presupuesto A/B de 1.452 pasos por rama. Procesos fallidos: {len(r['procesos_con_error'])}. Tiempo de orquestación, captura, procesos y análisis: {r['segundos']:.1f} s.",'',
        '## Localización antes y después de AdamW','',
        'Los archivos `paso_NN_pre.pt` se publican **antes** del step: pesos y momentos previos, logits TACO y auxiliares, pérdidas, gradientes TACO escalados, gradientes totales escalados, sin escala y con clipping, norma y huellas RNG. Los `post.pt` añaden pesos, momentos, contador, scaler y huellas posteriores. Los JSON de comparación contienen todos los parámetros y estados, no solo el primer tensor divergente.','',
        tabla(['Rama','Etapa, primer paso C1–C2 histórico','Exacta','Máx. delta absoluto','Elementos fuera'],primeras),'',
        'Una diferencia en gradientes con entradas, pesos y logits iguales nace antes de AdamW; la restauración todavía no ha ocurrido. Desde el paso 2 también pueden divergir los logits porque las actualizaciones anteriores ya cambiaron los pesos. Comparar solamente el último checkpoint no distingue estas causas.','',
        '## Resultados de las modalidades','',
        tabla(['Modalidad','Rama','Máx. Δ pesos C1–C2 paso 4','Máx. Δ pesos C1–R paso 4','Fuera C1–C2','Fuera C1–R','Recarga exacta','Todas etapas en tolerancia','Envolvente'],comparativas),'',
        '![Diferencias por paso](../resultados/v22_reproducibilidad_v20/informe/diferencias_por_paso.png)','',
        '![Continuos y reanudación](../resultados/v22_reproducibilidad_v20/informe/continuos_frente_reanudado.png)','',
        'Las modalidades se aislaron así:','',
        '- `historico`: opciones originales registradas, AMP y defaults de SDPA/clipping/AdamW.','- `sdpa_math`: solo backend SDPA MATH.','- `sin_autocast`: solo FP32 en los forwards; conserva el scaler original para no confundir precisión con escalado.','- `clip_serial`: solo clipping con foreach=False; no cambia acumulación ni backward.','- `adam_serial`: solo foreach=False y fused=False después de comprobar restauración exacta; se documenta ese cambio diagnóstico de grupos.','- `determinista`: cuBLAS `:4096:8` antes de iniciar Python, deterministic_algorithms=True, SDPA MATH, cuDNN deterministic=True/benchmark=False, TF32 desactivado, clipping y AdamW seriales. No se aplica al entrenador histórico.','',
        'Cada `entorno_inicio.json` y `entorno_reanudar.json` registra las opciones efectivas, versiones, GPU, PID, hilos y grupos AdamW. `operadores.json` identifica las operaciones de atención y foreach observadas con profiler CPU. No identifica por sí solo cada kernel CUDA nativo. La captura y el profiler pueden afectar planificación y temporización, por lo que este ensayo no promete reproducir exactamente el valor 0,0001954 de la ejecución anterior.','',
        '## Evidencia por hipótesis','',
        'El profiler histórico registra `aten::_scaled_dot_product_efficient_attention` y su backward. En A, los gradientes TACO del primer paso coinciden y la primera diferencia aparece tras el backward auxiliar: máximo 0,00390625 escalado, 1,90735e-6 sin escala y 2,81725e-7 después del clipping. La norma de clipping inicial coincide. En B ya aparecen diferencias en el backward TACO. Por tanto, ni el archivo de reanudación ni el clipping ni el step de AdamW son necesarios para que nazca la variación.','',
        'Restringir SDPA a MATH reduce la divergencia, pero no elimina toda la variación: en el primer backward TACO A, el único gradiente distinto es `tokenizador_contexto.weight`, con máximo 0,001953125 escalado. Esto localiza un residuo en el extractor convolucional; no identifica por sí solo el kernel nativo responsable. Desactivar autocast reduce el delta final de pesos a menos de 6e-7 en estos cuatro pares, pero no cumple todas las comprobaciones por etapa. La precisión amplifica la sensibilidad y no es una explicación exclusiva.','',
        'Clipping serial y AdamW serial conservan divergencia entre trayectorias. El replay de AdamW desde entradas idénticas es exacto para ambos defaults y frente al step archivado; serial frente a default difiere solo 1,49012e-8 y sigue dentro de tolerancia. La combinación determinista elimina las diferencias en todas las etapas ensayadas. El aislamiento de reducciones internas del backward es parcial: foreach=False solo controla el clipping, no todas las reducciones CUDA del extractor.','',
        '## Sesgos QKV y amplificación','',
        tabla(['Rama','Par','Parámetro','Partición','Máx. Δ peso paso 4','Máx. Δ gradiente clip del paso','Índice'],[[x['rama'],x['comparacion'],x['parametro'],x['parte'],f"{x['max_delta_peso']:.9g}",f"{x['max_delta_gradiente_clip']:.9g}",x['indice_max']] for x in peores]),'',
        'La partición K ocupa el tercio central del sesgo QKV. En aritmética exacta, añadir el mismo sesgo a todas las claves añade una constante por fila a las puntuaciones y softmax la cancela. Por ello su gradiente ideal es cero; los residuos de redondeo pueden ser pequeños y sensibles al orden de las reducciones. AdamW divide el momento de primer orden por la raíz del segundo más epsilon: cuando ambos son diminutos, diferencias pequeñas de gradientes pueden convertirse en diferencias mayores de pesos. `qkv_por_particion.json` guarda valores, momentos y gradientes de las coordenadas relevantes; no se cambia ni elimina este parámetro.','',
        '![Partición K](../resultados/v22_reproducibilidad_v20/informe/sensibilidad_sesgos_k.png)','',
        'La incidencia 21 no guardó el nombre ni los tensores del primer fallo. Estos ensayos muestran el mecanismo y su escala, pero no permiten reconstruir retrospectivamente todos los valores de aquel tensor ni atribuirle un nombre con certeza absoluta.','',
        '## Replay aislado de AdamW','',
        tabla(['Rama','Comparación','Exacto','Máx. delta','Elementos fuera'],[[b,n,x['exacto'],f"{x['max_diferencia_absoluta']:.9g}",x['elementos_fuera_tolerancia']] for b,rr in r['replay_adam'].items() for n,x in rr.items()]),'',
        'Los replays parten de los mismos pesos, momentos y gradientes recortados archivados del primer paso histórico de C1. No hay forward, backward, nuevo aumento ni actualización del modelo original. Se contrastan dos defaults, default contra el step realmente archivado y serial contra default. Un replay exacto descarta que AdamW genere espontáneamente aquella divergencia para estas entradas; no descarta su papel amplificando gradientes distintos ni garantiza equivalencia para todos los estados posibles.','',
        '## Condición de continuación y propuesta mínima','',
        f"Envolvente histórica satisfecha conjuntamente en A/B: **{hist}**. Modo determinista dentro de tolerancia en todos los pasos/estados: **{det}**. Identidad exacta comprobada en estos ensayos deterministas: **{exact_det}**. No se extrapola esta identidad a 1.452 pasos.",'',
        'La corrección mínima del control es conservar la comprobación exacta del estado recargado y añadir C1–C2 como referencia de variabilidad, con la evidencia por etapa ya implementada. El piloto histórico anterior sigue fallido. Si la envolvente histórica no pasa, este ensayo no autoriza continuar bajo ese protocolo, aunque los máximos globales de R sean menores que los de C1–C2.','',
        'Para exigir reproducibilidad numérica de trayectorias, la opción verificada es declarar explícitamente el modo determinista ensayado, regenerar diagnóstico y tratar el cambio de kernels como una modificación del protocolo de ejecución. El candidato mínimo para un próximo aislamiento es SDPA MATH más determinismo cuDNN, conservando AMP y AdamW originales; esa combinación reducida todavía no fue probada de forma aislada. Antes de proponerla para A/B debe comprobarse con el mismo C1/C2/R. No se cambia silenciosamente el entrenamiento histórico, no se amplía la tolerancia y no se adopta ningún peso diagnóstico.','',
        'Quedan abiertas la generalización a una ejecución larga, el aislamiento de cada kernel nativo residual, otros dispositivos/versiones y la distribución de variabilidad entre más pares continuos. El resultado es un diagnóstico de ejecución, no evidencia de mejora de materiales o detección.','',
        '## Artefactos y comandos','',
        '```powershell\npython scripts/diagnosticar_v20.py --salida resultados/v22_reproducibilidad_v20/preparacion\npython scripts/ejecutar_diagnostico_reproducibilidad_v20.py\npython scripts/analizar_reproducibilidad_v20.py\n# Entorno con matplotlib:\npython scripts/informar_reproducibilidad_v20.py\n```','',
        'Las rutas se publican sin reemplazo; repetir requiere un identificador nuevo. No ejecutar otra vez sobre esta entrega. No se genera `prueba_corta.json` de aprobación ni se habilita A/B.','',
        'Evidencia: [protocolo prefijado](../resultados/v22_reproducibilidad_v20/protocolo_ejecutado.json), [diagnóstico regenerado](../resultados/v22_reproducibilidad_v20/preparacion/diagnostico.json), [resumen completo](../resultados/v22_reproducibilidad_v20/resumen.json), [etapas](../resultados/v22_reproducibilidad_v20/evidencia_etapas.json), [CSV](../resultados/v22_reproducibilidad_v20/comparacion_por_etapa.csv), [QKV](../resultados/v22_reproducibilidad_v20/qkv_por_particion.json), [conclusión](../resultados/v22_reproducibilidad_v20/informe/conclusion.json), [integridad](../resultados/v22_reproducibilidad_v20/manifiesto_integridad.json). Los 36 checkpoints de paso 2, logits y gradientes están en los subdirectorios de modalidad/rama/trayectoria, con hashes en el manifiesto.','']
    with doc.open('x',encoding='utf-8') as f:f.write('\n'.join(texto))
    print(json.dumps(resultado,ensure_ascii=False))


if __name__=='__main__':main()
