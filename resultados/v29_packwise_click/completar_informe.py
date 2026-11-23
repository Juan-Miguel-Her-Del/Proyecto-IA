"""Lectura posthoc de resultados congelados; no carga modelos ni entrena."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'resultados/v29_packwise_click'
CLASES=['plastico','papel','carton','vidrio','metal','organico']
DOMINIOS=['historico_train','nuevo_train','nuevo_retenido']
MODELOS=['Original','A','B']


def leer(p):return json.loads(Path(p).read_text(encoding='utf8'))


def main():
    doc=ROOT/'documentacion/29_entrenamiento_packwise_click.md';texto=doc.read_text(encoding='utf8')
    if '## Lectura final de los resultados' in texto:raise FileExistsError('Complemento ya publicado')
    res=leer(OUT/'metricas.json')['resultados'];detalle={};lineas=['','## Lectura final de los resultados','',
        'Las siguientes diferencias se calculan sobre el mismo presupuesto y las mismas imágenes de cada cohorte. Se expresan en puntos porcentuales; son resultados de una semilla, sin prueba de significación ni garantía de independencia física.','',
        '| Cohorte, cinco vistas | B−A mAP50 | B−Original mAP50 | B−A mAP50:95 |','|---|---:|---:|---:|']
    for dom in DOMINIOS:
        a,b,o=[res[n][dom]['metricas']['5'] for n in ['A','B','Original']]
        lineas.append(f'| {dom} | {100*(b["map50"]-a["map50"]):+.3f} | {100*(b["map50"]-o["map50"]):+.3f} | {100*(b["map50_95"]-a["map50_95"]):+.3f} |')
    hist=res['B']['historico_train']['metricas']['5'];ret=res['B']['nuevo_retenido']['metricas']['5']
    ar=res['A']['nuevo_retenido']['metricas']['5'];origen=res['Original']['historico_train']['metricas']['5']
    if ret['map50']>ar['map50']:
        lineas+=['','B supera a A en la métrica agregada de la cohorte nueva retenida. La ganancia proviene principalmente de plástico: AP50 B=20.091 %, A=6.131 %, Original=4.544 %. Papel alcanza 0.704 % y metal permanece en 0 % con B. Esto apoya una mejora parcial, sin acreditar mejora global de seis clases ni atribuir el efecto exclusivamente a calidad de datos: B recibe más imágenes y cómputo.']
    else:
        lineas+=['','B no supera a A en mAP50 de la cohorte nueva retenida. No se considera demostrada una mejora de detección transferida por incorporar estas fuentes bajo este presupuesto. Una ganancia en train, si existe, acredita ajuste y no generalización.']
    if hist['map50']<origen['map50']:
        lineas+=['','B queda por debajo de la referencia original en mAP50 de train histórico. A también debe compararse con esa referencia: no debe atribuirse toda caída al dato nuevo cuando continuar el entrenamiento histórico también altera el resultado.']
    lineas+=['','El promedio histórico es sensible a la clase orgánica: solo hay cinco GT. La referencia logra un TP y AP50=10.396 %, mientras A y B tienen cero TP y AP50=0 %. Al promediar seis clases, esa pérdida aporta −1.733 puntos de mAP50, parcialmente compensados por otras clases. B aumenta TP totales de 592 a 634 y reduce FP de 16239 a 15613 frente al original, aunque su mAP50 agregado baja. No corresponde describirlo como un deterioro uniforme de todas las clases ni ocultar la pérdida del único acierto orgánico.','',
        'Papel también muestra un límite de transferencia: B obtiene AP50=8.298 % en train nuevo y 0.704 % en la cohorte retenida. Sus composiciones y grupos difieren; esta brecha merece revisión y no demuestra por sí sola un mecanismo exclusivo de memorización.']
    lineas+=['','**Decisión: mantener el checkpoint original como predeterminado y conservar A/B como experimentales.** La cohorte retenida no contiene cartón, vidrio u orgánico; no se puede concluir que se resolvieron sus fallos históricos de validación. Train nuevo solo añade seis instancias de cartón y ninguna positiva de vidrio u orgánico. La CE softmax sí aplica presión negativa a esos logits en otras clases y en fondo: ausencia de positivos no equivale a ausencia de gradiente.','',
        'El siguiente esfuerzo debe priorizar fuentes con instancias fiables de cartón, vidrio y restos alimentarios, y una colección independiente con procedencia/objetos/sesiones verificables. Antes de otro entrenamiento conviene contrastar en las trazas si la cabeza recupera material pero pierde geometría o separación de instancias. No se recomienda repetir otra pérdida ni ampliar este presupuesto para forzar una mejora.','',
        '### Recuperación desde la cabeza principal','',
        'Voto mayoritario sobre tokens GT válidos en una vista; se excluyen objetos sin tokens, padding, ignore y mezclas. “Con fondo” exige que el material gane a fondo; “condicional” compara solo seis materiales. Estas cifras proceden de `modelo.cabezal`, no de la cabeza auxiliar global.','',
        '| Cohorte | Modelo | Material | Instancias con tokens | Recuperadas con fondo | Recuperadas condicionales |','|---|---|---|---:|---:|---:|']
    for dom in DOMINIOS:
        detalle[dom]={}
        for n in MODELOS:
            d=leer(ROOT/res[n][dom]['analisis']);c=d['cabeza_principal'];cm=c['matriz_instancias_con_fondo'];cc=c['matriz_instancias_material_condicional']
            filas=[]
            for i,cl in enumerate(CLASES):
                total=sum(cm[i+1]);f={'material':cl,'instancias_con_tokens':total,'recuperadas_con_fondo':cm[i+1][i+1],
                    'confundidas_con_fondo':cm[i+1][0],'recuperadas_condicionales':cc[i][i]}
                filas.append(f)
                if total:lineas.append(f'| {dom} | {n} | {cl} | {total} | {f["recuperadas_con_fondo"]} | {f["recuperadas_condicionales"]} |')
            etapas=d['oportunidades_5v']['por_etapa'];geo=d['5']['geometria']['por_material']
            detalle[dom][n]={'recuperacion_material':filas,'oportunidades_5v':etapas,'geometria_5v':geo}
    lineas+=['','### Oportunidades y geometría de las cajas','',
        'Una oportunidad es una GT con alguna propuesta del material correcto e IoU≥0.5; permite que una propuesta cubra varias GT y no equivale a TP. Fusión/fragmentación son indicadores de cajas, sin adjudicación visual de identidad.','',
        '| Cohorte | Modelo | Oportunidades tras componentes | Oportunidades finales | Geometría insuficiente | Material erróneo con IoU≥0.5 | Indicadores de fusión |','|---|---|---:|---:|---:|---:|---:|']
    for dom in DOMINIOS:
        for n in MODELOS:
            d=detalle[dom][n];et=d['oportunidades_5v'];g=d['geometria_5v']
            total=lambda key:sum(v.get(key,0) for v in g.values())
            lineas.append(f'| {dom} | {n} | {sum(et["componentes"].values())} | {sum(et["final"].values())} | {total("geometria_insuficiente")} | {total("material_incorrecto_iou50")} | {total("indicador_fusion_bbox")} |')
    pruebas=leer(sorted((OUT/'pruebas').glob('*/pruebas.json'))[-1]);ctrl=leer(OUT/'controles/controles.json')
    lineas+=['','### Reproducibilidad, incidencias y lectura de los artefactos','',
        f'- Suite completa: {pruebas["resultados"]["suite"]["pruebas"]} casos únicos, conservando los 409 anteriores. El caso COCO omitido en el entorno CUDA se ejecuta además en la batería CPU de {pruebas["resultados"]["codec_cpu"]["pruebas"]} casos; no se suman como casos únicos distintos.',
        '- Las trayectorias diagnósticas C1/C2/R de cuatro pasos coinciden bit a bit en las comparaciones registradas de A y B. Esta observación no acredita identidad bit a bit universal de 600 pasos. El control A del paso 300 también coincide por huellas de pesos, AdamW, scaler y RNG con V28.2; véase `comparacion_control_historico_paso300.json`.',
        '- Las 600 entradas y aumentos históricos coinciden entre A y B. Las confirmaciones por paso y sus hashes son la autoridad del presupuesto y de la reanudación.',
        '- Los campos legacy `epoca=19`, `mejor_epoca` y `mejor_map` en los checkpoints experimentales describen el origen heredado por compatibilidad. No son nuevas métricas de validación. La identidad efectiva del experimento está en `v29.rama/paso/contrato` y `metricas.json`; véase `checkpoints_y_metadatos.json`.',
        '- La evaluación se pausó y reanudó. El coste registrado de B/nuevo_train corresponde al intento que cerró la captura, con relectura de logits y trabajo restante; no incluye todo el intento previo ni el intervalo de pausa. No constituye una medición aislada de latencia ni un total acumulado de evaluación. Los tiempos de entrenamiento corresponden a las etapas confirmadas completas.',
        '- La referencia histórica train se reutiliza con comprobación de pesos, entradas, metadatos, backend y hashes de capturas V28.2. No se vuelve a medir el 2.912 % de la validación histórica ni se evalúa test.',
        '- El evaluador propio maneja ignore mediante cajas conservadoras; los RLE originales y sus huecos permanecen en COCO y en las máscaras de entrenamiento. Los consumidores externos deben respetar el contrato `ignore` y no entrenar su categoría como un séptimo material.',
        '- En los ejemplos visuales se dibujan las GT positivas; los objetos ignorados pueden verse sin caja GT en ese panel. Las predicciones proceden de una inferencia previa sin GT.',
        '- Las incidencias de descarga, el ajuste de entorno de la suite y la exportación parcial por ruta relativa se conservan en `incidencias_operativas.json`. No se modificó la hipótesis, el presupuesto, el modelo ni la tolerancia para alterar resultados.',
        '- El estado final de integridad se consulta en `integridad_final.json` y su recibo. El test original y la colección de generalización siguen reservados. Las anotaciones de autores no se contabilizan como aceptación V26 ni como revisiones del usuario.','',
        '### Gráficas y referencias','',
        '![Comparación mAP50 cinco vistas](../resultados/v29_packwise_click/graficas/map50_5v.png)','',
        '![Curvas de pérdidas](../resultados/v29_packwise_click/graficas/curvas_train.png)','',
        '![Distribución de clases y tamaños](../resultados/v29_packwise_click/graficas/soportes_y_tamanos.png)','',
        '- Roming, L.; Rüger, S.; Kalb, M.; Beutler, F. PackWISE, versión 2. [DOI: 10.24406/fordatis/420.2](https://doi.org/10.24406/fordatis/420.2). CC BY 4.0; conservar también el README original.',
        '- Baishya, M. CLICK Trash Dataset, versión 2. [DOI: 10.17632/htcvvz6fm2.2](https://doi.org/10.17632/htcvvz6fm2.2). CC BY 4.0 según la ficha oficial de Mendeley; discrepancia descriptiva archivada.','']
    with (OUT/'analisis_complementario.json').open('x',encoding='utf8') as f:json.dump(detalle,f,ensure_ascii=False,indent=2)
    with doc.open('a',encoding='utf8') as f:f.write('\n'.join(lineas))
    print('Complemento de informe publicado',flush=True)


if __name__=='__main__':main()
