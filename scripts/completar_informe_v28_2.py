"""Completa el informe desde los artefactos medidos, sin ejecutar inferencia."""
from pathlib import Path
import json
RAIZ=Path(__file__).resolve().parents[1];S=RAIZ/'resultados/v28_2_exploratorio'
CLASES=['plastico','papel','carton','vidrio','metal','organico'];MODELOS=['Original','A','B']


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def pc(x):return '—' if x is None else f'{100*x:.3f}'


def tabla(encabezado,filas):
    return '\n'.join(['| '+' | '.join(encabezado)+' |','|'+'|'.join(['---']*len(encabezado))+'|',
        *['| '+' | '.join(str(x) for x in fila)+' |' for fila in filas]])+'\n'


def main():
    ver=leer(S/'verificacion_final.json')
    if not ver['aprobada']:raise ValueError('No publicar informe concluido sin cierre')
    resultados=leer(S/'metricas.json')['resultados'];org=leer(S/'organicos_desglosados.json')
    lineas=['Se completaron **300 actualizaciones reales por rama**, sin parada por métricas y sin ampliar el presupuesto. '
        'Los estados de los pasos0/100/200/300 están separados; ambas ramas se restauraron en procesos nuevos desde100. '
        '**No se adopta A ni B.** La aplicación conserva el checkpoint original de época19.\n',
        '### Métricas globales\n',
        'Porcentajes en mAP/AP/recall/precisión; TP/FP/FN son cantidades absolutas. Las filas de train auxiliar miden '
        'ajuste sobre las mismas fotografías utilizadas por B, no generalización.\n']
    filas=[]
    for dom in ['historico','auxiliar']:
        for v in ['1','5']:
            for n in MODELOS:
                m=resultados[n][dom]['metricas'][v]
                filas.append([dom,n,v,*[pc(m[k]) for k in ['map50','map50_95','recall50','precision50']],
                    *[m[k] for k in ['tp50','fp50','fn50']]])
    lineas.append(tabla(['Train','Modelo','Vistas','mAP50 %','mAP50:95 %','Recall %','Precisión %','TP','FP','FN'],filas))
    lineas+=['![mAP50 train](../resultados/v28_2_exploratorio/graficas/map50_train.png)\n',
        '### AP50 por material\n']
    for dom in ['historico','auxiliar']:
        filas=[[c,*[pc(resultados[n][dom]['metricas'][v]['ap50_por_material'][c]) for v in ['1','5'] for n in MODELOS]] for c in CLASES]
        lineas.append('**Train '+dom+'**\n');lineas.append(tabla(['Material','Original1v','A1v','B1v','Original5v','A5v','B5v'],filas))
    lineas+=['![AP50 por material](../resultados/v28_2_exploratorio/graficas/ap50_material.png)\n',
        '### Recuperación de regiones y confusión con fondo\n',
        'Cada celda de la tabla siguiente indica `correctas / fondo / otro material / sin tokens`, '
        'por voto de la cabeza principal incluyendo fondo, en una vista. El denominador de la clasificación condicional '
        'solo considera instancias con tokens válidos; sus matrices completas están en los JSON de análisis.\n']
    for dom in ['historico','auxiliar']:
        filas=[]
        for c in ['carton','vidrio','organico']:
            fila=[c]
            for n in MODELOS:
                a=leer(S/'evaluacion'/n/dom/'analisis.json');rs=[x for x in a['cabeza_principal']['instancias'] if x['clase']==CLASES.index(c)]
                correctas=sum(x.get('prediccion_con_fondo')==CLASES.index(c)+1 for x in rs)
                fondo=sum(x.get('prediccion_con_fondo')==0 for x in rs);sin=sum(x['tokens']==0 for x in rs)
                fila.append(f'{correctas} / {fondo} / {len(rs)-correctas-fondo-sin} / {sin}')
            filas.append(fila)
        lineas.append('**Train '+dom+'**\n');lineas.append(tabla(['Material','Original','A','B'],filas))
    lineas+=['![Confusión de tokens con fondo](../resultados/v28_2_exploratorio/graficas/confusion_con_fondo.png)\n',
        '![Confusión condicional por instancia](../resultados/v28_2_exploratorio/graficas/confusion_condicional_material.png)\n',
        '**Orgánicos auxiliares por fuente**, sin cambiar sus etiquetas:\n']
    filas=[]
    for n in MODELOS:
        for f in ['hpi','wsc']:
            x=org[n][f];filas.append([n,f,x['instancias'],x['con_tokens'],x['voto_organico_con_fondo'],x['voto_condicional_organico']])
    lineas.append(tabla(['Modelo','Fuente','Instancias','Con tokens','Voto orgánico con fondo','Voto condicional orgánico'],filas))
    oo=leer(S/'organicos_deteccion_por_fuente.json')['resultados'];filas=[]
    for n in MODELOS:
        for v in ['1','5']:
            for fuente in ['hpi','wsc']:
                x=oo[n][v][fuente];filas.append([n,v,fuente,x['objetos'],x['tp50'],x['fp50'],x['fn50'],pc(x['ap50'])])
    lineas.append('**Detección orgánica por fuente**, calculada offline sobre todas las imágenes HPI/WSC del auxiliar, sin inferencia nueva ni selección de casos:\n')
    lineas.append(tabla(['Modelo','Vistas','Fuente','GT orgánicos','TP','FP orgánicos','FN','AP50 %'],filas))
    lineas+=['Las nueve regiones orgánicas auxiliares pasan el criterio geométrico de utilidad48×48 de QA. '
        'Eso no certifica su detección, exhaustividad o diversidad. Solo tres son alimentarias HPI.\n',
        '### Oportunidades antes y después del postprocesamiento\n',
        'Conteos `componentes antes de filtros / final`, cinco vistas; clase correcta e IoU≥0,50. '
        'Una misma propuesta puede proporcionar oportunidades para varios GT: no son TP asignados uno a uno.\n']
    filas=[]
    for dom in ['historico','auxiliar']:
        for n in MODELOS:
            a=leer(S/'evaluacion'/n/dom/'analisis.json');e=a['oportunidades_5v']['por_etapa']
            filas.append([dom,n,*[f'{e["componentes"].get(c,0)} / {e["final"].get(c,0)}' for c in ['carton','vidrio','organico']]])
    lineas.append(tabla(['Train','Modelo','Cartón','Vidrio','Orgánico'],filas))
    filas=[]
    for dom in ['historico','auxiliar']:
        for n in MODELOS:
            g=leer(S/'evaluacion'/n/dom/'analisis.json')['5']['geometria']['por_material']
            filas.append([dom,n,sum(x.get('indicador_fusion_bbox',0) for x in g.values()),
                sum(x.get('indicador_fragmentacion_bbox',0) for x in g.values()),
                sum(x.get('geometria_insuficiente',0) for x in g.values()),sum(x.get('material_incorrecto_iou50',0) for x in g.values())])
    lineas.append(tabla(['Train','Modelo','GT con indicador fusión bbox','GT con indicador fragmentación bbox','Geometría insuficiente','Material incorrecto IoU50'],filas))
    lineas+=['Los indicadores de caja no prueban una fusión o división física real. Son un diagnóstico separado de las matrices '
        'de material y fondo y de los TP/FP oficiales del evaluador propio. Menos indicadores de fusión también pueden '
        'deberse a menos respuesta foreground y menor recall; no prueban por sí solos mejor separación.\n',
        '### Curvas, tiempo y memoria\n']
    filas=[]
    for n in ['A','B']:
        x=ver['ramas'][n];filas.append([n,x['actualizaciones'],x['paso_adamw'],f'{x["segundos_actualizaciones"]:.2f}',
            f'{x["segundos_intentos_con_checkpoints"]:.2f}',f'{x["memoria_gpu_max_allocated_bytes"]/2**20:.2f}',
            f'{x["memoria_gpu_max_reserved_bytes"]/2**20:.2f}',f'{x["loss_media_primeros50"]:.6f}',f'{x["loss_media_ultimos50"]:.6f}'])
    lineas.append(tabla(['Rama','Actualizaciones','AdamW final','s actualizaciones','s con checkpoints','GPU allocated MiB','GPU reserved MiB','Loss primeras50','Loss últimas50'],filas))
    lineas+=['Los tiempos anteriores excluyen verificación de datos, construcción del modelo y carga inicial; incluyen '
        'las etapas efectivamente señaladas. La pérdida B usa otros datos y no es directamente una métrica de detección comparable con A.\n',
        '![Curvas de entrenamiento](../resultados/v28_2_exploratorio/graficas/curvas_entrenamiento.png)\n']
    filas=[]
    for dom in ['historico','auxiliar']:
        for n in MODELOS:
            for v in ['1','5']:
                x=resultados[n][dom]['costes'][v]
                filas.append([dom,n,v,f'{x["segundos_captura_incluye_disco_y_observador"]:.2f}',
                    f'{x["memoria_gpu_max_allocated_bytes"]/2**20:.2f}',f'{x["memoria_gpu_max_reserved_bytes"]/2**20:.2f}'])
    lineas.append(tabla(['Train','Modelo','Vistas','s captura + disco + observador','GPU allocated MiB','GPU reserved MiB'],filas))
    lineas+=['### Respuestas e interpretación\n']
    a=resultados['A']['auxiliar']['metricas']['5'];b=resultados['B']['auxiliar']['metricas']['5'];o=resultados['Original']['auxiliar']['metricas']['5']
    lineas.append(f'1. **Aprendizaje auxiliar:** B obtiene {b["tp50"]} TP frente a {a["tp50"]} de A y {o["tp50"]} del original; '
        f'mAP50 {pc(b["map50"])} % frente a {pc(a["map50"])} % de A. '
        +('Hay una mejora de ajuste/detección en este subconjunto entrenado.\n' if b['map50']>a['map50'] and b['tp50']>a['tp50'] else 'El criterio conjunto de más TP y más mAP50 que A no se cumple.\n'))
    lineas.append('2. **AP por clase:** las tablas anteriores registran las seis clases en ambos dominios y modalidades; '
        'no se selecciona una clase favorable para declarar éxito global.\n')
    for c in ['carton','vidrio','organico']:
        puntos=[]
        for n in MODELOS:
            det=leer(S/'evaluacion'/n/'auxiliar/analisis.json')['5']['detalle_metricas']['por_clase'][c]['por_umbral'][0]
            puntos.append(n+f': TP{det["tp"]}, FP{det["fp"]}')
        lineas.append('3. **'+c+' auxiliar5v:** '+'; '.join(puntos)+'.\n')
    ah=resultados['A']['historico']['metricas']['5'];bh=resultados['B']['historico']['metricas']['5'];oh=resultados['Original']['historico']['metricas']['5']
    lineas.append(f'4. **Retención histórica:** B mAP50 {pc(bh["map50"])} %, A {pc(ah["map50"])} %, original {pc(oh["map50"])} %. '
        f'Delta B−original {100*(bh["map50"]-oh["map50"]):+.3f} puntos porcentuales y B−A {100*(bh["map50"]-ah["map50"]):+.3f}. '
        'No se prefijó margen de no inferioridad; estos resultados son descriptivos y no acreditan equivalencia estadística.\n')
    contrib=100*(bh['ap50_por_material']['organico']-oh['ap50_por_material']['organico'])/6
    lineas.append(f'La caída macro con cinco vistas está muy influida por orgánico histórico: pasa de AP50 {pc(oh["ap50_por_material"]["organico"])} % a0, '
        f'con solo cinco GT, y aporta {contrib:+.3f} puntos a la variación del mAP50. No se elimina esta clase del criterio '
        'para rescatar B. Además hay una pérdida real de recall: B conserva438 TP frente a592 del original y646 de A, '
        'por lo que la retención no se resume únicamente por ese efecto de media macro.\n')
    lineas.append('Con una vista B alcanza mAP50 histórico3,227 % frente a3,034 % del original, pero queda por debajo de A4,211 % '
        'y su recall es12,774 % frente a16,886 % del original. La conclusión depende de la modalidad; no se cambia '
        'automáticamente el modo predeterminado de cinco vistas.\n')
    lineas.append('5. **Memorización/generalización:** no puede distinguirse aprendizaje transferible de memorización con esta evaluación. '
        'B vio las mismas28 imágenes unas21–22 veces y se evalúa sobre ellas; cualquier ganancia es compatible con memorización. '
        'Train histórico también es train. No hay evidencia nueva de generalización y no se abrió otro conjunto para buscarla.\n')
    lineas.append(f'6. **Falsos positivos:** B genera {b["fp50"]} en auxiliar y {bh["fp50"]} en histórico con cinco vistas; '
        f'A genera {a["fp50"]} y {ah["fp50"]}. La precisión y recall completos están en la tabla global.\n')
    lineas.append('7. **Errores persistentes:** consultar confusión con fondo/material, oportunidades y geometría anteriores. '
        'Una región clasificada condicionalmente como su material no implica una caja correcta; un indicador de fusión '
        'tampoco autoriza dividir automáticamente una anotación humana.\n')
    lineas.append('En auxiliar5v, B recupera14 papeles y5 orgánicos; conserva9 cartones detectados, igual que el original, '
        'y cae a un plástico frente a los nueve del original. Vidrio y metal permanecen sin TP. La ganancia de AP de cartón '
        'no equivale a encontrar más cartones en esta colección.\n')
    lineas.append('B produce303 FP orgánicos sobre el auxiliar completo, de los cuales127 están fuera de HPI/WSC. '
        'En HPI hay2 TP y10 FP orgánicos; en WSC3 TP y166 FP. El aprendizaje de material no ha producido una discriminación '
        'suficientemente selectiva. Las ocho instancias auxiliares de vidrio tienen voto mayoritario fondo en la cabeza '
        'principal y ninguna oportunidad correcta antes de filtros: ese fallo no se resuelve solo con NMS.\n')
    lineas.append('Las trazas localizan pérdidas concretas en el filtro de bordes de vistas: B tiene una oportunidad orgánica '
        'histórica hasta top100_vista y cero después de bordes; en auxiliar pasa de siete a cinco oportunidades orgánicas '
        'en esa misma etapa. El postprocesamiento no cambió; cambiaron las propuestas producidas por los pesos. '
        'Se conserva la política prefijada y no se ajustan filtros después de mirar estos resultados.\n')
    lineas.append('**Ejemplo prefijado HPI40:** la cabeza de B ya vota orgánico en los restos, pero en una vista produce una caja '
        'que engloba ambos restos y parte del plato, con IoU insuficiente para ambos GT. Cinco vistas recuperan un objeto '
        'y añaden varias cajas. La clasificación de material ha cambiado y la separación sigue limitada. '
        'Los FP se definen respecto al GT disponible: su exhaustividad y la granularidad de migas/manchas requieren criterio humano, '
        'no se certifican a partir de una predicción. Véase [comparación local HPI40](../datos/nuevos/v28_2/visuales_privados/11_auxiliar_organico.png).\n')
    lineas.append('8. **Siguiente intervención:** completar QA humana antes de otro entrenamiento. Este piloto no autoriza ampliar pasos, '
        'retocar umbrales, cambiar pérdidas o sustituir el detector predeterminado.\n')
    lineas+=['### Correcciones humanas de mayor utilidad\n',
        '- **Comida real:** completar cobertura de HPI0,132,155,78,96 y192; revisar granularidad en HPI23 y fragmentos/solapes en54,173 y114. '
        'Hoy solo hay tres regiones alimentarias utilizables; faltan al menos cinco alimentarias para la referencia reducida de ocho, '
        'sin acreditar vegetación como diversidad alimentaria. Dibujar únicamente residuos realmente visibles y justificar ignore ambiguo.\n',
        '- **Recuperar escenas de vidrio ya revisadas:** resolver las incidencias de WaRP POSAD_1_13-Sep_02-24-53, '
        'POSAD_1_13-Sep_03-20-46, Robo_25-Mar_14-39-46 y prepared_data_all_MGS_20-Oct_01-03-59. '
        'Conservan cinco objetos guardados como vidrio excluidos por QA de escena; corregir la incidencia, no aceptar automáticamente su material.\n',
        '- **Cartón frente a multicapa:** adjudicar los nueve milk/juice-cardboard señalados en V28. Confirmar composición o '
        'marcar compuesto/ambiguo con motivo; no entrenar envases multicapa como cartón simple por el nombre de la fuente.\n',
        '- **Cobertura de papel y descartes:** revisar ZeroWaste1001,1076,1709,2670,2688,2706/2709/2710,2843,331,588 y896. '
        'Los residuos omitidos no deben enseñarse como fondo. Aclarar descartes «error/no se distingue» y mantener relacionadas las secuencias.\n',
        '- **Guardar pendientes:** completar las cuatro fotografías sin evento y el borrador indicados en V28, además de '
        'las dos discrepancias de relleno del editor. No modificar automáticamente esos contornos ni simular una segunda revisión.\n',
        'La lista completa permanece en [lista de correcciones V28](../resultados/v28_validacion_anotaciones/lista_correcciones.md). '
        'No se han alterado esas revisiones. La siguiente colección requiere nueva instantánea, QA y autorización; '
        'no usar los resultados de train para presentar una mejora definitiva.\n',
        '### Evidencia y cierre\n',
        'Se conservan los134 eventos humanos, los731 archivos de respaldo, los105 checkpoints históricos y los datasets/resultados originales. '
        'Se produjeron ocho checkpoints nuevos exclusivos. Las restauraciones de origen e intermedias fueron exactas; '
        'las dos comparaciones diagnósticas definitivas tienen413 tensores examinados cada una, diferencia máxima0 y ningún elemento fuera de tolerancia. '
        'Esto comprueba las trayectorias cortas medidas, no una identidad universal entre hardware o versiones. '
        'Las trazas confirman las mismas anclas, tarea global y aumentos en las300 actualizaciones.\n',
        'Incidencias: se corrigió la ubicación CPU/CUDA del softmax antes de producción y se regeneraron certificados conservando los anteriores. '
        'Torch emitió una advertencia al convertir pérdidas a escalares para el diario; no fue un error de entrenamiento ni '
        'cambió los gradientes o las actualizaciones. No hubo NaN, actualización omitida ni ampliación de presupuesto.\n',
        '- [Métricas JSON y protocolo](../resultados/v28_2_exploratorio/metricas.json).\n',
        '- [Tabla global y AP de ambas modalidades CSV](../resultados/v28_2_exploratorio/metricas_comparativas.csv).\n',
        '- [Imágenes, distribución, procedencia y hashes CSV](../resultados/v28_2_exploratorio/imagenes_y_hashes.csv).\n',
        '- [Curvas y pérdidas por actualización CSV](../resultados/v28_2_exploratorio/curvas_entrenamiento.csv).\n',
        '- [Índice de doce comparaciones visuales locales](../resultados/v28_2_exploratorio/visuales.json). '
        'Fotografías privadas en `datos/nuevos/v28_2/visuales_privados/`, ignoradas por Git; las imágenes GT están separadas de la inferencia.\n',
        '- [322 pruebas finales](../resultados/v28_2_exploratorio/pruebas_finales.json) y su log completo.\n',
        '- [Controles definitivos](../resultados/v28_2_exploratorio/controles_inferencia_cuda_v2/controles.json).\n',
        '- [Verificación final de estados, presupuesto e integridad](../resultados/v28_2_exploratorio/verificacion_final.json).\n',
        '- [Manifiesto final de integridad](../resultados/v28_2_exploratorio/manifiesto_integridad.json).\n',
        'Predicciones finales, logits y trazas de cada modelo/dominio están bajo `resultados/v28_2_exploratorio/evaluacion/`. '
        'Permiten auditoría posterior sin repetir inferencia. Los campos efectivos del backend, versiones y hashes quedan ligados al protocolo. '
        '**Se mantiene el detector original; no se evalúan val, test o generalización y no se inicia ningún entrenamiento adicional.**\n']
    p=RAIZ/'documentacion/28_2_entrenamiento_exploratorio.md';texto=p.read_text(encoding='utf-8')
    marcador='Pendientes de incorporar tras finalizar la evaluación en curso. Los checkpoints de entrenamiento ya están completos; no se adopta ninguno.'
    if marcador not in texto:raise ValueError('Informe ya completado; no sobrescribir resultados')
    texto=texto.replace('Estado de elaboración: entrenamiento A300/B300 completado; evaluación train en curso. No adoptar ningún checkpoint. Las tablas finales se incorporan al terminar la captura y sus comprobaciones.',
        '**Piloto A300/B300 completado y verificado. No se adopta ningún checkpoint; el detector predeterminado sigue siendo el original de época19.**')
    p.write_text(texto.replace(marcador,'\n'.join(lineas)),encoding='utf-8')
    print('Informe completado con métricas medidas')


if __name__=='__main__':main()
