"""Informe V31 desde JSON; no entrena ni realiza inferencia."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RAIZ=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
CL=['plastico','papel','carton','vidrio','metal','organico']


def leer(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pct(x):return 'sin GT' if x is None else f'{100*x:.3f} %'


def main():
    cfg=leer(RAIZ/'configuracion/instancias_v31.json');tests=leer(OUT/'pruebas.json');mem=leer(OUT/'controles/memorizacion.json');co=leer(OUT/'controles/reanudacion.json');datos=leer(OUT/'datos_y_pesos.json')
    doc=['# V31 — detección por instancias con CNN + Transformer propios','',
        'El usuario corrigió el objetivo: mejorar el detector, conservando el clasificador independiente. No se inició entrenamiento del clasificador ni se modificaron sus pesos. Esta fase implementa y prueba detección por instancia propia; no importa modelos ni pesos preentrenados de terceros o época 19.','',
        '## Problema e hipótesis','',
        'V30 mejoró el ajuste frente a su control pero quedó por debajo del detector original en desarrollo: 1,192 % frente a 2,912 % mAP50 cinco vistas. Sus confusiones materiales persistían y los mapas 48×48 perdían supervisión de objetos pequeños; solo una región orgánica tenía tokens interiores elegibles. Separar instancias no soluciona por sí solo ese problema de datos/material, por lo que se establece un control de extractor con la misma cabeza y supervisión por objeto.','',
        'La hipótesis es que las cajas GT pueden supervisar directamente localización y material por objeto, incluidos aquellos que no producen una región semántica suficiente. No se reconstruyen instancias mediante componentes ni se crean cajas desde etiquetas globales. Se conserva que solo hay cinco orgánicos históricos y que el vidrio se concentra en pocas escenas.','',
        '## Arquitectura','',
        '```mermaid','flowchart LR','I["RGB 384×384"] --> E["A: patches lineales / B: CNN residual propia"]','E --> T["Transformer local y global propios"]','T --> F["Fusión 48×48×96 y máscara de padding"]','F --> Q["100 consultas; dos bloques de atención cruzada"]','Q --> C["6 materiales y ausencia de objeto"]','Q --> B["Caja continua xyxy"]','```','',
        'Se reutilizan las capas propias de V30, Bloque/Atencion y la cabeza de consultas propia. A tiene 674129 parámetros y B 1068905. La CNN B usa Conv2d/GroupNorm/SiLU/residuales y está inicializada aleatoriamente. Se comparten las inicializaciones aleatorias de los parámetros compatibles entre A/B. El Transformer afecta las características que usan los clasificadores y regresores, con gradientes comprobados.','',
        'La asignación uno a uno y la predicción de conjuntos se relacionan con [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), sin descargar su código/modelo/pesos ni ResNet. La implementación de asignación es la del proyecto. La posición y las proyecciones Q/K/V son propias; se excluye padding de atención y se conservan regiones ignore en la pérdida.','',
        'Cien consultas bastan para los máximos train verificados: 53 objetos históricos y 62 nuevos. No se trunca GT. Cajas subtoken conservan sus cuatro coordenadas y su material. La inferencia de una vista no aplica componentes ni NMS; cinco vistas conserva el filtro de bordes y NMS 0,5 entre propuestas combinadas, que también puede suprimir objetos muy solapados: se evalúa por separado.','',
        '## Datos y protocolo','',
        '| Fuente | Imágenes train | Objetos GT |','|---|---:|---:|','| TACO histórico | 964 | 2505 |','| PackWISE/CLICK congelados V29 | 1050 | 4624 |','| Total | 2014 | 7129 |','',
        'Los conteos conjuntos son 5375 plástico, 732 papel, 155 cartón, 203 vidrio, 659 metal y 5 orgánico. No se añadieron nuevas fotos ni se acredita mayor diversidad física por sobremuestrear. Se mantienen las licencias/procedencia, las etiquetas de autores, polígonos/RLE originales y cajas derivadas de máscaras de V29. Ninguna propuesta IA pendiente se convierte en revisión humana o aceptación V26. No se incorpora aún WaRP completo ni se afirma que la colección sea mayor.','',
        'A/B comparten 8 imágenes históricas + 2 PackWISE + 2 CLICK y 16 ejemplos globales RealWaste por actualización. La clasificación global es una tarea auxiliar interna del detector con coeficiente 0,5: no modifica el clasificador independiente y no crea cajas. El microbatch nuevo usa coeficiente 0,5.','',
        'Se reutiliza la pérdida de consultas densidad_ponderada ya implementada en V10: CE positiva ponderada por material, CE de ausencia de objeto con factor 0,1, L1 de caja con factor 5 y GIoU con factor 2; normalización por objetos asignados. No se añade otra pérdida discriminativa V20. Pesos calculados exclusivamente con GT train, cap final 10: '+str(datos['pesos_material'])+'.','',
        'El muestreo histórico usa el promedio de pesos de los materiales presentes por imagen, normalizado y limitado a peso 3. No se interpreta ese peso como nuevas imágenes ni garantía de probabilidad exactamente triple tras normalizar. Los priores de tamaño se calculan únicamente de cajas train; son estadísticas globales, no GT por imagen en inferencia.','',
        'Semilla 3101, AdamW LR 0,0005, weight decay 0,01, AMP con GradScaler inicial 1024, clipping 1. Backend V23 completo: SDPA MATH, cuDNN determinista, benchmark false, algoritmos deterministas, TF32 false, CUBLAS_WORKSPACE_CONFIG=:4096:8 y foreach/fused false.','',
        'Presupuesto máximo fijo 2400 actualizaciones por rama desde cero. Checkpoints 0/400/800/1600/2400. Puerta train a 400: caída media de loss ≥10 % entre primeras/últimas 40 actualizaciones. Si se supera, se completa el presupuesto sin parada por métricas de desarrollo ni ampliación por resultados bajos. Mejor checkpoint por mAP50 desarrollo una vista entre 800/1600/2400, empate al primero; se reporta también el final.','',
        'A/B aíslan el extractor dentro de esta familia de consultas. Frente a V30 cambian cabeza/objetivo y muestreo, por lo que no se atribuye toda diferencia histórica únicamente a la CNN o a la cabeza. Una semilla no permite afirmar robustez estadística.','',
        '## Controles técnicos y aprendizaje pequeño','',
        f'La suite tiene {tests["resultados"]["suite"]["pruebas"]} casos, preservando los 457 anteriores y añadiendo 15. Un codec omitido en el entorno ML se comprueba en una batería CPU de 19 pruebas (no se suman como casos nuevos). Se cubren gradientes, vacío, padding, dos objetos cercanos, cajas subtoken, flips/inversa, guardado/carga y rechazo de protocolo/rama/presupuesto/preentrenados.','',
        'C1 y C2 independientes de cuatro AdamW, y R dos pasos + guardar/restaurar + dos: entradas, salidas, losses, gradientes antes/después de clipping, pesos, AdamW, scaler y RNG; tolerancia rtol=1e-4/atol=1e-5 y restauración de estados exigida. Exactitud observada de las ocho comparaciones por rama: '+str({n:all(x['exacto'] for x in r['comparaciones']) for n,r in co['ramas'].items()})+'. No se afirma identidad bit a bit universal entre versiones/hardware.','',
        'Memorización: ocho escenas train, mínimo ID por clase y escenas con objetos cercanos, 24 GT. Cuatrocientos pasos descartables por rama, sin aumentos, mismas cajas originales. Gates prefijados mAP50≥0,7, recall≥0,6 y al menos un TP de cada material. GT solo se usa en pérdida y análisis; la inferencia recibe imagen y geometría de padding.','',
        '| Rama | mAP50 | mAP50:95 | TP | FP | FN | Aprobada |','|---|---:|---:|---:|---:|---:|---|']
    for n,r in mem['ramas'].items():
        m=r['metricas'];doc.append('| '+' | '.join([n,pct(m['map50']),pct(m['map50_95']),str(m['tp50']),str(m['fp50']),str(m['fn50']),str(r['aprobada'])])+' |')
    doc+=['','Se descartan esos pesos; el resultado solo demuestra capacidad de ajuste de esas escenas, no generalización. Los ejemplos de orgánico subtoken quedan supervisados por sus cajas, aunque ello no añade comida distinta ni garantiza que otros alimentos sean reconocibles.','']
    figdir=OUT/'graficas';figdir.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,4))
    for n,r in mem['ramas'].items():ax.plot([h['paso'] for h in r['historial']],[h['loss'] for h in r['historial']],label=n)
    ax.legend();ax.set(xlabel='Actualización',ylabel='Loss',title='Memorización train-only, pesos descartados');fig.tight_layout();fig.savefig(figdir/'memorizacion.png',dpi=150);plt.close(fig)
    if mem['aprobada']:
        mets={n:leer(OUT/f'metricas_{n}.json') for n in ['A','B']};ref=leer(RAIZ/'resultados/v30_hibrido_revision02/metricas_piloto.json')['resultados']['Original'];filas=[]
        doc.insert(2,'Resultado sin adopción automática: A final 5v '+pct(mets['A']['pasos']['2400']['resultados']['desarrollo']['metricas']['5']['map50'])+', B final 5v '+pct(mets['B']['pasos']['2400']['resultados']['desarrollo']['metricas']['5']['map50'])+', referencia '+pct(ref['desarrollo']['metricas']['5']['map50'])+'. Ver decisión y diagnóstico cuantitativo al final.\n')
        doc+=['## Evaluación final y selección','',
            'Inferencia sin GT, umbral 0,05, ausencia de objeto descartada, top100, una y cinco vistas en desarrollo/retención. Train completo se mide con una vista para diagnóstico; no se inventa un mAP train cinco vistas. Evaluador propio de 101 recalls con ignore bbox, no COCOeval oficial.','',
            'Desarrollo histórico: 181 fotos/361 GT, usado previamente por el proyecto y para seleccionar estos checkpoints. Retenido V29: 240 fotos/1090 GT, solo plástico/papel/metal, independencia física no demostrada y ya observado en V29/V30. Test y colección de generalización V26 permanecen reservados. No se usa el retenido para seleccionar checkpoints.','',
            '| Variante | Dominio | Vistas | mAP50 | mAP50:95 | Recall | Precisión | TP | FP | FN |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for n,d in mets.items():
            for variante,datos_v in [('final',d['pasos']['2400']['resultados']),('mejor',d['mejor'])]:
                for dom,dr in datos_v.items():
                    for v,m in dr['metricas'].items():filas.append({'modelo':n,'variante':variante,'paso':2400 if variante=='final' else d['mejor_paso_por_map50_desarrollo_1v'],'dominio':dom,'vistas':v,**{k:m[k] for k in ['map50','map50_95','recall50','precision50','tp50','fp50','fn50']}})
        for dom in ['desarrollo','externo_retenido']:
            nombre=dom
            for v,m in ref[nombre]['metricas'].items():filas.append({'modelo':'Original','variante':'referencia','paso':0,'dominio':dom,'vistas':v,**{k:m[k] for k in ['map50','map50_95','recall50','precision50','tp50','fp50','fn50']}})
        with (OUT/'tabla_metricas.csv').open('x',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(filas[0]));w.writeheader();w.writerows(filas)
        for r in filas:doc.append('| '+' | '.join([r['modelo']+' '+r['variante'],r['dominio'],r['vistas'],*[pct(r[k]) for k in ['map50','map50_95','recall50','precision50']],*[str(r[k]) for k in ['tp50','fp50','fn50']]])+' |')
        doc+=['','| Mejor checkpoint, desarrollo | Paso | Vistas | '+' | '.join(CL)+' |','|---|---:|---:|'+'---:|'*6]
        for n,d in mets.items():
            for v,m in d['mejor']['desarrollo']['metricas'].items():doc.append('| '+n+' | '+str(d['mejor_paso_por_map50_desarrollo_1v'])+' | '+v+' | '+' | '.join(pct(m['ap50_por_material'][c]) for c in CL)+' |')
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for ax,v in zip(axs,['1','5']):
            ys=[ref['desarrollo']['metricas'][v]['map50'],*[mets[n]['mejor']['desarrollo']['metricas'][v]['map50'] for n in ['A','B']]]
            ax.bar(['Original','A mejor','B mejor'],[100*y for y in ys]);ax.set(title='Desarrollo '+v+' vista(s)',ylabel='mAP50 (%)')
        fig.tight_layout();fig.savefig(figdir/'map50_desarrollo.png',dpi=150);plt.close(fig)
        doc+=['','## Coste y checkpoints','', '| Rama | Actualizaciones | Segundos de bucle | Pico GPU |','|---|---:|---:|---:|']
        for n in ['A','B']:
            rs=[leer(p) for p in (OUT/f'entrenamiento/{n}/intentos').glob('*_fin.json')]
            doc.append(f'| {n} | {sum(r["actualizaciones"] for r in rs)} | {sum(r["segundos"] for r in rs):.2f} | {max(r["pico_gpu_bytes"] for r in rs)/1024**2:.1f} MiB |')
        doc+=['','Tiempos de captura incluyen lectura y postproceso; no se presentan como FPS de cámara. Pesos y estados nuevos viven en checkpoints/v31_instancias/A y B, guardados sin reemplazar originales, con hashes de confirmaciones, plan, fuentes, datos, versiones y backend.','',
            '## Interpretación','']
        for n,d in mets.items():
            m=d['mejor']['desarrollo']['metricas'];doc.append(f'- {n}, mejor paso {d["mejor_paso_por_map50_desarrollo_1v"]}: mAP50 1v {pct(m["1"]["map50"])} y 5v {pct(m["5"]["map50"])}. Comparar también AP individual, FP y geometría; no adoptar solo por el promedio.')
        doc+=['','Las causas geométricas (sin caja, material equivocado a IoU50, geometría insuficiente, posibles fusiones/fragmentos) están en cada analisis.json. Son indicadores de cajas, no adjudicación de identidad física. La clase orgánico tiene solo un GT en desarrollo: una mejora allí puede mover mucho el macro y no demuestra generalización alimentaria. No se afirma mejora de seis clases desde el retenido de tres.','',
            'La conclusión cuantitativa final se complementa tras revisar las tablas y ejemplos, sin cambiar entrenamiento, umbrales o presupuesto. Si la ganancia no supera la referencia o se limita a train/una clase, se conserva explícitamente como resultado negativo. No se inicia otra pérdida o búsqueda automática.','']
    else:doc+=['## Entrenamiento bloqueado','', 'El criterio de aprendizaje pequeño no se superó. No se ejecuta el presupuesto completo ni se fabrican métricas de validación. Se conservan los controles y pesos de diagnóstico descartados.','']
    doc+=['## Aplicación e integridad','',
        'scripts/detectar_instancias_v31.py permite cargar explícitamente un checkpoint propio para foto, MP4 o webcam solicitada, con seis materiales, orgánico/inorgánico, cajas, confianza y consulta al asistente propio. Las detecciones se calculan con CNN/Transformer en B, sin GT ni SAM. Se mantiene la advertencia experimental; no se declara cámara física probada en esta fase ni se modifica la aplicación predeterminada.','',
        'La selección original continúa en configuracion/arquitectura.json. El clasificador de 85,74 % no se reentrenó ni se sustituyó. No se borraron resultados V26–V30, revisiones humanas ni propuestas IA. La nueva exclusión Git de datos/nuevos/v31 protege futuros visuales de terceros; el .gitignore anterior se conserva.','',
        'Reproducción en una ubicación nueva: scripts/instancias_v31.py --fase pruebas/preparar/controles/memorizar; entrenar A/B hasta400 y luego2400 solo si pasa puerta train; evaluar checkpoints800/1600/2400 con el evaluador congelado o resultados/v31_instancias/evaluar_rama.py; generar este informe y ejecutar scripts/verificar_instancias_v31.py. Las publicaciones son inmutables y rechazan sobrescritura.','',
        'Integridad final: resultados/v31_instancias/integridad.json y recibo_integridad.json; pruebas, diagnóstico, estados, diarios y hashes quedan ligados al protocolo. Integridad aprobada no significa un modelo fiable ni adopción automática.','',
        '![Ajuste pequeño](../resultados/v31_instancias/graficas/memorizacion.png)','']
    if mem['aprobada']:doc+=['![mAP50 desarrollo](../resultados/v31_instancias/graficas/map50_desarrollo.png)','']
    with (RAIZ/'documentacion/31_deteccion_por_instancias.md').open('x',encoding='utf8') as f:f.write('\n'.join(doc))
    print('Informe V31 generado',flush=True)


if __name__=='__main__':main()
