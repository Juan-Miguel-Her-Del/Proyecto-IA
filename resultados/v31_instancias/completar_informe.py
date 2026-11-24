"""Tablas e interpretación de los resultados, sin cambiar el experimento."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RAIZ=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
CL=['plastico','papel','carton','vidrio','metal','organico']
def leer(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pc(x):return 'sin GT' if x is None else f'{100*x:.3f} %'
def main():
    ms={n:leer(OUT/f'metricas_{n}.json') for n in ['A','B']};extra=leer(OUT/'analisis_adicional.json');ref=leer(RAIZ/'resultados/v30_hibrido_revision02/metricas_piloto.json')['resultados']['Original'];doc=['','## Interpretación cuantitativa y diagnóstico por objeto','']
    a=ms['A']['mejor']['desarrollo']['metricas']['5'];b=ms['B']['mejor']['desarrollo']['metricas']['5'];r=ref['desarrollo']['metricas']['5']
    doc+=[f'Con cinco vistas, B alcanza {pc(b["map50"])} frente a {pc(a["map50"])} del control A y {pc(r["map50"])} de la referencia. Diferencia B−original: {100*(b["map50"]-r["map50"]):+.3f} puntos porcentuales; B−A: {100*(b["map50"]-a["map50"]):+.3f}. Se usan los mejores checkpoints seleccionados por el criterio de una vista ya fijado, no una búsqueda posterior del mejor resultado de cinco vistas.','']
    ganan=[c for c in CL if b['ap50_por_material'][c]>r['ap50_por_material'][c]]
    doc+=[f'AP50 mayor que la referencia en: {", ".join(ganan) or "ningún material"}. B produce {b["tp50"]} TP, {b["fp50"]} FP y {b["fn50"]} FN sobre 361 objetos de desarrollo; precisión {pc(b["precision50"])} y recall {pc(b["recall50"])} al umbral 0,05. Esos FP deben considerarse junto al AP, y la precisión operativa a otro umbral no se estima cambiándolo tras los resultados.','']
    if b['map50']>r['map50'] and b['map50']>a['map50']:
        doc+=['Hay una ganancia medida sobre esta validación de desarrollo para el híbrido por instancias. Justifica ofrecer su carga experimental explícita, pero una semilla, selección de checkpoints y desarrollo reutilizado no establecen fiabilidad en nuevas escenas ni autorizan sustituir el detector predeterminado.','']
    else:
        doc+=['El híbrido no supera simultáneamente al control y a la referencia en esta comparación. Se conserva como experimento no adoptado; no se amplían pasos, no se retocan umbrales y no se encadenan nuevas pérdidas para forzar una mejora.','']
    doc+=['### Ajuste train por material, final 2400, una vista','', '| Rama / dominio | '+' | '.join(CL)+' |','|---|'+'---:|'*6]
    for n,m in ms.items():
        for dom in ['historico_train','nuevo_train']:
            q=m['pasos']['2400']['resultados'][dom]['metricas']['1'];doc+=['| '+n+' / '+dom+' | '+' | '.join(pc(q['ap50_por_material'][c]) for c in CL)+' |']
    doc+=['','La mejora train mide ajuste y puede incluir memorización. El retenido externo no tiene GT de cartón, vidrio ni orgánico; esos AP se dejan sin GT. Su origen ya fue observado por el proyecto y no se demuestra independencia de objeto/sesión.','',
        '### Material, ausencia de objeto y geometría','',
        'El análisis adicional toma por cada GT la consulta raw de mayor IoU, sin utilizar su clase para elegirla. Si ninguna llega a IoU50 se registra «sin geometría»; en caso contrario se distingue material correcto, material incorrecto y ausencia de objeto. No es matching uno a uno ni reemplaza AP. La matriz condicional se restringe a objetos con geometría IoU50 y siempre escoge entre seis materiales, por lo que debe leerse junto a su denominador.','',
        '| Mejor / desarrollo 1v | Material GT | Correcto | Material erróneo | Ausencia objeto | Sin geometría IoU50 |','|---|---|---:|---:|---:|---:|']
    for n in ['A','B']:
        q=extra['resultados'][n]['desarrollo']['mejor_1v']
        for i,row in enumerate(q['matriz_6x8']):doc+=['| '+n+' | '+CL[i]+' | '+str(row[i])+' | '+str(sum(row[:6])-row[i])+' | '+str(row[6])+' | '+str(row[7])+' |']
        fig,ax=plt.subplots(figsize=(10,5));arr=np.array(q['matriz_6x8']);im=ax.imshow(arr,cmap='Blues');ax.set_xticks(range(8),q['columnas'],rotation=30,ha='right');ax.set_yticks(range(6),CL);ax.set(xlabel='Consulta de mayor IoU',ylabel='Material GT',title=n+' mejor: material / ausencia / geometría')
        for (i,j),v in np.ndenumerate(arr):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>arr.max()/2 else 'black')
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/f'graficas/confusion_{n}.png',dpi=150);plt.close(fig)
    sizes=leer(OUT/'tamano.json')
    doc+=['','### Efecto del tamaño, final 2400, una vista','',sizes['criterio'],'',
        '| Rama | Dominio | Tamaño | GT | Alguna consulta raw IoU50 | Material correcto en consulta de mayor IoU |','|---|---|---|---:|---:|---:|']
    for n,doms in sizes['resultados'].items():
        for dom,bins in doms.items():
            for grupo,x in bins.items():doc+=[f'| {n} | {dom} | {grupo} | {x["gt"]} | {x["con_geometria_iou50"]} | {x["material_correcto_mayor_iou"]} |']
    doc+=['','B carece de geometría IoU50 para 2143/2505 GT históricos train (85,55 %), 3465/4624 nuevos train (74,94 %) y 333/361 desarrollo (92,24 %), aun antes de filtrar por material, confianza o ausencia de objeto. No se explica exclusivamente por generalización: también hay bajo ajuste geométrico sobre train.','',
        'La diferencia por tamaño es fuerte: B tiene alguna consulta IoU50 para solo 30/1375 objetos pequeños históricos train (2,18 %), frente a 165/299 grandes (55,18 %). En desarrollo son 3/239 pequeños (1,26 %), 20/108 medianos y 5/14 grandes. Estas fracciones descriptivas no aíslan material, oclusión ni procedencia, y no demuestran que aumentar resolución por sí solo mejore AP.','',
        'Las 28 instancias de desarrollo que sí tienen una consulta IoU50 se reparten en 16 con ausencia de objeto, 9 con material equivocado y 3 correctas en la consulta de mayor IoU. La escasez de cajas razonables limita primero; también quedan problemas de material/confianza después.','',
        '### Oportunidades antes y después del postproceso, cinco vistas','', '| Rama | Material | Antes de bordes | Después de bordes | Después NMS/top100 |','|---|---|---:|---:|---:|']
    for n in ['A','B']:
        q=extra['resultados'][n]['desarrollo']['oportunidades_mejor_5v']['por_etapa']
        for c in CL:doc+=['| '+n+' | '+c+' | '+' | '.join(str(q[e][c]) for e in ['pre_bordes','post_bordes','final'])+' |']
    doc+=['','Una oportunidad exige alguna caja correcta a IoU50; no garantiza TP tras asignación uno a uno. Se verificó que recombinar los raw reproduce todas las cajas finales guardadas, sin otra inferencia ni cambios al postproceso. En B, cartón conserva 2 oportunidades y vidrio 0 antes/después; NMS/top100 no explica su pérdida principal. El AP50 cartón de B 5v es solo 0,0653 %, frente a 0 % histórico: ese valor aislado no compensa la caída global.','',
        '### Geometría y separación, mejor checkpoint cinco vistas','', '| Rama | Material | Correcto | Material erróneo IoU50 | Geometría insuficiente | Sin caja correcta solapada | Indicador fusión | Indicador fragmentos |','|---|---|---:|---:|---:|---:|---:|---:|']
    for n,d in ms.items():
        gg=leer(RAIZ/d['mejor']['desarrollo']['analisis'])['5']['geometria']['por_material']
        for c in CL:
            row=gg[c];doc+=['| '+n+' | '+c+' | '+' | '.join(str(row.get(k,0)) for k in ['correcto','material_incorrecto_iou50','geometria_insuficiente','sin_caja_correcta_solapada','indicador_fusion_bbox','indicador_fragmentacion_bbox'])+' |']
    doc+=['','Los indicadores de fusión/fragmentación se basan en cobertura de cajas y pueden marcar objetos anidados; no certifican identidad física ni segmentación.','',
        '### Exposición orgánica real','', '| Imagen train | Cajas orgánicas | Veces por rama |','|---|---:|---:|']
    for x in extra['exposicion_organicos']:doc+=[f'| {x["imagen"]} | {x["instancias_organicas"]} | {x["veces_muestreada_por_rama"]} |']
    doc+=['','Las cinco cajas tienen supervisión en asignación, incluso las subtoken; no hay nuevas instancias orgánicas en PackWISE/CLICK. Repetirlas no aumenta diversidad. Desarrollo solo tiene un orgánico, por lo que su AP no valida reconocimiento alimentario amplio.','',
        '### Curvas y ejemplos','', '![Entrenamiento](../resultados/v31_instancias/graficas/entrenamiento.png)','',
        '![Confusión A](../resultados/v31_instancias/graficas/confusion_A.png)','', '![Confusión B](../resultados/v31_instancias/graficas/confusion_B.png)','',
        'La selección visual se fijó antes de evaluar: menor ID de desarrollo que contiene cada material. Seis entradas corresponden a cinco fotos distintas (papel y cartón comparten escena). Las comparativas GT aparte/original/A/B usan todas las cajas guardadas a 0,05; no se escogieron por éxito de B. Permanecen privadas en datos/nuevos/v31/privado/ejemplos para respetar licencias. El índice y hashes están en ejemplos_visuales.json.','',
        '### Decisión y siguiente intervención','',
        'El modelo original sigue predeterminado. La prioridad técnica que respalda esta prueba es mejorar la localización de objetos pequeños, antes de otra pérdida de materiales o cambios de NMS. Un candidato posterior sería conservar características CNN más finas (p. ej. 96×96 frente a 48×48) hacia la misma cabeza por instancias, aislando esa resolución en un control y midiendo primero cajas pequeñas sobre train. Es una hipótesis, no una mejora demostrada, y requiere verificar memoria, padding y capacidad de ajuste antes de otro presupuesto grande. No se implementa ni entrena aquí.','',
        'La memorización casi perfecta de ocho escenas demuestra capacidad representacional limitada a esas escenas, pero no que el régimen actual optimice bien una colección diversa. Deben auditarse además matching, gradientes de cajas y ausencia de objeto por tamaño sobre train antes de atribuir el fallo solo a datos o solo a la CNN. Los experimentos de centros y consultas V10 ya tuvieron AP insuficiente: no se propone repetirlos sin un control nuevo.','',
        'La siguiente adquisición debe concentrarse en objetos reales distintos de vidrio, restos alimentarios y cartón, con cajas o polígonos por instancia y oclusiones/ignore coherentes. PackWISE/CLICK aumentan principalmente plástico y papel y no resuelven esa carencia. Completar las correcciones humanas pendientes y congelar una pequeña evaluación nueva por grupos sigue siendo necesario, sin abrir test ni generalización V26; añadir datos por sí solo no queda demostrado como solución al bajo ajuste geométrico actual. No se inicia otro experimento en esta fase.','']
    vel=leer(OUT/'velocidad.json');app=leer(OUT/'aplicacion.json');hist=leer(OUT/'referencia_verificada.json')
    doc+=['### Velocidad e integración ejecutadas','',
        'Foto prefijada, cinco calentamientos y veinte repeticiones por modelo/vistas, mismo backend V23, umbral visual 0,35 y propuestas 0,05. Incluye conversión del frame, letterbox, red y postproceso; excluye disco, dibujo y captura física. Una foto repetida no representa toda la aplicación ni la distribución de escenas.','',
        '| Modelo | Vistas | Mediana ms | Pico GPU MiB | Salida estable 20 repeticiones |','|---|---:|---:|---:|---|']
    for n,d in vel['resultados'].items():
        for v,x in d.items():doc+=[f'| {n} | {v} | {x["ms_mediana"]:.2f} | {x["pico_gpu_bytes"]/1024**2:.1f} | {x["predicciones_estables_20"]} |']
    doc+=['',f'Se ejecutó foto y MP4 con B paso {app["paso"]}, códigos de salida: '+str({k:v['codigo_salida'] for k,v in app['pruebas'].items()})+'. El MP4 son seis repeticiones de una foto real, exclusivamente comprobación de E/S y forward real; no se presenta como video independiente ni prueba física de webcam. Los archivos se guardan privados y sus hashes están en aplicacion.json.','',
        f'Carga experimental explícita (sin cambiar la aplicación predeterminada): `python scripts/detectar_instancias_v31.py --foto "ruta/foto.jpg" --checkpoint checkpoints/v31_instancias/B/{app["paso"]:04d}.pt --salida "datos/nuevos/v31/privado/mi_foto" --vistas 1`. La salida debe ser nueva. Para MP4 se sustituye `--foto` por `--video`; cámara solo cuando el usuario la solicite mediante `--camara 0 --mostrar --max-frames 300`.','',
        f'Se verificaron {len(hist["archivos"])} archivos de evidencia Original reutilizada (métricas, logits, trazas y predicciones), contra sus hashes guardados, sin recalcular métricas con otro backend. El hash del clasificador permanece {hist["clasificador_sha256"]} y el del detector original {hist["pesos_originales_sha256"]}.','',
        'Un verificador auxiliar inicialmente intentó leer la sección benchmark histórica como otro dominio y produjo KeyError antes de emitir certificado. Se corrigió únicamente ese lector y la denominación de externo_retenido del informe; la repetición terminó aprobada. El renderizador se adaptó a categorías ignoradas (`clase=null`) y reanudó su salida parcial tras comprobar identidad de bytes, sin sobrescribirla ni cambiar GT. No fueron fallos del entrenamiento ni alteraciones del código congelado.','',
        'El análisis de oportunidades rechazó inicialmente la reproducción CPU porque softmax CPU/CUDA difiere hasta 1,7881393e-7 en puntuación. Se conservaron las predicciones originales y se repitió solo softmax/postproceso con CUDA como en la captura: todas las cajas, etiquetas y puntuaciones finales de ambas ramas coincidieron exactamente. Se guardan las diferencias CPU/CUDA por imagen/vista; no se amplió ninguna tolerancia para aprobar.','']
    fig,axs=plt.subplots(1,2,figsize=(12,4));rows={}
    for n in ['A','B']:
        ts={}
        for p in (OUT/f'entrenamiento/{n}/trazas').rglob('*.json'):
            x=leer(p);ts[x['paso']]=x
        rows[n]=[ts[k] for k in sorted(ts)]
        for ax,key in zip(axs,['loss_total','norma']):
            yy=np.array([r[key] for r in rows[n]]);xx=np.array([r['paso'] for r in rows[n]])
            ax.plot(xx[39:],np.convolve(yy,np.ones(40)/40,mode='valid'),label=n)
            ax.set(xlabel='Actualización',ylabel=key,title=key+' media móvil 40');ax.legend()
    fig.tight_layout();fig.savefig(OUT/'graficas/entrenamiento.png',dpi=150);plt.close(fig)
    p=RAIZ/'documentacion/31_deteccion_por_instancias.md'
    if '## Interpretación cuantitativa y diagnóstico por objeto' in p.read_text(encoding='utf8'):raise ValueError('No duplicar informe')
    with p.open('a',encoding='utf8') as f:f.write('\n'.join(doc))
    print('Interpretación y gráficas añadidas')
if __name__=='__main__':main()
