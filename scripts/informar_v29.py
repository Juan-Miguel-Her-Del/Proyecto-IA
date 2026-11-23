"""Informe de resultados ya calculados; no entrena, evalúa modelos o cambia selección."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RAIZ=Path(__file__).resolve().parents[1]
OUT=RAIZ/'resultados/v29_packwise_click';COL=RAIZ/'datos/nuevos/v29/coleccion'
CLASES=['plastico','papel','carton','vidrio','metal','organico'];MODELOS=['Original','A','B']
DOMINIOS=['historico_train','nuevo_train','nuevo_retenido']


def leer(p):return json.loads(Path(p).read_text(encoding='utf8'))
def guardar(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)


def historial(r):
    c=leer(OUT/f'entrenamiento/{r}/confirmaciones/paso_0600.json')
    return sorted([leer(RAIZ/p) for p in c['trazas']],key=lambda h:h['paso'])


def cajas_originales(bs,w,h):
    s=384/max(w,h);nw,nh=round(w*s),round(h*s);dx,dy=(384-nw)//2,(384-nh)//2
    return [[(b[0]*384-dx)*w/nw,(b[1]*384-dy)*h/nh,(b[2]*384-dx)*w/nw,(b[3]*384-dy)*h/nh] for b in bs]


def omitidos(bs,cl,scores,gt,etiquetas):
    usados=set()
    for i in sorted(range(len(bs)),key=lambda j:-scores[j]):
        a=bs[i];coinciden=[]
        for j,(b,c) in enumerate(zip(gt,etiquetas)):
            if j in usados or c!=cl[i]:continue
            inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
            iu=inter/max(union,1e-12)
            if iu>=.5:coinciden.append((iu,j))
        if coinciden:usados.add(max(coinciden)[1])
    return len(gt)-len(usados)


def visuales(res):
    fuente=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14);colores=['#35b9eb','#e3dd49','#f49538','#66d5bf','#c0a0ed','#82ca52']
    hist=[json.loads(s) for s in (RAIZ/'datos/procesados/deteccion/train.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    coco={sp:leer(COL/f'{sp}_coco.json') for sp in ['train','val']};out=RAIZ/'datos/nuevos/v29/privado/visuales';out.mkdir(parents=True,exist_ok=True);indice=[]
    for k,regla in enumerate(leer(OUT/'seleccion_visual_prefijada.json')):
        dom,iid=regla['dominio'],regla['id']
        if dom=='historico_train':
            r=next(x for x in hist if x['id']==iid);p=RAIZ/r['ruta'];w,h=r['ancho'],r['alto']
            aa=[a for a in r['cajas'] if a['clase'] in CLASES];gt=[a['bbox'] for a in aa];ls=[CLASES.index(a['clase']) for a in aa]
        else:
            d=coco['train' if dom=='nuevo_train' else 'val'];im=next(i for i in d['images'] if i['eco']['id']==iid)
            p=COL/im['file_name'];w,h=im['width'],im['height'];aa=[a for a in d['annotations'] if a['image_id']==im['id'] and not a['ignore']]
            gt=[a['bbox'] for a in aa];ls=[a['category_id']-1 for a in aa]
        with Image.open(p) as im:foto=im.convert('RGB')
        pw,ph=440,480;esc=min(pw/w,(ph-70)/h);nw,nh=round(w*esc),round(h*esc)
        lienzo=Image.new('RGB',(pw*4,ph*2),'#17212a');dr=ImageDraw.Draw(lienzo)
        for fila,v in enumerate([1,5]):
            for columna,nombre in enumerate(['GT',*MODELOS]):
                ox,oy=columna*pw,fila*ph;dx=(pw-nw)//2;dy=70+(ph-70-nh)//2;lienzo.paste(foto.resize((nw,nh)),(ox+dx,oy+dy))
                if nombre=='GT':bs=[[x,y,x+b,y+a] for x,y,b,a in gt];cl=ls;scores=None
                else:
                    base=(RAIZ/res[nombre][dom]['analisis']).parent
                    pred=next(p for p in leer(base/f'predicciones_{v}v.json') if p['id']==iid)
                    bs=cajas_originales(pred['boxes'],w,h);cl=pred['labels'];scores=pred['scores']
                faltan='' if scores is None else ' | omitidos '+str(omitidos(bs,cl,scores,[[x,y,x+b,y+a] for x,y,b,a in gt],ls))
                dr.text((ox+6,oy+5),f'{nombre} | {v}v | {len(bs)} cajas'+faltan,fill='white',font=fuente)
                dr.text((ox+6,oy+25),iid,fill='white',font=fuente)
                dr.text((ox+6,oy+45),'GT separado de inferencia; score >=0.05',fill='#bfcbd1',font=fuente)
                for j,(b,c) in enumerate(zip(bs,cl)):
                    b=[ox+dx+b[0]*esc,oy+dy+b[1]*esc,ox+dx+b[2]*esc,oy+dy+b[3]*esc]
                    if b[2]<b[0] or b[3]<b[1]:continue
                    dr.rectangle(b,outline=colores[c],width=2)
                    texto=CLASES[c]+('' if scores is None else f' {scores[j]:.2f}')
                    dr.text((max(ox,b[0]),max(oy+65,b[1]-14)),texto,font=fuente,fill=colores[c],stroke_width=1,stroke_fill='black')
        pp=out/f'{k+1:02d}_{dom}_{CLASES[regla["material"]]}.jpg';lienzo.save(pp,quality=90)
        indice.append({**regla,'archivo':pp.relative_to(RAIZ).as_posix(),'sha256':hashlib.sha256(pp.read_bytes()).hexdigest()})
    guardar(OUT/'visuales.json',{'regla':'Primer ID por material/dominio fijado antes de inferencia V29','ejemplos':indice})


def main():
    d=leer(OUT/'metricas.json');res=d['resultados'];col=leer(COL/'MANIFIESTO.json');g=OUT/'graficas';g.mkdir(exist_ok=True)
    filas=[]
    for nombre in MODELOS:
        for dom in DOMINIOS:
            for v in ['1','5']:
                m=res[nombre][dom]['metricas'][v]
                filas.append({'modelo':nombre,'dominio':dom,'vistas':int(v),**{k:x for k,x in m.items() if k!='ap50_por_material'},**{'AP50_'+c:m['ap50_por_material'][c] for c in CLASES}})
    with (OUT/'tabla_comparativa.csv').open('x',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=list(filas[0]));wr.writeheader();wr.writerows(filas)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,dom in zip(axs,DOMINIOS):
        ax.bar(MODELOS,[100*res[n][dom]['metricas']['5']['map50'] for n in MODELOS],color=['#7b8796','#4391ca','#64b77b'])
        ax.set_title(dom);ax.set_ylabel('mAP50 % (clases con GT)');ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(g/'map50_5v.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    hs={r:historial(r) for r in ['A','B']}
    for ax,key,titulo in zip(axs,['loss_historico','loss_global','loss_nuevos'],['Mapas históricos','Clasificación global','Mapas nuevos']):
        for r in ['A','B']:
            if key=='loss_nuevos' and r=='A':continue
            ys=np.array([h[key] for h in hs[r]]);smooth=np.convolve(ys,np.ones(20)/20,mode='valid')
            ax.plot(np.arange(20,len(ys)+1),smooth,label=r)
        ax.set_title(titulo);ax.set_xlabel('Actualización AdamW');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(g/'curvas_train.png',dpi=160);plt.close(fig)
    for dom in DOMINIOS:
        fig,axs=plt.subplots(1,3,figsize=(14,4))
        for ax,n in zip(axs,MODELOS):
            a=leer(RAIZ/res[n][dom]['analisis']);cm=np.array(a['cabeza_principal']['matriz_instancias_con_fondo'])
            ax.imshow(cm,cmap='Blues');ax.set_title(n);ax.set_xticks(range(7),['fondo',*CLASES],rotation=90);ax.set_yticks(range(7),['fondo',*CLASES]);ax.set_xlabel('Predicción');ax.set_ylabel('GT instancia')
        fig.suptitle(dom+' | cabeza principal, voto por instancia');fig.tight_layout();fig.savefig(g/f'confusion_{dom}.png',dpi=160);plt.close(fig)
    visuales(res)
    def pct(x):return 'sin GT' if x is None else f'{100*x:.3f} %'
    texto=['# V29 — PackWISE v2 y CLICK: entrenamiento controlado de datos','',
        'Experimento autorizado por el usuario. Se conservan el Transformer, la pérdida original, el checkpoint predeterminado y los históricos. No se usa test ni se desbloquea generalización.','',
        '## Fuentes y colección','',
        '[PackWISE v2, Fraunhofer](https://fordatis.fraunhofer.de/handle/fordatis/463.2): ZIP completo verificado con MD5 publicado; train/val originales 410/88 imágenes y 4669/990 instancias. Se preservan RLE y huecos. [CLICK v2](https://data.mendeley.com/datasets/htcvvz6fm2/2): COCO train/val originales 901/113 imágenes y 1802/221 instancias; los autores declaran corrección de propuestas SAM en Label Studio. Licencias oficiales CC BY 4.0. La descripción de CLICK conserva una frase de licencia pendiente, aunque la ficha oficial publica CC BY; se conserva esa discrepancia.','',
        'Las anotaciones se atribuyen a los autores. No constituyen decisiones del usuario, revisión independiente de EcoWasteFormer ni aceptación formal V26. Los datos originales permanecen en `datos/nuevos/v29/externos`; la colección congelada, COCO, máscaras, hashes y procedencia están en `datos/nuevos/v29/coleccion`. Las fotos están excluidas de Git. CLICK se extrae selectivamente por HTTP Range: se verifican CRC ZIP y SHA256 por archivo. Su SHA256 del ZIP completo permanece declarado, no verificado, porque no se descarga el archivo íntegro.','',
        '| Cohorte | Imágenes | Plástico | Papel | Cartón | Vidrio | Metal | Orgánico | Ignore |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for sp,info in col['resumen'].items():texto.append('| '+sp+' | '+str(info['imagenes'])+' | '+' | '.join(str(info['por_material'].get(c,0)) for c in CLASES+['ignore'])+' |')
    texto+=['','Se ignoran categorías sin correspondencia material suficiente: multicapa, recubiertos de aluminio, tapas sin material, vasos/envases de papel potencialmente laminados, paper-other amplio y objetos indefinidos. No se convierten etiquetas globales en cajas. Las cajas finales se derivan de las máscaras, conservando también las cajas originales.','',
        'Se reagrupan únicamente los train y val públicos en un protocolo nuevo: nuestras cohortes no equivalen a los splits publicados por los proveedores. Test permanece cerrado. PackWISE se agrupa por secuencia completa: Vollstrom1/2 para train y Vollstrom6 prevista para retención. CLICK se agrupa en bloques de veinte nombres, con selección determinista anterior a métricas. Esta agrupación heurística no demuestra independencia física. Coincidencias exactas y candidatos perceptuales unen grupos; si conectan train y retención, el componente entero se reserva para train. Coincidencias con train/val históricos excluyen el grupo nuevo. Los errores geométricos excluyen la imagen completa, evitando enseñar un objeto conocido como fondo. Véase `mapeo_y_agrupacion.json`.','',
        '**Estas fuentes no aportan nuevas regiones de vidrio u orgánico y el cartón es muy escaso. No completan las cuotas V26 ni solucionan por sí solas las clases deficientes.** Las instancias sin un componente de dos celdas en 48×48 se contabilizan por material en el manifiesto: una fotografía de alta resolución no garantiza que el objeto resulte representable en la entrada actual. Se inspeccionaron visualmente siete ejemplos con superposiciones y un detalle de cartón; esto es inspección del agente, sin cambiar etiquetas, y no revisión humana exhaustiva. Véase `inspeccion_visual_agente.json`.','',
        '## Protocolo fijado antes de medir','',
        'A/B parten de época 19 con pesos, AdamW, scaler y RNG restaurados. Backend V23 completo, AMP, LR 0.0005, clipping 1, fondo 3, pesos históricos por material y clasificación global de 16 imágenes con coeficiente 0.5. Sin warmup nuevo, nuevas pérdidas, cambio de resolución, parada temprana, selección por validación o adopción automática.','',
        'Cada rama realiza 600 actualizaciones: 4800 exposiciones históricas idénticas (ocho por paso). B añade dos imágenes PackWISE y dos CLICK por actualización, 2400 exposiciones nuevas en total. Se aplica a ese microbatch la misma pérdida original, con contribución prefijada 0.5; la rama A no recibe este microbatch. B tiene más cómputo y exposición total: la comparación mide el efecto de añadir esa supervisión conservando la exposición histórica, no una comparación de FLOPs iguales. Un RNG privado para los aumentos/dropout nuevos evita desplazar el flujo histórico. Checkpoints exclusivos en pasos 0, 100, 300 y 600.','',
        'El presupuesto busca unos pocos recorridos de cada fuente nueva y limita la adaptación desde época 19. No se reutilizan por defecto los 1452 pasos de V20 ni se amplía el presupuesto tras observar resultados. Solo se evalúa el paso final.','',
        '## Controles y reproducibilidad','',
        'Las pruebas cubren COCO RLE/polígonos, huecos, padding, ignore, mezclas, geometría equivalente C/Fortran, duplicados, aislamiento del RNG y rechazo de protocolos/rutas/test incompatibles. Antes del presupuesto se comparan C1, C2 y R (2+guardar+restaurar+2), cuatro AdamW reales por trayectoria y rama: entradas, logits, pérdidas, gradientes, clipping, pesos, AdamW, scaler y RNG. Restauración exacta y tolerancia fija rtol=1e-4, atol=1e-5. Los pesos diagnósticos se descartan. La igualdad bit a bit solo se informa si queda registrada en cada comparación.','',
        '## Métricas finales','',
        'Inferencia con argmax, umbral 0.05, área mínima 2, NMS 0.5, top100, una y cinco vistas. El evaluador propio interpola 101 recalls y maneja ignore; no es COCOeval oficial. mAP promedia las clases que tienen GT: revisar soportes y AP por material. Los valores de las cohortes nuevas no se comparan directamente con el 2.912 % histórico de validación. No se ha vuelto a usar esa validación para ajustar el piloto.','',
        '| Dominio | Modelo | Vistas | mAP50 | mAP50:95 | Recall | Precisión | TP | FP | FN |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for f in filas:texto.append('| '+' | '.join([f['dominio'],f['modelo'],str(f['vistas']),pct(f['map50']),pct(f['map50_95']),pct(f['recall50']),pct(f['precision50']),str(f['tp50']),str(f['fp50']),str(f['fn50'])])+' |')
    texto+=['','| Dominio, cinco vistas | Modelo | '+' | '.join(CLASES)+' |','|---|---|'+'---:|'*6]
    for dom in DOMINIOS:
        for n in MODELOS:texto.append('| '+dom+' | '+n+' | '+' | '.join(pct(res[n][dom]['metricas']['5']['ap50_por_material'][c]) for c in CLASES)+' |')
    texto+=['','| Rama | Actualizaciones | Tiempo de entrenamiento confirmado | Máxima memoria GPU asignada |','|---|---:|---:|---:|']
    for r in ['A','B']:
        intentos=[leer(p) for p in (OUT/f'entrenamiento/{r}/intentos').glob('*_fin.json')]
        texto.append(f'| {r} | {sum(i["actualizaciones"] for i in intentos)} | {sum(i["segundos"] for i in intentos):.2f} s | {max(i["memoria_gpu_max_allocated_bytes"] for i in intentos)/1024**2:.1f} MiB |')
    texto+=['','AP por material, costes y métricas por fuente: `tabla_comparativa.csv` y `metricas.json`. Las matrices de confusión, oportunidades antes/después del postprocesamiento y los indicadores de fusión/fragmentación están en cada `analisis.json`. Los indicadores de cajas no certifican identidad física ni errores de anotación. Se conservan logits, predicciones por imagen y trazas de cinco vistas para auditorías sin repetir inferencia.','',
        'Los ejemplos se escogen por primer ID por material/dominio antes de inferencia, nunca por mejora de B. GT se muestra en un panel separado. Véase `visuales.json`; fotos comparativas locales en `datos/nuevos/v29/privado/visuales`.','',
        '## Interpretación','']
    for dom in DOMINIOS:
        vals=[res[n][dom]['metricas']['5']['map50'] for n in MODELOS]
        texto.append(f'- {dom}, cinco vistas: Original {pct(vals[0])}; A {pct(vals[1])}; B {pct(vals[2])}. B−A: {100*(vals[2]-vals[1]):+.3f} puntos porcentuales.')
    texto+=['','Una mejora en nuevo_train demuestra ajuste; no demuestra generalización. La cohorte retenida ofrece evidencia exploratoria limitada por la procedencia compartida y los grupos heurísticos CLICK. La retención histórica debe juzgarse junto con las ganancias nuevas y los falsos positivos; una métrica aislada no justifica adoptar B.','',
        'Para mejorar cartón, vidrio y orgánicos siguen siendo necesarias instancias reales específicas, con escenas/objetos distintos y máscaras revisadas. Es preferible completar las propuestas visuales ya disponibles de esas clases o incorporar otra fuente compatible, antes de repetir pérdidas o entrenamientos con escaso soporte. No se deben reclasificar envases multicapa como cartón para inflar cuotas.','',
        '## Reproducción','',
        '1. Descargar/verificar fuentes con `fuentes_v29.py` y `descargar_cohortes_v29.py` (sin extraer test).',
        '2. `python scripts/preparar_v29.py`; congelar el recibo QA con `--congelar ruta/cache.json` en un destino nuevo.',
        '3. En entorno CUDA: `python scripts/probar_v29.py` y `python scripts/entrenar_v29.py --controles`.',
        '4. `python scripts/entrenar_v29.py --rama A --hasta 100`, después `--hasta 300` y `--hasta 600`; repetir B. El diario conserva protocolo y restaura todos los estados.',
        '5. `python scripts/evaluar_v29.py`; generar tablas/figuras con este script en el entorno que tiene matplotlib.',
        '6. `python scripts/verificar_v29.py`: comprobar presupuestos, huellas, históricos y cerrar el manifiesto de integridad.','',
        'Los resultados existentes nunca se sobrescriben. Para repetir el experimento, utilizar una versión y rutas nuevas. Se conserva el checkpoint predeterminado original.']
    p=RAIZ/'documentacion/29_entrenamiento_packwise_click.md'
    with p.open('x',encoding='utf8') as f:f.write('\n'.join(texto)+'\n')
    print('Informe',p,flush=True)


if __name__=='__main__':main()
