"""Tablas, figuras y ejemplos de la auditoría. Solo consume resultados ya capturados."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon
from matplotlib.colors import ListedColormap
RAIZ=Path(__file__).resolve().parents[1]
CLASES=['plastico','papel','carton','vidrio','metal','organico']
CAUSAS=['correcto','perdida_postprocesamiento','fusion_objetos','fragmentacion',
    'material_incorrecto','ausencia_respuesta_fg','geometria_insuficiente','sin_soporte_gt_no_ambiguo']
COLORES=['#268970','#9b59b6','#da7445','#e8b344','#4278b0','#777777','#abc4dd','#a39a8d']
NOMBRES=['TP final','Pérdida en filtros','Fusión','Fragmentación','Otro material','Sin respuesta FG','Geometría','GT ambiguo/sin soporte']


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def guardar(fig,path):
    if path.exists():raise FileExistsError('Figura histórica: '+str(path))
    fig.savefig(path,dpi=150);plt.close(fig)


def ejemplo(ficha,mapas,base,salida):
    j=ficha['indice'];o=ficha['objeto'];g=ficha['geometria'];z=ficha['ventana']
    with Image.open(RAIZ/ficha['ruta']) as im:crop=im.convert('RGB').crop(z)
    nuevo=crop.resize((round(crop.width*g['escala_x']),round(crop.height*g['escala_y'])),Image.Resampling.BILINEAR)
    lienzo=Image.new('RGB',(384,384),(114,114,114));lienzo.paste(nuevo,(g['relleno_x'],g['relleno_y']))
    fig,axs=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    axs[0].imshow(lienzo);axs[0].set_title('Imagen y GT (amarillo)')
    im=axs[1].imshow(mapas[f'p{j}'],vmin=0,vmax=1,cmap='magma',extent=[0,384,384,0])
    axs[1].set_title('Probabilidad del material GT');fig.colorbar(im,ax=axs[1],shrink=.75)
    paleta=ListedColormap(['#202020','#42a1de','#e5d883','#b2784c','#55b89f','#b49fc9','#78b748'])
    axs[2].imshow(mapas[f'argmax{j}'],vmin=0,vmax=6,cmap=paleta,interpolation='nearest',extent=[0,384,384,0])
    axs[2].set_title('Argmax y componentes (verde: final)')
    for ax in axs:
        for p in ficha['poligonos_originales']:
            puntos=[((x-z[0])*g['escala_x']+g['relleno_x'],(y-z[1])*g['escala_y']+g['relleno_y']) for x,y in zip(p[::2],p[1::2])]
            ax.add_patch(Polygon(puntos,fill=False,edgecolor='yellow',linewidth=1.5))
        ax.set_xlim(0,384);ax.set_ylim(384,0);ax.set_xticks([]);ax.set_yticks([])
    relacionados={c['id'] for v in o['vistas'] if v['vista']==ficha['vista'] for c in v['componentes_tocan_gt']}
    relacionados.update(o['ids_correctos_pre_filtros']);relacionados.update(o['ids_otros_materiales_iou50'])
    seleccion=[n for n in ficha['nodos'] if n['id'] in relacionados]
    seleccion=sorted(seleccion,key=lambda n:(-n['area_celdas'],n['id']))[:10]
    for n in seleccion:
        x0,y0,x1,y1=np.array(n['caja_vista'])*384;color='lime' if 'final' in n['etapas'] else '#ff5c61'
        axs[2].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,linewidth=1.2))
        axs[2].text(x0,y0,f"{n['material']} {n['puntuacion']:.2f}",fontsize=6,color=color,backgroundcolor='#202020')
    fig.suptitle(f"{o['imagen']} / GT {o['id_gt']} · {o['material']} · {o['causa_principal']} · vista {ficha['vista']}\n"
        f"Lado menor en entrada completa: {o['lado_menor_384']:.1f} px; sin modificación de propuestas",fontsize=10)
    path=salida/f'ejemplo_{j:02d}_{o["material"]}_{o["causa_principal"]}.png';guardar(fig,path)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--entrada',type=Path,default=RAIZ/'resultados/auditoria_propuestas_internas_v1')
    a=p.parse_args();base=a.entrada.resolve();r=leer(base/'analisis.json');cap=leer(base/'captura.json');tests=leer(base/'pruebas.json')
    filas=leer(base/'objetos_gt.json');res=r['resumen'];salida=base/'graficas';salida.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9})
    fig,ax=plt.subplots(figsize=(11,5.4),layout='constrained');bottom=np.zeros(6);xs=np.arange(6)
    for causa,nombre,color in zip(CAUSAS,NOMBRES,COLORES):
        vals=np.array([res['por_material'][c]['causas'][causa] for c in CLASES]);ax.bar(xs,vals,bottom=bottom,label=nombre,color=color);bottom+=vals
    ax.set(xticks=xs,xticklabels=CLASES,ylabel='Instancias GT val',title='Causa principal por material; sin entrenamiento')
    ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8);guardar(fig,salida/'causas_por_material.png')
    etapas=list(r['conteos_propuestas']);labels=['CC','Área≥2','Clip','NMS vista*','Top100 vista','Bordes','NMS global*','Top100 global','Final .05','Visual .35']
    fig,ax=plt.subplots(figsize=(12,4.5),layout='constrained')
    for c in CLASES:
        g=res['por_material'][c];ax.plot(range(len(etapas)),[100*g['oportunidades_por_etapa'][e]/g['gt'] for e in etapas],marker='o',label=c)
    ax.set(xticks=range(len(etapas)),xticklabels=labels,ylabel='GT con alguna caja correcta IoU≥.5 (%)',ylim=(-2,102),title='Techo por GT y etapa (sin asignación uno a uno); *NMS ilimitado contrafactual')
    ax.tick_params(axis='x',rotation=25);ax.legend(ncol=3);guardar(fig,salida/'oportunidades_por_etapa.png')
    tamanos=list(res['por_tamano']);fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');bottom=np.zeros(4)
    for causa,nombre,color in zip(CAUSAS,NOMBRES,COLORES):
        vals=np.array([100*res['por_tamano'][t]['causas'][causa]/max(1,res['por_tamano'][t]['gt']) for t in tamanos])
        axs[0].bar(range(4),vals,bottom=bottom,label=nombre,color=color);bottom+=vals
    axs[0].set(xticks=range(4),xticklabels=[f"{t}\nn={res['por_tamano'][t]['gt']}" for t in tamanos],ylabel='GT (%)',title='Tamaño en la entrada completa de 384 px')
    axs[0].tick_params(axis='x',rotation=20)
    for c,color in [('carton','#b2784c'),('vidrio','#269a8b')]:
        fs=[f for f in filas if f['material']==c]
        respuesta=[max((v['fraccion_material_correcto'] for v in f['vistas'] if v['util']),default=0.) for f in fs]
        axs[1].scatter([f['lado_menor_384'] for f in fs],respuesta,label=c,color=color,alpha=.8)
    axs[1].set(xscale='log',xlabel='Lado menor del GT en entrada completa (px)',ylabel='Máxima fracción de GT con argmax correcto',title='Solo vistas útiles (visibilidad bbox ≥80 %)')
    axs[1].axvline(8,linestyle='--',color='gray',linewidth=1);axs[1].legend()
    guardar(fig,salida/'tamano_y_respuesta.png')
    mapas=np.load(base/'ejemplos_mapas.npz');ejemplos=leer(base/'ejemplos.json')
    rutas_ejemplos=[ejemplo(e,mapas,base,salida) for e in ejemplos]
    conteos=res['total']['causas'];head=conteos['material_incorrecto']+conteos['ausencia_respuesta_fg']
    local=conteos['geometria_insuficiente']+conteos['fusion_objetos']+conteos['fragmentacion'];post=conteos['perdida_postprocesamiento']
    perdidas_etapa={}
    for f in filas:
        if f['causa_principal']!='perdida_postprocesamiento':continue
        previo=True
        for etapa in etapas:
            if etapa=='visual035':break
            actual=f['etapas'][etapa]['correctos_iou50']>0
            if previo and not actual:
                perdidas_etapa[etapa]=perdidas_etapa.get(etapa,0)+1;break
            previo=actual
    recomendacion='cabeza_de_mapas' if head>=local else 'localizacion_de_instancias'
    if post>max(head,local):recomendacion='postprocesamiento'
    rel=base.relative_to(RAIZ).as_posix();doc=RAIZ/'documentacion/19_auditoria_propuestas_internas.md'
    if doc.exists():raise FileExistsError('Informe existente')
    lineas=['# Auditoría de propuestas internas del detector seleccionado','',
        '**Auditoría sin entrenamiento. Se conserva el detector original de época 19; no se adopta la variante regional B.**','',
        '## Alcance e integridad','',
        f"Se auditaron **181 imágenes de validación, 905 vistas y 361 GT**. Pasaron **{tests['ejecutadas']} pruebas** (136 anteriores y 13 nuevas). La instrumentación conservó exactamente cajas, materiales, puntuaciones y orden final de las 181 escenas. Cada vista se comprobó también contra `cajas_desde_mapas` y cada combinación contra `combinar_vistas`. Se verificaron {r['historicos_intactos']} archivos protegidos: pesos, arquitectura, configuración de entrenamiento, código de producción y resultados históricos.",'',
        f"Se realizaron {cap['llamadas_modelo']} forwards del modelo, cada uno con cinco vistas; {cap['imagenes_logits_reutilizados']} escenas reutilizaron logits previos de la caché. Los mismos logits alimentaron todos los análisis y regresiones. Caché local: `{cap['cache']}`, con hashes de tensores, checkpoint, val, código de producción y entorno. Las trazas tensoriales y logits están excluidos de Git como los demás `.pt`; se reconstruyen con el script. No hay nuevas pérdidas, entrenamiento, checkpoints ni cambios de umbrales.",'',
        'La captura acepta solo ID, ruta, tamaño y hash de imagen. No recibe GT. Las anotaciones y polígonos se abren después de cerrar la captura. El análisis usa exclusivamente registros val; no se selecciona ni evalúa la partición test. El archivo TACO original contiene el índice general de anotaciones, pero solo se consultan los polígonos correspondientes a val.','',
        '## Etapas y procedencia','',
        'Se registran las celdas elegibles antes de componentes, los componentes conexos de ocho vecinos (incluidos los de área uno), el filtro de dos celdas, recorte a contenido, NMS y top100 por vista, filtros de bordes internos, proyección a imagen completa, NMS y top100 globales, salida de evaluación a 0,05 y filtro visual posterior a 0,35.','',
        'Cada propuesta conserva ID estable, imagen, vista, material, puntuación, caja local y/o global, etapa, motivo de descarte y supresor cuando corresponde. Las celdas apuntan al componente que las agrupa. `vistas.json` permite deshacer letterbox y recuperar coordenadas originales. El registro `propuestas.jsonl.gz` conserva las etapas sin depender de figuras.','',
        'El NMS original se detiene al elegir 100 cajas. Las curvas con asterisco muestran **NMS ilimitado contrafactual**, calculado solo para separar supresión y límite: el prefijo de 100 se comprueba contra producción. Una caja no examinada por alcanzar 100 no se atribuye a una supresión posterior hipotética. Los contadores de `descartes_reales` conservan esa distinción.','',
        '| Etapa | Propuestas totales | Plástico | Papel | Cartón | Vidrio | Metal | Orgánico | GT con caja correcta IoU≥0,5 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for etapa,cs in r['conteos_propuestas'].items():
        lineas.append('| '+etapa+' | '+str(sum(cs.values()))+' | '+' | '.join(str(cs[c]) for c in CLASES)+' | '+str(res['total']['oportunidades_por_etapa'][etapa])+' |')
    lineas+=['','El conteo de GT con caja razonable es un techo por existencia de candidato, no recall real ni AP: permite que una misma caja coincida con varios GT y no penaliza FP. Las métricas reales se calculan con la asignación habitual uno a uno. Las uniones de fragmentos también son oráculos de análisis; nunca se convierten en propuestas del detector.','',
        'Las 2.560 cajas finales incluyen 112 predicciones que el evaluador omite por regiones GT ignoradas. No se eliminan durante inferencia: por eso TP+FP=2.448 no coincide con el total de propuestas. El GT interviene en esta contabilidad solo durante el análisis.','',
        f"De las {post} pérdidas de oportunidad, la primera etapa que deja al GT sin caja correcta se distribuye así: **{perdidas_etapa}**. No se pierde ninguna oportunidad por top100 ni por NMS de una vista en esta validación. Esto no garantiza que relajar filtros recupere ese número de TP: también reaparecerían FP y cambiaría la competencia entre cajas.",'',
        f"![Oportunidades por etapa](../{rel}/graficas/oportunidades_por_etapa.png)",'',
        '## Criterios de diagnóstico','',
        'La causa principal usa esta prioridad: TP real; pérdida en postprocesamiento si existía una caja correcta IoU≥0,5 antes de filtros y ya no existe al final; fusión sustentada por máscaras; fragmentación sustentada por máscaras; material incorrecto; ausencia de respuesta foreground; geometría insuficiente. Se separan los GT sin soporte no ambiguo. Los indicadores de fusión y fragmentación no excluyentes se guardan además de la causa principal.','',
        '- **Fusión:** el mismo componente cubre al menos 50 % de los píxeles GT de dos o más instancias útiles. Para atribuirla al GT auditado, el componente lleva su material. No basta una caja grande que encierre dos objetos.',
        '- **Fragmentación:** al menos dos componentes del material correcto cubren cada uno ≥10 % del GT; juntos cubren ≥50 %, su envolvente tiene IoU≥0,5 y ninguno alcanza individualmente 0,5.',
        '- **Material incorrecto:** hay una caja razonable de otro material sin caja razonable correcta, o hay respuesta FG dentro del polígono y ninguna celda del material correcto.',
        '- **Ausencia de respuesta FG:** todas las celdas observables del polígono quedan en fondo en las vistas útiles. No equivale a que la probabilidad bruta del material sea exactamente cero.',
        '- **Geometría insuficiente:** respuesta parcial o forma/tamaño/desplazamiento que no alcanza una caja correcta y no satisface las evidencias anteriores. Puede incluir asignación compartida sin evidencia suficiente de fusión.','',
        'La respuesta se mide con cobertura fraccional de polígonos rasterizados a 384 y reducidos a 48. Se excluyen padding, celdas con varios materiales GT y regiones ignoradas. Una vista es útil si contiene ≥80 % de la caja GT; la vista completa permanece disponible. Esto evita interpretar la falta de respuesta en un recorte que apenas ve el objeto. Se conservan respuesta correcta, respuesta FG, fondo, probabilidad GT media/máxima y probabilidad GT≥0,05 por objeto y vista.','',
        '## Diagnóstico por clase','',
        '| Material | GT | TP | Pérdida filtros | Fusión | Fragmentación | Material incorrecto | Sin FG | Geometría | GT sin soporte |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in CLASES:
        grupo=res['por_material'][c];lineas.append('| '+c+' | '+str(grupo['gt'])+' | '+' | '.join(str(grupo['causas'][ca]) for ca in CAUSAS)+' |')
    lineas+=['',f"![Causas por material](../{rel}/graficas/causas_por_material.png)",'','### Cartón y vidrio','']
    for c in ['carton','vidrio']:
        g=res['por_material'][c];cs=g['causas'];o=g['oportunidades_por_etapa']
        lineas.append(f"**{c}:** {g['gt']} GT, {cs['correcto']} TP. Existían {o['componentes']} GT con alguna caja correcta antes de filtros y {o['final']} al final. Causas principales: {dict((k,v) for k,v in cs.items() if v)}. Indicadores adicionales: {g['indicadores_fragmentacion']} fragmentación y {g['indicadores_fusion']} fusión.")
        lineas.append('')
    lineas+=['La fragmentación no alcanzó el criterio conservador de confirmación en esta validación: cero uniones con cobertura e IoU suficientes. Esto no demuestra que nunca existan mapas fragmentados; los componentes dispersos que no cubren suficiente GT quedan en geometría insuficiente. Las fusiones se concentran en plástico y metal, con 29 indicadores y 21 causas principales; algunas cajas fusionadas todavía pueden contar como TP para uno de los objetos.','',
        '## Efecto del tamaño','',
        '| Lado menor a 384 px | GT | TP | Sin FG | Material incorrecto | Geometría/fusión/fragmentación | Pérdida filtros | Supera 8 px en alguna vista útil |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for t,g in res['por_tamano'].items():
        cs=g['causas'];lineas.append(f"| {t} | {g['gt']} | {cs['correcto']} | {cs['ausencia_respuesta_fg']} | {cs['material_incorrecto']} | {cs['geometria_insuficiente']+cs['fusion_objetos']+cs['fragmentacion']} | {cs['perdida_postprocesamiento']} | {g['ganancia_resolucion_tokens']} |")
    lineas+=['',f"![Tamaño y respuesta](../{rel}/graficas/tamano_y_respuesta.png)",'',
        'El tamaño es una asociación observacional, no una intervención causal de resolución. Las cinco vistas aumentan como máximo aproximadamente 1/0,65 la escala para un objeto contenido en un recorte. Que un objeto supere ocho píxeles no demuestra que el token sea puro ni que la clase esté aprendida. El JSON incluye el cruce de seis materiales y cuatro tamaños para evitar que la distribución de plástico determine por sí sola la conclusión.','',
        'Para cartón no hay GT menores de ocho píxeles: 12 de 25 tienen lado menor ≥32 px y 10 de esos 12 fallan por material incorrecto. En vidrio hay un GT <8 px y ocho de 8–16 px; siete de los nueve casos sin FG pertenecen a esas dos bandas. Pero los 13 vidrios ≥16 px también fallan: 11 por material y dos sin FG. **La resolución agrava la ausencia de respuesta en objetos pequeños, pero no explica por sí sola el AP cero de cartón y vidrio.**','',
        '## Qué etapa conviene investigar después','',
        f"En los FN, las causas principales vinculadas a respuesta/material suman **{head}**; geometría/fusión/fragmentación suman **{local}**; pérdida real de una oportunidad de caja por filtros suma **{post}**. La prioridad general respaldada por esta partición es **{recomendacion}**. Las categorías son diagnósticos conservadores y dependientes de sus criterios; no constituyen una prueba causal de que una modificación concreta funcionará.",'',
        'Orden recomendado para el siguiente trabajo:','',
        '1. **Cabeza de mapas/representación semántica:** estudiar la respuesta espacial de las siete clases sobre GT real, especialmente cartón y vidrio. No usar la mejora de la cabeza auxiliar global de V17/V18 como sustituto de evidencia en mapas. La captura muestra cero cajas correctas de esos materiales antes de los filtros, incluso con objetos grandes.',
        '2. **Localización de instancias:** investigar separación de componentes que mezclan objetos y geometría de regiones. Es un problema secundario importante, sobre todo en plástico; 21 FN tienen fusión como causa principal y 70 geometría insuficiente.',
        '3. **Resolución de pequeños:** plantear una comparación controlada aparte para GT <8 px y vidrio de 8–16 px. La asociación con fondo es fuerte, pero aumentar resolución no resolvería la confusión observada en cartones grandes.',
        '4. **Postprocesamiento:** prioridad menor para el objetivo cartón/vidrio. Top100 no está saturando ni elimina oportunidades; los filtros solo explican 14 de 287 FN, ninguno de cartón o vidrio. Mantener por ahora la política seleccionada.','',
        'No se recomienda retirar filtros, aumentar top100 ni implementar otra pérdida automáticamente. Primero deben usarse los IDs y mapas de esta auditoría para definir una sola intervención verificable, manteniendo la referencia actual y una comparación controlada.','',
        '## Ejemplos visuales','',
        'Selección determinista del primer GT por material y causa (orden ID imagen/instancia), con prioridad a cartón y vidrio. Se añaden evidencias de filtros, fragmentación y fusión cuando existen. Son ejemplos explicativos, no una estimación de prevalencia. Verde indica que el componente llegó a la salida final de alguna vista; rojo, que fue descartado. El contorno original puede extenderse fuera del contenido del recorte, incluso sobre el padding visual; el análisis excluye ese padding y usa solo la porción visible.','']
    for ficha,path in zip(ejemplos,rutas_ejemplos):
        o=ficha['objeto'];lineas+=[f"### {o['imagen']} / GT {o['id_gt']}: {o['material']} — {o['causa_principal']}",'',
            f"![Ejemplo {ficha['indice']}](../{path.relative_to(RAIZ).as_posix()})",'']
    mm=r['resumen_metricas_finales']
    lineas+=['## Reproducción, métricas y límites','',
        f"La salida final reproduce mAP@50 **{100*mm['map50']:.6f} %**, mAP@50:95 **{100*mm['map50_95']:.6f} %**, recall **{100*mm['recall50']:.6f} %**, precisión **{100*mm['precision50']:.6f} %**, TP={mm['tp50']}, FP={mm['fp50']}, FN={mm['fn50']}. AP50 de cartón/vidrio sigue en cero. No se modificaron pesos ni selección.",'',
        '```powershell',r"$pythonEco = Join-Path $env:USERPROFILE '.venvs\EcoWasteFormer\Scripts\python.exe'",
        '& $pythonEco scripts/auditar_propuestas_internas.py',
        '& $pythonEco scripts/auditar_propuestas_internas.py --fase captura --salida resultados/auditoria_nueva',
        '& $pythonEco scripts/auditar_propuestas_internas.py --fase analisis --salida resultados/auditoria_nueva',
        'python scripts/informar_propuestas_internas.py',
        '& $pythonEco scripts/verificar_propuestas_internas.py','```','',
        'Las rutas existentes se rechazan para preservar evidencia. Para repetir, usar salida nueva; la caché permite cero nuevos forwards. El análisis de trazas ya capturadas se ejecuta en CPU. El observador necesita CUDA al reproducir softmax de logits AMP para conservar exactamente el redondeo de producción. No se afirma equivalencia numérica entre backends distintos.','',
        f'- [Resumen y métricas completas](../{rel}/analisis.json).',f'- [361 objetos GT](../{rel}/objetos_gt.json).',
        f'- [Tabla por objeto](../{rel}/objetos_gt.csv).',f'- [Tabla por etapa/material](../{rel}/etapas.csv).',
        f'- [Propuestas con procedencia y descarte](../{rel}/propuestas.jsonl.gz).',f'- [Metadatos de captura y caché](../{rel}/captura.json).','',
        'La validación se ha reutilizado durante el proyecto: sus resultados no son evidencia independiente ni una estimación de test. Las coincidencias de caja a IoU 0,5 y las máscaras rasterizadas no prueban por sí solas identidad física de instancias; solapes, anotaciones ambiguas y objetos pequeños limitan las atribuciones. Se conserva el caso sin soporte GT para no culpar al modelo por un diagnóstico no observable. El AP es el evaluador propio de 101 puntos, no COCOeval oficial. No se ha probado una corrección ni entrenado otra variante.','']
    doc.write_text('\n'.join(lineas),encoding='utf-8')
    resumen={'recomendacion_general':recomendacion,'respuesta_y_material':head,'localizacion_instancias':local,
        'postprocesamiento':post,'perdidas_por_primera_etapa':perdidas_etapa,
        'sha256_informe':sha(doc),'sha256_script_figuras':sha(Path(__file__)),
        'figuras':{p.relative_to(RAIZ).as_posix():sha(p) for p in salida.glob('*.png')},'test_evaluado':False,'entrenamiento':False}
    destino=base/'informe.json'
    if destino.exists():raise FileExistsError('Resumen existente')
    destino.write_text(json.dumps(resumen,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(resumen,ensure_ascii=False))


if __name__=='__main__':main()
