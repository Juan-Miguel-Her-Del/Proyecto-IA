"""Tablas, gráficos y comparaciones posteriores; no entrena ni carga modelos."""
import csv
import hashlib
import json
import sys
import importlib.metadata
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAIZ=Path(__file__).resolve().parents[1]
SALIDA=RAIZ/'resultados/v28_2_exploratorio'
COLECCION=RAIZ/'datos/nuevos/v28_2/coleccion'
MATERIALES=['plastico','papel','carton','vidrio','metal','organico']
MODELOS=['Original','A','B'];COLORES=['#35b9eb','#f5d142','#e99533','#70dcba','#b7a5e9','#74ca52']


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def guardar(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)


def csv_nuevo(p,filas):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(filas[0]));w.writeheader();w.writerows(filas)


def historia(rama):
    ev=leer(SALIDA/rama/'confirmaciones/paso_0300.json')
    # Trazas confirmadas, excluyendo cualquier intento huérfano.
    return sorted([leer(RAIZ/p) for p in ev['trazas']],key=lambda x:x['paso'])


def cajas_originales(boxes,w,h):
    escala=384/max(w,h);nw,nh=round(w*escala),round(h*escala)
    dx,dy=(384-nw)//2,(384-nh)//2;sx,sy=nw/w,nh/h
    return [[(b[0]*384-dx)/sx,(b[1]*384-dy)/sy,(b[2]*384-dx)/sx,(b[3]*384-dy)/sy] for b in boxes]


def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0]));iy=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=ix*iy
    return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1e-12)


def asignar(pred,boxes,labels):
    usados=set();tp=[]
    for i in sorted(range(len(pred['scores'])),key=lambda j:-pred['scores'][j]):
        posibles=[j for j,c in enumerate(labels) if c==pred['labels'][i] and j not in usados and iou(pred['boxes'][i],boxes[j])>=.5]
        if posibles:j=max(posibles,key=lambda j:iou(pred['boxes'][i],boxes[j]));usados.add(j);tp.append(i)
    return usados,tp


def visuales():
    reglas=leer(SALIDA/'seleccion_visual_prefijada.json');coco=leer(COLECCION/'annotations.json')
    cats={c['id']:c['name'] for c in coco['categories']}
    historicos=[json.loads(l) for l in (RAIZ/'datos/procesados/deteccion/train.jsonl').read_text(encoding='utf-8').splitlines()]
    privados=RAIZ/'datos/nuevos/v28_2/visuales_privados';privados.mkdir(exist_ok=True)
    fuente='C:/Windows/Fonts/arial.ttf';font=ImageFont.truetype(fuente,16);pequena=ImageFont.truetype(fuente,12)
    indice=[]
    for i,regla in enumerate(reglas['imagenes']):
        iid,dom=regla['imagen'],regla['dominio']
        if dom=='auxiliar':
            im=next(x for x in coco['images'] if x['eco']['imagen_id_auxiliar']==iid);path=COLECCION/im['file_name'];w,h=im['width'],im['height']
            anns=[a for a in coco['annotations'] if a['image_id']==im['id'] and cats[a['category_id']] in MATERIALES]
            gt=[[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in anns];labels=[MATERIALES.index(cats[a['category_id']]) for a in anns]
        else:
            im=next(x for x in historicos if x['id']==iid);path=RAIZ/im['ruta'];w,h=im['ancho'],im['alto']
            anns=[a for a in im['cajas'] if a['clase'] in MATERIALES]
            gt=[[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in anns];labels=[MATERIALES.index(a['clase']) for a in anns]
        with Image.open(path) as pil:photo=pil.convert('RGB')
        pw,ph=580,540;s=min(pw/w,(ph-70)/h);nw,nh=round(w*s),round(h*s)
        canvas=Image.new('RGB',(pw*4,ph*2),'#161d28');d=ImageDraw.Draw(canvas);detalle=[]
        for row,v in enumerate([1,5]):
            for col,nombre in enumerate(['GT',*MODELOS]):
                ox,oy=col*pw,row*ph;dx=(pw-nw)//2;dy=65+(ph-65-nh)//2
                canvas.paste(photo.resize((nw,nh)),(ox+dx,oy+dy))
                if nombre=='GT':bs,ls,scores=gt,labels,None;titulo=f'{iid} | GT separado | {len(gt)} objetos'
                else:
                    pred=next(x for x in leer(SALIDA/'evaluacion'/nombre/dom/f'predicciones_{v}v.json') if x['id']==iid)
                    bs=cajas_originales(pred['boxes'],w,h);ls=pred['labels'];scores=pred['scores']
                    # Asignación sobre las coordenadas comunes originales; solo para visual posthoc.
                    emparejado,_=asignar({**pred,'boxes':bs},gt,labels)
                    titulo=f'{nombre} | {v} vistas | {len(bs)} cajas | {len(gt)-len(emparejado)} omitidos'
                    detalle.append({'modelo':nombre,'vistas':v,'detecciones':len(bs),'omitidos_iou50':len(gt)-len(emparejado)})
                d.text((ox+8,oy+8),titulo,fill='white',font=font)
                d.text((ox+8,oy+32),'Predicciones >=0.05; GT nunca entra a inferencia' if scores is not None else 'GT mostrado solo en este panel',fill='#aeb9c8',font=pequena)
                for j,(b,c) in enumerate(zip(bs,ls)):
                    puntos=[ox+dx+b[0]*s,oy+dy+b[1]*s,ox+dx+b[2]*s,oy+dy+b[3]*s]
                    d.rectangle(puntos,outline=COLORES[c],width=2)
                    texto=MATERIALES[c]+('' if scores is None else f' {scores[j]:.2f}')
                    d.text((max(ox,puntos[0]),max(oy+52,puntos[1]-12)),texto,fill=COLORES[c],font=pequena,stroke_width=1,stroke_fill='black')
        p=privados/f'{i+1:02d}_{dom}_{regla["material_prioritario"]}.png';canvas.save(p)
        indice.append({**regla,'archivo_privado':p.relative_to(RAIZ).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'comparacion':detalle})
    guardar(SALIDA/'visuales.json',{'seleccion_prefijada':reglas,'redistribuir_fotos':False,'ejemplos':indice})


def main():
    guardar(SALIDA/'entorno_graficas.json',{'python':sys.version,'ejecutable':sys.executable,
        'paquetes':{n:importlib.metadata.version(n) for n in ['matplotlib','numpy','pillow']}})
    r=leer(SALIDA/'metricas.json')['resultados'];out=SALIDA/'graficas';out.mkdir(exist_ok=True)
    filas=[]
    for modelo in MODELOS:
        for dom in ['historico','auxiliar']:
            for v in ['1','5']:
                m=r[modelo][dom]['metricas'][v]
                filas.append({'modelo':modelo,'dominio':dom,'vistas':int(v),**{k:m[k] for k in ['map50','map50_95','recall50','precision50','tp50','fp50','fn50']},
                    **{'ap50_'+c:m['ap50_por_material'][c] for c in MATERIALES}})
    csv_nuevo(SALIDA/'metricas_comparativas.csv',filas)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,dom in zip(axs,['historico','auxiliar']):
        for off,v in [(-.18,'1'),(.18,'5')]:
            ax.bar(np.arange(3)+off,[100*r[n][dom]['metricas'][v]['map50'] for n in MODELOS],.35,label=v+' vistas')
        ax.set_xticks(range(3),MODELOS);ax.set_ylabel('mAP50 (%)');ax.set_title('Train '+dom);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('V28.2: ajuste en train, sin evidencia de generalización');fig.tight_layout();fig.savefig(out/'map50_train.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    for row,dom in enumerate(['historico','auxiliar']):
        for col,v in enumerate(['1','5']):
            ax=axs[row,col]
            for j,n in enumerate(MODELOS):ax.bar(np.arange(6)+(j-1)*.26,[100*(r[n][dom]['metricas'][v]['ap50_por_material'][c] or 0) for c in MATERIALES],.25,label=n)
            ax.set_xticks(range(6),MATERIALES,rotation=20);ax.set_ylabel('AP50 (%)');ax.set_title(f'Train {dom}, {v} vistas');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'ap50_material.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True);hist={n:historia(n) for n in ['A','B']}
    for n in ['A','B']:
        for ax,key in zip(axs,['loss_mapas','loss_global']):
            ys=np.array([x[key] for x in hist[n]]);ax.plot(np.arange(1,301),ys,alpha=.2)
            ax.plot(np.arange(10,301),np.convolve(ys,np.ones(10)/10,'valid'),label=n+' media móvil 10')
            ax.set_ylabel(key);ax.legend();ax.grid(alpha=.2)
    axs[-1].set_xlabel('Actualización AdamW');fig.tight_layout();fig.savefig(out/'curvas_entrenamiento.png',dpi=160);plt.close(fig)
    csv_nuevo(SALIDA/'curvas_entrenamiento.csv',[{'rama':n,**{k:x[k] for k in ['paso','loss_mapas','loss_global','loss_total','norma_gradiente_pre_clip','segundos','paso_adamw']}} for n in ['A','B'] for x in hist[n]])
    coco=leer(COLECCION/'annotations.json');cats={c['id']:c['name'] for c in coco['categories']}
    csv_nuevo(SALIDA/'imagenes_y_hashes.csv',[{'imagen':im['eco']['imagen_id_auxiliar'],'archivo':im['file_name'],
        'sha256':im['eco']['sha256_archivo'],'fuente':im['eco']['fuente'],'revision_huella':im['eco']['revision_huella'],
        **{c:sum(a['image_id']==im['id'] and cats[a['category_id']]==c for a in coco['annotations']) for c in MATERIALES}}
        for im in coco['images']])
    # Desglose orgánico alimentario frente a vegetación, sin reetiquetar ni nueva inferencia.
    organicos={}
    for n in MODELOS:
        a=leer(SALIDA/'evaluacion'/n/'auxiliar/analisis.json');organicos[n]={}
        for fuente in ['hpi','wsc']:
            rr=[x for x in a['cabeza_principal']['instancias'] if x['clase']==5 and x['imagen'].startswith(fuente+':')]
            organicos[n][fuente]={'instancias':len(rr),'con_tokens':sum(x['tokens']>0 for x in rr),
                'voto_organico_con_fondo':sum(x.get('prediccion_con_fondo')==6 for x in rr),
                'voto_condicional_organico':sum(x.get('prediccion_condicional_material')==5 for x in rr)}
    guardar(SALIDA/'organicos_desglosados.json',organicos)
    for tipo,nombres,archivo in [('semantica',['fondo',*MATERIALES],'confusion_con_fondo'),
        ('condicional',MATERIALES,'confusion_condicional_material')]:
        fig,axs=plt.subplots(2,3,figsize=(17,10))
        for row,dom in enumerate(['historico','auxiliar']):
            for col,n in enumerate(MODELOS):
                d=leer(SALIDA/'evaluacion'/n/dom/'analisis.json')['cabeza_principal']
                cm=np.array(d['semantica']['matriz_confusion'] if tipo=='semantica' else d['matriz_instancias_material_condicional'])
                normal=cm/np.maximum(cm.sum(1,keepdims=True),1);ax=axs[row,col]
                ax.imshow(normal,vmin=0,vmax=1,cmap='Blues')
                ax.set_xticks(range(len(nombres)),nombres,rotation=55,ha='right')
                ax.set_yticks(range(len(nombres)),[f'{c} (n={cm[i].sum()})' for i,c in enumerate(nombres)])
                ax.set_title(n+' / train '+dom);ax.set_xlabel('Predicción');ax.set_ylabel('GT')
                for i in range(len(nombres)):
                    for j in range(len(nombres)):
                        if normal[i,j]>=.005:ax.text(j,i,f'{100*normal[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if normal[i,j]>.5 else 'black')
        fig.suptitle('Cabeza principal: '+tipo+'; % normalizados por fila GT');fig.tight_layout()
        fig.savefig(out/(archivo+'.png'),dpi=150);plt.close(fig)
    visuales();print('Tablas, curvas y 12 comparaciones visuales privadas generadas')


if __name__=='__main__':main()
