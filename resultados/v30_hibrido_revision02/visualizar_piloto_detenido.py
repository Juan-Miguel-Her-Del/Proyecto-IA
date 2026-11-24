"""Paneles locales: GT aparte, Original/A800/B800, selección por ID prefijada."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw
RAIZ=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
CL=['plastico','papel','carton','vidrio','metal','organico']


def leer(p):return json.loads(Path(p).read_text(encoding='utf8'))
def jsonl(p):return [json.loads(l) for l in Path(p).read_text(encoding='utf8').splitlines() if l.strip()]


def originales():
    out={}
    for dom,split in [('historico_train','train'),('desarrollo','val')]:
        out[dom]={r['id']:{**r,'gt':[(a['bbox'],CL.index(a['clase'])) for a in r['cajas'] if a['clase'] in CL],
            'ign':[a['bbox'] for a in r['cajas'] if a['clase'] not in CL]} for r in jsonl(RAIZ/f'datos/procesados/deteccion/{split}.jsonl')}
    for dom,split in [('nuevo_train','train'),('externo_retenido','val')]:
        d=leer(RAIZ/f'datos/nuevos/v29/coleccion/{split}_coco.json');cats={c['id']:c['name'] for c in d['categories']}
        out[dom]={}
        for im in d['images']:
            gt=[(a['bbox'],CL.index(cats[a['category_id']])) for a in d['annotations'] if a['image_id']==im['id'] and cats[a['category_id']] in CL]
            ign=[a['bbox'] for a in d['annotations'] if a['image_id']==im['id'] and cats[a['category_id']] not in CL]
            out[dom][im['eco']['id']]={'id':im['eco']['id'],'ruta':im['eco']['ruta_original'],'ancho':im['width'],'alto':im['height'],'gt':gt,'ign':ign}
    return out


def desletterbox(box,w,h):
    scale=384/max(w,h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale));px,py=(384-nw)//2,(384-nh)//2
    sx,sy=nw/w,nh/h
    return [(box[0]*384-px)/sx,(box[1]*384-py)/sy,(box[2]*384-px)/sx,(box[3]*384-py)/sy]


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    ar=lambda z:max(0,z[2]-z[0])*max(0,z[3]-z[1])
    return inter/max(1e-12,ar(a)+ar(b)-inter)


def main():
    met=leer(OUT/'metricas_piloto.json');seleccion=leer(OUT/'ejemplos_prefijados.json');data=originales();cache={};index=[]
    colores=['#2288bb','#995599','#bd7711','#00a099','#aa4444','#228833']
    for k,e in enumerate(seleccion):
        dom,ident=e['dominio'],e['id'];r=data[dom][ident]
        with Image.open(RAIZ/r['ruta']) as pil:im=pil.convert('RGB')
        w,h=440,min(580,max(120,round(im.height*440/im.width)))
        thumb=im.resize((w,h));panel=Image.new('RGB',(w*4,(h+58)*2),'white')
        gt=[([b[0],b[1],b[0]+b[2],b[1]+b[3]],c) for b,c in r['gt']];medidas={}
        for fila,v in enumerate(['1','5']):
            for col,n in enumerate(['GT separado','Original','A','B']):
                pieza=thumb.copy();draw=ImageDraw.Draw(pieza);omitidos=None
                if col==0:bs=[b for b,c in gt];ls=[c for b,c in gt];ss=None
                else:
                    key=(n,dom,v)
                    if key not in cache:
                        folder=RAIZ/met['resultados'][n][dom]['captura'];cache[key]={p['id']:p for p in leer(folder/f'predicciones_{v}v.json')}
                    p=cache[key][ident];bs=[desletterbox(b,im.width,im.height) for b in p['boxes']];ls=p['labels'];ss=p['scores']
                    usados=set()
                    for j in sorted(range(len(bs)),key=lambda j:-ss[j]):
                        candidatos=[(iou(bs[j],g),i) for i,(g,c) in enumerate(gt) if c==ls[j] and i not in usados and iou(bs[j],g)>=.5]
                        if candidatos:usados.add(max(candidatos)[1])
                    omitidos=len(gt)-len(usados);medidas[f'{n}_{v}']={'gt':len(gt),'gt_sin_match_material_iou50_score05':omitidos}
                visibles=0
                for j,(b,c) in enumerate(zip(bs,ls)):
                    if ss is not None and ss[j]<.35:continue
                    xy=[b[0]*w/im.width,b[1]*h/im.height,b[2]*w/im.width,b[3]*h/im.height]
                    color=colores[c];draw.rectangle(xy,outline=color,width=2)
                    texto=CL[c]+(f' {ss[j]:.2f}' if ss is not None else '')
                    draw.text((xy[0],max(0,xy[1]-12)),texto,fill=color,stroke_width=1,stroke_fill='white');visibles+=1
                if col==0:
                    for b in r['ign']:
                        xy=[b[0]*w/im.width,b[1]*h/im.height,(b[0]+b[2])*w/im.width,(b[1]+b[3])*h/im.height]
                        draw.rectangle(xy,outline='#999999',width=2)
                        draw.text((xy[0],max(0,xy[1]-12)),'IGNORE',fill='#555555',stroke_width=1,stroke_fill='white')
                ox,oy=col*w,fila*(h+58);panel.paste(pieza,(ox,oy+58))
                caption=f'{n} | {v}v | {ident}\n{visibles} cajas visibles >=.35; metricas >=.05'
                if col==0:caption=f'{n} | {v}v | {ident}\n{visibles} positivos; {len(r["ign"])} ignore (gris); sin umbral GT'
                if omitidos is not None:caption+=f'\nGT sin match material/IoU50: {omitidos}/{len(gt)}'
                ImageDraw.Draw(panel).text((ox+4,oy+3),caption,fill='black')
        path=RAIZ/f'datos/nuevos/v30/privado/visuales_revision02/{k+1:02d}_{dom}_{CL[e["clase"]]}.jpg'
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:panel.save(f,format='JPEG',quality=88)
        index.append({**e,'archivo':path.relative_to(RAIZ).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'comparaciones':medidas,'regla':'mínimo ID por material/dominio antes de captura; ningún criterio favorece B'})
    with (OUT/'visuales_revision02.json').open('x',encoding='utf8') as f:json.dump(index,f,ensure_ascii=False,indent=2)
    print('Paneles locales',len(index),flush=True)


if __name__=='__main__':main()
