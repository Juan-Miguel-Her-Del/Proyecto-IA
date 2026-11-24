"""Selección previa a métricas; visuales privados desde predicciones guardadas."""
import argparse,json,hashlib,io
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
RAIZ=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
CL=['plastico','papel','carton','vidrio','metal','organico']
COL=['#e53935','#1565c0','#8d6e63','#00a99d','#a020f0','#36b329']
def leer(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guardar(p,d):
    with p.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def seleccionar():
    rs=[json.loads(l) for l in (RAIZ/'datos/procesados/deteccion/val.jsonl').read_text(encoding='utf8').splitlines()]
    elegidos=[]
    for c in CL:
        r=min((r for r in rs if any(o['clase']==c for o in r['cajas'])),key=lambda r:r['id_original'])
        elegidos.append({'material_priorizado':c,'id':r['id'],'ruta':r['ruta'],'sha256':r['sha256_archivo']})
    guardar(OUT/'seleccion_visual_previa.json',{'regla':'menor id_original en val que contiene cada material; antes de observar métricas V31; sin selección por aciertos','imagenes':elegidos})
    print(elegidos)
def panel(im,pred,titulo,gt=False):
    im=im.copy();im.thumbnail((600,600));x0=(600-im.width)//2;y0=35+(600-im.height)//2
    out=Image.new('RGB',(600,635),'#222222');out.paste(im,(x0,y0));d=ImageDraw.Draw(out);d.text((8,8),titulo,fill='white')
    if gt:
        for o in pred['cajas']:
            x,y,w,h=o['bbox'];b=[x,y,x+w,y+h];c=CL.index(o['clase']) if o['clase'] in CL else None;bb=[x0+b[0]*im.width/pred['ancho'],y0+b[1]*im.height/pred['alto'],x0+b[2]*im.width/pred['ancho'],y0+b[3]*im.height/pred['alto']]
            color=COL[c] if c is not None else '#bbbbbb';label=CL[c] if c is not None else 'IGNORE'
            d.rectangle(bb,outline=color,width=2);d.text((bb[0],bb[1]),label,fill=color,stroke_width=1,stroke_fill='black')
    else:
        # Las cajas finales están normalizadas sobre el letterbox de 384.
        g=pred['_geo'];sx=im.width/(g['ancho_original']*g['escala_x']);sy=im.height/(g['alto_original']*g['escala_y'])
        for b,c,s in zip(pred['boxes'],pred['labels'],pred['scores']):
            bb=[x0+(b[0]*384-g['relleno_x'])*sx,y0+(b[1]*384-g['relleno_y'])*sy,x0+(b[2]*384-g['relleno_x'])*sx,y0+(b[3]*384-g['relleno_y'])*sy]
            d.rectangle(bb,outline=COL[c],width=2);d.text((bb[0],bb[1]),f'{CL[c]} {s:.2f}',fill=COL[c],stroke_width=1,stroke_fill='black')
    return out
def renderizar():
    from sys import path
    path.insert(0,str(RAIZ))
    from ecowaste.datasets import geometria_letterbox
    rs={r['id']:r for r in [json.loads(l) for l in (RAIZ/'datos/procesados/deteccion/val.jsonl').read_text(encoding='utf8').splitlines()]}
    ref=leer(RAIZ/'resultados/v30_hibrido_revision02/metricas_piloto.json')['resultados']['Original']['desarrollo']
    paths={'Original':RAIZ/ref['captura']/'predicciones_5v.json'}
    for n in ['A','B']:
        d=leer(OUT/f'metricas_{n}.json');paths[n]=RAIZ/d['mejor']['desarrollo']['analisis'];paths[n]=paths[n].parent/'predicciones_5v.json'
    ps={n:{r['id']:r for r in leer(p)} for n,p in paths.items()}
    dest=RAIZ/'datos/nuevos/v31/privado/ejemplos';dest.mkdir(parents=True,exist_ok=True);files={}
    for e in leer(OUT/'seleccion_visual_previa.json')['imagenes']:
        r=rs[e['id']];p=RAIZ/r['ruta']
        if sha(p)!=e['sha256']:raise ValueError('Foto alterada')
        with Image.open(p) as f:im=f.convert('RGB')
        g=geometria_letterbox(r['ancho'],r['alto'],384);panels=[panel(im,r,'GT, solo análisis posterior',True)]
        for n in ['Original','A','B']:
            pred=dict(ps[n][r['id']]);pred['_geo']=g;panels.append(panel(im,pred,n+' - 5 vistas, umbral 0.05'))
        canvas=Image.new('RGB',(1200,1270))
        for k,a in enumerate(panels):canvas.paste(a,((k%2)*600,(k//2)*635))
        q=dest/(e['material_priorizado']+'_'+r['id'].replace(':','_')+'.jpg')
        buf=io.BytesIO();canvas.save(buf,format='JPEG',quality=92);datos=buf.getvalue()
        if q.exists():
            if q.read_bytes()!=datos:raise ValueError('Visual parcial diferente; conservar e investigar')
        else:
            with q.open('xb') as f:f.write(datos)
        files[q.relative_to(RAIZ).as_posix()]=sha(q)
    guardar(OUT/'ejemplos_visuales.json',{'seleccion':sha(OUT/'seleccion_visual_previa.json'),'predicciones':{str(p.relative_to(RAIZ)):sha(p) for p in paths.values()},'archivos_privados':files,'gt_solo_analisis':True,'todas_las_detecciones_mostradas':True,'redistribuir_fotos':False})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fase',choices=['seleccionar','renderizar'],required=True);a=p.parse_args();seleccionar() if a.fase=='seleccionar' else renderizar()
