"""Figuras exportables del A/B congelado, sin volver a ejecutar el modelo."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from PIL import Image,ImageOps
import numpy as np

RAIZ=Path(__file__).resolve().parents[1]


def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8'))


def main():
    exp=leer('resultados/mapas_v6_ab_propuestas_material.json')
    colores={'argmax7':'#216ca5','por_material':'#c76b17'}
    fig,ax=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    fig.suptitle('Mismo checkpoint congelado · propuestas 0,05 · 181 imágenes val / 361 objetos')
    for a,clave,titulo in zip(ax,['map50','recall50','fp50'],['mAP@50','Recall@50','Falsos positivos @ IoU 0,5']):
        maximo=0
        for dx,pol in [(-.18,'argmax7'),(.18,'por_material')]:
            vals=[exp['resultados'][f'{v}_vistas_{pol}']['por_filtro_caja']['0.05']['resumen'][clave] for v in [1,5]]
            maximo=max(maximo,*vals)
            barras=a.bar(np.arange(2)+dx,vals,width=.35,color=colores[pol],label='Actual: argmax' if pol=='argmax7' else 'Por material')
            a.bar_label(barras,labels=[f'{v:.2%}' if clave!='fp50' else f'{v:,}'.replace(',','.') for v in vals],padding=3,fontsize=9)
        a.set(xticks=[0,1],xticklabels=['1 vista','5 vistas'],title=titulo,ylim=(0,maximo*1.25))
        if clave!='fp50':a.yaxis.set_major_formatter(PercentFormatter(1))
        a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    ax[0].legend(loc='upper right',fontsize=8)
    for ext in ['png','svg']:fig.savefig(RAIZ/f'resultados/figuras/propuestas_material_ab.{ext}',dpi=170)
    plt.close(fig)
    registro=json.loads((RAIZ/'datos/procesados/deteccion/val.jsonl').read_text(encoding='utf-8').splitlines()[0])
    with Image.open(RAIZ/registro['ruta']) as im:imagen=ImageOps.exif_transpose(im).convert('RGB')
    ancho,alto=imagen.size
    imagen.thumbnail((900,900))
    fig,ax=plt.subplots(1,3,figsize=(13,6.5),layout='constrained')
    for a in ax:
        a.imshow(imagen,extent=(0,ancho,alto,0));a.set(xlim=(0,ancho),ylim=(alto,0));a.axis('off')
        for c in registro['cajas']:
            if c['clase']:
                x,y,w,h=c['bbox'];a.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#228be6',linewidth=1.2))
    ax[0].set_title('Primera escena val: taco:686\nGT azul: botella y tapa de plástico',fontsize=10)
    s=384/max(ancho,alto);nw,nh=round(ancho*s),round(alto*s)
    px,py=(384-nw)//2,(384-nh)//2;sx,sy=nw/ancho,nh/alto
    clases=['plástico','papel','cartón','vidrio','metal','orgánico']
    for a,pol in zip(ax[1:],['argmax7','por_material']):
        r=exp['resultados'][f'5_vistas_{pol}']
        p=leer(r['predicciones_val'])[0]
        if p['id']!=registro['id']:raise ValueError('Escena distinta')
        orden=sorted(range(len(p['boxes'])),key=lambda i:-p['scores'][i])[:10]
        for i in orden:
            b=p['boxes'][i];x0,y0=(b[0]*384-px)/sx,(b[1]*384-py)/sy
            x1,y1=(b[2]*384-px)/sx,(b[3]*384-py)/sy
            a.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,
                edgecolor='#2f9e44' if pol=='argmax7' else colores[pol],linewidth=1))
            a.text(x0,y0,f"{clases[p['labels'][i]]} {p['scores'][i]:.2f}",fontsize=5,color='black',
                   bbox=dict(facecolor='white',alpha=.7,pad=.3,edgecolor='none'))
        a.set_title(f"{'Actual: argmax' if pol=='argmax7' else 'Sin argmax'} · cinco vistas\n{len(orden)} cajas de {len(p['boxes'])}, ordenadas por puntuación",fontsize=10)
    fig.suptitle('Ejemplo fijado por orden de validación · cajas formadas con umbral 0,05',fontsize=12)
    for ext in ['png','svg']:fig.savefig(RAIZ/f'resultados/figuras/propuestas_material_ejemplo.{ext}',dpi=170)
    plt.close(fig)
    print('Comparación y primera escena val guardadas en PNG/SVG')


if __name__=='__main__':main()
