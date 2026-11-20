"""Gráficos y galería de revisión desde resultados guardados; sin inferencia."""
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
CLASES=['plastico','papel','carton','vidrio','metal','organico']
NOMBRES=['Plástico','Papel','Cartón','Vidrio','Metal','Orgánico']
ESTADOS=['correcto','sin_cobertura_suficiente','geometria_incorrecta','material_equivocado']
ETIQUETAS=['Correcto','Sin cobertura suficiente','Geometría incorrecta','Material equivocado']
COLORES=['#22836f','#8a9299','#df8d2f','#8754ad']


def guardar(fig,n):
    for ext in ['png','svg']:fig.savefig(RAIZ/f'resultados/figuras/{n}.{ext}',dpi=160)
    plt.close(fig)


def main():
    r=json.loads((RAIZ/'resultados/auditoria_clase_tamano.json').read_text(encoding='utf-8'))
    c=r['condiciones']['5_vistas_score_0.05'];por=c['diagnostico_gt']['por_clase']
    fig,ax=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    y=np.arange(6);antes=np.zeros(6);antes_p=np.zeros(6)
    counts=np.array([por[k]['objetos'] for k in CLASES])
    for e,label,color in zip(ESTADOS,ETIQUETAS,COLORES):
        vals=np.array([por[k][e] for k in CLASES]);p=vals/np.maximum(counts,1)
        ax[0].barh(y,vals,left=antes,label=label,color=color);ax[1].barh(y,p,left=antes_p,color=color)
        for i,n in enumerate(vals):
            if n:ax[0].text(antes[i]+n/2,i,str(n),ha='center',va='center',color='white',fontsize=8)
        antes+=vals;antes_p+=p
    for a in ax:a.set(yticks=y,yticklabels=[f'{n} (n={v})' for n,v in zip(NOMBRES,counts)]);a.invert_yaxis()
    ax[0].set(title='Recuentos por GT',xlabel='Objetos anotados');ax[1].set(title='Distribución dentro de cada clase',xlim=(0,1))
    ax[1].xaxis.set_major_formatter(PercentFormatter(1));fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,fontsize=9)
    fig.suptitle('Cajas finales existentes · cinco vistas · puntuación ≥0,05 · IoU razonable ≥0,5')
    guardar(fig,'auditoria_fallos_clase')
    grupos=list(c['diagnostico_gt']['por_tamano']);stats=c['diagnostico_gt']['por_tamano']
    fig,ax=plt.subplots(1,2,figsize=(12,4.6),layout='constrained');x=np.arange(4)
    vals=[stats[t]['recall50'] or 0 for t in grupos]
    bars=ax[0].bar(x,vals,color='#216ca5')
    ax[0].bar_label(bars,labels=[f"{v:.1%}\n{stats[t]['correcto']}/{stats[t]['objetos']}" for v,t in zip(vals,grupos)],padding=3,fontsize=9)
    ax[0].set(ylim=(0,.4),title='Recall@50 por tamaño de GT');ax[0].yaxis.set_major_formatter(PercentFormatter(1))
    anterior=np.zeros(4)
    for e,label,color in zip(ESTADOS,ETIQUETAS,COLORES):
        v=np.array([stats[t][e] for t in grupos]);ax[1].bar(x,v,bottom=anterior,label=label,color=color);anterior+=v
    ax[1].set(title='Partición de los mismos GT',ylabel='Objetos');ax[1].legend(fontsize=8)
    for a in ax:a.set(xticks=x,xticklabels=['<8 px','8–16 px','16–32 px','≥32 px'],xlabel='Lado menor en el lienzo completo 384');a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    fig.suptitle('No son categorías COCO: se conserva la asignación global de detecciones')
    guardar(fig,'auditoria_tamanos')
    cm=np.array(c['confusion_diagnostica_gt']['matriz']);norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
    fig,ax=plt.subplots(figsize=(10,5.8),layout='constrained')
    im=ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
    ax.set(yticks=range(6),yticklabels=[f'{n} (n={v})' for n,v in zip(NOMBRES,counts)],
           xticks=range(7),xticklabels=NOMBRES+['Sin asignación\ngeométrica unívoca'],ylabel='Material real',xlabel='Material de caja testigo / ausencia geométrica')
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right')
    for i in range(6):
        for j in range(7):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if norm[i,j]>.5 else 'black')
    fig.colorbar(im,ax=ax,shrink=.8,label='Fracción por clase real')
    ax.set_title('Diagnóstico con GT y cajas finales · no es clasificación autónoma')
    guardar(fig,'auditoria_materiales')
    ejemplos=[]
    for material,estado,maximo in [('carton','material_equivocado',2),('vidrio','sin_cobertura_suficiente',2),('organico','sin_cobertura_suficiente',1)]:
        ejemplos.extend([o for o in c['objetos'] if o['material_real']==material and o['estado']==estado][:maximo])
    fig,ax=plt.subplots(len(ejemplos),2,figsize=(11,3.2*len(ejemplos)),layout='constrained',squeeze=False)
    for fila,o in enumerate(ejemplos):
        with Image.open(RAIZ/o['ruta_imagen']) as im:imagen=ImageOps.exif_transpose(im).convert('RGB')
        W,H=imagen.size;x,y,w,h=o['bbox_gt_original_xywh']
        lado=max(64,max(w,h)*2.5);cw,ch=min(W,lado),min(H,lado)
        x0=max(0,min(W-cw,x+w/2-cw/2));y0=max(0,min(H-ch,y+h/2-ch/2))
        ventana=(int(x0),int(y0),int(x0+cw),int(y0+ch))
        gt=(x,y,x+w,y+h);testigo=None
        if o['caja_testigo'] is not None:
            s=384/max(W,H);nw,nh=round(W*s),round(H*s);px,py=(384-nw)//2,(384-nh)//2
            b=o['caja_testigo'];testigo=((b[0]*384-px)/(nw/W),(b[1]*384-py)/(nh/H),(b[2]*384-px)/(nw/W),(b[3]*384-py)/(nh/H))
        for a,zoom in zip(ax[fila],[False,True]):
            if zoom:
                foto=imagen.crop(ventana);extent=(ventana[0],ventana[2],ventana[3],ventana[1])
            else:foto=imagen.copy();foto.thumbnail((900,900));extent=(0,W,H,0)
            a.imshow(foto,extent=extent);a.set(xlim=(extent[0],extent[1]),ylim=(extent[2],extent[3]));a.axis('off')
            for b,color in [(gt,'#228be6'),(testigo,'#df8d2f')]:
                if b is not None:a.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor=color,linewidth=1.6))
            if zoom:a.set_title(f"Acercamiento original · testigo={o['material_testigo_nombre']} · IoU={o['iou_testigo'] if o['iou_testigo'] is not None else 0:.3f}",fontsize=9)
            else:a.set_title(f"{o['id_imagen']} / GT {o['id_anotacion']} · {o['material_real']}\n{o['estado']} · lado menor {o['lado_menor_384']:.2f} px",fontsize=9)
    fig.suptitle('Casos fijados por orden del manifiesto · azul=GT objetivo · naranja=caja testigo',fontsize=12)
    guardar(fig,'auditoria_ejemplos')
    print('Cuatro figuras de auditoría guardadas en PNG/SVG; no se ejecutó el detector')


if __name__=='__main__':main()
