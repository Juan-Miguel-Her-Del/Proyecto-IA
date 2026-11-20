"""Figuras científicas de la ablación: aprendizaje, confusiones y detección."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

RAIZ=Path(__file__).resolve().parents[1]


def leer(p):return json.loads((RAIZ/p).read_text(encoding='utf-8'))


def guardar(fig,nombre):
    for e in ['png','svg']:fig.savefig(RAIZ/f'resultados/figuras/{nombre}.{e}',dpi=170)
    plt.close(fig)


def main():
    r=leer('resultados/ablacion_fondo_ep19.json');origen=leer('resultados/detector_mapas_v6_auxiliar_ext48.json')
    colores={'A':'#216ca5','B':'#c76b17'}
    fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for b in ['A','B']:
        h=r['ramas'][b]['historial']
        ax[0].plot([19]+[t['epoca'] for t in h],[origen['val']['map50']]+[t['map50_val'] for t in h],
                   marker='o',markersize=3,color=colores[b],label=f'{b}: fondo={3 if b=="A" else 1}')
    ax[0].set(title='mAP@50 val · una vista',xlabel='Época absoluta',ylim=(0,None))
    ax[0].yaxis.set_major_formatter(PercentFormatter(1));ax[0].legend(fontsize=8)
    estados=[r['semantica_train_origen'],r['ramas']['A']['semantica_train_final'],r['ramas']['B']['semantica_train_final']]
    vals=[m['miou_materiales'] for m in estados]
    barras=ax[1].bar(['Origen 19','A final 31','B final 31'],vals,color=['#777777',colores['A'],colores['B']])
    ax[1].bar_label(barras,labels=[f'{x:.1%}' for x in vals],padding=3)
    ax[1].set(title='mIoU materiales · train',ylim=(0,max(vals)*1.2));ax[1].yaxis.set_major_formatter(PercentFormatter(1))
    x=np.arange(3)
    for dx,k,n,c in [(-.18,'material_predicho_fondo','Material → fondo','#216ca5'),(.18,'fondo_predicho_material','Fondo → material','#c76b17')]:
        bars=ax[2].bar(x+dx,[s[k] for s in estados],width=.35,label=n,color=c)
        ax[2].bar_label(bars,labels=[f'{s[k]/1000:.1f}k' for s in estados],padding=3,fontsize=8)
    ax[2].set(title='Confusión de celdas · train',xticks=x,xticklabels=['Origen 19','A final 31','B final 31'],ylim=(0,105000))
    ax[2].legend(fontsize=8)
    fig.suptitle('Mismo estado de época 19 · 1.452 actualizaciones por rama · test reservado')
    for a in ax:a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    guardar(fig,'ablacion_fondo_aprendizaje')
    fig,ax=plt.subplots(1,2,figsize=(12,5.7),layout='constrained')
    nombres=['Fondo','Plástico','Papel','Cartón','Vidrio','Metal','Orgánico']
    for a,b in zip(ax,['A','B']):
        cm=np.array(r['ramas'][b]['semantica_train_final']['matriz_confusion'])
        norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
        im=a.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        a.set(title=f'{b}: fondo={3 if b=="A" else 1} · final época 31',xticks=range(7),yticks=range(7),
              xticklabels=nombres,yticklabels=nombres,xlabel='Predicción',ylabel='Etiqueta train')
        plt.setp(a.get_xticklabels(),rotation=40,ha='right')
        for i in range(7):
            for j in range(7):a.text(j,i,f'{norm[i,j]:.0%}',ha='center',va='center',fontsize=8,color='white' if norm[i,j]>.5 else 'black')
    fig.colorbar(im,ax=ax,shrink=.8,label='Fracción de cada clase real')
    fig.suptitle('Mismas máscaras train · celdas ignoradas excluidas · normalización por fila')
    guardar(fig,'ablacion_fondo_confusiones')
    origen5=leer('resultados/detector_mapas_v6_mosaico.json')['val']
    m=[origen5]+[leer(f'resultados/ablacion_fondo_ep19_{b}_{s}_cinco_vistas.json')['val'] for b,s in [('A','final'),('A','mejor'),('B','final'),('B','mejor')]]
    etiquetas=['Origen 19','A final 31','A mejor 24','B final 31','B mejor 24']
    fig,ax=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    for a,k,titulo in zip(ax,['map50','recall50','fp50'],['mAP@50','Recall@50','Falsos positivos @ IoU 0,5']):
        vals=[v[k] if k!='fp50' else sum(c['por_umbral'][0]['fp'] for c in v['por_clase'].values()) for v in m]
        bars=a.bar(range(5),vals,color=['#777777',colores['A'],colores['A'],colores['B'],colores['B']])
        a.bar_label(bars,labels=[f'{v:.2%}' if k!='fp50' else str(v) for v in vals],padding=3,fontsize=8)
        a.set(title=titulo,xticks=range(5),xticklabels=etiquetas,ylim=(0,max(vals)*1.25));a.tick_params(axis='x',labelrotation=30)
        if k!='fp50':a.yaxis.set_major_formatter(PercentFormatter(1))
        a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    fig.suptitle('Validación de cinco vistas · mejores seleccionados por una vista · umbral 0,05')
    guardar(fig,'ablacion_fondo_cinco_vistas')
    print('Tres figuras guardadas en PNG/SVG')


if __name__=='__main__':main()
