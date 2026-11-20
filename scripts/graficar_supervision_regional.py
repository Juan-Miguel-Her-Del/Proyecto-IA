"""Figuras exportables del diagnóstico y, cuando existan, de A/B completo."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
RAIZ=Path(__file__).resolve().parents[1]


def main():
    base=RAIZ/'resultados/supervision_regional_preparacion'
    cfg=json.loads((RAIZ/'configuracion/supervision_regional_materiales.json').read_text(encoding='utf-8'))
    d=json.loads((RAIZ/cfg['diagnostico']).read_text(encoding='utf-8'))
    if not d['aprobada']:raise ValueError('Diagnóstico no aprobado')
    plt.rcParams.update({'font.size':10,'figure.dpi':140})
    nombres=d['estadistica_regiones']['orden_materiales'];est=d['estadistica_regiones']
    fig,axs=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    x=np.arange(6);bottom=np.zeros(6)
    for modo,color in [('interior','#187c66'),('fraccion_poligono','#d9a63e'),('vacia','#b65859')]:
        v=np.array([est['modos_por_material'][c].get(modo,0) for c in nombres])
        axs[0].bar(x,v,bottom=bottom,label=modo,color=color);bottom+=v
    axs[0].set(xticks=x,xticklabels=nombres,ylabel='Instancias TACO train',title='Regiones elegibles y omisiones')
    axs[0].tick_params(axis='x',rotation=30);axs[0].legend(fontsize=8)
    axs[1].bar(x,est['pesos_clase'],color='#497baf')
    axs[1].set(xticks=x,xticklabels=nombres,ylabel='Peso de clase (escala log)',yscale='log',title='Frecuencias de regiones elegibles de train')
    axs[1].tick_params(axis='x',rotation=30)
    fig.savefig(base/'regiones_y_pesos.png');plt.close(fig)
    h=d['memorizacion']['historial'];fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    t=[v['paso'] for v in h]
    axs[0].plot(t,[v['ce_regional'] for v in h],marker='o',label='CE regional')
    axs[0].set(xlabel='Paso de diagnóstico',ylabel='CrossEntropy ponderada',title='Memorización train; pesos descartados')
    for k,label in [('accuracy_regional','Exactitud regional'),('accuracy_auxiliar_global','Exactitud global'),('map50_memorizacion','AP50 memorización')]:
        axs[1].plot(t,[v[k] for v in h],marker='o',label=label)
    axs[1].set(xlabel='Paso de diagnóstico',ylabel='Fracción',ylim=(0,1.05),title='Diagnóstico funcional; no generalización');axs[1].legend()
    fig.savefig(base/'memorizacion.png');plt.close(fig)
    cm=np.array(d['confusion_regional_train_origen']['matriz']);tot=cm.sum(1,keepdims=True)
    norm=np.divide(cm,tot,out=np.zeros_like(cm,dtype=float),where=tot>0)
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained');im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
    for i in range(6):
        for j in range(6):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if norm[i,j]>.6 else 'black')
    ax.set(xticks=range(6),yticks=range(6),xticklabels=nombres,yticklabels=nombres,
        xlabel='Material predicho',ylabel='GT poligonal train',title='Cabeza compartida del origen época 19; conteos y color por fila')
    ax.tick_params(axis='x',rotation=30);fig.colorbar(im,ax=ax,label='Fracción por GT')
    fig.savefig(base/'confusion_train_origen.png');plt.close(fig)
    resultado=RAIZ/f"resultados/{cfg['nombre']}.json"
    evaluacion=RAIZ/f"resultados/{cfg['nombre']}_evaluacion.json"
    if resultado.exists() and evaluacion.exists():
        r=json.loads(resultado.read_text(encoding='utf-8'));e=json.loads(evaluacion.read_text(encoding='utf-8'))
        fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for rama in ['A','B']:
            hist=r['ramas'][rama]['historial']
            axs[0].plot([v['actualizaciones'] for v in hist],[100*v['map50_val'] for v in hist],marker='o',label=rama)
        axs[0].set(xlabel='Actualizaciones adicionales',ylabel='mAP@50 val (%)',title='Una vista: mismo presupuesto');axs[0].legend()
        for j,rama in enumerate(['A','B']):
            ap=e['resultados'][rama+'_ultimo']['vistas']['5']['resumen']['ap50_por_material']
            axs[1].bar(x+(j-.5)*.36,[100*ap[c] for c in nombres],width=.36,label=rama)
        axs[1].set(xticks=x,xticklabels=nombres,ylabel='AP@50 val (%)',title='Cinco vistas: checkpoints finales');axs[1].tick_params(axis='x',rotation=30);axs[1].legend()
        fig.savefig(base/'comparacion_ab.png');plt.close(fig)
    print('Figuras guardadas en',base)


if __name__=='__main__':main()
