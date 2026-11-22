"""Figuras V25, con requirements-graficas y sin PyTorch."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
RAIZ=Path(__file__).resolve().parents[1]
SALIDA=RAIZ/'resultados/v25_datos_representacion'
CLASES=['plastico','papel','carton','vidrio','metal','organico']
COLORES=['#4b9b70','#a886ce','#b07b3a','#3aa8ba','#718495','#db9146']
def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def graficas(ana,censo,oracle):
    carpeta=SALIDA/'figuras';carpeta.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for j,s in enumerate(['train','val']):
        vals=[censo['splits'][s]['censo']['por_material'][c]['instancias'] for c in CLASES]
        axs[j].bar(CLASES,vals,color=COLORES);axs[j].set_title(f'Instancias GT {s}');axs[j].tick_params(axis='x',rotation=25)
        for i,v in enumerate(vals):axs[j].text(i,v,str(v),ha='center',va='bottom')
    fig.savefig(carpeta/'balance.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for j,s in enumerate(['train','val']):
        pp=ana['splits'][s]['propuestas']['por_material'];causas=list(ana['splits'][s]['propuestas']['total']['causas']);bottom=np.zeros(6)
        for causa in causas:
            vals=np.array([pp[c]['causas'][causa]/pp[c]['gt']*100 for c in CLASES]);axs[j].bar(CLASES,vals,bottom=bottom,label=causa);bottom+=vals
        axs[j].set_title(f'Causas principales, cinco vistas: {s}');axs[j].tick_params(axis='x',rotation=25);axs[j].set_ylabel('% GT')
    axs[1].legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
    fig.savefig(carpeta/'errores_train_val.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for i,s in enumerate(['train','val']):
        for j,(key,labels) in enumerate([('confusion_instancia_voto_principal',['fondo',*CLASES]),('confusion_instancia_voto_condicional',CLASES)]):
            mat=np.array(ana['splits'][s]['principal'][key]);mat=mat[1:] if j==0 else mat
            norm=mat/np.maximum(mat.sum(1,keepdims=True),1)*100
            ax=axs[i,j];ax.imshow(norm,vmin=0,vmax=100,cmap='Blues');ax.set_xticks(range(len(labels)),labels,rotation=45,ha='right');ax.set_yticks(range(6),CLASES)
            for y in range(6):
                for x in range(len(labels)):ax.text(x,y,str(int(mat[y,x])),ha='center',va='center',color='white' if norm[y,x]>55 else 'black',fontsize=9)
            ax.set_title(f'{s}: '+('voto 7 clases' if j==0 else 'material condicionado a GT'))
    fig.savefig(carpeta/'confusiones_principales.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained');x=np.arange(6)
    for i,s in enumerate(['train','val']):
        for j,n in enumerate(['48','96','384']):
            vals=[100*oracle[s][n][c]['caja_semantica_iou50']/oracle[s][n][c]['instancias'] for c in CLASES]
            axs[i].bar(x+(j-1)*.25,vals,width=.25,label=n+'x'+n)
        axs[i].set_xticks(x,CLASES,rotation=25);axs[i].set_ylim(0,105);axs[i].set_title(f'GT → componentes: {s} (no AP)');axs[i].set_ylabel('% GT con caja IoU≥.5');axs[i].legend()
    fig.savefig(carpeta/'oraculo_geometria.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    graficas(leer(SALIDA/'analisis/resumen.json'),leer(SALIDA/'censo/resumen.json'),leer(SALIDA/'contrafactuales_resumen.json'))
