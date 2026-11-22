"""Gráfica científica de los conteos observados; no genera fotografías ni GT."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    salida=Path(__file__).resolve().parents[1]/'resultados/v28_validacion_anotaciones'
    filas=json.loads((salida/'tabla_informe_materiales.json').read_text(encoding='utf-8'))[:6]
    qa=json.loads((salida/'suficiencia_exploratoria.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained');x=np.arange(6);w=.25
    for dx,label,vals,color in [(-w,'Guardados activos',[r['objetos_guardados'] for r in filas],'#728b99'),
        (0,'En imágenes declaradas completas',[r['objetos_en_imagenes_completas'] for r in filas],'#27815e'),
        (w,'En candidatas sin incidencias identificadas',[qa['materiales'][r['material']] for r in filas],'#e6a440')]:
        bs=ax.bar(x+dx,vals,w,label=label,color=color);ax.bar_label(bs,padding=3,fontsize=8)
    ax.set_xticks(x,['Plástico','Papel','Cartón','Vidrio','Metal','Orgánico']);ax.set_ylabel('Instancias anotadas, no identidades físicas acreditadas')
    ax.set_title('V28 · Distribución por estado de revisión');ax.set_ylim(0,165);ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,fontsize=8,loc='upper right');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.08,-.03,'28 imágenes candidatas; QA visual pendiente. Entrenamiento no autorizado.',fontsize=9)
    for ext in ['png','svg']:
        p=salida/('distribucion_materiales.'+ext)
        if p.exists():raise FileExistsError('No sobrescribir gráfica')
        fig.savefig(p,dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
