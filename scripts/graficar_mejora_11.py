"""Comparación estática exportable: clasificación e inferencias de detección."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

RAIZ=Path(__file__).resolve().parents[1]


def leer(n):return json.loads((RAIZ/f'resultados/{n}.json').read_text(encoding='utf-8'))


def main():
    cls=[leer(n) for n in ['clasificador_piloto_v1','clasificador_v2_coseno80','clasificador_v3_regularizado']]
    det=[leer(n) for n in ['detector_mapas_v6_auxiliar_ext48','detector_mapas_v6_mosaico',
                          'detector_centros_v11','detector_centros_v11_mosaico']]
    fig,ax=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    fig.suptitle('Validación de desarrollo · modelos propios · test reservado')
    x=np.arange(3)
    for dx,clave,label,color in [(-.18,'accuracy','Exactitud','#226ca3'),(.18,'f1_macro','F1 macro','#138573')]:
        barras=ax[0].bar(x+dx,[r['val'][clave] for r in cls],width=.35,label=label,color=color)
        ax[0].bar_label(barras,labels=[f"{r['val'][clave]:.1%}" for r in cls],padding=3,fontsize=8)
    ax[0].set(xticks=x,xticklabels=['Inicial: 15 épocas','V2: coseno','V3: regularizado'],ylim=(0,1),
              title='Clasificación: 610 imágenes val')
    ax[0].yaxis.set_major_formatter(PercentFormatter(1))
    ax[0].legend(loc='lower right')
    x=np.arange(4)
    for dx,clave,label,color in [(-.18,'map50','mAP@50','#226ca3'),(.18,'map50_95','mAP@50:95','#138573')]:
        barras=ax[1].bar(x+dx,[r['val'][clave] for r in det],width=.35,label=label,color=color)
        ax[1].bar_label(barras,labels=[f"{r['val'][clave]:.2%}" for r in det],padding=3,fontsize=8)
    ax[1].set(xticks=x,xticklabels=['Mapas: 1 vista','Mapas: 5 vistas','Centros: 1 vista','Centros: 5 vistas'],
              ylim=(0,max(r['val']['map50'] for r in det)*1.35),title='Detección: 181 imágenes / 361 objetos val')
    ax[1].yaxis.set_major_formatter(PercentFormatter(1))
    ax[1].tick_params(axis='x',labelrotation=15)
    ax[1].legend(loc='upper right')
    for a in ax:a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    for extension in ['png','svg']:fig.savefig(RAIZ/f'resultados/figuras/comparacion_mejora_11.{extension}',dpi=170)
    plt.close(fig)
    print('Comparación PNG/SVG guardada')


if __name__=='__main__':main()
