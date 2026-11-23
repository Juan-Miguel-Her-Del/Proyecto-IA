"""Análisis descriptivo posthoc de GT; no inferencia, entrenamiento o anotaciones nuevas."""
from pathlib import Path
import csv,hashlib,json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RAIZ=Path(__file__).resolve().parents[2]
OUT=RAIZ/'resultados/v29_packwise_click';COL=RAIZ/'datos/nuevos/v29/coleccion'
CLASES=['plastico','papel','carton','vidrio','metal','organico']


def main():
    resumen={};filas=[]
    for sp in ['train','val']:
        path=COL/f'{sp}_coco.json';d=json.loads(path.read_text(encoding='utf8'));ims={i['id']:i for i in d['images']}
        objetos=[]
        for a in d['annotations']:
            if a['ignore']:continue
            im=ims[a['image_id']];s=384/max(im['width'],im['height']);w,h=a['bbox'][2]*s,a['bbox'][3]*s
            r={'split':sp,'imagen':im['eco']['id'],'fuente':im['eco']['fuente'],'material':CLASES[a['category_id']-1],
                'id_instancia':a['id'],'ancho_bbox_en_384':w,'alto_bbox_en_384':h,'area_mask_aproximada_en_384':a['area']*s*s,
                'lado_menor_bbox_menor_8px':min(w,h)<8,'fraccion_bbox_imagen':a['bbox'][2]*a['bbox'][3]/(im['width']*im['height'])}
            objetos.append(r);filas.append(r)
        por={}
        for c in CLASES:
            rs=[r for r in objetos if r['material']==c]
            por[c]={'instancias':len(rs),'bbox_lado_menor_inferior_a_un_token':sum(r['lado_menor_bbox_menor_8px'] for r in rs),
                'mediana_lado_menor_en_384':float(np.median([min(r['ancho_bbox_en_384'],r['alto_bbox_en_384']) for r in rs])) if rs else None}
        resumen[sp]={'por_material':por,'imagenes_por_fuente':dict(Counter(i['eco']['fuente'] for i in d['images'])),
            'grupos_captura_conservadores':len({i['eco']['grupo_componente'] for i in d['images']}),
            'sha256_COCO':hashlib.sha256(path.read_bytes()).hexdigest()}
    out=OUT/'tamanos_y_soportes.json'
    with out.open('x',encoding='utf8') as f:json.dump({'resumen':resumen,'nota':'GT solo para análisis descriptivo. Un token es 8x8 a entrada 384; bbox pequeño no certifica imposibilidad de detección ni equivale a máscara/token elegible. Grupos heurísticos no son objetos físicos.'},f,ensure_ascii=False,indent=2)
    with (OUT/'tamanos_instancias.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(filas[0]));w.writeheader();w.writerows(filas)
    g=OUT/'graficas';g.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for sp,offset in [('train',-.18),('val',.18)]:
        axs[0].bar(np.arange(6)+offset,[resumen[sp]['por_material'][c]['instancias'] for c in CLASES],width=.35,label=sp)
        valores=[min(r['ancho_bbox_en_384'],r['alto_bbox_en_384']) for r in filas if r['split']==sp]
        axs[1].hist(valores,bins=[0,4,8,16,32,64,128,256,384],alpha=.5,label=sp)
    axs[0].set_xticks(range(6),CLASES,rotation=30);axs[0].set_title('Instancias positivas; no identidades físicas');axs[0].legend()
    axs[1].axvline(8,color='black',linestyle='--',label='un token');axs[1].set_title('Lado menor de bbox en entrada 384');axs[1].legend()
    fig.tight_layout();fig.savefig(g/'soportes_y_tamanos.png',dpi=160);plt.close(fig)
    print(resumen,flush=True)


if __name__=='__main__':main()
