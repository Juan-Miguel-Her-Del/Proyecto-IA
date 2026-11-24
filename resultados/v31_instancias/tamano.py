"""Desglose descriptivo por tamaño, solo GT train/desarrollo y raw ya guardados."""
import sys
from pathlib import Path
from types import SimpleNamespace
RAIZ=Path(__file__).resolve().parents[2];sys.path.insert(0,str(RAIZ));OUT=Path(__file__).resolve().parent
import torch
from ecowaste.ejecucion_detector import CacheDeteccion
from ecowaste.incorporacion import leer
from ecowaste.reanudacion_regional import publicar
def main():
    torch.set_num_threads(4);cfg=leer(RAIZ/'configuracion/instancias_v31.json');ex=leer(OUT/'analisis_adicional.json');ds={'historico_train':CacheDeteccion(RAIZ,'train',.1),'desarrollo':CacheDeteccion(RAIZ,'val',.1),'nuevo_train':SimpleNamespace(**torch.load(RAIZ/cfg['coleccion']/'train.pt',weights_only=True))};out={}
    for n,doms in ex['resultados'].items():
        out[n]={}
        for dom,q in doms.items():
            objetivos={o['id']:o for o in ds[dom].objetivos};bins={b:{'gt':0,'con_geometria_iou50':0,'material_correcto_mayor_iou':0,'ausencia_objeto':0} for b in ['pequeno','mediano','grande']}
            for r in q['final_1v']['objetos']:
                o=objetivos[r['imagen']];b=o['boxes'][r['indice_gt']];area=float((b[2:]-b[:2]).prod())*384**2;grupo='pequeno' if area<32**2 else 'mediano' if area<96**2 else 'grande';z=bins[grupo];z['gt']+=1;z['con_geometria_iou50']+=int(r['iou']>=.5);z['material_correcto_mayor_iou']+=int(r['prediccion']==int(o['labels'][r['indice_gt']]));z['ausencia_objeto']+=int(r['prediccion']==6)
            out[n][dom]=bins
    publicar(OUT/'tamano.json',{'resultados':out,'paso':2400,'vistas':1,'criterio':'Área bbox en lienzo letterbox 384; pequeño <32², mediano <96², grande resto. Desglose posthoc descriptivo, no AP COCO por área original. Consulta raw mayor IoU, no asignación uno a uno.','inferencia_repetida':False})
if __name__=='__main__':main()
