"""Diagnóstico de instancias que se funden al convertir contornos en mapas semánticos."""
import cv2
import numpy as np


def analizar_componentes(etiquetas,instancias,area_minima=2):
    compartidas=set();fragmentadas=[];sin_componente=[];fusiones=[]
    for clase in range(6):
        _,componentes,stats,_=cv2.connectedComponentsWithStats((etiquetas==clase+1).astype(np.uint8),connectivity=8)
        validos={i for i in range(1,len(stats)) if stats[i,cv2.CC_STAT_AREA]>=area_minima}
        asociados={}
        for ident,c,mascara in instancias:
            if c!=clase:continue
            ids=set(np.unique(componentes[mascara.astype(bool)]).tolist())&validos
            if not ids:sin_componente.append(ident)
            if len(ids)>1:fragmentadas.append(ident)
            for i in ids:asociados.setdefault(i,[]).append(ident)
        for i,ids in asociados.items():
            if len(ids)>1:
                compartidas.update(ids)
                fusiones.append({'clase':clase,'componente':i,'anotaciones':ids})
    return {'objetos':len(instancias),'instancias_en_componentes_compartidos':sorted(compartidas),
        'instancias_fragmentadas':fragmentadas,'instancias_sin_componente_minimo':sin_componente,'fusiones':fusiones}
