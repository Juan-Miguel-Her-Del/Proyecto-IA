"""Desglose train-only HPI/comida y WSC/vegetación sobre predicciones ya guardadas."""
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer
from ecowaste.reanudacion_regional import publicar
from ecowaste.datos_v28_2 import AuxiliarExperimental
from ecowaste.evaluacion_v28_2 import cargar_predicciones
from ecowaste.evaluacion_detector import evaluar_detecciones


def main():
    s=RAIZ/'resultados/v28_2_exploratorio';ds=AuxiliarExperimental(RAIZ,RAIZ/'datos/nuevos/v28_2/coleccion');sal={}
    for n in ['Original','A','B']:
        sal[n]={}
        for v in [1,5]:
            ps=cargar_predicciones(s/'evaluacion'/n/'auxiliar',v);sal[n][str(v)]={}
            for fuente in ['hpi','wsc']:
                ix=[i for i,r in enumerate(ds.registros) if r['fuente']==fuente]
                m=evaluar_detecciones([ps[i] for i in ix],[ds.objetivos[i] for i in ix])['por_clase']['organico']
                p=m['por_umbral'][0]
                sal[n][str(v)][fuente]={'imagenes':len(ix),'objetos':m['objetos'],'tp50':p['tp'],'fp50':p['fp'],
                    'fn50':m['objetos']-p['tp'],'ap50':m['ap50'],'precision50':p['precision'],'recall50':p['recall']}
    publicar(s/'organicos_deteccion_por_fuente.json',{'resultados':sal,'solo_train':True,
        'inferencias_nuevas':0,'interpretacion':'subgrupos pequeños de fuentes ya entrenadas; no estimación de generalización'})
    print(sal,flush=True)


if __name__=='__main__':main()
