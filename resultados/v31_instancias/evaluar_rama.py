"""Evaluación por rama tras presupuesto fijo; reutiliza evaluador congelado V31."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,sys
from pathlib import Path
from types import SimpleNamespace
RAIZ=Path(__file__).resolve().parents[2];sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.experimento_v31 import OUT,contexto,contrato,Diario
from ecowaste.datos_v29 import cargar_coleccion
from ecowaste.ejecucion_detector import CacheDeteccion
from ecowaste.incorporacion import leer,sha
from ecowaste.reanudacion_regional import publicar
from ecowaste.backend_numerico import aplicar_backend,politica,registro_efectivo
from scripts.evaluar_instancias_v31 import checkpoint,capturar,analizar,meta


def ejecutar(m,dominios,base):
    res={}
    for dom,(ds,vistas) in dominios.items():
        folder=base/dom;cap=capturar(m,ds.imagenes,meta(ds),folder,vistas);q=folder/'analisis.json'
        if q.exists():an=leer(q)
        else:an=analizar(folder,ds,vistas);publicar(q,an)
        res[dom]={'analisis':q.relative_to(RAIZ).as_posix(),'sha256':sha(q),'metricas':{str(v):an[str(v) if str(v) in an else v]['metricas'] for v in vistas},'costes':cap['costes']}
        print(dom,{v:r['map50'] for v,r in res[dom]['metricas'].items()},flush=True)
    return res


def main():
    p=argparse.ArgumentParser();p.add_argument('--rama',choices=['A','B'],required=True);a=p.parse_args()
    cfg,ht,nu,gl,pm=contexto();del gl,pm
    plan=torch.load(OUT/'plan.pt',weights_only=True);c=contrato(cfg,plan)
    if c!=leer(OUT/'protocolo.json'):raise ValueError('Protocolo/código cambiados')
    final,_=Diario(a.rama,c).leer()
    if final['paso']!=2400:raise ValueError('Entrenamiento incompleto')
    del final
    v=CacheDeteccion(RAIZ,'val',.1);va=SimpleNamespace(imagenes=v.imagenes,objetivos=v.objetivos,registros=v.origen.registros)
    ret=cargar_coleccion(RAIZ,RAIZ/cfg['coleccion'],'val');todo={}
    with aplicar_backend(politica('completo')):
        for k in [800,1600,2400]:
            base=OUT/f'evaluacion/{a.rama}/{k:04d}';m,ck=checkpoint(a.rama,k,c)
            dominios={'desarrollo':(va,[1])} if k!=2400 else {'historico_train':(ht,[1]),'nuevo_train':(nu,[1]),'desarrollo':(va,[1,5]),'externo_retenido':(ret,[1,5])}
            resultados=ejecutar(m,dominios,base)
            out={'rama':a.rama,'paso':k,'resultados':resultados,'backend':registro_efectivo(),'test':False,'adopcion':False}
            if not (base/'resumen.json').exists():publicar(base/'resumen.json',out)
            else:
                ant=leer(base/'resumen.json')
                if ant['resultados']!=resultados:raise ValueError('Resumen previo diferente')
            todo[str(k)]=out;del m,ck;torch.cuda.empty_cache()
        mejor=max([800,1600,2400],key=lambda k:todo[str(k)]['resultados']['desarrollo']['metricas']['1']['map50'])
        if mejor!=2400:
            m,ck=checkpoint(a.rama,mejor,c)
            elegida=ejecutar(m,{'desarrollo':(va,[1,5]),'externo_retenido':(ret,[1,5])},OUT/f'evaluacion_mejor/{a.rama}/{mejor:04d}')
            del m,ck;torch.cuda.empty_cache()
        else:elegida={d:todo['2400']['resultados'][d] for d in ['desarrollo','externo_retenido']}
        publicar(OUT/f'metricas_{a.rama}.json',{'rama':a.rama,'pasos':todo,'mejor_paso_por_map50_desarrollo_1v':mejor,'mejor':elegida,
            'seleccion_prefijada':True,'test':False,'generalizacion_desbloqueada':False,'adopcion':False,'backend':registro_efectivo()})
    print('Rama',a.rama,'evaluada; mejor paso',mejor,flush=True)


if __name__=='__main__':main()
