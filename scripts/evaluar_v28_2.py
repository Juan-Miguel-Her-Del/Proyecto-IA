"""Original/A300/B300 sobre train histórico y auxiliar; sin val, test o adopción."""
import os
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in [None,':4096:8']:raise RuntimeError('cuBLAS incompatible')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.experimento_v28_2 import contexto,codigo,Diario,SALIDA,CERTIFICADOS
from ecowaste.incorporacion import leer,sha
from ecowaste.reanudacion_regional import publicar
from ecowaste.backend_numerico import aplicar_backend,registro_efectivo
from ecowaste.detector_mapas import DetectorMapas
from ecowaste.datasets import CLASES,geometria_letterbox
from ecowaste.poligonos import ContornosTACO
from ecowaste.validacion_anotaciones import mask_poligonos,celdas
from ecowaste.evaluacion_v28_2 import capturar,analizar


def preparar_regiones(ds,auxiliar=False):
    regiones=[];contornos=ContornosTACO(RAIZ,1/64) if not auxiliar else None
    for r in ds.registros:
        g=geometria_letterbox(r['ancho'],r['alto'],384);filas=[]
        if auxiliar:
            im=next(i for i in ds.coco['images'] if i['eco']['imagen_id_auxiliar']==r['id'])
            cats={c['id']:c['name'] for c in ds.coco['categories']}
            for a in ds.coco['annotations']:
                if a['image_id']!=im['id'] or cats[a['category_id']] not in CLASES:continue
                mask=mask_poligonos({'poligonos':a['eco']['poligonos_usuario'],'huecos':a['eco']['huecos_usuario']},r['ancho'],r['alto'],geometria=g)
                filas.append((CLASES.index(cats[a['category_id']]),celdas(mask)>=1/64,a['eco']['id_auxiliar']))
        else:
            for a in r['cajas']:
                if a['clase'] is None:continue
                mask=mask_poligonos({'poligonos':contornos.contornos(r,a)},r['ancho'],r['alto'],geometria=g)
                filas.append((CLASES.index(a['clase']),celdas(mask)>=1/64,a['id_interno']))
        regiones.append(filas)
    return regiones


def main():
    cfg,origen,ht,nu,gl,pm=contexto();c=leer(CERTIFICADOS/'protocolo_congelado.json')
    if c['codigo']!=codigo() or c['config']!=cfg:raise ValueError('Integridad del evaluador incompatible')
    ramas={}
    for rama in ['A','B']:
        ck,_=Diario(rama,c).leer()
        if ck is None or ck['v28_2']['paso']!=300:raise ValueError('Entrenamiento incompleto')
        ramas[rama]=ck
    del gl,pm
    salida=SALIDA/'evaluacion';salida.mkdir(exist_ok=True);res={}
    with aplicar_backend(cfg['backend_numerico']):
        for nombre,ck in [('Original',origen),*ramas.items()]:
            modelo=DetectorMapas(**ck['configuracion']).cuda();modelo.load_state_dict(ck['pesos'],strict=True);modelo.eval();res[nombre]={}
            for dominio,ds in [('historico',ht),('auxiliar',nu)]:
                carpeta=salida/nombre/dominio;q=carpeta/'analisis.json'
                ms=[{**{k:r[k] for k in ['id','ruta','ancho','alto']},'sha256_imagen':sha(RAIZ/r['ruta'])} for r in ds.registros]
                cert=capturar(modelo,ds.imagenes,ms,carpeta,RAIZ)
                if q.exists():analisis=leer(q)
                else:
                    analisis=analizar(carpeta,ds,preparar_regiones(ds,dominio=='auxiliar'));publicar(q,analisis)
                res[nombre][dominio]={'metricas':{v:analisis[v]['metricas'] for v in ['1','5']},
                    'costes':cert['costes'],'analisis':q.relative_to(RAIZ).as_posix(),'sha256_analisis':sha(q)}
                print(nombre,dominio,{v:res[nombre][dominio]['metricas'][v]['map50'] for v in ['1','5']},flush=True)
            del modelo;torch.cuda.empty_cache()
        p=SALIDA/'metricas.json'
        if not p.exists():publicar(p,{'resultados':res,'backend':registro_efectivo(),'protocolo':c,
            'ambito':'aprendizaje en train; no demuestra generalización','evaluacion_val_test_generalizacion':False,'adopcion':False})


if __name__=='__main__':main()
