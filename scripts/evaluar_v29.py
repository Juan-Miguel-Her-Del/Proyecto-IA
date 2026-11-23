"""Evaluación fija Original/A600/B600: nuevo train, cohorte retenida y retención histórica.

No evalúa validación histórica, test ni colección de generalización bloqueada.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.experimento_v29 import contexto,codigo,Diario,SALIDA
from ecowaste.datos_v29 import cargar_coleccion
from ecowaste.incorporacion import leer,sha
from ecowaste.reanudacion_regional import publicar
from ecowaste.backend_numerico import aplicar_backend,registro_efectivo
from ecowaste.detector_mapas import DetectorMapas
from ecowaste.evaluacion_v28_2 import capturar,analizar,cargar_predicciones
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.ablacion_fondo import resumir_deteccion
from scripts.evaluar_v28_2 import preparar_regiones


def main():
    cfg,origen,ht,nu,gl,pm=contexto();c=leer(SALIDA/'controles/protocolo_congelado.json')
    if c['codigo']!=codigo() or c['config']!=cfg:raise ValueError('Integridad de evaluación incompatible')
    va=cargar_coleccion(RAIZ,RAIZ/cfg['coleccion'],'val');ramas={}
    for rama in ['A','B']:
        ck,_=Diario(rama,c).leer()
        if ck is None or ck['v29']['paso']!=600:raise ValueError('Entrenamiento incompleto')
        ramas[rama]=ck
    a,b=[ramas[r]['v29']['historial'] for r in ['A','B']]
    comunes=all(x['huella_entradas_comunes']==y['huella_entradas_comunes'] and
        x['rng_comun_antes']==y['rng_comun_antes'] and x['rng_comun_despues']==y['rng_comun_despues'] for x,y in zip(a,b))
    if not comunes:raise ValueError('Exposición/aumentos históricos divergentes: detener evaluación comparativa')
    del gl,pm
    dominios={'historico_train':ht,'nuevo_train':nu,'nuevo_retenido':va}
    regiones={'historico_train':preparar_regiones(ht)}
    for n,ds in [('nuevo_train',nu),('nuevo_retenido',va)]:
        regiones[n]=[[(c,m.numpy(),ident) for c,m,ident in rs] for rs in ds.regiones]
    # Ejemplos definidos por IDs antes de obtener métricas/predicciones V29.
    ejemplos=[]
    for nombre,ds in dominios.items():
        for cl in [2,3,5,1,4,0]:
            candidatos=[(r['id'],i) for i,r in enumerate(ds.registros) if bool((ds.objetivos[i]['labels']==cl).any())]
            if candidatos:
                iid,i=min(candidatos);ejemplos.append({'dominio':nombre,'material':cl,'id':iid,'indice':i})
    p=SALIDA/'seleccion_visual_prefijada.json'
    if p.exists():
        if leer(p)!=ejemplos:raise ValueError('Selección visual cambió')
    else:publicar(p,ejemplos)
    salida=SALIDA/'evaluacion';res={}
    with aplicar_backend(cfg['backend_numerico']):
        for nombre,ck in [('Original',origen),*ramas.items()]:
            m=DetectorMapas(**ck['configuracion']).cuda();m.load_state_dict(ck['pesos'],strict=True);m.eval();res[nombre]={}
            for dominio,ds in dominios.items():
                carpeta=salida/nombre/dominio;reusada=False
                anterior=RAIZ/'resultados/v28_2_exploratorio'
                if nombre=='Original' and dominio=='historico_train' and (anterior/'metricas.json').exists():
                    previo=leer(anterior/'metricas.json');pc=anterior/'evaluacion/Original/historico'
                    if previo['backend']==registro_efectivo() and (pc/'captura.json').exists() and (pc/'analisis.json').exists():
                        carpeta=pc;reusada=True
                q=carpeta/'analisis.json'
                meta=[{**{k:r[k] for k in ['id','ruta','ancho','alto']},'sha256_imagen':r.get('sha256_archivo') or sha(RAIZ/r['ruta'])} for r in ds.registros]
                cert=capturar(m,ds.imagenes,meta,carpeta,RAIZ)
                if q.exists():an=leer(q)
                else:an=analizar(carpeta,ds,regiones[dominio]);publicar(q,an)
                res[nombre][dominio]={'metricas':{v:an[v]['metricas'] for v in ['1','5']},'costes':cert['costes'],
                    'analisis':q.relative_to(RAIZ).as_posix(),'sha256_analisis':sha(q),'captura_reutilizada_y_verificada':reusada}
                if dominio!='historico_train':
                    por={}
                    for fuente in ['packwise','click']:
                        idx=[i for i,r in enumerate(ds.registros) if r['fuente']==fuente]
                        if not idx:continue
                        por[fuente]={}
                        for v in [1,5]:
                            ps=cargar_predicciones(carpeta,v)
                            por[fuente][str(v)]=resumir_deteccion(evaluar_detecciones([ps[i] for i in idx],[ds.objetivos[i] for i in idx]))
                    res[nombre][dominio]['por_fuente']=por
                print(nombre,dominio,{v:res[nombre][dominio]['metricas'][v]['map50'] for v in ['1','5']},flush=True)
            del m;torch.cuda.empty_cache()
        out=SALIDA/'metricas.json'
        if not out.exists():publicar(out,{'resultados':res,'backend':registro_efectivo(),'protocolo':c,
            'exposicion_historica_igual_A_B':comunes,'validacion_historica_usada':False,'test_usado':False,
            'generalizacion_desbloqueada':False,'adopcion':False,
            'limites':'Cohorte nueva retenida por secuencia PackWISE y bloques heurísticos CLICK; no independencia física demostrada ni evaluación final. mAP promedia solo clases con soporte, no comparable directamente a 2.912% de validación histórica.'})


if __name__=='__main__':main()
