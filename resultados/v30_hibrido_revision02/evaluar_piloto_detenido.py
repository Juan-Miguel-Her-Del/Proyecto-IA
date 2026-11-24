"""Auditoría posthoc del piloto detenido: cero actualizaciones adicionales.

No elude el entrenador ni altera su gate. Evalúa exclusivamente A/B800, no
crea metricas.json de producción ni habilita el motor híbrido de la aplicación.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
from types import SimpleNamespace
RAIZ = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.experimento_v30 import SALIDA, contexto, contrato, Diario
from ecowaste.detector_hibrido import DetectorHibrido
from ecowaste.detector_mapas import DetectorMapas
from ecowaste.datos_v29 import cargar_coleccion
from ecowaste.incorporacion import leer, sha
from ecowaste.reanudacion_regional import publicar
from ecowaste.ablacion_fondo import huella
from ecowaste.backend_numerico import aplicar_backend, politica, registro_efectivo
from scripts.evaluar_v30 import capturar, analizar, benchmark
from scripts.evaluar_v28_2 import preparar_regiones
from scripts.entrenar_detector_mapas import DatosMapas
from ecowaste.poligonos import ContornosTACO


def main():
    cfg, ht, nu, gl, pm = contexto(); del gl, pm
    plan = torch.load(SALIDA/'plan.pt', weights_only=True)
    c = contrato(cfg, plan)
    if c != leer(SALIDA/'protocolo.json'): raise ValueError('Integridad distinta')
    gate = leer(SALIDA/'piloto_800/puerta.json')
    if gate['aprobada'] or gate['contrato'] != c: raise ValueError('Este análisis solo admite piloto detenido con protocolo original')
    ramas = {}
    for rama in ['A', 'B']:
        ck, _ = Diario(rama, c).leer()
        if ck['paso'] != 800: raise ValueError('Presupuesto distinto')
        ramas[rama] = ck
    comunes = all(a['entradas_sha256'] == b['entradas_sha256'] for a,b in zip(ramas['A']['historial'],ramas['B']['historial']))
    if not comunes: raise ValueError('Entradas A/B diferentes')
    base = DatosMapas('val', ContornosTACO(RAIZ, 1/64)).base
    # DatosMapas construye máscaras; conservarlas de esa instancia para análisis.
    valmap = DatosMapas('val', ContornosTACO(RAIZ, 1/64))
    va = SimpleNamespace(registros=base.origen.registros, objetivos=base.objetivos, imagenes=base.imagenes, mascaras=valmap.mascaras)
    ret = cargar_coleccion(RAIZ, RAIZ/cfg['coleccion'], 'val')
    datasets = {'historico_train':ht,'nuevo_train':nu,'desarrollo':va,'externo_retenido':ret}
    regiones = {'historico_train':preparar_regiones(ht),'desarrollo':preparar_regiones(va)}
    for n, ds in [('nuevo_train',nu),('externo_retenido',ret)]:
        regiones[n] = [[(cl,m.numpy(),ident) for cl,m,ident in rs] for rs in ds.regiones]
    ejemplos = []
    for n, ds in datasets.items():
        for cl in [2,3,5,4,1,0]:
            ix = [(r['id'],i) for i,r in enumerate(ds.registros) if bool((ds.objetivos[i]['labels']==cl).any())]
            if ix: ejemplos.append({'dominio':n,'clase':cl,'indice':min(ix)[1],'id':min(ix)[0]})
    ep = SALIDA/'ejemplos_prefijados.json'
    if not ep.exists(): publicar(ep,ejemplos)
    elif leer(ep)!=ejemplos: raise ValueError('Regla visual alterada')
    origen = torch.load(RAIZ/cfg['checkpoint_referencia'], weights_only=True)
    v29 = leer(RAIZ/'resultados/v29_packwise_click/metricas.json'); res = {}
    with aplicar_backend(politica('completo')):
        for nombre, ck in [('Original',origen), *ramas.items()]:
            m = DetectorMapas(**ck['configuracion']) if nombre=='Original' else DetectorHibrido(**ck['configuracion_hibrido'])
            m.cuda().eval(); m.load_state_dict(ck['pesos'], strict=True); res[nombre] = {}
            for dom, ds in datasets.items():
                folder = SALIDA/'evaluacion_piloto'/nombre/dom
                equivalencia = {'historico_train':'historico_train','nuevo_train':'nuevo_train','externo_retenido':'nuevo_retenido'}
                reusada = nombre=='Original' and dom in equivalencia
                if reusada:
                    prev = v29['resultados']['Original'][equivalencia[dom]]
                    q = RAIZ/prev['analisis']; folder = q.parent
                    if sha(q)!=prev['sha256_analisis']: raise ValueError('Análisis histórico alterado')
                    cert = leer(folder/'captura.json')
                    if cert['clave']['pesos']!=huella(m.state_dict()) or cert['clave']['entradas']!=huella(ds.imagenes): raise ValueError('Referencia ajena')
                    for p,h in cert['archivos'].items():
                        if sha(folder/p)!=h: raise ValueError('Captura histórica alterada')
                    an = leer(q)
                else:
                    cert = capturar(m,ds,folder); q = folder/'analisis.json'
                    if q.exists(): an = leer(q)
                    else:
                        an = analizar(folder,ds,regiones[dom]); publicar(q,an)
                res[nombre][dom] = {'metricas':{v:an[v]['metricas'] for v in ['1','5']},'analisis':q.relative_to(RAIZ).as_posix(),
                    'sha256_analisis':sha(q),'captura':folder.relative_to(RAIZ).as_posix(),'costes':cert['costes'],'referencia_reutilizada':reusada}
                print('Piloto detenido',nombre,dom,{v:res[nombre][dom]['metricas'][v]['map50'] for v in ['1','5']},flush=True)
            res[nombre]['benchmark'] = benchmark(m,va)
            del m; torch.cuda.empty_cache()
        publicar(SALIDA/'metricas_piloto.json',{'resultados':res,'codigo':c['codigo'],'protocolo':c,'backend':registro_efectivo(),
            'presupuesto_maximo':2400,'paso_evaluado':800,'presupuesto_completo_ejecutado':False,
            'motivo_parada':'B no supera mAP50>=0.01 en gate de 800, criterio prefijado',
            'puerta_integracion_superada':False,'entradas_A_B_identicas':comunes,
            'test':False,'generalizacion_desbloqueada':False,'adopcion':False,
            'advertencia':'Desarrollo histórico; retenido externo solo plástico/papel/metal e independencia física no demostrada.'})


if __name__=='__main__': main()
