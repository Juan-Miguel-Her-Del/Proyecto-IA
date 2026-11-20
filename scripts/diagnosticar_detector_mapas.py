"""Diagnóstico de representabilidad y memorización del detector de mapas débiles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from torch.nn import functional as F
from ecowaste.datasets import ResiduosDataset, objetivo_deteccion
from ecowaste.detector_mapas import DetectorMapas, rasterizar, perdida_mapas, cajas_desde_mapas
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.poligonos import ContornosTACO, POLITICA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fuente',choices=['cajas','poligonos'],default='cajas')
    parser.add_argument('--nombre',default='diagnostico_detector_mapas')
    parser.add_argument('--cobertura-minima',type=float,default=.25)
    args=parser.parse_args()
    if Path(args.nombre).name!=args.nombre:raise ValueError('Nombre inválido')
    salida=RAIZ/f'resultados/{args.nombre}.json'
    if salida.exists():raise FileExistsError('Diagnóstico existente; usar otro nombre')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    ds=ResiduosDataset(RAIZ,'deteccion','train')
    contornos=ContornosTACO(RAIZ,args.cobertura_minima) if args.fuente=='poligonos' else None
    def mascara(i,o):
        return contornos.rasterizar(ds.registros[i]) if contornos else rasterizar(o)
    indices=[]
    for c in range(6):
        candidatos=[]
        for i,r in enumerate(ds.registros):
            if i in indices: continue
            o=objetivo_deteccion(r)
            if len(o['boxes'])>3 or not bool((o['labels']==c).any()):continue
            tam=(o['boxes'][:,2:]-o['boxes'][:,:2]).min(-1).values*384
            if bool((tam<16).any()): continue
            candidatos.append((float(tam[o['labels']==c].max()),i))
        elegido=None
        for _,i in sorted(candidatos,reverse=True):
            o=objetivo_deteccion(ds.registros[i])
            target=mascara(i,o)
            if not bool((target==c+1).any()):continue
            oracle=F.one_hot(target.clamp(max=6),7).permute(2,0,1).float().unsqueeze(0)*40
            oracle[:,0][target.unsqueeze(0)==255]=50
            m=evaluar_detecciones(cajas_desde_mapas(oracle, area_minima=2,geometrias=[o['geometria']]),[o])
            if m['recall50']<.99:continue
            elegido=i;break
        if elegido is None: raise RuntimeError(f'No hay escena representable para clase {c}')
        indices.append(elegido)
    lote=[ds[i] for i in indices]
    x=torch.stack([t for t,_ in lote]).cuda()
    objetivos=[o for _,o in lote]
    target=torch.stack([mascara(i,o) for i,o in zip(indices,objetivos)]).cuda()
    modelo=DetectorMapas().cuda()
    opt=torch.optim.AdamW(modelo.parameters(),lr=.001,weight_decay=0)
    inicio=time.perf_counter()
    historial=[]
    aprobado=False
    for paso in range(1,401):
        modelo.train(); opt.zero_grad(set_to_none=True)
        loss=perdida_mapas(modelo(x),target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(),1,error_if_nonfinite=True)
        opt.step()
        if paso%25==0:
            modelo.eval()
            with torch.no_grad(): pred=cajas_desde_mapas(modelo(x),umbral=.35,area_minima=2,geometrias=[o['geometria'] for o in objetivos])
            m=evaluar_detecciones(pred,objetivos)
            historial.append({'paso':paso,'loss':float(loss.detach()),'map50':m['map50'],'recall50':m['recall50'],'precision50':m['precision50']})
            print(f"Paso {paso}: loss={float(loss.detach()):.3f}, AP50={m['map50']:.3f}, recall={m['recall50']:.3f}, precision={m['precision50']:.3f}",flush=True)
            if m['map50']>=.8 and m['recall50']>=.9 and m['precision50']>=.7:
                aprobado=True; break
    resultado={'aprobada':aprobado,'tipo':'mapas_de_cobertura_de_cajas_no_mascaras_precisas' if contornos is None else contornos.politica,
        'area_minima_celdas':2,'cobertura_minima':args.cobertura_minima,
        'postprocesamiento':'componentes_solo_contenido_real_v2',
        'ultimas_predicciones_train':[{k:v.tolist() for k,v in p.items()} for p in pred],
        'objetivos_train':[{'id':o['id'],'boxes':o['boxes'].tolist(),'labels':o['labels'].tolist()} for o in objetivos],
        'fuente_supervision':args.fuente,'sha256_anotaciones':contornos.sha256 if contornos else None,
        'sha256_train':hashlib.sha256((RAIZ/'datos/procesados/deteccion/train.jsonl').read_bytes()).hexdigest(),
        'objetos':sum(len(o['boxes']) for o in objetivos),
        'ids_train':[o['id'] for o in objetivos],'historial':historial,'duracion_segundos':time.perf_counter()-inicio,
        'parametros':sum(p.numel() for p in modelo.parameters()),'test_evaluado':False,
        'limite':'Escenas fáciles con representabilidad comprobada; no mide generalización. No se reutilizan estos pesos.'}
    salida.write_text(json.dumps(resultado,indent=2),encoding='utf-8')
    print('Diagnóstico aprobado:',aprobado,flush=True)
    return 0 if aprobado else 1


if __name__=='__main__':raise SystemExit(main())
