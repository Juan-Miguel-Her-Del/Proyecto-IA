"""Memorización de seis recortes train representables; no reutiliza los pesos."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from torch.nn import functional as F
from ecowaste.datasets import CLASES,ResiduosDataset
from ecowaste.poligonos import ContornosTACO
from ecowaste.recortes import CacheRecortes,objetivo_recorte
from ecowaste.detector_mapas import DetectorMapas,cajas_desde_mapas,perdida_mapas
from ecowaste.evaluacion_detector import evaluar_detecciones
from scripts.entrenar_clasificador import escribir_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--carpeta',default='datos/cache/recortes_multiescala_v1')
    p.add_argument('--nombre',default='diagnostico_recortes')
    a=p.parse_args()
    if Path(a.nombre).name!=a.nombre:raise ValueError('Nombre inválido')
    salida=RAIZ/f'resultados/{a.nombre}.json'
    if salida.exists():raise FileExistsError('Diagnóstico existente')
    torch.set_num_threads(4);torch.manual_seed(42)
    c=ContornosTACO(RAIZ,1/64)
    hashes={'train':hashlib.sha256((RAIZ/'datos/procesados/deteccion/train.jsonl').read_bytes()).hexdigest(),
        'anotaciones_poligonos':c.sha256,'inventario_procedencia':c.sha256_inventario}
    cache=CacheRecortes(RAIZ,a.carpeta,hashes)
    registros=ResiduosDataset(RAIZ,'deteccion','train').registros
    indices=[];objetivos=[]
    for clase in CLASES:
        elegido=None
        for i,r in enumerate(cache.plan):
            if r['clase_ancla']!=clase or i in indices:continue
            o,y=objetivo_recorte(registros[r['indice_registro']],r['ventana_xyxy'],c)
            if len(o['boxes'])>3:continue
            dimensiones=(o['boxes'][:,2:]-o['boxes'][:,:2])*384
            if bool((dimensiones.min(-1).values<16).any()):continue
            logits=F.one_hot(y.clamp(max=6),7).permute(2,0,1).float()[None]*40
            logits[:,0][y[None]==255]=50
            m=evaluar_detecciones(cajas_desde_mapas(logits,area_minima=2,geometrias=[o['geometria']]),[o])
            if m['recall50']>=.99:
                elegido=i;objetivos.append(o);break
        if elegido is None:raise ValueError(f'Sin recorte representable para {clase}')
        indices.append(elegido)
    x=torch.stack([cache[i][0] for i in indices]).cuda().float()/255
    target=torch.stack([cache[i][1] for i in indices]).cuda()
    modelo=DetectorMapas().cuda();opt=torch.optim.AdamW(modelo.parameters(),lr=.001,weight_decay=0)
    inicio=time.perf_counter();historial=[];aprobada=False
    for paso in range(1,401):
        modelo.train();opt.zero_grad(set_to_none=True)
        loss=perdida_mapas(modelo(x),target);loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(),1,error_if_nonfinite=True);opt.step()
        if paso%25==0:
            modelo.eval()
            with torch.no_grad():pred=cajas_desde_mapas(modelo(x),.35,2,[o['geometria'] for o in objetivos])
            m=evaluar_detecciones(pred,objetivos)
            historial.append({'paso':paso,'loss':float(loss.detach()),**{k:m[k] for k in ['map50','recall50','precision50']}})
            print(f"Paso {paso}: loss={loss.item():.3f}, AP50={m['map50']:.3f}, recall={m['recall50']:.3f}, precision={m['precision50']:.3f}",flush=True)
            if m['map50']>=.8 and m['recall50']>=.9 and m['precision50']>=.7:aprobada=True;break
    escribir_json(salida,{'aprobada':aprobada,'recortes_ids':[cache.plan[i]['id'] for i in indices],
        'sha256_metadata_recortes':hashlib.sha256((RAIZ/a.carpeta/'metadata.json').read_bytes()).hexdigest(),
        'historial':historial,'objetos':sum(len(o['boxes']) for o in objetivos),
        'parametros':sum(p.numel() for p in modelo.parameters()),'duracion_segundos':time.perf_counter()-inicio,
        'pesos_reutilizados':False,'test_evaluado':False,
        'limite':'Solo escenas fáciles train con representabilidad comprobada; no mide generalización.'})
    return 0 if aprobada else 1


if __name__=='__main__':raise SystemExit(main())
