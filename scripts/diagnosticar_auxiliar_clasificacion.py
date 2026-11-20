"""Diagnóstico conjunto en escenas TACO y fotos RealWaste de train, pesos descartados."""
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
from ecowaste.datasets import ResiduosDataset
from ecowaste.poligonos import ContornosTACO
from ecowaste.auxiliar_clasificacion import DatosClasificacionAuxiliar
from ecowaste.detector_mapas import DetectorMapas,perdida_mapas,cajas_desde_mapas
from ecowaste.evaluacion_detector import evaluar_detecciones
from scripts.entrenar_clasificador import escribir_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ventanas-desplazadas',action='store_true')
    parser.add_argument('--nombre',default='diagnostico_auxiliar_clasificacion')
    args=parser.parse_args()
    if Path(args.nombre).name!=args.nombre:raise ValueError('Nombre inválido')
    salida=RAIZ/f'resultados/{args.nombre}.json'
    if salida.exists():raise FileExistsError('Diagnóstico existente')
    torch.set_num_threads(4);torch.manual_seed(42)
    aux=DatosClasificacionAuxiliar(RAIZ)
    ia=[int(torch.where(aux.etiquetas==c)[0][0]) for c in range(6)]
    xa=torch.stack([aux[i][0] for i in ia]).cuda().float()/255;ya=aux.etiquetas[ia].cuda()
    origen=json.loads((RAIZ/'resultados/diagnostico_detector_mapas_poligonos_padding.json').read_text(encoding='utf-8'))
    ds=ResiduosDataset(RAIZ,'deteccion','train');contornos=ContornosTACO(RAIZ,1/64)
    indices=[next(i for i,r in enumerate(ds.registros) if r['id']==ident) for ident in origen['ids_train']]
    lote=[ds[i] for i in indices];objetivos=[o for _,o in lote]
    x=torch.stack([x for x,_ in lote]).cuda();y=torch.stack([contornos.rasterizar(ds.registros[i]) for i in indices]).cuda()
    m=DetectorMapas(auxiliar_clasificacion=True,ventanas_desplazadas=args.ventanas_desplazadas).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0)
    inicio=time.perf_counter();historial=[];aprobada=False
    for paso in range(1,401):
        m.train();opt.zero_grad(set_to_none=True)
        loss_det=perdida_mapas(m(x),y);loss_aux=F.cross_entropy(m.clasificar(xa),ya)
        loss=loss_det+.5*loss_aux;loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
        if paso%25==0:
            m.eval()
            with torch.no_grad():
                pred=cajas_desde_mapas(m(x),.35,2,[o['geometria'] for o in objetivos])
                acc=float((m.clasificar(xa).argmax(1)==ya).float().mean())
            met=evaluar_detecciones(pred,objetivos)
            historial.append({'paso':paso,'loss':loss.item(),'accuracy_aux':acc,**{k:met[k] for k in ['map50','recall50','precision50']}})
            print(f"Paso {paso}: AP50={met['map50']:.3f}, recall={met['recall50']:.3f}, precision={met['precision50']:.3f}, clasif={acc:.3f}",flush=True)
            if met['map50']>=.8 and met['recall50']>=.9 and met['precision50']>=.7 and acc==1:aprobada=True;break
    escribir_json(salida,{'aprobada':aprobada,'ventanas_desplazadas':args.ventanas_desplazadas,'ids_taco_train':origen['ids_train'],'ids_realwaste_train':[aux.ids[i] for i in ia],
        'sha256_train':hashlib.sha256((RAIZ/'datos/procesados/deteccion/train.jsonl').read_bytes()).hexdigest(),
        'sha256_clasificacion_train':aux.metadata['sha256_manifiesto'],'sha256_anotaciones':contornos.sha256,
        'historial':historial,'parametros':sum(p.numel() for p in m.parameters()),'duracion_segundos':time.perf_counter()-inicio,
        'pesos_reutilizados':False,'test_evaluado':False,'limite':'Memorización de ejemplos fáciles de train, no generalización.'})
    return 0 if aprobada else 1


if __name__=='__main__':raise SystemExit(main())
