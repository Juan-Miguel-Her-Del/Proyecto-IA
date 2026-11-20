"""Comprueba cajas individuales y clasificación auxiliar en ejemplos train fijados."""
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
from ecowaste.auxiliar_clasificacion import DatosClasificacionAuxiliar
from ecowaste.detector_consultas import DetectorConsultas,perdida_consultas,postprocesar_consultas,xyxy_a_cxcywh,VERSION_CONSULTAS,VERSION_PONDERADA,pesos_materiales
from ecowaste.datasets import CLASES
from ecowaste.evaluacion_detector import evaluar_detecciones
from scripts.entrenar_clasificador import escribir_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nombre',default='diagnostico_consultas_v2')
    p.add_argument('--centros-atencion',action='store_true')
    p.add_argument('--objetivo',choices=['hardneg','densidad_ponderada'],default='hardneg')
    p.add_argument('--posicion-por-capa',action='store_true')
    args=p.parse_args()
    if Path(args.nombre).name!=args.nombre:raise ValueError('Nombre inválido')
    salida=RAIZ/f'resultados/{args.nombre}.json'
    if salida.exists():raise FileExistsError('Diagnóstico existente')
    torch.set_num_threads(4);torch.manual_seed(42)
    ds=ResiduosDataset(RAIZ,'deteccion','train')
    origen=json.loads((RAIZ/'resultados/diagnostico_auxiliar_clasificacion.json').read_text(encoding='utf-8'))
    indices=[next(i for i,r in enumerate(ds.registros) if r['id']==ident) for ident in origen['ids_taco_train']]
    lote=[ds[i] for i in indices];objetivos=[o for _,o in lote]
    x=torch.stack([x for x,_ in lote]).cuda()
    aux=DatosClasificacionAuxiliar(RAIZ)
    ia=[aux.ids.index(ident) for ident in origen['ids_realwaste_train']]
    xa=torch.stack([aux[i][0] for i in ia]).cuda().float()/255;ya=aux.etiquetas[ia].cuda()
    m=DetectorConsultas(centros_atencion=args.centros_atencion,posicion_por_capa=args.posicion_por_capa).cuda()
    pesos=None
    if args.objetivo=='densidad_ponderada':
        labels=torch.tensor([CLASES.index(a['clase']) for r in ds.registros for a in r['cajas'] if a['clase']])
        pesos=pesos_materiales(torch.bincount(labels,minlength=6)).cuda()
    version=VERSION_PONDERADA if args.objetivo=='densidad_ponderada' else VERSION_CONSULTAS
    # Para este diagnóstico de memorización, priors calculados solo en su lote train.
    m.configurar_priores(xyxy_a_cxcywh(torch.cat([o['boxes'] for o in objetivos]))[:,2:].median(0).values)
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0)
    historial=[];aprobada=False;inicio=time.perf_counter()
    for paso in range(1,401):
        m.train();opt.zero_grad(set_to_none=True)
        loss,detalle=perdida_consultas(m(x),objetivos,args.objetivo,pesos)
        loss_aux=F.cross_entropy(m.clasificar(xa),ya)
        (loss+.5*loss_aux).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
        if paso%25==0:
            m.eval()
            with torch.no_grad():
                pred=postprocesar_consultas(m(x),[o['geometria'] for o in objetivos],umbral=.35,descartar_nula=args.objetivo!='densidad_ponderada')
                acc=float((m.clasificar(xa).argmax(1)==ya).float().mean())
            met=evaluar_detecciones(pred,objetivos)
            historial.append({'paso':paso,'loss':loss.item(),'accuracy_aux':acc,**{k:met[k] for k in ['map50','recall50','precision50']},**detalle})
            print(f"Paso {paso}: AP50={met['map50']:.3f}, recall={met['recall50']:.3f}, precision={met['precision50']:.3f}, aux={acc:.3f}",flush=True)
            if met['map50']>=.8 and met['recall50']>=.9 and met['precision50']>=.7 and acc==1:aprobada=True;break
    escribir_json(salida,{'aprobada':aprobada,'tipo':'consultas_espaciales_asignacion_univoca','version':version,'objetivo':args.objetivo,
        'arquitectura':m.configuracion_consultas,'parametros':sum(p.numel() for p in m.parameters()),
        'sha256_train':hashlib.sha256((RAIZ/'datos/procesados/deteccion/train.jsonl').read_bytes()).hexdigest(),
        'sha256_clasificacion_train':aux.metadata['sha256_manifiesto'],'ids_taco_train':origen['ids_taco_train'],
        'historial':historial,'duracion_segundos':time.perf_counter()-inicio,'pesos_reutilizados':False,
        'test_evaluado':False,'limite':'Memorización de ejemplos fáciles train; no mide generalización. Pesos descartados.'})
    return 0 if aprobada else 1


if __name__=='__main__':raise SystemExit(main())
