"""Comprueba memorización de la salida separada en las seis escenas train fijadas."""
import hashlib
import json
from pathlib import Path
import sys
import time

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.datasets import ResiduosDataset
from ecowaste.poligonos import ContornosTACO
from ecowaste.detector_mapas import DetectorMapas,cajas_desde_mapas,perdida_mapas
from ecowaste.evaluacion_detector import evaluar_detecciones
from scripts.entrenar_clasificador import escribir_json


def main():
    salida=RAIZ/'resultados/diagnostico_presencia_material.json'
    if salida.exists():raise FileExistsError('Diagnóstico existente')
    torch.set_num_threads(4);torch.manual_seed(42)
    origen=json.loads((RAIZ/'resultados/diagnostico_detector_mapas_poligonos_padding.json').read_text(encoding='utf-8'))
    ds=ResiduosDataset(RAIZ,'deteccion','train');contornos=ContornosTACO(RAIZ,1/64)
    indices=[next(i for i,r in enumerate(ds.registros) if r['id']==ident) for ident in origen['ids_train']]
    lote=[ds[i] for i in indices];objetivos=[o for _,o in lote]
    x=torch.stack([x for x,_ in lote]).cuda()
    y=torch.stack([contornos.rasterizar(ds.registros[i]) for i in indices]).cuda()
    modelo=DetectorMapas(formulacion='presencia_y_material').cuda()
    opt=torch.optim.AdamW(modelo.parameters(),lr=.001,weight_decay=0)
    inicio=time.perf_counter();historial=[];aprobada=False
    for paso in range(1,401):
        modelo.train();opt.zero_grad(set_to_none=True)
        loss=perdida_mapas(modelo(x),y,formulacion='presencia_y_material')
        loss.backward();torch.nn.utils.clip_grad_norm_(modelo.parameters(),1,error_if_nonfinite=True);opt.step()
        if paso%25==0:
            modelo.eval()
            with torch.no_grad():p=cajas_desde_mapas(modelo(x),.35,2,[o['geometria'] for o in objetivos],formulacion='presencia_y_material')
            m=evaluar_detecciones(p,objetivos)
            historial.append({'paso':paso,'loss':loss.item(),**{k:m[k] for k in ['map50','recall50','precision50']}})
            print(f"Paso {paso}: loss={loss.item():.3f}, AP50={m['map50']:.3f}, recall={m['recall50']:.3f}, precision={m['precision50']:.3f}",flush=True)
            if m['map50']>=.8 and m['recall50']>=.9 and m['precision50']>=.7:aprobada=True;break
    escribir_json(salida,{'aprobada':aprobada,'formulacion':'presencia_y_material','ids_train':origen['ids_train'],
        'sha256_train':hashlib.sha256((RAIZ/'datos/procesados/deteccion/train.jsonl').read_bytes()).hexdigest(),
        'sha256_anotaciones':contornos.sha256,'historial':historial,'duracion_segundos':time.perf_counter()-inicio,
        'objetos':sum(len(o['boxes']) for o in objetivos),'pesos_reutilizados':False,'test_evaluado':False})
    return 0 if aprobada else 1


if __name__=='__main__':raise SystemExit(main())
