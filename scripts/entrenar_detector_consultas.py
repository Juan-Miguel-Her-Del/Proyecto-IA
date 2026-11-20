"""Entrena el detector propio de consultas con cajas TACO y clasificación auxiliar."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler
from ecowaste.datasets import CLASES,collate_deteccion
from ecowaste.ejecucion_detector import CacheDeteccion,aumentar_lote
from ecowaste.auxiliar_clasificacion import DatosClasificacionAuxiliar
from ecowaste.detector_consultas import DetectorConsultas,perdida_consultas,postprocesar_consultas,xyxy_a_cxcywh,VERSION_CONSULTAS,VERSION_PONDERADA,pesos_materiales
from ecowaste.evaluacion_detector import evaluar_detecciones
from scripts.entrenar_clasificador import escribir_json,guardar_checkpoint


@torch.inference_mode()
def evaluar(m,dl,device,cfg):
    m.eval();pred=[];objetivos=[]
    for x,lote in dl:
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):raw=m(x.to(device).float()/255)
        pred.extend(postprocesar_consultas(raw,[o['geometria'] for o in lote],cfg.get('umbral_evaluacion',.05),descartar_nula=cfg.get('descartar_nula',True)));objetivos.extend(lote)
    return evaluar_detecciones(pred,objetivos),pred


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=RAIZ/'configuracion/entrenamiento_detector_consultas.json')
    parser.add_argument('--reanudar',type=Path)
    a=parser.parse_args();cfg=json.loads(a.config.read_text(encoding='utf-8'))
    if Path(cfg['nombre']).name!=cfg['nombre'] or cfg['evaluar_test']:raise ValueError('Nombre o partición inválidos')
    reporte_path=RAIZ/f"resultados/{cfg['nombre']}.json";mejor_path=RAIZ/f"checkpoints/{cfg['nombre']}_mejor.pt";ultimo_path=RAIZ/f"checkpoints/{cfg['nombre']}_ultimo.pt"
    if not a.reanudar and any(p.exists() for p in [reporte_path,mejor_path,ultimo_path]):raise FileExistsError('Experimento existente')
    valid=json.loads((RAIZ/'datos/reportes/validacion.json').read_text(encoding='utf-8'));hashes={}
    if not valid['validacion_estructural_correcta']:raise ValueError('Datos sin validación estructural')
    for split in ['train','val']:
        p=f'datos/procesados/deteccion/{split}.jsonl';hashes[split]=hashlib.sha256((RAIZ/p).read_bytes()).hexdigest()
        if hashes[split]!=valid['sha256_manifiestos'][p]:raise ValueError('Datos alterados')
    torch.set_num_threads(4);torch.manual_seed(cfg['semilla']);device='cuda' if torch.cuda.is_available() else 'cpu';cuda=device=='cuda'
    train=CacheDeteccion(RAIZ,'train',.1);val=CacheDeteccion(RAIZ,'val',.1);aux=DatosClasificacionAuxiliar(RAIZ)
    hashes['clasificacion_auxiliar_train']=aux.metadata['sha256_manifiesto']
    hashes['cache_clasificacion_auxiliar']=hashlib.sha256(aux.metadata_path.read_bytes()).hexdigest()
    diag=json.loads((RAIZ/cfg['diagnostico']).read_text(encoding='utf-8'))
    objetivo=cfg.get('objetivo','hardneg');version=VERSION_PONDERADA if objetivo=='densidad_ponderada' else VERSION_CONSULTAS
    if (not diag['aprobada'] or diag['version']!=version or diag['sha256_train']!=hashes['train']
            or diag['sha256_clasificacion_train']!=hashes['clasificacion_auxiliar_train']):raise ValueError('Diagnóstico incompatible')
    rng=torch.Generator().manual_seed(cfg['semilla']);rng_aux=torch.Generator().manual_seed(cfg['semilla']+1)
    dl=DataLoader(train,batch_size=cfg['batch_size'],shuffle=True,generator=rng,collate_fn=collate_deteccion,pin_memory=cuda)
    dlval=DataLoader(val,batch_size=cfg['batch_size'],collate_fn=collate_deteccion,pin_memory=cuda)
    cuenta=torch.bincount(aux.etiquetas,minlength=6)
    sampler=WeightedRandomSampler(1/cuenta[aux.etiquetas].double(),len(dl)*16,replacement=True,generator=rng_aux)
    dla=DataLoader(aux,batch_size=16,sampler=sampler,generator=rng_aux,pin_memory=cuda)
    m=DetectorConsultas(**cfg['arquitectura']).to(device)
    arquitectura_diag=dict(diag['arquitectura']);arquitectura_diag.setdefault('centros_atencion',False)
    arquitectura_diag.setdefault('posicion_por_capa',False)
    if m.configuracion_consultas!=arquitectura_diag:raise ValueError('Arquitectura distinta del diagnóstico')
    if max(len(o['boxes']) for o in train.objetivos)>cfg['arquitectura']['consultas']:raise ValueError('Consultas insuficientes')
    priors=xyxy_a_cxcywh(torch.cat([o['boxes'] for o in train.objetivos]))[:,2:].median(0).values
    m.configurar_priores(priors)
    pesos=None
    conteos=torch.bincount(torch.cat([o['labels'] for o in train.objetivos]),minlength=6)
    if objetivo=='densidad_ponderada':pesos=pesos_materiales(conteos).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=cuda,init_scale=1024)
    inicio_epoca=1;historial=[];mejor=-1;epoca_mejor=0;paciencia=0;duracion_previa=0
    if a.reanudar:
        ck=torch.load(a.reanudar,map_location=device,weights_only=True)
        if ck['config_entrenamiento']!=cfg or ck['hashes_manifiestos']!=hashes or ck['version']!=version:raise ValueError('Estado de reanudación incompatible')
        m.load_state_dict(ck['pesos']);opt.load_state_dict(ck['optimizador']);scaler.load_state_dict(ck['scaler'])
        torch.set_rng_state(ck['rng_torch'].cpu());rng.set_state(ck['rng_loader'].cpu());rng_aux.set_state(ck['rng_auxiliar'].cpu())
        if cuda:torch.cuda.set_rng_state_all([t.cpu() for t in ck['rng_cuda']])
        inicio_epoca=ck['epoca']+1;historial=ck['historial'];mejor=ck['mejor_map'];epoca_mejor=ck['mejor_epoca'];paciencia=ck['paciencia'];duracion_previa=ck['duracion_segundos']
    if cuda:torch.cuda.reset_peak_memory_stats()
    inicio=time.perf_counter()
    for epoca in range(inicio_epoca,cfg['epocas']+1):
        m.train();total=n=total_aux=correctos_aux=na=0;it_aux=iter(dla);asignados=0
        for g in opt.param_groups:g['lr']=cfg['learning_rate']*min(epoca/2,1)
        for x,objetivos in dl:
            x=x.to(device).float()/255
            x,objetivos=aumentar_lote(x,objetivos,{'flip_horizontal':.5,'brillo':[.85,1.15],'contraste':[.85,1.15]},.1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=cuda):raw=m(x)
            loss,detalle=perdida_consultas(raw,objetivos,objetivo,pesos)
            if not bool(torch.isfinite(loss)):raise RuntimeError('Pérdida detector no finita')
            scaler.scale(loss).backward()
            xa,ya=next(it_aux);xa=xa.to(device).float()/255;ya=ya.to(device)
            flip=torch.rand(len(xa),1,1,1,device=device)<.5;xa=torch.where(flip,xa.flip(-1),xa)
            brillo=torch.empty(len(xa),1,1,1,device=device).uniform_(.85,1.15);contraste=torch.empty_like(brillo).uniform_(.85,1.15)
            xa=((xa-.5)*contraste+.5).mul(brillo).clamp(0,1)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=cuda):la=m.clasificar(xa)
            loss_aux=F.cross_entropy(la.float(),ya)
            if not bool(torch.isfinite(loss_aux)):raise RuntimeError('Pérdida auxiliar no finita')
            scaler.scale(.5*loss_aux).backward();scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);scaler.step(opt);scaler.update()
            total+=loss.item()*len(x);n+=len(x);total_aux+=loss_aux.item()*len(xa);na+=len(xa);correctos_aux+=int((la.argmax(1)==ya).sum());asignados+=detalle['objetos_asignados']
        metricas,_=evaluar(m,dlval,device,cfg)
        historial.append({'epoca':epoca,'loss_train':total/n,'loss_aux_train':total_aux/na,'loss_total_train':total/n+.5*total_aux/na,
            'accuracy_aux_train':correctos_aux/na,'map50_val':metricas['map50'],'map50_95_val':metricas['map50_95'],'recall50_val':metricas['recall50'],
            'objetos_asignados_train':asignados,'muestras_train':n,'muestras_aux_train':na})
        mejora=metricas['map50']>mejor+1e-8
        if mejora:mejor=metricas['map50'];epoca_mejor=epoca;paciencia=0
        else:paciencia+=1
        ck={'tipo':'ecowaste_consultas_desde_cero_v1','version':version,'configuracion':m.configuracion_consultas,
            'clases':CLASES,'pesos':m.state_dict(),'optimizador':opt.state_dict(),'scaler':scaler.state_dict(),'config_entrenamiento':cfg,
            'epoca':epoca,'mejor_map':mejor,'mejor_epoca':epoca_mejor,'paciencia':paciencia,'historial':historial,
            'hashes_manifiestos':hashes,'rng_torch':torch.get_rng_state(),'rng_loader':rng.get_state(),'rng_auxiliar':rng_aux.get_state(),
            'rng_cuda':torch.cuda.get_rng_state_all() if cuda else [],'pesos_externos':False,'duracion_segundos':duracion_previa+time.perf_counter()-inicio}
        guardar_checkpoint(ultimo_path,ck)
        if mejora:guardar_checkpoint(mejor_path,ck)
        escribir_json(reporte_path,{'estado':'en_curso','configuracion':cfg,'historial':historial,'test_evaluado':False})
        print(f"Época {epoca}/{cfg['epocas']}: loss={total/n:.3f}, AP50={metricas['map50']:.3f}, AP50:95={metricas['map50_95']:.3f}, recall={metricas['recall50']:.3f}",flush=True)
        if epoca>=cfg['min_epocas'] and paciencia>=cfg['paciencia']:break
    elegido=torch.load(mejor_path,map_location=device,weights_only=True);m.load_state_dict(elegido['pesos'])
    metricas,pred=evaluar(m,dlval,device,cfg)
    escribir_json(RAIZ/f"resultados/{cfg['nombre']}_predicciones_val.json",[{'id':o['id'],**{k:v.tolist() for k,v in p.items()}} for o,p in zip(val.objetivos,pred)])
    escribir_json(reporte_path,{'estado':'piloto_completado','fecha_utc':datetime.now(timezone.utc).isoformat(),'version':version,
        'arquitectura':m.configuracion_consultas,'parametros':sum(p.numel() for p in m.parameters()),'configuracion':cfg,'historial':historial,
        'mejor_epoca':epoca_mejor,'epocas_ejecutadas':len(historial),'val':metricas,'hashes_manifiestos':hashes,'priores_wh_train':priors.tolist(),
        'clasificacion_auxiliar':{'coeficiente_perdida':.5,'imagenes_train':len(aux),'conteos_clase_train':cuenta.tolist()},
        'conteos_materiales_deteccion_train':conteos.tolist(),'pesos_materiales_deteccion_train':pesos.tolist() if pesos is not None else None,
        'checkpoint_mejor':mejor_path.relative_to(RAIZ).as_posix(),'checkpoint_ultimo':ultimo_path.relative_to(RAIZ).as_posix(),
        'pesos_externos':False,'test_evaluado':False,'duracion_segundos':duracion_previa+time.perf_counter()-inicio,
        'vram_pico_asignada_bytes':torch.cuda.max_memory_allocated() if cuda else None,
        'limites':['Modelo nuevo desde cero; diagnóstico aprobado no garantiza generalización.','Cinco instancias orgánicas train y una val.','Sin NMS ni fusión semántica; consultas duplicadas cuentan como falsos positivos.']})
    print(json.dumps({'map50':metricas['map50'],'map50_95':metricas['map50_95'],'mejor_epoca':epoca_mejor}),flush=True)


if __name__=='__main__':main()
