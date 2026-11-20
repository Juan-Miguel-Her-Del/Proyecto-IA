"""Entrena desde cero el Transformer de mapas derivados de cajas o polígonos."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from ecowaste.datasets import CLASES
from ecowaste.ejecucion_detector import CacheDeteccion
from ecowaste.detector_mapas import DetectorMapas, rasterizar, perdida_mapas, cajas_desde_mapas
from ecowaste.evaluacion_detector import evaluar_detecciones
from scripts.entrenar_clasificador import guardar_checkpoint, escribir_json
from ecowaste.poligonos import ContornosTACO, POLITICA
from ecowaste.recortes import CacheRecortes,MezclaRecortes
from ecowaste.auxiliar_clasificacion import DatosClasificacionAuxiliar
from ecowaste.continuacion import validar_continuacion


class DatosMapas(Dataset):
    def __init__(self,split,contornos=None):
        self.base=CacheDeteccion(RAIZ,split,.1)
        self.mascaras=torch.stack([contornos.rasterizar(r) if contornos else rasterizar(o)
            for r,o in zip(self.base.origen.registros,self.base.objetivos)])
    def __len__(self):return len(self.base)
    def __getitem__(self,i):return self.base.imagenes[i],self.mascaras[i],i


@torch.inference_mode()
def evaluar(modelo,loader,dispositivo,amp,cfg):
    modelo.eval()
    pred,objetivos=[],[]
    for x,_,indices in loader:
        x=x.to(dispositivo,non_blocking=True).float()/255
        lote=[loader.dataset.base.objetivos[i] for i in indices.tolist()]
        with torch.autocast(device_type=dispositivo,dtype=torch.float16,enabled=amp):raw=modelo(x)
        pred.extend(cajas_desde_mapas(raw,cfg['umbral_evaluacion'],cfg['area_minima_celdas'],[o['geometria'] for o in lote],
            formulacion=modelo.configuracion_mapas.get('formulacion','softmax7')))
        objetivos.extend(lote)
    return evaluar_detecciones(pred,objetivos),pred


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=RAIZ/'configuracion/entrenamiento_detector_mapas.json')
    modos=parser.add_mutually_exclusive_group()
    modos.add_argument('--reanudar',type=Path)
    modos.add_argument('--continuar-desde',type=Path)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if Path(cfg['nombre']).name!=cfg['nombre'] or cfg['evaluar_test']:raise ValueError('Nombre no válido o test solicitado')
    salida=RAIZ/f"resultados/{cfg['nombre']}.json"
    mejor_path=RAIZ/f"checkpoints/{cfg['nombre']}_mejor.pt"
    ultimo_path=RAIZ/f"checkpoints/{cfg['nombre']}_ultimo.pt"
    if not args.reanudar and any(p.exists() for p in [salida,mejor_path,ultimo_path]):raise FileExistsError('Experimento existente')
    fuente=cfg.get('fuente_supervision','cajas')
    if fuente not in ['cajas','poligonos']:raise ValueError('Fuente desconocida')
    contornos=ContornosTACO(RAIZ,cfg.get('cobertura_minima',.25)) if fuente=='poligonos' else None
    diag=json.loads((RAIZ/cfg.get('diagnostico','resultados/diagnostico_detector_mapas.json')).read_text(encoding='utf-8'))
    valid=json.loads((RAIZ/'datos/reportes/validacion.json').read_text(encoding='utf-8'))
    if not diag['aprobada'] or not valid['validacion_estructural_correcta']:raise ValueError('Se requieren diagnóstico y datos válidos')
    hashes={}
    for s in ['train','val']:
        ruta=f'datos/procesados/deteccion/{s}.jsonl'
        hashes[s]=hashlib.sha256((RAIZ/ruta).read_bytes()).hexdigest()
        if hashes[s]!=valid['sha256_manifiestos'][ruta]:raise ValueError('Los datos cambiaron')
    if contornos:
        if (diag.get('tipo')!=contornos.politica or diag.get('postprocesamiento')!=cfg.get('postprocesamiento') or diag.get('area_minima_celdas')!=cfg['area_minima_celdas'] or diag.get('sha256_anotaciones')!=contornos.sha256
                or diag.get('sha256_train')!=hashes['train']):raise ValueError('Diagnóstico de polígonos incompatible')
        hashes['anotaciones_poligonos']=contornos.sha256
        hashes['inventario_procedencia']=contornos.sha256_inventario
    formulacion=cfg.get('formulacion','softmax7')
    if formulacion=='presencia_y_material':
        if contornos is None:raise ValueError('La nueva formulación requiere diagnóstico con polígonos')
        df=json.loads((RAIZ/cfg['diagnostico_formulacion']).read_text(encoding='utf-8'))
        if not df['aprobada'] or df['formulacion']!=formulacion or df['sha256_train']!=hashes['train'] or df['sha256_anotaciones']!=contornos.sha256:
            raise ValueError('Diagnóstico de formulación incompatible')
    if cfg.get('recortes_train') and cfg.get('cobertura_minima')!=1/64:
        raise ValueError('La caché de recortes requiere cobertura de cualquier píxel')
    torch.set_num_threads(4);torch.manual_seed(cfg['semilla'])
    dispositivo='cuda' if torch.cuda.is_available() else 'cpu'
    es_cuda=dispositivo=='cuda'
    amp=es_cuda and cfg['precision_mixta_cuda']
    train,val=DatosMapas('train',contornos),DatosMapas('val',contornos)
    rng=torch.Generator().manual_seed(cfg['semilla'])
    dlval=DataLoader(val,batch_size=cfg['batch_size'],pin_memory=es_cuda)
    conteos=torch.bincount(train.mascaras[train.mascaras!=255].flatten(),minlength=7)[1:].float()
    frecuencias=conteos.clone();detalle_recortes=None
    if cfg.get('recortes_train'):
        if contornos is None:raise ValueError('Los recortes requieren polígonos originales')
        recortes=CacheRecortes(RAIZ,cfg['recortes_train'],hashes)
        if recortes.metadata.get('cobertura_minima')!=contornos.cobertura_minima:
            raise ValueError('Políticas de máscara distintas')
        metadata_path=RAIZ/cfg['recortes_train']/'metadata.json'
        hashes['recortes_metadata']=hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        diagnostico= json.loads((RAIZ/cfg['diagnostico_recortes']).read_text(encoding='utf-8'))
        if not diagnostico['aprobada'] or diagnostico['sha256_metadata_recortes']!=hashes['recortes_metadata']:
            raise ValueError('Diagnóstico de recortes incompatible')
        probabilidad=cfg['probabilidad_recorte']
        balancear=cfg.get('balancear_anclas',False)
        mezcla=MezclaRecortes(train,recortes,probabilidad,balancear)
        sampler=WeightedRandomSampler(mezcla.pesos_muestreo,len(train),replacement=True,generator=rng)
        dltrain=DataLoader(mezcla,batch_size=cfg['batch_size'],sampler=sampler,generator=rng,pin_memory=es_cuda)
        conteos_recortes=torch.tensor(recortes.metadata['celdas_fondo_y_clases'][1:],dtype=torch.float32)
        ids_clase_recorte=torch.tensor([CLASES.index(r['clase_ancla']) for r in recortes.plan])
        frecuencia_recortes=conteos_recortes/len(recortes)
        if balancear:
            import numpy as np
            por_ancla=torch.zeros(6,7,dtype=torch.float64)
            for i in range(len(recortes)):
                y=recortes.mascaras[i]
                por_ancla[ids_clase_recorte[i]]+=torch.from_numpy(np.bincount(y[y!=255],minlength=7))
            numero_anclas=torch.bincount(ids_clase_recorte,minlength=6)
            frecuencia_recortes=(por_ancla[:,1:]/numero_anclas[:,None]).mean(0).float()
        frecuencias=(1-probabilidad)*conteos/len(train)+probabilidad*frecuencia_recortes
        detalle_recortes={'version':recortes.metadata['version'],'recortes':len(recortes),
            'probabilidad_recorte':probabilidad,'muestras_por_epoca':len(train),'muestreo_con_reemplazo':True,
            'celdas_por_clase_recortes':conteos_recortes.long().tolist(),
            'frecuencia_esperada_celdas_por_muestra':frecuencias.tolist(),
            'balanceo_de_anclas_por_clase':balancear}
    else:
        dltrain=DataLoader(train,batch_size=cfg['batch_size'],shuffle=True,generator=rng,pin_memory=es_cuda)
    pesos=frecuencias.clamp(min=1e-6).rsqrt();pesos=(pesos/pesos.mean()).clamp(max=3).to(dispositivo)
    aux_cfg=cfg.get('clasificacion_auxiliar');detalle_aux=None;rng_aux=None
    if aux_cfg:
        if contornos is None:raise ValueError('La tarea auxiliar requiere diagnóstico conjunto con polígonos')
        aux=DatosClasificacionAuxiliar(RAIZ)
        hashes['clasificacion_auxiliar_train']=aux.metadata['sha256_manifiesto']
        hashes['cache_clasificacion_auxiliar']=hashlib.sha256(aux.metadata_path.read_bytes()).hexdigest()
        da=json.loads((RAIZ/aux_cfg['diagnostico']).read_text(encoding='utf-8'))
        if (not da['aprobada'] or da['sha256_train']!=hashes['train']
                or da['sha256_clasificacion_train']!=hashes['clasificacion_auxiliar_train']
                or da.get('ventanas_desplazadas',False)!=cfg.get('ventanas_desplazadas',False)
                or da['sha256_anotaciones']!=contornos.sha256):raise ValueError('Diagnóstico auxiliar incompatible')
        rng_aux=torch.Generator().manual_seed(cfg['semilla']+1)
        cuenta=torch.bincount(aux.etiquetas,minlength=6)
        sampler_aux=WeightedRandomSampler(1/cuenta[aux.etiquetas].double(),len(dltrain)*aux_cfg['batch_size'],replacement=True,generator=rng_aux)
        dlaux=DataLoader(aux,batch_size=aux_cfg['batch_size'],sampler=sampler_aux,generator=rng_aux,pin_memory=es_cuda)
        detalle_aux={'imagenes_train':len(aux),'conteos_clase_train':cuenta.tolist(),'balanceo_por_clase':True,
            'coeficiente_perdida':aux_cfg['coeficiente_perdida'],'muestras_por_epoca':len(dltrain)*aux_cfg['batch_size'],
            'cajas_o_mascaras_inventadas':False}
    modelo=DetectorMapas(formulacion=formulacion,auxiliar_clasificacion=bool(aux_cfg),ventanas_desplazadas=cfg.get('ventanas_desplazadas',False)).to(dispositivo)
    opt=torch.optim.AdamW(modelo.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=1024)
    historial=[];mejor=-1;mejor_epoca=0;paciencia=0;inicio_epoca=1;tiempo_previo=0
    origen_continuacion=None
    fuente_estado=args.reanudar or args.continuar_desde
    if fuente_estado:
        previo=torch.load(fuente_estado,map_location=dispositivo,weights_only=True)
        if args.continuar_desde:
            validar_continuacion(previo,cfg,hashes)
            mejor_origen=fuente_estado.parent/f"{previo['config_entrenamiento']['nombre']}_mejor.pt"
            mejor_anterior=torch.load(mejor_origen,map_location='cpu',weights_only=True)
            if (mejor_anterior['hashes_manifiestos']!=hashes or mejor_anterior['epoca']!=previo['mejor_epoca']
                    or mejor_anterior['mejor_map']!=previo['mejor_map']):raise ValueError('Mejor checkpoint de origen incompatible')
            origen_continuacion={'checkpoint':str(fuente_estado),'sha256':hashlib.sha256(fuente_estado.read_bytes()).hexdigest(),
                'mejor_checkpoint':str(mejor_origen),'sha256_mejor':hashlib.sha256(mejor_origen.read_bytes()).hexdigest(),
                'experimento':previo['config_entrenamiento']['nombre'],'epoca_inicio':previo['epoca']+1,
                'cadena_anterior':previo.get('origen_continuacion'),
                'nota':'Mismo aprendizaje propio desde cero; conserva optimizador, scaler y estados aleatorios de ambas tareas.'}
            mejor_anterior['config_entrenamiento']=cfg;mejor_anterior['origen_continuacion']=origen_continuacion
            guardar_checkpoint(mejor_path,mejor_anterior)
        elif previo['config_entrenamiento']!=cfg or previo['hashes_manifiestos']!=hashes:raise ValueError('Configuración o datos distintos')
        else:origen_continuacion=previo.get('origen_continuacion')
        modelo.load_state_dict(previo['pesos']);opt.load_state_dict(previo['optimizador']);scaler.load_state_dict(previo['scaler'])
        torch.set_rng_state(previo['rng_torch'].cpu());rng.set_state(previo['rng_loader'].cpu())
        if rng_aux is not None:rng_aux.set_state(previo['rng_auxiliar'].cpu())
        if es_cuda:torch.cuda.set_rng_state_all([t.cpu() for t in previo['rng_cuda']])
        historial=previo['historial'];mejor=previo['mejor_map'];mejor_epoca=previo['mejor_epoca'];paciencia=previo['paciencia']
        inicio_epoca=previo['epoca']+1;tiempo_previo=previo['duracion_segundos']
    if es_cuda:torch.cuda.reset_peak_memory_stats()
    inicio=time.perf_counter()
    for epoca in range(inicio_epoca,cfg['epocas']+1):
        modelo.train();total=0;n=0;muestras_completas=0;muestras_recorte=0;anclas_por_clase=torch.zeros(6,dtype=torch.long)
        total_aux=0;correctos_aux=0;n_aux=0
        if aux_cfg:it_aux=iter(dlaux)
        for g in opt.param_groups:g['lr']=cfg['learning_rate']*min(epoca/cfg['warmup_epocas'],1)
        for x,y,indices in dltrain:
            cantidad_recortes=int((indices>=len(train)).sum()) if detalle_recortes else 0
            if cantidad_recortes:
                anclas_por_clase+=torch.bincount(ids_clase_recorte[indices[indices>=len(train)]-len(train)],minlength=6)
            muestras_recorte+=cantidad_recortes;muestras_completas+=len(x)-cantidad_recortes
            x=x.to(dispositivo,non_blocking=True).float()/255;y=y.to(dispositivo,non_blocking=True)
            flip=torch.rand(len(x),1,1,1,device=dispositivo)<.5
            x=torch.where(flip,x.flip(-1),x);y=torch.where(flip[:,0],y.flip(-1),y)
            brillo=torch.empty(len(x),1,1,1,device=dispositivo).uniform_(.85,1.15)
            contraste=torch.empty_like(brillo).uniform_(.85,1.15)
            x=((x-.5)*contraste+.5).mul(brillo).clamp(0,1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dispositivo,dtype=torch.float16,enabled=amp):raw=modelo(x)
            loss=perdida_mapas(raw,y,pesos,formulacion)
            if not bool(torch.isfinite(loss)):raise RuntimeError('Pérdida no finita')
            scaler.scale(loss).backward()
            if aux_cfg:
                xa,ya=next(it_aux);xa=xa.to(dispositivo,non_blocking=True).float()/255;ya=ya.to(dispositivo,non_blocking=True)
                fa=torch.rand(len(xa),1,1,1,device=dispositivo)<.5
                xa=torch.where(fa,xa.flip(-1),xa)
                ba=torch.empty(len(xa),1,1,1,device=dispositivo).uniform_(.85,1.15)
                ca=torch.empty_like(ba).uniform_(.85,1.15)
                xa=((xa-.5)*ca+.5).mul(ba).clamp(0,1)
                with torch.autocast(device_type=dispositivo,dtype=torch.float16,enabled=amp):raw_aux=modelo.clasificar(xa)
                loss_aux=F.cross_entropy(raw_aux.float(),ya)
                if not bool(torch.isfinite(loss_aux)):raise RuntimeError('Pérdida auxiliar no finita')
                scaler.scale(aux_cfg['coeficiente_perdida']*loss_aux).backward()
                total_aux+=loss_aux.item()*len(xa);correctos_aux+=int((raw_aux.argmax(1)==ya).sum());n_aux+=len(xa)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(),1,error_if_nonfinite=True)
            scaler.step(opt);scaler.update();total+=loss.item()*len(x);n+=len(x)
        metrica,_=evaluar(modelo,dlval,dispositivo,amp,cfg)
        historial.append({'epoca':epoca,'loss_train':total/n,'map50_val':metrica['map50'],'map50_95_val':metrica['map50_95'],'recall50_val':metrica['recall50'],
            'muestras_completas_train':muestras_completas,'muestras_recorte_train':muestras_recorte,'pasos_optimizacion':len(dltrain),
            'anclas_recortes_por_clase':anclas_por_clase.tolist()})
        if aux_cfg:historial[-1].update(loss_aux_train=total_aux/n_aux,accuracy_aux_train=correctos_aux/n_aux,muestras_aux_train=n_aux,
            loss_total_train=total/n+aux_cfg['coeficiente_perdida']*total_aux/n_aux)
        mejora=metrica['map50']>mejor+1e-8
        if mejora:mejor=metrica['map50'];mejor_epoca=epoca;paciencia=0
        else:paciencia+=1
        checkpoint={'tipo':'ecowaste_mapas_desde_cero_v1','configuracion':modelo.configuracion_mapas,'clases':CLASES,
            'pesos':modelo.state_dict(),'optimizador':opt.state_dict(),'scaler':scaler.state_dict(),'config_entrenamiento':cfg,
            'epoca':epoca,'mejor_map':mejor,'mejor_epoca':mejor_epoca,'paciencia':paciencia,'historial':historial,
            'hashes_manifiestos':hashes,'rng_torch':torch.get_rng_state(),'rng_loader':rng.get_state(),
            'recortes_train':detalle_recortes,
            'clasificacion_auxiliar':detalle_aux,'rng_auxiliar':rng_aux.get_state() if rng_aux is not None else None,
            'origen_continuacion':origen_continuacion,
            'rng_cuda':torch.cuda.get_rng_state_all() if es_cuda else [],'pesos_externos':False,
            'duracion_segundos':tiempo_previo+time.perf_counter()-inicio}
        guardar_checkpoint(ultimo_path,checkpoint)
        if mejora:guardar_checkpoint(mejor_path,checkpoint)
        escribir_json(salida,{'estado':'en_curso','configuracion':cfg,'historial':historial,'test_evaluado':False})
        print(f"Época {epoca}/{cfg['epocas']}: loss={total/n:.3f}, AP50 val={metrica['map50']:.3f}, AP50:95={metrica['map50_95']:.3f}, recall={metrica['recall50']:.3f}",flush=True)
        if epoca>=cfg['min_epocas'] and paciencia>=cfg['paciencia']:break
    elegido=torch.load(mejor_path,map_location=dispositivo,weights_only=True);modelo.load_state_dict(elegido['pesos'])
    metrica,pred=evaluar(modelo,dlval,dispositivo,amp,cfg)
    escribir_json(RAIZ/f"resultados/{cfg['nombre']}_predicciones_val.json",[{'id':o['id'],**{k:v.tolist() for k,v in p.items()}} for o,p in zip(val.base.objetivos,pred)])
    reporte={'estado':'piloto_completado','fecha_utc':datetime.now(timezone.utc).isoformat(),'modelo_tipo':contornos.politica if contornos else 'mapas_de_cobertura_de_cajas',
        'arquitectura':modelo.configuracion_mapas,'parametros':sum(p.numel() for p in modelo.parameters()),'configuracion':cfg,
        'mejor_epoca':mejor_epoca,'epocas_ejecutadas':len(historial),'historial':historial,'val':metrica,
        'hashes_manifiestos':hashes,'celdas_por_clase_train':conteos.long().tolist(),'pesos_clase_train':pesos.tolist(),
        'recortes_train':detalle_recortes,
        'clasificacion_auxiliar':detalle_aux,
        'origen_continuacion':origen_continuacion,
        'gpu':torch.cuda.get_device_name(0) if es_cuda else 'cpu','torch':str(torch.__version__),'pesos_externos':False,
        'duracion_segundos':tiempo_previo+time.perf_counter()-inicio,'vram_pico_asignada_bytes':torch.cuda.max_memory_allocated() if es_cuda else None,
        'checkpoint_mejor':mejor_path.relative_to(RAIZ).as_posix(),'checkpoint_ultimo':ultimo_path.relative_to(RAIZ).as_posix(),'test_evaluado':False,
        'limites':[('Contornos originales cuantizados a 48x48; no equivale a segmentación de instancias a resolución original.' if contornos else 'Cobertura rectangular débil; no equivale a segmentación real.'),'Componentes de igual material pueden fusionarse y objetos pequeños perderse por cuantización.','Solo cinco orgánicos train y uno val.']}
    escribir_json(salida,reporte)
    print(json.dumps({'map50':metrica['map50'],'map50_95':metrica['map50_95'],'mejor_epoca':mejor_epoca,'test_evaluado':False}),flush=True)


if __name__=='__main__':main()
