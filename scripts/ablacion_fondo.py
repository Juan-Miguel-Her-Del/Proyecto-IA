"""Dos continuaciones pareadas desde época 19; solo cambia el coeficiente de fondo."""
import argparse
import copy
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
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from ecowaste.datasets import CLASES,letterbox
from ecowaste.detector_mapas import DetectorMapas,perdida_mapas
from ecowaste.poligonos import ContornosTACO
from ecowaste.auxiliar_clasificacion import DatosClasificacionAuxiliar
from ecowaste.ablacion_fondo import (huella,estado_aleatorio,restaurar_aleatorio,pasos_adam,
    metricas_semanticas,resumir_deteccion,parametros_experimento)
from scripts.entrenar_detector_mapas import DatosMapas,evaluar
from scripts.entrenar_clasificador import escribir_json,guardar_checkpoint


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class AuxIndice(Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        x,y=self.base[i]
        return x,y,i


@torch.inference_mode()
def semantica(m,ds,device,amp):
    m.eval();cm=torch.zeros(49,device=device,dtype=torch.long)
    rng_eval=torch.Generator().manual_seed(0)
    for x,y,_ in DataLoader(ds,batch_size=8,generator=rng_eval):
        y=y.to(device)
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):raw=m(x.to(device).float()/255)
        pred=raw.argmax(1);ok=y!=255
        cm+=torch.bincount((7*y[ok]+pred[ok]).flatten(),minlength=49)
    return metricas_semanticas(cm.reshape(7,7).cpu())


def aumentar(x,y=None):
    flip=torch.rand(len(x),1,1,1,device=x.device)<.5
    x=torch.where(flip,x.flip(-1),x)
    if y is not None:y=torch.where(flip[:,0],y.flip(-1),y)
    brillo=torch.empty(len(x),1,1,1,device=x.device).uniform_(.85,1.15)
    contraste=torch.empty_like(brillo).uniform_(.85,1.15)
    x=((x-.5)*contraste+.5).mul(brillo).clamp(0,1)
    return x,y,{'flip':flip.detach().cpu(),'brillo':brillo.detach().cpu(),'contraste':contraste.detach().cpu()}


def guardar_pred(path,pred,objetivos):
    escribir_json(path,[{'id':o['id'],**{k:v.tolist() for k,v in p.items()}} for o,p in zip(objetivos,pred)])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=RAIZ/'configuracion/ablacion_fondo.json')
    a=p.parse_args();cfg=json.loads(a.config.read_text(encoding='utf-8'))
    if Path(cfg['nombre']).name!=cfg['nombre']:raise ValueError('Nombre inválido')
    salida=RAIZ/f"resultados/{cfg['nombre']}.json"
    nombres={r:f"{cfg['nombre']}_{r}" for r in cfg['ramas']}
    protegidos=[salida]+[RAIZ/f'checkpoints/{n}_{s}.pt' for n in nombres.values() for s in ['ultimo','mejor','mejor_incluyendo_origen']]
    if any(p.exists() for p in protegidos):raise FileExistsError('Experimento existente; usar otro nombre')
    origen=RAIZ/cfg['checkpoint_origen'];hash_origen=sha(origen)
    if hash_origen!=cfg['sha256_checkpoint_origen']:raise ValueError('Origen diferente')
    ck=torch.load(origen,map_location='cpu',weights_only=True);parametros_experimento(ck,cfg)
    if cfg['parada_temprana'] or cfg['reiniciar_warmup']:raise ValueError('Presupuesto o calentamiento alterados')
    base=ck['config_entrenamiento'];hashes=ck['hashes_manifiestos'];hash_seleccion=sha(RAIZ/'configuracion/arquitectura.json')
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu';cuda=device=='cuda'
    if not cuda or len(ck['rng_cuda'])!=torch.cuda.device_count():raise ValueError('Se requiere reproducir el entorno CUDA original')
    amp=base['precision_mixta_cuda']
    valid=json.loads((RAIZ/'datos/reportes/validacion.json').read_text(encoding='utf-8'))
    for ruta,h in valid['sha256_manifiestos'].items():
        if sha(RAIZ/ruta)!=h:raise ValueError('Manifiesto modificado')
    contornos=ContornosTACO(RAIZ,base['cobertura_minima'])
    if contornos.sha256!=hashes['anotaciones_poligonos'] or contornos.sha256_inventario!=hashes['inventario_procedencia']:
        raise ValueError('Máscaras de otros datos')
    train=DatosMapas('train',contornos);val=DatosMapas('val',contornos);aux=DatosClasificacionAuxiliar(RAIZ)
    if aux.metadata['sha256_manifiesto']!=hashes['clasificacion_auxiliar_train'] or sha(aux.metadata_path)!=hashes['cache_clasificacion_auxiliar']:
        raise ValueError('Clasificación auxiliar diferente')
    conteos=torch.bincount(train.mascaras[train.mascaras!=255].flatten(),minlength=7)[1:].float()
    pesos=conteos.clamp(min=1e-6).rsqrt();pesos=(pesos/pesos.mean()).clamp(max=3).to(device)
    hash_mascaras=huella(train.mascaras);cuenta=torch.bincount(aux.etiquetas,minlength=6)
    pasos_epoca=(len(train)+base['batch_size']-1)//base['batch_size']
    pasos_fijos=pasos_epoca*cfg['epocas_adicionales'];resultados={};inicio=time.perf_counter()
    esperados={'pesos':huella(ck['pesos']),'adamw':huella(ck['optimizador']),'scaler':huella(ck['scaler']),
        'aleatorios':huella({k:ck[k] for k in ['rng_torch','rng_cuda','rng_loader','rng_auxiliar']})}
    comun={'configuracion':cfg,'checkpoint_origen':cfg['checkpoint_origen'],'sha256_origen':hash_origen,
        'epoca_origen':19,'huellas_estado_origen':esperados,'hashes_manifiestos':hashes,
        'sha256_manifiestos_invariantes':valid['sha256_manifiestos'],'sha256_seleccion_antes':hash_seleccion,
        'huella_mascaras_train':hash_mascaras,'pesos_materiales':pesos.cpu().tolist(),
        'pasos_por_epoca':pasos_epoca,'actualizaciones_por_rama':pasos_fijos,
        'seleccion_mejor':'mAP@50 val de una vista; origen separado del mejor durante actualizaciones',
        'test_evaluado':False,'cambio_predeterminado':False,'reinicio_warmup':False}
    escribir_json(salida,{'estado':'en_curso',**comun})
    origen_semantica=None
    for rama,coef in cfg['ramas'].items():
        nombre=nombres[rama];m=DetectorMapas(**ck['configuracion']).to(device)
        m.load_state_dict(ck['pesos'])
        opt=torch.optim.AdamW(m.parameters(),lr=base['learning_rate'],weight_decay=base['weight_decay'])
        opt.load_state_dict(copy.deepcopy(ck['optimizador']))
        scaler=torch.amp.GradScaler('cuda',enabled=amp);scaler.load_state_dict(copy.deepcopy(ck['scaler']))
        rng=torch.Generator();rng_aux=torch.Generator()
        restaurar_aleatorio(ck,rng,rng_aux)
        dl=DataLoader(train,batch_size=base['batch_size'],shuffle=True,generator=rng,pin_memory=True)
        dlval=DataLoader(val,batch_size=base['batch_size'],pin_memory=True)
        aux_cfg=base['clasificacion_auxiliar']
        sampler=WeightedRandomSampler(1/cuenta[aux.etiquetas].double(),len(dl)*aux_cfg['batch_size'],replacement=True,generator=rng_aux)
        dla=DataLoader(AuxIndice(aux),batch_size=aux_cfg['batch_size'],sampler=sampler,generator=rng_aux,pin_memory=True)
        inicial={'pesos':huella(m.state_dict()),'adamw':huella(opt.state_dict()),'scaler':huella(scaler.state_dict()),
                 'aleatorios':huella(estado_aleatorio(rng,rng_aux))}
        if inicial!=esperados:raise ValueError(f'La rama {rama} no replica todo el estado del origen')
        paso_inicial=pasos_adam(opt)
        if origen_semantica is None:origen_semantica=semantica(m,train,device,amp)
        restaurar_aleatorio(ck,rng,rng_aux)
        historia=[];mejor=-1.;epoca_mejor=None;traza_total=hashlib.sha256();pasos=0
        cfg_rama={**base,'nombre':nombre,'coeficiente_fondo':coef,'epocas':19+cfg['epocas_adicionales'],
            'min_epocas':19+cfg['epocas_adicionales'],'paciencia':999,'protocolo_ablacion':cfg['nombre']}
        mejor_path=RAIZ/f'checkpoints/{nombre}_mejor.pt';ultimo_path=RAIZ/f'checkpoints/{nombre}_ultimo.pt'
        print(f'Rama {rama}: fondo={coef:g}, AdamW paso {paso_inicial}, LR={opt.param_groups[0]["lr"]}, {pasos_fijos} actualizaciones',flush=True)
        for epoca in range(20,20+cfg['epocas_adicionales']):
            m.train();it_aux=iter(dla);total=n=total_aux=na=0;traza=hashlib.sha256()
            for x,y,indices in dl:
                x=x.to(device,non_blocking=True).float()/255;y=y.to(device,non_blocking=True)
                x,y,aug=aumentar(x,y)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):raw=m(x)
                loss=perdida_mapas(raw,y,pesos,base['formulacion'],coeficiente_fondo=coef)
                if not bool(torch.isfinite(loss)):raise RuntimeError('Pérdida detector no finita')
                scaler.scale(loss).backward()
                xa,ya,ia=next(it_aux);xa=xa.to(device).float()/255;ya=ya.to(device)
                xa,_,auga=aumentar(xa)
                with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):logits_aux=m.clasificar(xa)
                la=F.cross_entropy(logits_aux.float(),ya)
                if not bool(torch.isfinite(la)):raise RuntimeError('Pérdida auxiliar no finita')
                scaler.scale(aux_cfg['coeficiente_perdida']*la).backward()
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
                scaler.step(opt);scaler.update();pasos+=1
                traza.update(huella({'indices_train':indices,'aumentos_train':aug,'indices_aux':ia,'aumentos_aux':auga}).encode())
                total+=loss.item()*len(x);n+=len(x);total_aux+=la.item()*len(xa);na+=len(xa)
            metricas,_=evaluar(m,dlval,device,amp,base)
            if pasos_adam(opt)!=paso_inicial+pasos:raise RuntimeError('Se omitió una actualización de AdamW')
            h=traza.hexdigest();traza_total.update(h.encode())
            historia.append({'epoca':epoca,'actualizaciones_desde_origen':pasos,'paso_adamw':pasos_adam(opt),
                'learning_rate':opt.param_groups[0]['lr'],'loss_train':total/n,'loss_aux_train':total_aux/na,
                'map50_val':metricas['map50'],'map50_95_val':metricas['map50_95'],'recall50_val':metricas['recall50'],
                'huella_orden_y_aumentos':h,'muestras_train':n,'muestras_aux':na})
            mejora=metricas['map50']>mejor+1e-8
            if mejora:mejor=metricas['map50'];epoca_mejor=epoca
            estado={'tipo':ck['tipo'],'configuracion':m.configuracion_mapas,'clases':CLASES,'pesos':m.state_dict(),
                'optimizador':opt.state_dict(),'scaler':scaler.state_dict(),'config_entrenamiento':cfg_rama,
                'epoca':epoca,'mejor_map':mejor,'mejor_epoca':epoca_mejor,'paciencia':0,'historial':historia,
                'hashes_manifiestos':hashes,**estado_aleatorio(rng,rng_aux),'recortes_train':None,
                'clasificacion_auxiliar':ck['clasificacion_auxiliar'],'pesos_externos':False,
                'origen_ablacion':{'checkpoint':cfg['checkpoint_origen'],'sha256':hash_origen,'epoca':19,
                    'coeficiente_fondo':coef,'adamw_escalador_rng_restaurados':True},
                'duracion_segundos':time.perf_counter()-inicio,'actualizaciones_desde_origen':pasos}
            guardar_checkpoint(ultimo_path,estado)
            if mejora:guardar_checkpoint(mejor_path,estado)
            escribir_json(RAIZ/f'resultados/{nombre}.json',{'estado':'en_curso','historial':historia,'coeficiente_fondo':coef,'test_evaluado':False})
            print(f'Rama {rama}, época {epoca}: loss={total/n:.3f}, AP50={metricas["map50"]:.3%}, paso+{pasos}',flush=True)
        if pasos!=pasos_fijos:raise ValueError('Presupuesto incompleto')
        sem_final=semantica(m,train,device,amp)
        val_final,pred=evaluar(m,dlval,device,amp,base)
        guardar_pred(RAIZ/f'resultados/{nombre}_final_predicciones_val.json',pred,val.base.objetivos)
        mejor_ck=torch.load(mejor_path,map_location=device,weights_only=True);m.load_state_dict(mejor_ck['pesos'])
        sem_mejor=semantica(m,train,device,amp);val_mejor,pred=evaluar(m,dlval,device,amp,base)
        guardar_pred(RAIZ/f'resultados/{nombre}_mejor_predicciones_val.json',pred,val.base.objetivos)
        incl_path=RAIZ/f'checkpoints/{nombre}_mejor_incluyendo_origen.pt'
        if mejor>ck['mejor_map']:guardar_checkpoint(incl_path,mejor_ck);incl_epoca=epoca_mejor
        else:guardar_checkpoint(incl_path,ck);incl_epoca=19
        fuente=json.loads((RAIZ/'resultados/detector_mapas_v6_auxiliar_ext48.json').read_text(encoding='utf-8'))
        control=[]
        if rama=='A':
            por_epoca={e['epoca']:e for e in fuente['historial']}
            for h in historia:
                old=por_epoca[h['epoca']]
                control.append({'epoca':h['epoca'],'delta_loss':h['loss_train']-old['loss_train'],
                    'delta_map50':h['map50_val']-old['map50_val']})
        resultado={'estado':'ablacion_rama_completada','nombre':nombre,'rama':rama,'coeficiente_fondo':coef,
            'huellas_estado_inicial':inicial,'paso_adamw_inicial':paso_inicial,'paso_adamw_final':paso_inicial+pasos,
            'actualizaciones':pasos,'historial':historia,'huella_orden_y_aumentos':traza_total.hexdigest(),
            'semantica_train_final':sem_final,'semantica_train_mejor':sem_mejor,
            'val_final_una_vista':val_final,'val_mejor_una_vista':val_mejor,
            'resumen_final':resumir_deteccion(val_final),'resumen_mejor':resumir_deteccion(val_mejor),
            'mejor_epoca_durante_actualizaciones':epoca_mejor,'mejor_epoca_incluyendo_origen':incl_epoca,
            'checkpoint_final':ultimo_path.relative_to(RAIZ).as_posix(),'checkpoint_mejor':mejor_path.relative_to(RAIZ).as_posix(),
            'checkpoint_mejor_incluyendo_origen':incl_path.relative_to(RAIZ).as_posix(),
            'regresion_control_historico':control,'test_evaluado':False,'pesos_externos':False}
        escribir_json(RAIZ/f'resultados/{nombre}.json',resultado);resultados[rama]=resultado
        escribir_json(salida,{'estado':'en_curso',**comun,'semantica_train_origen':origen_semantica,'ramas':resultados})
        del m,opt,scaler,mejor_ck
        torch.cuda.empty_cache()
    if resultados['A']['huella_orden_y_aumentos']!=resultados['B']['huella_orden_y_aumentos']:
        raise ValueError('Las ramas no recibieron exactamente las mismas muestras y aumentos')
    if any(a['huella_orden_y_aumentos']!=b['huella_orden_y_aumentos'] for a,b in zip(resultados['A']['historial'],resultados['B']['historial'])):
        raise ValueError('Las trazas por época difieren')
    if sha(origen)!=hash_origen or sha(RAIZ/'configuracion/arquitectura.json')!=hash_seleccion:
        raise ValueError('Se alteró el origen o la selección')
    informe={'estado':'ablacion_entrenamiento_completada',**comun,'fecha_utc':datetime.now(timezone.utc).isoformat(),
        'semantica_train_origen':origen_semantica,'ramas':resultados,'orden_y_aumentos_iguales':True,
        'sha256_origen_despues':sha(origen),'sha256_seleccion_despues':sha(RAIZ/'configuracion/arquitectura.json'),
        'duracion_segundos':time.perf_counter()-inicio,'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__)}
    escribir_json(salida,informe)
    print('Ablación completada: estados iniciales y entradas idénticos; selección intacta',flush=True)


if __name__=='__main__':main()
