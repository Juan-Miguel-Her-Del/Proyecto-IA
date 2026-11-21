"""A/B V20 con presupuesto fijo, estados completos y diarios inmutables por época."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import uuid
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ecowaste.reanudacion_regional import publicar,sha,bloqueo,huellas_estado
from ecowaste.reanudacion_v20 import DiarioV20
from ecowaste.ablacion_fondo import huella,pasos_adam,restaurar_aleatorio,estado_aleatorio
from ecowaste.poligonos import ContornosTACO
from ecowaste.discriminacion_mapas import VERSION
from scripts.preparar_v20 import (leer_cfg,protocolo,contexto_train,calcular_lote,exigir_autorizacion,huellas_codigo)
from scripts.supervision_regional_materiales import restaurar,loaders
from scripts.ablacion_fondo import aumentar
from scripts.entrenar_detector_mapas import DatosMapas,evaluar


def paso_adamw(m,opt,scaler,x,y,indices,xa,ya,ia,regiones,pm,coef):
    """Una sola pasada TACO; la auxiliar global y clipping conservan la operación histórica."""
    m.train();x,y,aug=aumentar(x.cuda().float()/255,y.cuda());opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):
        loss,raw,lo,ln,d=calcular_lote(m,x,y,[regiones[i] for i in indices.tolist()],pm,coef,aug['flip'].flatten())
    if not torch.isfinite(loss):raise RuntimeError('Pérdida no finita; detener')
    scaler.scale(loss).backward()
    xa,_,auga=aumentar(xa.cuda().float()/255)
    with torch.autocast('cuda',dtype=torch.float16):logits=m.clasificar(xa)
    la=F.cross_entropy(logits.float(),ya.cuda())
    if not torch.isfinite(la):raise RuntimeError('Auxiliar no finita; detener')
    scaler.scale(.5*la).backward();scaler.unscale_(opt)
    norma=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
    anterior=pasos_adam(opt);scaler.step(opt);scaler.update()
    if pasos_adam(opt)!=anterior+1:raise RuntimeError('AdamW omitió una actualización; detener')
    return {'loss_original':float(lo.detach()),'loss_nueva':float(ln.detach()),'loss_global':float(la.detach()),
        'norma_antes_clip':float(norma),'huella_entrada':huella({'indices_train':indices,'aumentos_train':aug,
            'indices_aux':ia,'aumentos_aux':auga}),'detalle':d}


def cargar_diagnostico(cfg):
    d=json.loads((RAIZ/cfg['diagnostico']).read_text(encoding='utf-8'));d['sha256_archivo']=sha(RAIZ/cfg['diagnostico'])
    return d


def entrenar(cfg,ramas=None,reanudar=False,hasta_epoca=31):
    exigir_autorizacion(cfg)
    salida=RAIZ/f"resultados/{cfg['nombre']}.json"
    if salida.exists():raise FileExistsError('Experimento completo existente')
    if not 20<=hasta_epoca<=31:raise ValueError('Época inválida')
    ck,train,aux,regiones,pm,est=contexto_train(cfg);diag=cargar_diagnostico(cfg);codigos=huellas_codigo()
    if huella(regiones)!=diag['huella_regiones'] or huella(train.mascaras)!=diag['huella_mascaras_originales']:
        raise ValueError('Geometría diferente del diagnóstico')
    val=DatosMapas('val',ContornosTACO(RAIZ,1/64))
    if sha(RAIZ/'datos/procesados/deteccion/val.jsonl')!=ck['hashes_manifiestos']['val']:raise ValueError('Val diferente')
    dlval=DataLoader(val,batch_size=8,generator=torch.Generator().manual_seed(0))
    protegidos=json.loads((RAIZ/'resultados/v20_ejecucion/estado_previo.json').read_text(encoding='utf-8'))['historicos']
    if any(sha(RAIZ/p)!=h for p,h in protegidos.items()):raise ValueError('Históricos alterados')
    for rama in (ramas or list(cfg['ramas'])):
      diario=DiarioV20(RAIZ,cfg,rama,diag,codigos);coef=cfg['ramas'][rama]
      with bloqueo(diario.directorio/'ejecucion.lock'):
        if not reanudar and (any(diario.directorio.glob('epoca_*.json')) or any((RAIZ/'checkpoints').glob(diario.nombre+'*.pt'))):
            raise FileExistsError('Rama existente: reanudación explícita requerida')
        anterior,registros,previo=diario.leer(recuperar=reanudar)
        if not registros and any((RAIZ/'checkpoints'/f'{diario.nombre}_{s}.pt').exists() for s in ['ultimo','mejor']):
            raise ValueError('Aliases sin diario propio')
        diario.actualizar_alias(registros)
        if anterior and anterior['epoca']>=hasta_epoca:continue
        origen=anterior or ck;m,opt,scaler,rng,ra,real=restaurar(origen)
        dl,da=loaders(train,aux,rng,ra);restaurar_aleatorio(origen,rng,ra)
        hist=copy.deepcopy(anterior['historial']) if anterior else [];pasos=121*len(hist)
        mejor=anterior['mejor_map'] if anterior else -1.;mejor_ep=anterior['mejor_epoca'] if anterior else None
        inicial=anterior['reanudacion']['estado_inicial_origen'] if anterior else real
        acumulado=anterior['reanudacion']['segundos_acumulados'] if anterior else 0.
        run=uuid.uuid4().hex;inicio=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        publicar(diario.directorio/'ejecuciones'/f'{run}.json',{'rama':rama,'desde_epoca':origen['epoca'],
            'reanudar':reanudar,'pasos_comprometidos':pasos,'huellas_restauradas':real,'hashes_codigo_v20':codigos})
        for ep in range(origen['epoca']+1,hasta_epoca+1):
            itaux=iter(da);traza=[];n=0;lo=ln=lg=0.;normas=[];org=[];contrib={c:0. for c in est['orden_materiales']}
            for x,y,indices in dl:
                xa,ya,ia=next(itaux)
                s=paso_adamw(m,opt,scaler,x,y,indices,xa,ya,ia,regiones,pm,coef);pasos+=1
                if pasos_adam(opt)!=2299+pasos:raise RuntimeError('Contador divergente')
                traza.append(s['huella_entrada']);n+=len(x)
                lo+=s['loss_original']*len(x);ln+=s['loss_nueva']*len(x);lg+=s['loss_global']*len(x);normas.append(s['norma_antes_clip'])
                for c,d in s['detalle']['por_material'].items():contrib[c]+=d['contribucion']*len(x)
                for f in s['detalle']['filas']:
                    if f['clase']==5:
                        imagen=train.base.origen.registros[int(indices[f['imagen_lote']])]['id']
                        org.append({**f,'imagen':imagen,'actualizacion':pasos,'contribucion_loss_total':coef*f['contribucion']})
            import hashlib
            h=hashlib.sha256(''.join(traza).encode()).hexdigest()
            if h!=diag['trazas_preparadas'][rama][ep-20]['huella_orden_y_aumentos']:raise RuntimeError('Entradas o aumentos divergentes; detener')
            if len(org)!=1:raise RuntimeError('Presencia orgánica distinta; detener')
            met,_=evaluar(m,dlval,'cuda',True,ck['config_entrenamiento'])
            if met['map50']>mejor:mejor=met['map50'];mejor_ep=ep
            hist.append({'epoca':ep,'actualizaciones':pasos,'paso_adamw':pasos_adam(opt),'learning_rate':opt.param_groups[0]['lr'],
                'coeficiente_discriminativo':coef,'loss_original':lo/n,'loss_nueva':ln/n,'loss_global':lg/n,
                'norma_antes_clip_max':max(normas),'norma_antes_clip_media':sum(normas)/len(normas),
                'contribucion_por_material':{c:v/n for c,v in contrib.items()},
                'region_organica':{'presencias':len(org),'eventos':org},'map50_val':met['map50'],'map50_95_val':met['map50_95'],
                'huella_orden_y_aumentos':h,'trazas_lotes':traza})
            cfgbase={**ck['config_entrenamiento'],'nombre':diario.nombre,'v20':{'coeficiente':coef,'margen':1.,'fraccion':.25,
                'delta_huber':1.,'pesos':[1.,1.,2.,2.,1.,1.]}}
            estado={**ck,'pesos':m.state_dict(),'configuracion':m.configuracion_mapas,'optimizador':opt.state_dict(),
                'scaler':scaler.state_dict(),**estado_aleatorio(rng,ra),'config_entrenamiento':cfgbase,
                'historial':hist,'epoca':ep,'mejor_epoca':mejor_ep,'mejor_map':mejor,'actualizaciones_desde_origen':pasos,
                'sha256_origen':cfg['sha256_checkpoint_origen'],'protocolo_v20':protocolo(cfg),
                'hashes_codigo_v20':codigos,'huella_regiones':diag['huella_regiones']}
            memoria=anterior['reanudacion'] if anterior else {}
            estado['reanudacion']={'tipo':'v20_margen_huber','rama':rama,'epoca_completa':True,'commit_previo_sha256':previo,
                'sha256_diagnostico':diag['sha256_archivo'],'huellas_estado':huellas_estado(estado),'estado_inicial_origen':inicial,
                'segundos_acumulados':acumulado+time.perf_counter()-inicio,
                'memoria_gpu_max_allocated_bytes':max(memoria.get('memoria_gpu_max_allocated_bytes',0),torch.cuda.max_memory_allocated()),
                'memoria_gpu_max_reserved_bytes':max(memoria.get('memoria_gpu_max_reserved_bytes',0),torch.cuda.max_memory_reserved())}
            previo=diario.confirmar(estado,previo);registros.append(json.loads(diario.registro(ep).read_text(encoding='utf-8')))
            diario.actualizar_alias(registros)
            print(f'V20 {rama} época {ep}: +{pasos}/1452, AP50 1 vista={met["map50"]:.4%}',flush=True)
        publicar(diario.directorio/'ejecuciones'/f'{run}.fin.json',{'epoca':ep,'pasos':pasos,'commit_sha256':previo})
        del m,opt,scaler,estado,x,y,xa,ya;torch.cuda.empty_cache()
    resultados={}
    for rama in cfg['ramas']:
        diario=DiarioV20(RAIZ,cfg,rama,diag,codigos);final,registros,_=diario.leer()
        if final is None or final['actualizaciones_desde_origen']!=1452:
            print('Presupuesto pendiente:',rama,flush=True);return
        rr=final['reanudacion']
        resultados[rama]={'actualizaciones':1452,'historial':final['historial'],'mejor_epoca':final['mejor_epoca'],
            'estado_inicial':rr['estado_inicial_origen'],'segundos':rr['segundos_acumulados'],
            'memoria_gpu_max_allocated_bytes':rr['memoria_gpu_max_allocated_bytes'],
            'memoria_gpu_max_reserved_bytes':rr['memoria_gpu_max_reserved_bytes'],'checkpoints_epocas':registros}
    if resultados['A']['estado_inicial']!=resultados['B']['estado_inicial']:raise ValueError('Distintos estados iniciales')
    if any(sha(RAIZ/p)!=h for p,h in protegidos.items()):raise ValueError('Históricos alterados')
    publicar(salida,{'estado':'entrenamiento_v20_completo','protocolo':protocolo(cfg),'ramas':resultados,
        'test_evaluado':False,'cambio_predeterminado':False,'hashes_codigo':codigos})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rama',choices=['A','B','ambas'],default='ambas')
    p.add_argument('--reanudar',action='store_true');p.add_argument('--hasta-epoca',type=int,default=31);a=p.parse_args()
    entrenar(leer_cfg(),None if a.rama=='ambas' else [a.rama],a.reanudar,a.hasta_epoca)


if __name__=='__main__':main()
