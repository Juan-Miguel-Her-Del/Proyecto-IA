"""Preparación y continuación pareada regional. Entrenar exige habilitación explícita."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from ecowaste.datasets import CLASES
from ecowaste.detector_mapas import DetectorMapas,perdida_mapas
from ecowaste.poligonos import ContornosTACO
from ecowaste.auxiliar_clasificacion import DatosClasificacionAuxiliar
from ecowaste.ablacion_fondo import huella,estado_aleatorio,restaurar_aleatorio,pasos_adam,metricas_semanticas
from ecowaste.supervision_regional import preparar_regiones,perdida_regional,resumen_confusion,pool_regiones,VERSION
from scripts.entrenar_detector_mapas import DatosMapas,evaluar
from scripts.entrenar_clasificador import escribir_json,guardar_checkpoint
from scripts.ablacion_fondo import AuxIndice,aumentar
from ecowaste.reanudacion_regional import DiarioRama,bloqueo,publicar,huellas_estado,restaurar_estados

CONFIG=RAIZ/'configuracion/supervision_regional_materiales.json'
CODIGOS=['ecowaste/supervision_regional.py','ecowaste/detector_mapas.py',
    'ecowaste/reanudacion_regional.py','pruebas/test_reanudacion_regional.py',
    'ecowaste/detector.py','ecowaste/poligonos.py','ecowaste/auxiliar_clasificacion.py',
    'ecowaste/transformer.py','ecowaste/clasificador.py','ecowaste/datasets.py',
    'ecowaste/mosaico.py','ecowaste/evaluacion_detector.py','ecowaste/cajas.py',
    'ecowaste/ejecucion_detector.py','ecowaste/auditoria_deteccion.py','scripts/entrenar_clasificador.py',
    'ecowaste/ablacion_fondo.py','scripts/entrenar_detector_mapas.py','scripts/ablacion_fondo.py',
    'scripts/supervision_regional_materiales.py','scripts/diagnosticar_supervision_regional.py',
    'scripts/evaluar_supervision_regional.py','pruebas/test_supervision_regional.py',
    'scripts/verificar_supervision_regional.py']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hashes_codigo():return {p:sha(RAIZ/p) for p in CODIGOS}


def protocolo(cfg):return {k:v for k,v in cfg.items() if k!='entrenamiento_completo_habilitado'}


def validar_protocolo(cfg):
    esperados={'ramas':{'A':0.,'B':.25},'coeficiente_fondo':3.,'coeficiente_auxiliar_global':.5,
        'epocas_adicionales':12,'actualizaciones_por_rama':1452,'criterio_mejor':'map50_val_una_vista',
        'desempate':'primera_epoca','parada_temprana':False,'reiniciar_warmup':False,
        'evaluar_test':False,'cambiar_predeterminado':False,'politica_mapas':'argmax7',
        'umbral_evaluacion':.05,'area_minima_celdas':2,'vistas_evaluacion':[1,5]}
    if any(cfg.get(k)!=v for k,v in esperados.items()):raise ValueError('Protocolo regional modificado')
    if Path(cfg['nombre']).name!=cfg['nombre']:raise ValueError('Nombre inseguro')


def exigir_habilitacion(cfg):
    validar_protocolo(cfg)
    if not cfg.get('entrenamiento_completo_habilitado',False):
        raise ValueError('Entrenamiento completo deshabilitado: diagnóstico y revisión pendientes de autorización')
    d=json.loads((RAIZ/cfg['diagnostico']).read_text(encoding='utf-8'))
    if not d['aprobada'] or d['protocolo']!=protocolo(cfg) or d['hashes_codigo']!=hashes_codigo():
        raise ValueError('Diagnóstico no aprobado o desactualizado')
    if str(torch.__version__)!=d['torch'] or not torch.cuda.is_available() or torch.cuda.get_device_name()!=d['gpu']:
        raise ValueError('El entorno difiere del diagnóstico')
    v=json.loads((RAIZ/cfg['verificacion']).read_text(encoding='utf-8'))
    if not v['aprobada'] or v['sha256_diagnostico']!=sha(RAIZ/cfg['diagnostico']) or v['hashes_codigo']!=hashes_codigo():
        raise ValueError('Verificación congelada ausente o desactualizada')


def contexto(cfg):
    validar_protocolo(cfg);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise ValueError('Se requiere CUDA como en el origen')
    if sha(RAIZ/cfg['checkpoint_origen'])!=cfg['sha256_checkpoint_origen']:raise ValueError('Checkpoint alterado')
    ck=torch.load(RAIZ/cfg['checkpoint_origen'],map_location='cpu',weights_only=True)
    base=ck['config_entrenamiento'];h=ck['hashes_manifiestos']
    if ck['epoca']!=19 or ck['pesos_externos'] or ck['recortes_train'] is not None:raise ValueError('Origen incompatible')
    if not ck['configuracion'].get('auxiliar_clasificacion') or base['formulacion']!='softmax7':raise ValueError('Cabeza incompatible')
    if base['batch_size']!=8 or base['clasificacion_auxiliar']['batch_size']!=16:raise ValueError('Lotes diferentes')
    if base['clasificacion_auxiliar']['coeficiente_perdida']!=.5:raise ValueError('Auxiliar global distinta')
    if len(ck['rng_cuda'])!=torch.cuda.device_count():raise ValueError('Entorno CUDA diferente')
    if any(g['lr']!=base['learning_rate'] for g in ck['optimizador']['param_groups']):raise ValueError('LR fuera de fase estable')
    valid=json.loads((RAIZ/'datos/reportes/validacion.json').read_text(encoding='utf-8'))
    for p,v in valid['sha256_manifiestos'].items():
        if sha(RAIZ/p)!=v:raise ValueError('Manifiesto cambiado: '+p)
    c=ContornosTACO(RAIZ,base['cobertura_minima'])
    if c.sha256!=h['anotaciones_poligonos'] or c.sha256_inventario!=h['inventario_procedencia']:raise ValueError('Polígonos distintos')
    train=DatosMapas('train',c);aux=DatosClasificacionAuxiliar(RAIZ)
    if aux.metadata['sha256_manifiesto']!=h['clasificacion_auxiliar_train'] or sha(aux.metadata_path)!=h['cache_clasificacion_auxiliar']:
        raise ValueError('Auxiliar global diferente')
    regiones,w,estadistica=preparar_regiones(train.base.origen.registros,c)
    if not estadistica['todas_las_clases_elegibles']:raise ValueError('Hay materiales sin regiones train elegibles')
    if ((len(train)+7)//8)*12!=1452:raise ValueError('Presupuesto incompatible con train')
    n=torch.bincount(train.mascaras[train.mascaras!=255].flatten(),minlength=7)[1:].float()
    pm=n.clamp(min=1e-6).rsqrt();pm=(pm/pm.mean()).clamp(max=3).cuda()
    return ck,train,aux,regiones,w.cuda(),estadistica,pm


def restaurar(ck):
    m=DetectorMapas(**ck['configuracion']).cuda()
    base=ck['config_entrenamiento']
    opt=torch.optim.AdamW(m.parameters(),lr=base['learning_rate'],weight_decay=base['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=base['precision_mixta_cuda'])
    rng=torch.Generator();ra=torch.Generator()
    real=restaurar_estados(m,opt,scaler,ck,rng,ra)
    return m,opt,scaler,rng,ra,real


def loaders(train,aux,rng,ra,solo_indices=False):
    d=TensorDataset(torch.arange(len(train))) if solo_indices else train
    a=TensorDataset(torch.arange(len(aux))) if solo_indices else AuxIndice(aux)
    dl=DataLoader(d,batch_size=8,shuffle=True,generator=rng,pin_memory=not solo_indices)
    n=torch.bincount(aux.etiquetas,minlength=6)
    sampler=WeightedRandomSampler(1/n[aux.etiquetas].double(),len(dl)*16,replacement=True,generator=ra)
    da=DataLoader(a,batch_size=16,sampler=sampler,generator=ra,pin_memory=not solo_indices)
    return dl,da


def trazar_sin_entrenar(ck,train,aux):
    """Simula el consumo RNG original; no evalúa redes ni actualiza parámetros."""
    rng=torch.Generator();ra=torch.Generator();restaurar_aleatorio(ck,rng,ra)
    dl,da=loaders(train,aux,rng,ra,True);trazas=[]
    for epoca in range(20,32):
        ia=iter(da);h=hashlib.sha256();pasos=0
        for (indices,) in dl:
            _,_,aug=aumentar(torch.zeros(len(indices),1,1,1,device='cuda'))
            (indaux,)=next(ia)
            _,_,auga=aumentar(torch.zeros(len(indaux),1,1,1,device='cuda'))
            h.update(huella({'indices_train':indices,'aumentos_train':aug,'indices_aux':indaux,'aumentos_aux':auga}).encode());pasos+=1
        trazas.append({'epoca':epoca,'pasos':pasos,'huella_orden_y_aumentos':h.hexdigest()})
    return trazas


@torch.inference_mode()
def evaluar_regiones(m,train,regiones,w,amp=True):
    m.eval();cm=torch.zeros(6,6,dtype=torch.long);numerador=den=0.;por={c:0. for c in CLASES}
    cm_mapas=torch.zeros(7,7,dtype=torch.long);cm_pixeles=cm_mapas.clone()
    for x,mascaras,indices in DataLoader(train,batch_size=8,generator=torch.Generator().manual_seed(0)):
        rs=[regiones[i] for i in indices.tolist()]
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            raw,f=m(x.cuda().float()/255,True)
            _,d,p=perdida_regional(m,f,rs,w)
        pm=pool_regiones(raw.permute(0,2,3,1),rs)
        ym=pm['labels'].cpu()+1;predm=pm['vectores'].argmax(1).cpu()
        cm_mapas+=torch.bincount(7*ym+predm,minlength=49).reshape(7,7)
        predpix=raw.argmax(1).cpu();ok=mascaras!=255
        cm_pixeles+=torch.bincount(7*mascaras[ok]+predpix[ok],minlength=49).reshape(7,7)
        y=p['labels'].cpu();pred=p['logits'].argmax(1).cpu()
        cm+=torch.bincount(6*y+pred,minlength=36).reshape(6,6)
        den+=d['denominador_pesos'] if d['regiones'] else 0
        for c in CLASES:por[c]+=d['por_material'][c]['numerador_ponderado']
    numerador=sum(por.values())
    return {**resumen_confusion(cm),'ce_regional':numerador/den if den else None,
        'contribucion_ce_por_material':{c:v/den if den else None for c,v in por.items()},
        'cabeza_medida':'auxiliar global compartida aplicada a pooling GT regional, distinta de la cabeza de mapas',
        'mapas_logits_promediados_por_region':metricas_semanticas(cm_mapas),
        'semantica_pixeles_train':metricas_semanticas(cm_pixeles),
        'fuente':'GT polígonos TACO train; solo diagnóstico, nunca propuestas de inferencia'}


def entrenar(cfg,ramas=None,reanudar=False,hasta_epoca=31):
    # Esta comprobación sucede antes de cargar datos, redes o escribir checkpoints.
    exigir_habilitacion(cfg)
    salida=RAIZ/f"resultados/{cfg['nombre']}.json"
    if not 20<=hasta_epoca<=31:raise ValueError('Límite de época inválido')
    if salida.exists():raise FileExistsError('Resultado completo existente; no se sobrescribe')
    ck,train,aux,regiones,w,estadistica,pm=contexto(cfg)
    diagnostico=json.loads((RAIZ/cfg['diagnostico']).read_text(encoding='utf-8'))
    diagnostico['sha256_archivo']=sha(RAIZ/cfg['diagnostico'])
    if diagnostico['huella_regiones']!=huella(regiones):raise ValueError('Regiones diferentes del diagnóstico')
    val=DatosMapas('val',ContornosTACO(RAIZ,ck['config_entrenamiento']['cobertura_minima']))
    dlval=DataLoader(val,batch_size=8,generator=torch.Generator().manual_seed(0))
    protegidos={p:sha(RAIZ/p) for p in ['configuracion/arquitectura.json',cfg['checkpoint_origen']]}
    resultados={};base=ck['config_entrenamiento'];amp=base['precision_mixta_cuda'];codigos=hashes_codigo()
    organicas=[{'imagen':r['id'],'id_region':ident,'indice_train':i} for i,(r,rs) in enumerate(zip(train.base.origen.registros,regiones))
        for ident,y in zip(rs['ids'],rs['labels']) if int(y)==5]
    if len(organicas)!=1:raise ValueError('Cambió la presencia orgánica elegible')
    for rama in (ramas or list(cfg['ramas'])):
      coef=cfg['ramas'][rama];diario=DiarioRama(RAIZ,cfg,rama,diagnostico,codigos)
      with bloqueo(diario.directorio/'ejecucion.lock'):
        if not reanudar and (any(diario.directorio.glob('epoca_*.json')) or any((RAIZ/'checkpoints').glob(diario.nombre+'*.pt'))):
            raise FileExistsError('Rama existente; usar --reanudar explícitamente')
        anterior,registros,previo=diario.leer(recuperar=reanudar)
        if not registros and any((RAIZ/'checkpoints'/f'{diario.nombre}_{s}.pt').exists() for s in ['ultimo','mejor']):
            raise ValueError('Aliases sin diario; no se adoptan checkpoints de procedencia desconocida')
        diario.actualizar_alias(registros)
        if anterior is not None and anterior['epoca']>=hasta_epoca:continue
        origen=anterior if anterior is not None else ck
        m,opt,scaler,rng,ra,restaurados=restaurar(origen);dl,da=loaders(train,aux,rng,ra)
        restaurar_aleatorio(origen,rng,ra);inicio=time.perf_counter()
        historia=copy.deepcopy(anterior['historial']) if anterior else []
        mejor=anterior['mejor_map'] if anterior else -1.;pasos=121*len(historia)
        mejor_epoca=anterior['mejor_epoca'] if anterior else None
        inicial=anterior['reanudacion']['estado_inicial_origen'] if anterior else restaurados
        acumulado=anterior['reanudacion']['segundos_acumulados'] if anterior else 0.
        runid=uuid.uuid4().hex
        publicar(diario.directorio/'ejecuciones'/f'{runid}.json',{'rama':rama,'reanudar':reanudar,
            'desde_epoca':origen['epoca'],'pasos_comprometidos':pasos,'huellas_restauradas':restaurados,
            'protocolo':protocolo(cfg),'hashes_codigo':codigos})
        torch.cuda.reset_peak_memory_stats();m.train()
        for epoca in range(origen['epoca']+1,hasta_epoca+1):
            m.train();itaux=iter(da);traza=hashlib.sha256();tot=glob=reg=0.;n=0
            por={c:{'regiones':0,'numerador_ponderado':0.} for c in CLASES};den=0.;eventos_organicos=[]
            for x,y,indices in dl:
                x,y,aug=aumentar(x.cuda().float()/255,y.cuda());opt.zero_grad(set_to_none=True)
                rs=[regiones[i] for i in indices.tolist()]
                with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                    raw,f=m(x,True)
                    lr,d,_=perdida_regional(m,f,rs,w,aug['flip'].flatten(),con_gradiente=coef>0)
                ld=perdida_mapas(raw,y,pm,base['formulacion'],coeficiente_fondo=3.)
                loss=ld+coef*lr if coef else ld
                if d['por_material']['organico']['regiones']:
                    eventos_organicos.append({'actualizacion':pasos+1,**organicas[0],
                        'peso_sin_modificar':float(w[5]),'denominador_lote':d['denominador_pesos'],
                        **d['por_material']['organico'],
                        'contribucion_loss_total':coef*d['por_material']['organico']['contribucion_ce'],
                        'con_gradiente_regional':coef>0})
                if not torch.isfinite(loss):raise RuntimeError('Pérdida regional/mapas no finita')
                scaler.scale(loss).backward()
                xa,ya,ia=next(itaux);xa,_,auga=aumentar(xa.cuda().float()/255);ya=ya.cuda()
                with torch.autocast('cuda',dtype=torch.float16,enabled=amp):logits=m.clasificar(xa)
                la=F.cross_entropy(logits.float(),ya)
                if not torch.isfinite(la):raise RuntimeError('Pérdida global no finita')
                scaler.scale(.5*la).backward();scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
                scaler.step(opt);scaler.update();pasos+=1
                if pasos_adam(opt)!=2299+pasos:raise RuntimeError('Actualización omitida')
                traza.update(huella({'indices_train':indices,'aumentos_train':aug,'indices_aux':ia,'aumentos_aux':auga}).encode())
                tot+=ld.item()*len(x);glob+=la.item()*len(x);reg+=lr.item()*len(x);n+=len(x)
                den+=d['denominador_pesos'] if d['regiones'] else 0
                for c in CLASES:
                    for k in por[c]:por[c][k]+=d['por_material'][c][k]
            metricas,_=evaluar(m,dlval,'cuda',amp,base)
            h=traza.hexdigest();esperada=diagnostico['trazas_preparadas'][rama][epoca-20]['huella_orden_y_aumentos']
            if h!=esperada:raise RuntimeError('Orden o aumentos distintos al plan preparado')
            if por['organico']['regiones']!=1 or len(eventos_organicos)!=1:raise RuntimeError('Presencia orgánica inesperada')
            historia.append({'epoca':epoca,'actualizaciones':pasos,'paso_adamw':pasos_adam(opt),
                'learning_rate':opt.param_groups[0]['lr'],'loss_mapas':tot/n,'loss_global':glob/n,
                'loss_regional_media_lotes_ponderada_imagen':reg/n,'coeficiente_regional':coef,
                'region_organica':{'presencia':1,'peso':float(w[5]),'eventos':eventos_organicos},
                'contribucion_ce_regional_por_material':{c:{**d,'contribucion_ce':d['numerador_ponderado']/den if den else 0.} for c,d in por.items()},
                'map50_val':metricas['map50'],'map50_95_val':metricas['map50_95'],'huella_orden_y_aumentos':h})
            mejora=metricas['map50']>mejor
            if mejora:mejor=metricas['map50'];mejor_epoca=epoca
            estado={**ck,'configuracion':m.configuracion_mapas,'pesos':m.state_dict(),'optimizador':opt.state_dict(),
                'scaler':scaler.state_dict(),**estado_aleatorio(rng,ra),'epoca':epoca,'mejor_epoca':mejor_epoca,
                'mejor_map':mejor,'historial':historia,'config_entrenamiento':{**base,'nombre':f"{cfg['nombre']}_{rama}",
                    'supervision_regional':{'coeficiente':coef,'version':VERSION,'pesos_train':w.cpu().tolist()}},
                'protocolo_regional':cfg,'sha256_origen':cfg['sha256_checkpoint_origen'],'hashes_codigo':codigos,
                'huella_regiones':huella(regiones),'actualizaciones_desde_origen':pasos}
            memoria_previa=anterior['reanudacion'] if anterior else {}
            estado['reanudacion']={'version':1,'rama':rama,'epoca_completa':True,
                'commit_previo_sha256':previo,'sha256_diagnostico':diagnostico['sha256_archivo'],
                'estado_inicial_origen':inicial,'huellas_estado':huellas_estado(estado),
                'segundos_acumulados':acumulado+time.perf_counter()-inicio,
                'memoria_gpu_max_allocated_bytes':max(memoria_previa.get('memoria_gpu_max_allocated_bytes',0),torch.cuda.max_memory_allocated()),
                'memoria_gpu_max_reserved_bytes':max(memoria_previa.get('memoria_gpu_max_reserved_bytes',0),torch.cuda.max_memory_reserved())}
            previo=diario.confirmar(estado,previo)
            registros.append(json.loads(diario.registro(epoca).read_text(encoding='utf-8')))
            diario.actualizar_alias(registros)
            print(f'{rama} época {epoca}, paso {pasos}/1452, AP50 {metricas["map50"]:.4%}',flush=True)
        torch.cuda.synchronize()
        _,registros,_=diario.leer();diario.actualizar_alias(registros)
        publicar(diario.directorio/'ejecuciones'/f'{runid}.fin.json',{'rama':rama,'epoca':epoca,
            'actualizaciones_comprometidas':pasos,'commit_sha256':previo})
        del m,opt,scaler,estado,raw,f,lr,ld,loss,la,logits,xa,x,y,ya
        torch.cuda.empty_cache()
    for rama in cfg['ramas']:
        diario=DiarioRama(RAIZ,cfg,rama,diagnostico,codigos);final,regs,_=diario.leer()
        if final is None or final['actualizaciones_desde_origen']!=1452:
            print('Rama pendiente de completar:',rama,flush=True);return
        re=final['reanudacion']
        resultados[rama]={'estado_inicial':re['estado_inicial_origen'],'actualizaciones':1452,'historial':final['historial'],
            'mejor_epoca':final['mejor_epoca'],'mejor_map50':final['mejor_map'],'criterio_mejor':cfg['criterio_mejor'],
            **{k:re[k] for k in ['memoria_gpu_max_allocated_bytes','memoria_gpu_max_reserved_bytes']},
            'segundos':re['segundos_acumulados'],'checkpoints_epocas':regs,
            'checkpoint_final':f"checkpoints/{diario.nombre}_ultimo.pt",'checkpoint_mejor':f"checkpoints/{diario.nombre}_mejor.pt"}
    if resultados['A']['estado_inicial']!=resultados['B']['estado_inicial']:raise RuntimeError('Estados iniciales distintos')
    if any(sha(RAIZ/p)!=v for p,v in protegidos.items()):raise RuntimeError('Origen/selección alterados')
    publicar(salida,{'estado':'entrenamiento_completado_evaluacion_pendiente','configuracion':cfg,
        'estadistica_regiones':estadistica,'ramas':resultados,'test_evaluado':False,'cambio_predeterminado':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--entrenar',action='store_true');p.add_argument('--reanudar',action='store_true')
    p.add_argument('--rama',choices=['A','B','ambas'],default='ambas')
    p.add_argument('--hasta-epoca',type=int,default=31);a=p.parse_args()
    cfg=json.loads(a.config.read_text(encoding='utf-8'));validar_protocolo(cfg)
    if a.entrenar:entrenar(cfg,None if a.rama=='ambas' else [a.rama],a.reanudar,a.hasta_epoca)
    else:print(json.dumps({'protocolo':protocolo(cfg),'entrenamiento_lanzado':False,
        'siguiente':'Ejecutar scripts/diagnosticar_supervision_regional.py; revisar diagnóstico antes de habilitar entrenamiento.'},ensure_ascii=False))


if __name__=='__main__':main()
