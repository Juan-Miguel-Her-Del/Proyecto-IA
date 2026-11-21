"""Diagnóstico regional train-only, regresión congelada y preparación A/B sin entrenarla."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ecowaste.detector_mapas import DetectorMapas,perdida_mapas,cajas_desde_mapas
from ecowaste.datasets import letterbox
from ecowaste.mosaico import predecir_mosaico
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.ablacion_fondo import huella,pasos_adam
from ecowaste.supervision_regional import perdida_regional,resumen_confusion
from scripts.entrenar_clasificador import escribir_json
from scripts.supervision_regional_materiales import (CONFIG,contexto,restaurar,trazar_sin_entrenar,
    sha,hashes_codigo,protocolo,evaluar_regiones,loaders)


@torch.inference_mode()
def regresion(ck,train,indices,preparacion):
    # Fuente exacta de HEAD capturada antes de editar; no una implementación duplicada a mano.
    codigo=(preparacion/'detector_mapas_previo.py').read_bytes()
    previo=json.loads((preparacion/'estado_previo.json').read_text(encoding='utf-8'))
    import hashlib
    if hashlib.sha256(codigo).hexdigest()!=previo['sha256_codigo_previo']:raise ValueError('Baseline de código alterada')
    ns={'__name__':'ecowaste._mapas_previo'};exec(compile(codigo,'detector_mapas_previo.py','exec'),ns)
    viejo=ns['DetectorMapas'](**ck['configuracion']).cuda().eval()
    nuevo=DetectorMapas(**ck['configuracion']).cuda().eval()
    viejo.load_state_dict(ck['pesos'],strict=True);nuevo.load_state_dict(ck['pesos'],strict=True)
    if huella(viejo.state_dict())!=huella(nuevo.state_dict()):raise ValueError('Parámetros modificados')
    comprobaciones=[]
    for i in indices:
        r=train.base.origen.registros[i]
        with Image.open(RAIZ/r['ruta']) as im:imagen=im.convert('RGB')
        x,_=letterbox(imagen,384);x=x[None].cuda()
        for amp in [False,True]:
            with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                a=viejo(x);b=nuevo(x);b2,f=nuevo(x,True)
            if not torch.equal(a,b) or not torch.equal(a,b2):raise ValueError('Logits predeterminados diferentes')
            aa=cajas_desde_mapas(a,.05,2,[train.base.objetivos[i]['geometria']])
            bb=cajas_desde_mapas(b,.05,2,[train.base.objetivos[i]['geometria']])
            if huella(aa)!=huella(bb):raise ValueError('Cajas una vista diferentes')
            aa=predecir_mosaico(viejo,imagen,.05,amp=amp)
            bb=predecir_mosaico(nuevo,imagen,.05,amp=amp)
            if huella(aa)!=huella(bb):raise ValueError('Cajas cinco vistas diferentes')
            comprobaciones.append({'id_train':r['id'],'amp':amp,'logits_exactos':True,
                'cajas_una_vista_exactas':True,'cajas_cinco_vistas_exactas':True})
    return {'aprobada':True,'parametros':sum(p.numel() for p in nuevo.parameters()),
        'strict_load':True,'huella_pesos':huella(nuevo.state_dict()),'casos':comprobaciones,
        'alcance':'Comparación exacta en seis escenas train, FP32 y AMP, una/cinco vistas; no prueba universal de todos los dispositivos.'}


def memorizar(train,aux,regiones,w,pm,indices,ia,pasos=150):
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    m=DetectorMapas(auxiliar_clasificacion=True).cuda()
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0)
    x=train.base.imagenes[indices].cuda().float()/255;y=train.mascaras[indices].cuda()
    xa=torch.stack([aux[i][0] for i in ia]).cuda().float()/255;ya=aux.etiquetas[ia].cuda()
    rs=[regiones[i] for i in indices];historia=[];inicio=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for paso in range(1,pasos+1):
        m.train();opt.zero_grad(set_to_none=True)
        raw,f=m(x,True);lr,detalle,pr=perdida_regional(m,f,rs,w)
        ld=perdida_mapas(raw,y,pm,coeficiente_fondo=3.)
        la=F.cross_entropy(m.clasificar(xa),ya);loss=ld+.5*la+.25*lr
        if not torch.isfinite(loss):raise RuntimeError('Memorización no finita')
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
        if paso%25==0 or paso==pasos:
            m.eval()
            with torch.no_grad():
                raw,f=m(x,True);lr,detalle,pr=perdida_regional(m,f,rs,w)
                yp=pr['logits'].argmax(1);et=pr['labels'];cm=torch.bincount(6*et+yp,minlength=36).reshape(6,6)
                acc=(m.clasificar(xa).argmax(1)==ya).float().mean().item()
                pred=cajas_desde_mapas(raw,.35,2,[train.base.objetivos[i]['geometria'] for i in indices])
                ev=evaluar_detecciones(pred,[train.base.objetivos[i] for i in indices])
            historia.append({'paso':paso,'loss':loss.item(),'ce_regional':lr.item(),
                'accuracy_regional':(yp==et).float().mean().item(),'accuracy_auxiliar_global':acc,
                'map50_memorizacion':ev['map50'],'recall50_memorizacion':ev['recall50'],
                'precision50_memorizacion':ev['precision50'],'contribuciones':detalle['por_material']})
            print('Memorización:',{k:v for k,v in historia[-1].items() if k!='contribuciones'},flush=True)
    torch.cuda.synchronize();fin=historia[-1]
    resultado={'aprobada':fin['accuracy_regional']==1 and fin['accuracy_auxiliar_global']==1
        and fin['map50_memorizacion']>=.8 and fin['recall50_memorizacion']>=.9 and fin['precision50_memorizacion']>=.7,
        'criterios_prefijados':{'accuracy_regional':1,'accuracy_auxiliar_global':1,'map50_min':.8,'recall_min':.9,'precision_min':.7},
        'pasos':pasos,'precision':'FP32; prueba funcional, no réplica del entrenamiento AMP',
        'inicio':'aleatorio propio semilla42; no el origen A/B','historial':historia,
        'confusion_regional_final':resumen_confusion(cm),'segundos':time.perf_counter()-inicio,
        'memoria_gpu_max_allocated_bytes':torch.cuda.max_memory_allocated(),
        'pesos_guardados':False,'pesos_reutilizados':False,'solo_train':True,
        'limite':'Memorizar seis escenas fáciles de train no demuestra generalización.'}
    del m,opt,loss,raw,f;gc.collect();torch.cuda.empty_cache()
    return resultado


def medir_coste(ck,train,aux,regiones,w,pm,indices,ia):
    resultado={}
    for rama,coef in [('A',0.),('B',.25)]:
        m,opt,scaler,rng,ra,inicial=restaurar(ck);m.train()
        dl,da=loaders(train,aux,rng,ra,solo_indices=True)
        # Mismo primer lote y muestreo de la continuación, con tamaños reales 8/16.
        itaux=iter(da);indices=next(iter(dl))[0].tolist();ia=next(itaux)[0].tolist()
        x=train.base.imagenes[indices].cuda().float()/255;y=train.mascaras[indices].cuda()
        xa=torch.stack([aux[i][0] for i in ia]).cuda().float()/255;ya=aux.etiquetas[ia].cuda()
        tiempos=[];conteos=[];torch.cuda.reset_peak_memory_stats()
        for j in range(4):
            opt.zero_grad(set_to_none=True);torch.cuda.synchronize();inicio=time.perf_counter()
            with patch.object(m,'caracteristicas',wraps=m.caracteristicas) as spy:
                with torch.autocast('cuda',dtype=torch.float16):
                    raw,f=m(x,True);lr,_,_=perdida_regional(m,f,[regiones[i] for i in indices],w,con_gradiente=coef>0)
                ld=perdida_mapas(raw,y,pm,coeficiente_fondo=3.)
                scaler.scale(ld+coef*lr if coef else ld).backward()
                llamadas_taco=spy.call_count
                with torch.autocast('cuda',dtype=torch.float16):la=F.cross_entropy(m.clasificar(xa).float(),ya)
                scaler.scale(.5*la).backward()
                conteos.append({'taco':llamadas_taco,'total_con_lote_realwaste_distinto':spy.call_count})
            torch.cuda.synchronize();tiempos.append(time.perf_counter()-inicio)
        if any(c!={'taco':1,'total_con_lote_realwaste_distinto':2} for c in conteos):raise ValueError('Extractor duplicado')
        resultado[rama]={'coeficiente':coef,'segundos_forward_backward_media_sin_primero':sum(tiempos[1:])/3,
            'memoria_gpu_max_allocated_bytes':torch.cuda.max_memory_allocated(),
            'memoria_gpu_max_reserved_bytes':torch.cuda.max_memory_reserved(),'llamadas_extractor':conteos,
            'paso_adamw_sin_cambios':pasos_adam(opt),'estado_inicial':inicial,'actualizaciones_optimizador':0,
            'lote_taco':len(x),'lote_global':len(xa),'indices_taco':indices,'indices_auxiliar_global':ia,
            'limite':'Microbenchmark local de forward/backward, sin AdamW, aumentos ni carga de datos; no coste total de A/B.'}
        del m,opt,scaler,raw,f,ld,lr,la,spy,x,y,xa,ya
        gc.collect()
        torch.cuda.empty_cache()
    return resultado


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--salida',type=Path)
    a=p.parse_args()
    cfg=json.loads(a.config.read_text(encoding='utf-8'))
    a.salida=a.salida or RAIZ/cfg['diagnostico']
    if a.salida.exists():raise FileExistsError('Diagnóstico existente; usar otra salida')
    inicio=time.perf_counter()
    suite=unittest.defaultTestLoader.discover(str(RAIZ/'pruebas'),pattern='test_*.py')
    torch.set_num_threads(4);res=unittest.TextTestRunner(verbosity=1).run(suite)
    pruebas={'ejecutadas':res.testsRun,'fallos':len(res.failures),'errores':len(res.errors),'omitidas':len(res.skipped),'aprobadas':res.wasSuccessful() and not res.skipped}
    if not pruebas['aprobadas']:
        escribir_json(a.salida,{'aprobada':False,'pruebas':pruebas,'entrenamiento_ab_lanzado':False});raise RuntimeError('Pruebas fallidas')
    ck,train,aux,regiones,w,estadistica,pm=contexto(cfg)
    preparacion=RAIZ/'resultados/supervision_regional_preparacion'
    escribir_json(a.salida.with_name(a.salida.stem+'_regiones_train.json'),estadistica)
    fijo=json.loads((RAIZ/'resultados/diagnostico_auxiliar_clasificacion.json').read_text(encoding='utf-8'))
    indices=[next(i for i,r in enumerate(train.base.origen.registros) if r['id']==ident) for ident in fijo['ids_taco_train']]
    ia=[aux.ids.index(ident) for ident in fijo['ids_realwaste_train']]
    if set(torch.cat([regiones[i]['labels'] for i in indices]).tolist())!=set(range(6)):raise ValueError('Memorización no cubre los seis materiales')
    reg=regresion(ck,train,indices,preparacion)
    memoria=memorizar(train,aux,regiones,w,pm,indices,ia)
    if not memoria['aprobada']:
        escribir_json(a.salida,{'aprobada':False,'pruebas':pruebas,'memorizacion':memoria,'entrenamiento_ab_lanzado':False})
        raise RuntimeError('Falló memorización; no preparar entrenamiento completo')
    coste=medir_coste(ck,train,aux,regiones,w,pm,indices,ia)
    trazas={r:trazar_sin_entrenar(ck,train,aux) for r in cfg['ramas']}
    if trazas['A']!=trazas['B']:raise ValueError('Trazas A/B diferentes')
    m,opt,scaler,rng,ra,inicial=restaurar(ck);confusion=evaluar_regiones(m,train,regiones,w)
    previo=json.loads((preparacion/'estado_previo.json').read_text(encoding='utf-8'))
    intactos={p:sha(RAIZ/p)==v for p,v in previo['sha256_protegidos'].items()}
    if not all(intactos.values()):raise ValueError('Artefactos anteriores cambiados')
    resultado={'aprobada':True,'estado':'diagnostico_preparado_entrenamiento_completo_no_iniciado','protocolo':protocolo(cfg),
        'hashes_codigo':hashes_codigo(),'sha256_checkpoint':sha(RAIZ/cfg['checkpoint_origen']),
        'hashes_datos':ck['hashes_manifiestos'],'huella_mascaras_semanticas_originales':huella(train.mascaras),
        'huella_regiones':huella(regiones),'pruebas':pruebas,'regresion_inferencia':reg,
        'estadistica_regiones':estadistica,'ids_taco_memorizacion':fijo['ids_taco_train'],
        'region_organica_elegible':next({'imagen':r['id'],'id_region':ident,'indice_train':i,'peso':float(w[5])}
            for i,(r,rs) in enumerate(zip(train.base.origen.registros,regiones))
            for ident,y in zip(rs['ids'],rs['labels']) if int(y)==5),
        'ids_auxiliar_global_memorizacion':fijo['ids_realwaste_train'],'memorizacion':memoria,
        'coste_diagnostico':coste,'trazas_preparadas':trazas,'estados_restaurados_iguales':coste['A']['estado_inicial']==coste['B']['estado_inicial'],
        'confusion_regional_train_origen':confusion,'protegidos_intactos':intactos,
        'epoca_origen':19,'paso_adamw_origen':pasos_adam(opt),'lr_restaurado':opt.param_groups[0]['lr'],
        'scaler_restaurado':scaler.state_dict(),'actualizaciones_ab_ejecutadas':0,'test_evaluado':False,
        'entrenamiento_ab_lanzado':False,'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),
        'duracion_segundos':time.perf_counter()-inicio}
    escribir_json(a.salida,resultado);print(json.dumps({'aprobada':True,'pruebas':pruebas,'actualizaciones_ab':0},ensure_ascii=False))


if __name__=='__main__':main()
