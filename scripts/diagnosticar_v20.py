"""V20: evidencia train-only de gradientes y memorización; nunca lanza A/B."""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time
import unittest
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from torch.nn import functional as F
from ecowaste.detector_mapas import DetectorMapas,perdida_mapas,cajas_desde_mapas
from ecowaste.discriminacion_mapas import (perdida_discriminativa,ce_seleccionada,comparar_gradientes,
    descomponer_original,metricas_instancias)
from ecowaste.ablacion_fondo import huella,pasos_adam,restaurar_aleatorio
from ecowaste.reanudacion_regional import publicar,sha
from ecowaste.evaluacion_detector import evaluar_detecciones
from scripts.preparar_v20 import (CONFIG,leer_cfg,protocolo,huellas_codigo,contexto_train,
    calcular_lote,evaluar_principal_train)
from scripts.supervision_regional_materiales import restaurar,loaders,trazar_sin_entrenar
from scripts.ablacion_fondo import aumentar
from scripts.diagnosticar_supervision_regional import regresion


def proteger(salida):
    archivos=[*list((RAIZ/'resultados').rglob('*')),*list((RAIZ/'checkpoints').glob('*.pt')),
        *list((RAIZ/'configuracion').glob('*.json')),*list((RAIZ/'ecowaste').glob('*.py')),
        *list((RAIZ/'documentacion').glob('*.md'))]
    return {p.relative_to(RAIZ).as_posix():sha(p) for p in archivos if p.is_file() and not p.is_relative_to(salida)}


def gradientes_parametros(m,loss):
    ps=list(m.parameters());grads=torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True)
    return grads,{n:float(g.float().norm()) if g is not None else None for (n,_),g in zip(m.named_parameters(),grads)}


def vector(grads):return torch.cat([g.detach().float().flatten() for g in grads if g is not None])


def margenes_fijos(z,detalle):
    valores=[]
    for f in detalle['filas']:
        if not f['indices_tokens']:continue
        r=z[f['imagen_lote'],1:].permute(1,2,0).reshape(-1,6)
        ix=torch.tensor(f['indices_tokens'],device=z.device);riv=torch.tensor(f['rivales'],device=z.device)
        valores.append(r[ix,f['clase']]-r[ix,riv])
    return torch.cat(valores) if valores else z.new_empty(0)


def probar_gradientes(m,train,regiones,pm,ids):
    m.eval();x=train.base.imagenes[ids].cuda().float()/255;y=train.mascaras[ids].cuda();rs=[regiones[i] for i in ids]
    raw=m(x);nuevo,d=perdida_discriminativa(raw,rs);partes=descomponer_original(raw,y,pm)
    if not torch.equal(partes['total'],perdida_mapas(raw,y,pm)):raise AssertionError('Descomposición original distinta')
    ce7=ce_seleccionada(raw,rs,d);ce6=ce_seleccionada(raw,rs,d,True)
    gn=torch.autograd.grad(nuevo,raw,retain_graph=True)[0]
    refs={'ce_foreground_existente':partes['ce_foreground'],'perdida_mapas_completa':partes['total'],
        'ce7_mismos_tokens_pesos':ce7,'ce6_mismos_tokens_pesos':ce6}
    comparaciones={};grads_refs={}
    for nombre,loss in refs.items():
        g=torch.autograd.grad(loss,raw,retain_graph=True)[0];grads_refs[nombre]=g
        comparaciones[nombre]=comparar_gradientes(gn,g)
        a=gn[:,1:]-gn[:,1:].mean(1,keepdim=True);b=g[:,1:]-g[:,1:].mean(1,keepdim=True)
        comparaciones[nombre]['solo_contrastes_materiales']=comparar_gradientes(a,b)
    gparams,normas=gradientes_parametros(m,nuevo)
    gf,normas_fg=gradientes_parametros(m,partes['ce_foreground'])
    # Comparar el mismo conjunto de parámetros evita que None en la cabeza auxiliar cambie la dimensión.
    pareados=[(a,b) for a,b in zip(gparams,gf) if a is not None and b is not None]
    parametros=comparar_gradientes(torch.cat([a.flatten() for a,b in pareados]),torch.cat([b.flatten() for a,b in pareados]))
    assert all(g is None or torch.isfinite(g).all() for g in gparams)
    assert all(v is None for n,v in normas.items() if n.startswith('clasificador_auxiliar'))
    assert normas['cabezal.2.weight']>0 and normas['tokenizador_detalle.weight']>0 and normas['tokenizador_contexto.weight']>0
    assert float(gn[:,0].abs().sum())==0
    paso=.01/max(float(gn.abs().max()),1e-12);z2=raw.detach()-paso*gn
    antes=margenes_fijos(raw.detach(),d);despues=margenes_fijos(z2,d)
    contraste={'cantidad':len(antes),'margen_gt_rival_medio_antes':float(antes.mean()),
        'margen_gt_rival_medio_despues':float(despues.mean()),
        'fraccion_mejorados':float((despues>antes).float().mean()),'fraccion_no_empeorados':float((despues>=antes).float().mean()),
        'max_cambio_logit':float((z2-raw.detach()).abs().max()),'cambio_fondo':float((z2[:,0]-raw.detach()[:,0]).abs().max()),
        'loss_antes':float(nuevo.detach()),'loss_reseleccionando_despues':float(perdida_discriminativa(z2,rs)[0])}
    # Comprobación adicional en parámetros del Transformer, restaurados exactamente al acabar.
    originales=[p.detach().clone() for p in m.parameters()];huella_antes=huella(m.state_dict());pruebas_descenso=[]
    mayor=max(float(g.abs().max()) for g in gparams if g is not None)
    for max_delta in [1e-4,1e-5]:
        with torch.no_grad():
            for p,v,g in zip(m.parameters(),originales,gparams):
                p.copy_(v)
                if g is not None:p.add_(g,alpha=-max_delta/max(mayor,1e-12))
            despues_loss=float(perdida_discriminativa(m(x),rs)[0])
        pruebas_descenso.append({'max_delta_parametro':max_delta,'loss_nueva_despues':despues_loss,
            'delta':despues_loss-float(nuevo.detach())})
    with torch.no_grad():
        for p,v in zip(m.parameters(),originales):p.copy_(v)
    assert huella(m.state_dict())==huella_antes
    por_clase={}
    for c in range(6):
        fg=(y==c+1);n=int(fg.sum())
        por_clase[str(c)]={'tokens_fg':n,'ce_media':float(F.cross_entropy(raw.float(),y,ignore_index=255,reduction='none')[fg].mean()) if n else None,
            'grad_ce_gt_norma':float(grads_refs['ce_foreground_existente'][:,c+1][fg].norm()) if n else 0.}
    resultado={'ids_train':[train.base.origen.registros[i]['id'] for i in ids],
        'loss_original':{k:float(v.detach()) for k,v in partes.items()},'loss_nueva':float(nuevo.detach()),
        'comparacion_logits':comparaciones,'comparacion_parametros_vs_ce_fg':parametros,
        'normas_parametros_nueva':normas,'normas_parametros_ce_fg':normas_fg,
        'grad_directo_fondo_nuevo_l1':float(gn[:,0].abs().sum()),'margenes':contraste,
        'descenso_local_parametros':pruebas_descenso,'parametros_restaurados_exactos':True,
        'detalle_instancias':d,'ce_original_por_gt':por_clase}
    del raw,nuevo,gparams,gf,partes,grads_refs,x,y;gc.collect();torch.cuda.empty_cache()
    return resultado


def prueba_amp(ck,train,aux,regiones,pm):
    resultados={}
    for rama,coef in [('A',0.),('B',.25)]:
        m,opt,scaler,rng,ra,inicial=restaurar(ck);m.train();dl,da=loaders(train,aux,rng,ra)
        itaux=iter(da);x,y,ix=next(iter(dl));x,y,aug=aumentar(x.cuda().float()/255,y.cuda());opt.zero_grad(set_to_none=True)
        estado_rng=huella({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})
        with torch.autocast('cuda',dtype=torch.float16):
            loss,raw,lo,ln,d=calcular_lote(m,x,y,[regiones[i] for i in ix.tolist()],pm,coef,aug['flip'].flatten())
        assert estado_rng==huella({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})
        scaler.scale(loss).backward();xa,ya,ia=next(itaux);xa,_,auga=aumentar(xa.cuda().float()/255)
        with torch.autocast('cuda',dtype=torch.float16):la=F.cross_entropy(m.clasificar(xa).float(),ya.cuda())
        scaler.scale(.5*la).backward();scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
        assert huella(m.state_dict())==inicial['pesos'] and huella(opt.state_dict())==inicial['adamw'] and pasos_adam(opt)==2299
        resultados[rama]={'inicial':inicial,'loss_original':float(lo.detach()),'loss_nueva':float(ln.detach()),
            'norma_gradiente_antes_clip':float(norm),'gradientes_finitos':True,'rng_no_consumido_por_perdida':True,
            'huella_entradas':huella({'ix':ix,'ia':ia,'aug':aug,'auga':auga}),'paso_adamw':2299,
            'actualizaciones':0,'scaler':scaler.state_dict(),'detalle':d}
        del m,opt,scaler,raw,loss,lo,ln,la,x,y,xa;gc.collect();torch.cuda.empty_cache()
    assert resultados['A']['inicial']==resultados['B']['inicial']
    assert resultados['A']['huella_entradas']==resultados['B']['huella_entradas']
    return resultados


def memorizacion(train,aux,regiones,pm,ids,ia):
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    m=DetectorMapas(auxiliar_clasificacion=True).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0)
    x=train.base.imagenes[ids].cuda().float()/255;y=train.mascaras[ids].cuda();rs=[regiones[i] for i in ids]
    xa=torch.stack([aux[i][0] for i in ia]).cuda().float()/255;ya=aux.etiquetas[ia].cuda();hist=[];inicio=time.perf_counter()
    for paso in range(1,201):
        m.train();opt.zero_grad(set_to_none=True);loss,raw,lo,ln,d=calcular_lote(m,x,y,rs,pm,.25)
        la=F.cross_entropy(m.clasificar(xa),ya);total=loss+.5*la
        total.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
        if paso%25==0:
            m.eval()
            with torch.no_grad():
                raw=m(x);rr=metricas_instancias(raw,rs);ln2,dd=perdida_discriminativa(raw,rs)
                pred=cajas_desde_mapas(raw,.35,2,[train.base.objetivos[i]['geometria'] for i in ids])
                met=evaluar_detecciones(pred,[train.base.objetivos[i] for i in ids])
                acc=float((m.clasificar(xa).argmax(1)==ya).float().mean())
            fila={'paso':paso,'loss_original':float(lo.detach()),'loss_discriminativa':float(ln2),
                'recuperadas_principal':sum(f['recuperada_mayoria_gt'] for f in rr['filas']),
                'recuperadas_condicional':sum(f['condicional_correcta_mayoria'] for f in rr['filas']),
                'instancias':rr['instancias'],'accuracy_global':acc,'map50_mem':met['map50'],'recall_mem':met['recall50'],'precision_mem':met['precision50']}
            hist.append(fila);print('Memorización V20:',fila,flush=True)
    final=hist[-1]
    aprobo=final['recuperadas_principal']==final['instancias'] and final['recuperadas_condicional']==final['instancias'] and final['accuracy_global']==1 and final['map50_mem']>=.8 and final['recall_mem']>=.9 and final['precision_mem']>=.7
    final_principal=evaluar_principal_train(m,train,regiones,False,ids)
    resultado={'aprobada':aprobo,'pasos':200,'seed':42,'inicio':'aleatorio propio, sin pesos de origen ni diagnóstico anterior',
        'criterios_prefijados':{'recuperacion_principal_y_condicional':1.,'accuracy_global':1.,'map50_min':.8,'recall_min':.9,'precision_min':.7},
        'historial':hist,'principal_final':final_principal,'segundos':time.perf_counter()-inicio,
        'pesos_guardados':False,'pesos_reutilizados':False,'solo_train':True,'no_es_evidencia_de_generalizacion':True}
    del m,opt,raw,total,loss,lo,ln,la,x,y,xa;gc.collect();torch.cuda.empty_cache()
    return resultado


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--salida',type=Path);a=p.parse_args()
    cfg=leer_cfg();salida=(a.salida or RAIZ/'resultados/v20_preparacion').resolve()
    if salida.exists():raise FileExistsError('Diagnóstico existente: usar una carpeta nueva')
    torch.set_num_threads(4);inicio=time.perf_counter()
    suite=unittest.defaultTestLoader.discover(str(RAIZ/'pruebas'),pattern='test_*.py');r=unittest.TextTestRunner(verbosity=1).run(suite)
    if not r.wasSuccessful() or r.skipped:raise RuntimeError('Pruebas fallidas; no usar GPU en diagnóstico')
    salida.mkdir(parents=True);publicar(salida/'protegidos.json',proteger(salida))
    pruebas={'ejecutadas':r.testsRun,'aprobadas':True,'fallos':0,'errores':0,'omitidas':0}
    ck,train,aux,regiones,pm,est=contexto_train(cfg)
    historico=json.loads((RAIZ/'resultados/diagnostico_auxiliar_clasificacion.json').read_text(encoding='utf-8'))
    ids=[next(i for i,r in enumerate(train.base.origen.registros) if r['id']==ident) for ident in historico['ids_taco_train']]
    ia=[aux.ids.index(ident) for ident in historico['ids_realwaste_train']]
    m,opt,scaler,rng,ra,inicial=restaurar(ck)
    regres=regresion(ck,train,ids,RAIZ/'resultados/supervision_regional_preparacion')
    publicaciones=[];lotes=[ids,list(range(8))]
    for c in [2,3]:lotes.append([i for i,r in enumerate(regiones) if bool((r['labels']==c).any())][:8])
    pruebas_grad=[probar_gradientes(m,train,regiones,pm,lote) for lote in lotes]
    principal=evaluar_principal_train(m,train,regiones)
    publicar(salida/'principal_train_origen.json',principal)
    amp=prueba_amp(ck,train,aux,regiones,pm)
    memoria=memorizacion(train,aux,regiones,pm,ids,ia)
    trazas={rama:trazar_sin_entrenar(ck,train,aux) for rama in cfg['ramas']}
    assert trazas['A']==trazas['B']
    protegido=json.loads((salida/'protegidos.json').read_text(encoding='utf-8'))
    assert all(sha(RAIZ/p)==h for p,h in protegido.items())
    ort=[g['comparacion_logits']['ce6_mismos_tokens_pesos']['fraccion_ortogonal'] for g in pruebas_grad]
    util=all(g['margenes']['margen_gt_rival_medio_despues']>g['margenes']['margen_gt_rival_medio_antes'] for g in pruebas_grad)
    descenso=sum(any(v['delta']<0 for v in g['descenso_local_parametros']) for g in pruebas_grad)
    aprobada=memoria['aprobada'] and util and descenso>=3 and sorted(ort)[len(ort)//2]>.01
    resultado={'aprobada':aprobada,'estado':'preparacion_train_only_ab_no_iniciado','protocolo':protocolo(cfg),
        'pruebas':pruebas,'revision_informes':['08','13','17','18','19'],'hashes_codigo':huellas_codigo(),
        'hashes_datos':ck['hashes_manifiestos'],'huella_regiones':huella(regiones),
        'huella_mascaras_originales':huella(train.mascaras),'pesos_fg_originales':pm.cpu().tolist(),'regiones_train':est,
        'comparacion_gradientes':pruebas_grad,'senal_adicional':{'margenes_seleccionados_mejoran':util,
            'probes_con_descenso_local_parametros':descenso,'fracciones_ortogonales_vs_ce6_matched':ort},
        'amp':amp,'memorizacion':memoria,'regresion_inferencia_train':regres,
        'trazas_preparadas':trazas,'estado_inicial':inicial,'paso_adamw_origen':pasos_adam(opt),
        'lr':opt.param_groups[0]['lr'],'scaler_origen':scaler.state_dict(),
        'coeficiente_prefijado_antes_val':.25,'validacion_evaluada_en_v20':False,'test_evaluado':False,
        'actualizaciones_ab_ejecutadas':0,'pesos_diagnostico_descartados':True,'historicos_intactos':len(protegido),
        'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),'segundos':time.perf_counter()-inicio}
    publicar(salida/'diagnostico.json',resultado)
    print(json.dumps({'aprobada':aprobada,'senal_adicional':resultado['senal_adicional'],'actualizaciones_ab':0},ensure_ascii=False))
    if not aprobada:raise RuntimeError('Diagnóstico insuficiente: no habilitar entrenamiento')


if __name__=='__main__':main()
