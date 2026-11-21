"""Verifica la preparación y reproduce val congelada; no entrena ni selecciona."""
import argparse
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from torch.nn import functional as F
from ecowaste.detector_mapas import DetectorMapas,perdida_mapas
from ecowaste.supervision_regional import perdida_regional
from ecowaste.ablacion_fondo import huella,pasos_adam
from ecowaste.poligonos import ContornosTACO
from scripts.entrenar_detector_mapas import DatosMapas
from scripts.entrenar_clasificador import escribir_json
from scripts.supervision_regional_materiales import (CONFIG,sha,hashes_codigo,protocolo,exigir_habilitacion,
    contexto,restaurar,loaders)
from scripts.ablacion_fondo import aumentar
from scripts.evaluar_supervision_regional import evaluar_vistas


def gradientes_amp(cfg):
    ck,train,aux,regiones,w,estadistica,pm=contexto(cfg);resultados={}
    for rama,coef in cfg['ramas'].items():
        m,opt,scaler,rng,ra,inicial=restaurar(ck);m.train()
        dl,da=loaders(train,aux,rng,ra);itaux=iter(da)
        x,y,indices=next(iter(dl));x,y,aug=aumentar(x.cuda().float()/255,y.cuda())
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            raw,f=m(x,True)
            lr,_,_=perdida_regional(m,f,[regiones[i] for i in indices.tolist()],w,aug['flip'].flatten(),con_gradiente=coef>0)
        ld=perdida_mapas(raw,y,pm,coeficiente_fondo=3.)
        scaler.scale(ld+coef*lr if coef else ld).backward()
        xa,ya,ia=next(itaux);xa,_,auga=aumentar(xa.cuda().float()/255)
        with torch.autocast('cuda',dtype=torch.float16):la=F.cross_entropy(m.clasificar(xa).float(),ya.cuda())
        scaler.scale(.5*la).backward();scaler.unscale_(opt)
        norma=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('Gradiente AMP no finito')
        if huella(m.state_dict())!=inicial['pesos'] or huella(opt.state_dict())!=inicial['adamw'] or pasos_adam(opt)!=2299:
            raise ValueError('El diagnóstico alteró pesos/AdamW')
        resultados[rama]={'gradientes_finitos':True,'norma_antes_clip':float(norma),
            'loss_mapas':float(ld.detach()),'loss_regional':float(lr.detach()),'loss_global':float(la.detach()),
            'huella_entradas_y_aumentos':huella({'indices':indices,'aux':ia,'aug':aug,'auga':auga}),
            'actualizaciones_optimizador':0,'paso_adamw':2299,'pesos_y_adamw_intactos':True}
        del m,opt,scaler,raw,f,lr,ld,la,x,y,xa
        torch.cuda.empty_cache()
    if resultados['A']['huella_entradas_y_aumentos']!=resultados['B']['huella_entradas_y_aumentos']:
        raise ValueError('Entradas distintas en prueba AMP A/B')
    return resultados


def main():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'));base=RAIZ/'resultados/supervision_regional_preparacion'
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--salida',type=Path,default=RAIZ/cfg['verificacion'])
    salida=p.parse_args().salida
    if salida.exists():raise FileExistsError('Verificación existente')
    diag=json.loads((RAIZ/cfg['diagnostico']).read_text(encoding='utf-8'))
    if not diag['aprobada'] or diag['hashes_codigo']!=hashes_codigo() or diag['protocolo']!=protocolo(cfg):raise ValueError('Diagnóstico desactualizado')
    if cfg['entrenamiento_completo_habilitado']:raise ValueError('Debe seguir deshabilitado')
    try:exigir_habilitacion(cfg)
    except ValueError as e:bloqueo=str(e)
    else:raise ValueError('No se bloqueó entrenamiento')
    if list((RAIZ/'checkpoints').glob('supervision_regional*')):raise ValueError('Hay checkpoints del experimento sin autorización')
    amp=gradientes_amp(cfg)
    torch.set_num_threads(4)
    ck=torch.load(RAIZ/cfg['checkpoint_origen'],map_location='cpu',weights_only=True)
    m=DetectorMapas(**ck['configuracion']).cuda();m.load_state_dict(ck['pesos'],strict=True);m.eval()
    val=DatosMapas('val',ContornosTACO(RAIZ,ck['config_entrenamiento']['cobertura_minima']))
    resultados={}
    for vistas,ruta in [(1,'resultados/detector_mapas_v6_auxiliar_ext48_predicciones_val.json'),
                        (5,'resultados/detector_mapas_v6_mosaico_predicciones_val.json')]:
        ev,pred=evaluar_vistas(m,val,ck['config_entrenamiento'],vistas)
        actual=[{'id':o['id'],**{k:v.tolist() for k,v in p.items()}} for o,p in zip(val.base.objetivos,pred)]
        antes=json.loads((RAIZ/ruta).read_text(encoding='utf-8'))
        iguales=actual==antes
        if not iguales:raise ValueError(f'Predicciones guardadas difieren: {vistas} vistas')
        resultados[str(vistas)]={**ev,'predicciones_exactas_archivo_anterior':iguales,'referencia':ruta,'imagenes':len(pred)}
    protegidos=json.loads((base/'estado_previo.json').read_text(encoding='utf-8'))['sha256_protegidos']
    if any(sha(RAIZ/p)!=h for p,h in protegidos.items()):raise ValueError('Artefactos anteriores alterados')
    escribir_json(salida,{'aprobada':True,'sha256_diagnostico':sha(RAIZ/cfg['diagnostico']),
        'hashes_codigo':hashes_codigo(),'sha256_script_verificacion':sha(Path(__file__)),
        'gradientes_amp_origen':amp,'val_congelada':resultados,'bloqueo_entrenamiento':bloqueo,
        'actualizaciones_ab':0,'test_evaluado':False,'cambio_predeterminado':False})
    print(json.dumps({'aprobada':True,'val':{v:r['resumen'] for v,r in resultados.items()}},ensure_ascii=False))


if __name__=='__main__':main()
