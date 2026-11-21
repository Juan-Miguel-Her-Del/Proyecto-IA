"""Cuatro pasos reales por rama: continuo frente a 2+restauración+2; descartar pesos."""
import copy
import json
from pathlib import Path
import sys
import tempfile
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.ablacion_fondo import huella,estado_aleatorio,pasos_adam
from ecowaste.reanudacion_regional import publicar,sha,huellas_estado
from scripts.preparar_v20 import leer_cfg,protocolo,huellas_codigo,contexto_train
from scripts.supervision_regional_materiales import restaurar,loaders
from scripts.entrenar_v20 import paso_adamw


def estado(ck,m,opt,scaler,rng,ra):
    return copy.deepcopy({**ck,'pesos':m.state_dict(),'optimizador':opt.state_dict(),
        'scaler':scaler.state_dict(),**estado_aleatorio(rng,ra)})


def comparar(a,b):
    maxdelta=0.;todo_exacto=True
    def visitar(x,y):
        nonlocal maxdelta,todo_exacto
        if torch.is_tensor(x):
            x=x.cpu();y=y.cpu();todo_exacto&=torch.equal(x,y)
            if x.is_floating_point():
                maxdelta=max(maxdelta,float((x-y).abs().max()) if x.numel() else 0.)
                torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)
            elif not torch.equal(x,y):raise AssertionError('Estado entero distinto')
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:visitar(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for xx,yy in zip(x,y):visitar(xx,yy)
        else:assert x==y
    visitar(a,b)
    return {'exacto':todo_exacto,'max_delta_absoluto':maxdelta,'rtol':1e-4,'atol':1e-5}


def main():
    cfg=leer_cfg();out=RAIZ/cfg['prueba_corta']
    if out.exists():raise FileExistsError('Prueba corta existente')
    d=json.loads((RAIZ/cfg['diagnostico']).read_text(encoding='utf-8'))
    if not d['aprobada'] or d['hashes_codigo']!=huellas_codigo() or d['protocolo']!=protocolo(cfg):raise ValueError('Diagnóstico no vigente')
    ck,train,aux,regiones,pm,est=contexto_train(cfg)
    m,opt,sc,rng,ra,inicial=restaurar(ck)
    assert inicial==d['estado_inicial']
    dl,da=loaders(train,aux,rng,ra,solo_indices=True);itaux=iter(da);plan=[]
    for (ix,) in dl:
        (ia,)=next(itaux);plan.append((ix,ia))
        if len(plan)==4:break
    r_final=rng.get_state();ra_final=ra.get_state();del m,opt,sc;torch.cuda.empty_cache()
    def paso(k,coef,m,opt,sc,rng,ra):
        ix,ia=plan[k];x=train.base.imagenes[ix];y=train.mascaras[ix]
        xa=torch.stack([aux[int(i)][0] for i in ia]);ya=aux.etiquetas[ia]
        return paso_adamw(m,opt,sc,x,y,ix,xa,ya,ia,regiones,pm,coef)
    ramas={}
    with tempfile.TemporaryDirectory(prefix='v20_piloto_',dir=RAIZ/'datos/cache') as tmp:
        tmp_path=Path(tmp).resolve()
        if not tmp_path.is_relative_to((RAIZ/'datos/cache').resolve()):raise ValueError('Temporal fuera de caché')
        for rama,coef in cfg['ramas'].items():
            m,opt,sc,rng,ra,real=restaurar(ck);assert real==inicial
            rng.set_state(r_final);ra.set_state(ra_final);continuos=[]
            for k in range(4):continuos.append(paso(k,coef,m,opt,sc,rng,ra))
            continuo=estado(ck,m,opt,sc,rng,ra)
            assert pasos_adam(opt)==2303
            del m,opt,sc;torch.cuda.empty_cache()
            m,opt,sc,rng,ra,real=restaurar(ck);rng.set_state(r_final);ra.set_state(ra_final);reiniciados=[]
            for k in range(2):reiniciados.append(paso(k,coef,m,opt,sc,rng,ra))
            cp=tmp_path/f'{rama}_paso2.pt';publicar(cp,estado(ck,m,opt,sc,rng,ra),tensor=True)
            antes=sha(cp);del m,opt,sc;torch.cuda.empty_cache()
            retomado=torch.load(cp,map_location='cpu',weights_only=True)
            m,opt,sc,rng,ra,real=restaurar(retomado);assert real==huellas_estado(retomado)
            for k in range(2,4):reiniciados.append(paso(k,coef,m,opt,sc,rng,ra))
            final=estado(ck,m,opt,sc,rng,ra);assert pasos_adam(opt)==2303 and sha(cp)==antes
            assert [s['huella_entrada'] for s in continuos]==[s['huella_entrada'] for s in reiniciados]
            pesos=comparar(continuo['pesos'],final['pesos']);adam=comparar(continuo['optimizador'],final['optimizador'])
            assert huella(continuo['scaler'])==huella(final['scaler'])
            assert huella({k:continuo[k] for k in ['rng_torch','rng_cuda','rng_loader','rng_auxiliar']})==huella({k:final[k] for k in ['rng_torch','rng_cuda','rng_loader','rng_auxiliar']})
            ramas[rama]={'estado_inicial':inicial,'continuo':continuos,'interrumpido_reanudado':reiniciados,
                'paso_adamw_final':2303,'pesos':pesos,'adamw':adam,'scaler_y_rng_exactos':True,
                'max_delta_loss':max(abs(a['loss_original']-b['loss_original']) for a,b in zip(continuos,reiniciados))}
            print(f'Prueba corta {rama}: cuatro pasos reales y reanudación aprobados; delta pesos={pesos["max_delta_absoluto"]}',flush=True)
            del m,opt,sc,continuo,final,retomado;torch.cuda.empty_cache()
    assert [s['huella_entrada'] for s in ramas['A']['continuo']]==[s['huella_entrada'] for s in ramas['B']['continuo']]
    assert sha(RAIZ/cfg['checkpoint_origen'])==cfg['sha256_checkpoint_origen']
    publicar(out,{'aprobada':True,'hashes_codigo':huellas_codigo(),'protocolo':protocolo(cfg),
        'sha256_diagnostico':sha(RAIZ/cfg['diagnostico']),'ramas':ramas,'actualizaciones_por_trayectoria':4,
        'actualizaciones_totales_diagnosticas':16,'pesos_guardados':False,'pesos_descartados':True,
        'produccion_reanuda_en_limites_de_epoca':True,'piloto_reanuda_con_plan_restante_serializado':True,
        'identidad_numerica':'Comparada solo en las trayectorias cortas indicadas; no se garantiza para 1452 pasos.',
        'test_evaluado':False,'entrenamiento_completo_iniciado':False})


if __name__=='__main__':main()
