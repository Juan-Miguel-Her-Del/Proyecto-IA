"""V31: diagnóstico, control de reanudación y experimento limitado por instancias."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,json,re,subprocess,sys,tempfile,time,uuid
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.experimento_v31 import (OUT,CONFIG,codigo,contexto,planificar,contrato,iniciar,paso,instantanea,comprobar,Diario)
from ecowaste.incorporacion import leer,sha,publicar_bytes
from ecowaste.reanudacion_regional import publicar,bloqueo
from ecowaste.backend_numerico import aplicar_backend,politica,registro_efectivo
from ecowaste.reproducibilidad_v20 import comparar_arbol
from ecowaste.detector_consultas import perdida_consultas,postprocesar_consultas
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.ablacion_fondo import resumir_deteccion,huella
from ecowaste.detector_hibrido import mascara_letterbox


def preparar():
    cfg,ht,nu,gl,pm=contexto();plan=planificar(cfg,ht,nu,gl)
    publicar(OUT/'plan.pt',plan,tensor=True);publicar(OUT/'protocolo.json',contrato(cfg,plan))
    archivos={}
    for folder in ['checkpoints','ecowaste','scripts','pruebas','configuracion','documentacion','datos/procesados','datos/nuevos/v27_2/proyecto/seleccion_72']:
        for p in (RAIZ/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and 'v31' not in p.name:
                archivos[p.relative_to(RAIZ).as_posix()]=sha(p)
    # Certificados de históricos conservados como evidencia, no regenerados.
    for p in (RAIZ/'resultados').glob('*/integridad*.json'):archivos[p.relative_to(RAIZ).as_posix()]=sha(p)
    publicar(OUT/'protegidos.json',archivos)
    counts={}
    for name,ds in [('historico',ht),('nuevo',nu)]:
        counts[name]={'imagenes':len(ds.registros),'objetos':sum(len(o['boxes']) for o in ds.objetivos),
            'por_material':torch.bincount(torch.cat([o['labels'] for o in ds.objetivos]),minlength=6).tolist(),
            'max_por_imagen':max(len(o['boxes']) for o in ds.objetivos)}
    publicar(OUT/'datos_y_pesos.json',{'conteos':counts,'pesos_material':pm.cpu().tolist(),'priores_wh':plan['priores_wh'].tolist(),
        'peso_muestreo_maximo':float(plan['pesos_muestreo'].max()),'clasificador_intacto':True,'supervision':'cajas GT, sin umbral de tamaño ni rasterización'})
    print('Preparado',counts,flush=True)


def pruebas():
    env=os.environ.copy();env.pop('CUBLAS_WORKSPACE_CONFIG',None);res={}
    for n,cmd in [('suite',[sys.executable,'-m','unittest','discover','-s','pruebas']),
                  ('codec',[str(Path.home()/'.venvs/EcoWasteAnotacion/Scripts/python.exe'),'-m','unittest','pruebas.test_datos_y_entrenamiento_v29'])]:
        r=subprocess.run(cmd,cwd=RAIZ,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf8',errors='replace')
        p=OUT/f'pruebas/{n}.log';publicar_bytes(p,r.stdout.encode('utf8'));k=re.search(r'Ran (\d+) tests',r.stdout)
        res[n]={'returncode':r.returncode,'pruebas':int(k[1]) if k else 0,'sha256':sha(p)};print(n,res[n],flush=True)
    ok=res['suite']['returncode']==0 and res['suite']['pruebas']>=472 and res['codec']['returncode']==0
    publicar(OUT/'pruebas.json',{'aprobada':ok,'codigo':codigo(),'resultados':res})
    if not ok:raise RuntimeError('Pruebas fallidas')


def controles():
    cfg,ht,nu,gl,pm=contexto();plan=torch.load(OUT/'plan.pt',weights_only=True);c=contrato(cfg,plan)
    if c!=leer(OUT/'protocolo.json') or not leer(OUT/'pruebas.json')['aprobada']:raise ValueError('Pruebas/protocolo pendientes')
    res={}
    with aplicar_backend(politica('completo')):
        for rama in ['A','B']:
            refs=[];comps=[];trs={}
            with tempfile.TemporaryDirectory(dir=OUT,prefix='descartar_') as td:
                for camino in ['C1','C2','R']:
                    m,opt,sc,rg,ra=iniciar(rama,cfg,plan);hist=[]
                    for k in range(4):
                        tr,detalle=paso(m,opt,sc,ht,nu,gl,pm,plan,k,rg,ra,True);hist.append(tr)
                        if camino=='C1':refs.append(detalle)
                        else:
                            a=comparar_arbol(refs[k],detalle);publicar(OUT/f'controles/{rama}/{camino}_{k+1}.json',a)
                            comps.append({'camino':camino,'paso':k+1,'exacto':a['exacto'],'tolerancia':a['dentro_tolerancia']})
                        if camino=='R' and k==1:
                            ck=instantanea(m,opt,sc,rg,ra,rama,2,hist,c,None);p=Path(td)/'dos.pt';publicar(p,ck,tensor=True)
                            ck=torch.load(p,weights_only=True);comprobar(ck,c,rama);del m,opt,sc
                            m,opt,sc,rg,ra=iniciar(rama,cfg,plan,ck)
                    trs[camino]=hist;del m,opt,sc;torch.cuda.empty_cache()
            res[rama]={'aprobada':all(a['tolerancia'] for a in comps),'comparaciones':comps,'trazas':trs,'pesos_descartados':True}
            if not res[rama]['aprobada']:raise RuntimeError('Reanudación divergente')
    iguales=all(a['entradas_sha256']==b['entradas_sha256'] for a,b in zip(res['A']['trazas']['C1'],res['B']['trazas']['C1']))
    publicar(OUT/'controles/reanudacion.json',{'aprobada':iguales,'ramas':res,'codigo':codigo(),'rtol':1e-4,'atol':1e-5})
    if not iguales:raise ValueError('Entradas A/B distintas')
    print('Reanudación comprobada',flush=True)


@torch.inference_mode()
def evaluar_modelo(m,imagenes,objetivos):
    m.eval();ps=[]
    for k in range(0,len(objetivos),8):
        os=objetivos[k:k+8];v=mascara_letterbox([o['geometria'] for o in os],'cuda')
        with torch.autocast('cuda',dtype=torch.float16):s=m(imagenes[k:k+8].cuda().float()/255,v)
        ps.extend(postprocesar_consultas(s,[o['geometria'] for o in os],.05,True))
    return resumir_deteccion(evaluar_detecciones(ps,objetivos)),ps


def memorizar():
    cfg,ht,nu,gl,pm=contexto();plan=torch.load(OUT/'plan.pt',weights_only=True)
    if leer(OUT/'protocolo.json')['codigo']!=codigo() or not leer(OUT/'controles/reanudacion.json')['aprobada']:raise ValueError('Controles pendientes')
    ids=[]
    for cl in [5,3,2,4,1,0]:
        ix=min((r['id'],i) for i,r in enumerate(ht.registros) if bool((ht.objetivos[i]['labels']==cl).any()))[1]
        if ix not in ids:ids.append(ix)
    for iid,i in sorted((r['id'],i) for i,r in enumerate(ht.registros)):
        o=ht.objetivos[i];centros=(o['boxes'][:,:2]+o['boxes'][:,2:])/2
        cerca=any(int(o['labels'][a])==int(o['labels'][b]) and float((centros[a]-centros[b]).norm())<=.2 for a in range(len(centros)) for b in range(a+1,len(centros)))
        if cerca and i not in ids:ids.append(i)
        if len(ids)>=8:break
    publicar(OUT/'controles/seleccion_memorizacion.json',{'indices':ids,'ids':[ht.registros[i]['id'] for i in ids],'gt':sum(len(ht.objetivos[i]['boxes']) for i in ids),'regla':'mínimo ID por clase + dos escenas con objetos cercanos del mismo material'})
    x=ht.imagenes[ids].cuda().float()/255;os=[ht.objetivos[i] for i in ids];v=ht.validos[ids].cuda();res={}
    with aplicar_backend(politica('completo')):
        for rama in ['A','B']:
            m,opt,sc,rg,ra=iniciar(rama,cfg,plan);hist=[];inicio=time.perf_counter()
            for k in range(cfg['memorizacion']['pasos']):
                m.train();opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):s=m(x,v)
                l,d=perdida_consultas(s,os,'densidad_ponderada',pm)
                if not torch.isfinite(l):raise RuntimeError('Memorización no finita')
                sc.scale(l).backward();sc.unscale_(opt);norma=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True,foreach=False);sc.step(opt);sc.update()
                hist.append({'paso':k+1,'loss':float(l.detach()),'norma':float(norma),**d})
                if (k+1)%50==0:print('Memoria',rama,k+1,'loss',float(l.detach()),flush=True)
            metricas,ps=evaluar_modelo(m,ht.imagenes[ids],os)
            # TP por clase usando evaluador completo, sin usar GT para inferir.
            detalle=evaluar_detecciones(ps,os);tps={c:d['por_umbral'][0]['tp'] for c,d in detalle['por_clase'].items()}
            gate=cfg['memorizacion'];ok=metricas['map50']>=gate['map50_minimo'] and metricas['recall50']>=gate['recall_minimo'] and min(tps.values())>=1
            res[rama]={'aprobada':ok,'metricas':metricas,'tp_por_material':tps,'historial':hist,'segundos':time.perf_counter()-inicio,'pesos_descartados':True}
            del m,opt,sc;torch.cuda.empty_cache();print('Memorización',rama,metricas,flush=True)
    publicar(OUT/'controles/memorizacion.json',{'aprobada':all(r['aprobada'] for r in res.values()),'ramas':res,'codigo':codigo()})
    if not all(r['aprobada'] for r in res.values()):raise RuntimeError('Falla aprendizaje pequeño; no entrenamiento completo')


def entrenar(rama,hasta):
    cfg,ht,nu,gl,pm=contexto();plan=torch.load(OUT/'plan.pt',weights_only=True);c=contrato(cfg,plan)
    if c!=leer(OUT/'protocolo.json') or hasta not in cfg['checkpoints'][1:]:raise ValueError('Protocolo/presupuesto distinto')
    for n in ['pruebas','controles/reanudacion','controles/memorizacion']:
        z=leer(OUT/(n+'.json'))
        if not z['aprobada'] or z['codigo']!=codigo():raise ValueError('Controles no aprobados')
    diario=Diario(rama,c)
    with bloqueo(diario.base/'entrenar.lock'),aplicar_backend(politica('completo')):
        ck,previo=diario.leer();m,opt,sc,rg,ra=iniciar(rama,cfg,plan,ck);hist=copy.deepcopy(ck['historial']) if ck else [];k0=ck['paso'] if ck else 0
        if not ck:previo=diario.guardar(instantanea(m,opt,sc,rg,ra,rama,0,hist,c,previo))
        if k0>=hasta:raise ValueError('Etapa ya completada')
        if k0<400 and hasta>400:raise ValueError('Primero completar la etapa 400')
        if k0>=400 and not leer(diario.base/'puerta_400.json')['aprobada']:raise ValueError('No hay señal de aprendizaje')
        intento=uuid.uuid4().hex;inicio=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        publicar(diario.base/f'intentos/{intento}_inicio.json',{'paso_inicial':k0,'backend':registro_efectivo(opt),'huellas_restauradas':ck['huellas'] if ck else None})
        for k in range(k0,hasta):
            tr,_=paso(m,opt,sc,ht,nu,gl,pm,plan,k,rg,ra);p=diario.base/f'trazas/{intento}/{k+1:04d}.json';publicar(p,tr)
            hist.append({**tr,'archivo':p.relative_to(RAIZ).as_posix(),'sha256':sha(p)})
            if k+1 in cfg['checkpoints']:previo=diario.guardar(instantanea(m,opt,sc,rg,ra,rama,k+1,hist,c,previo))
            if (k+1)%50==0:print(rama,k+1,'/',cfg['actualizaciones'],'loss',tr['loss_total'],flush=True)
        publicar(diario.base/f'intentos/{intento}_fin.json',{'hasta':hasta,'actualizaciones':hasta-k0,'segundos':time.perf_counter()-inicio,'pico_gpu_bytes':torch.cuda.max_memory_allocated()})
        if hasta==400:
            r=1-sum(h['loss_total'] for h in hist[-40:])/sum(h['loss_total'] for h in hist[:40]);ok=r>=cfg['puerta_400']['reduccion_loss_minima']
            publicar(diario.base/'puerta_400.json',{'aprobada':ok,'reduccion_loss':r,'train_only':True})
            if not ok:raise RuntimeError('Puerta train fallida')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fase',choices=['preparar','pruebas','controles','memorizar','entrenar'],required=True)
    p.add_argument('--rama',choices=['A','B']);p.add_argument('--hasta',type=int,choices=[400,800,1600,2400],default=400);a=p.parse_args()
    try:
        if a.fase=='preparar':preparar()
        elif a.fase=='pruebas':pruebas()
        elif a.fase=='controles':controles()
        elif a.fase=='memorizar':memorizar()
        else:entrenar(a.rama,a.hasta)
    except Exception:
        import traceback
        publicar(OUT/f'incidencias/{uuid.uuid4().hex}.json',{'error':traceback.format_exc(),'entrenamiento_detenido':True});raise


if __name__=='__main__':main()
