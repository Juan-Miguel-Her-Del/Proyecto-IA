"""Análisis posthoc de capturas inmutables, sin forward ni selección nueva."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,sys
from pathlib import Path
from types import SimpleNamespace
RAIZ=Path(__file__).resolve().parents[2];sys.path.insert(0,str(RAIZ));OUT=Path(__file__).resolve().parent
import torch
from ecowaste.ejecucion_detector import CacheDeteccion
from ecowaste.datasets import CLASES
from ecowaste.cajas import iou_pares,deshacer_letterbox
from ecowaste.detector_consultas import postprocesar_consultas
from ecowaste.mosaico import proyectar_al_lienzo,combinar_vistas
from ecowaste.incorporacion import leer,sha
from ecowaste.reanudacion_regional import publicar
from ecowaste.backend_numerico import aplicar_backend,politica


def matrices(folder,ds):
    cm=torch.zeros(6,8,dtype=torch.long);cond=torch.zeros(6,6,dtype=torch.long);filas=[]
    paths=sorted((folder/'raw_1v').glob('*.pt'))
    for path in paths:
        r=torch.load(path,weights_only=True);k=int(path.stem)
        for j,meta in enumerate(r['meta']):
            o=ds.objetivos[k+j]
            if o['id']!=meta['id']:raise ValueError('GT y captura distintos')
            g=r['geometrias'][j];b=r['salida']['boxes'][j].float().clone();z=r['salida']['logits'][j].float()
            x0,y0=g['relleno_x']/384,g['relleno_y']/384;x1=x0+g['ancho_original']*g['escala_x']/384;y1=y0+g['alto_original']*g['escala_y']/384
            centros=(b[:,:2]+b[:,2:])/2
            b[:,[0,2]]=b[:,[0,2]].clamp(x0,x1);b[:,[1,3]]=b[:,[1,3]].clamp(y0,y1)
            valid=(b[:,2]>b[:,0])&(b[:,3]>b[:,1])&(centros[:,0]>=x0)&(centros[:,0]<=x1)&(centros[:,1]>=y0)&(centros[:,1]<=y1)
            iu=iou_pares(b,o['boxes']);iu[~valid]=0
            for t,c in enumerate(o['labels'].tolist()):
                q=int(iu[:,t].argmax());v=float(iu[q,t]);p=int(z[q].argmax()) if v>=.5 else 7;pc=int(z[q,:6].argmax());cm[c,p]+=1
                if v>=.5:cond[c,pc]+=1
                filas.append({'imagen':o['id'],'indice_gt':t,'material':CLASES[c],'consulta_mayor_iou':q,'iou':v,'prediccion':p,'material_condicional':pc,'score_material':float(z[q].softmax(-1)[:6].max())})
    return {'matriz_6x8':cm.tolist(),'columnas':CLASES+['ausencia_objeto','sin_geometria_iou50'],
            'matriz_condicional_solo_iou50':cond.tolist(),'objetos':filas,
            'criterio':'Por GT se consulta la caja raw de mayor IoU sin usar el material. No es asignación uno a uno ni AP. GT solo en análisis. Condicional excluye GT sin caja IoU50.'}


def oportunidades(folder,ds):
    cuentas={e:{c:0 for c in CLASES} for e in ['pre_bordes','post_bordes','final']};files={};cpu_diffs=[]
    finales={r['id']:r for r in leer(folder/'predicciones_5v.json')}
    for path in sorted((folder/'raw_5v').glob('*.pt')):
        r=torch.load(path,weights_only=True);k=int(path.stem);o=ds.objetivos[k];meta=r['meta'][0]
        if meta['id']!=o['id']:raise ValueError('Correspondencia distinta')
        gs,zs=r['geometrias'],r['zonas']
        cpu=postprocesar_consultas(r['salida'],gs,.05,True)
        with aplicar_backend(politica('completo')):
            ps=postprocesar_consultas({n:t.cuda() for n,t in r['salida'].items()},gs,.05,True)
        diferencias=[]
        for a,b in zip(cpu,ps):
            if a['scores'].shape==b['scores'].shape:
                diferencias.append(float((a['scores']-b['scores']).abs().max()) if len(a['scores']) else 0.)
            else:diferencias.append(None)
        cpu_diffs.append({'imagen':o['id'],'max_diff_score_por_vista':diferencias,'mismos_scores':all(torch.equal(a['scores'],b['scores']) for a,b in zip(cpu,ps))})
        etapas={'pre_bordes':[],'post_bordes':[]}
        for p,g,z in zip(ps,gs,zs):
            global_b=proyectar_al_lienzo(p['boxes'],g,z,meta['ancho'],meta['alto']);orig=deshacer_letterbox(p['boxes'],g);ok=torch.ones(len(orig),dtype=torch.bool)
            if z[0]>0:ok&=orig[:,0]>1
            if z[1]>0:ok&=orig[:,1]>1
            if z[2]<meta['ancho']:ok&=orig[:,2]<(z[2]-z[0]-1)
            if z[3]<meta['alto']:ok&=orig[:,3]<(z[3]-z[1]-1)
            etapas['pre_bordes'].append((global_b,p['labels']));etapas['post_bordes'].append((global_b[ok],p['labels'][ok]))
        fin=combinar_vistas(ps,gs,zs,meta['ancho'],meta['alto']);serial={n:t.tolist() for n,t in fin.items()}
        if any(serial[n]!=finales[o['id']][n] for n in serial):raise ValueError('Postproceso no reproduce captura')
        for e,partes in etapas.items():
            bb=torch.cat([p[0] for p in partes]);ll=torch.cat([p[1] for p in partes]);iu=iou_pares(bb,o['boxes'])
            for t,c in enumerate(o['labels'].tolist()):cuentas[e][CLASES[c]]+=int(bool(((ll==c)&(iu[:,t]>=.5)).any()))
        iu=iou_pares(fin['boxes'],o['boxes'])
        for t,c in enumerate(o['labels'].tolist()):cuentas['final'][CLASES[c]]+=int(bool(((fin['labels']==c)&(iu[:,t]>=.5)).any()))
        files[path.relative_to(RAIZ).as_posix()]=sha(path)
    return {'por_etapa':cuentas,'criterio':'GT con al menos una caja correcta a IoU50; no asignación uno a uno; softmax sobre CUDA como en captura, sin nuevo forward; reproducción final exacta exigida','capturas_sha256':files,'comparacion_softmax_cpu_cuda':cpu_diffs,'predicciones_finales_reproducidas_exactamente':True}


def main():
    torch.set_num_threads(4);cfg=leer(RAIZ/'configuracion/instancias_v31.json');ht=CacheDeteccion(RAIZ,'train',.1);va=CacheDeteccion(RAIZ,'val',.1)
    nu=SimpleNamespace(**torch.load(RAIZ/cfg['coleccion']/'train.pt',weights_only=True));ds={'historico_train':ht,'nuevo_train':nu,'desarrollo':va}
    res={};plan=torch.load(OUT/'plan.pt',weights_only=True);exposiciones=[]
    for i,o in enumerate(ht.objetivos):
        n=int((plan['historico']==i).sum())
        if bool((o['labels']==5).any()):exposiciones.append({'imagen':o['id'],'instancias_organicas':int((o['labels']==5).sum()),'veces_muestreada_por_rama':n})
    for n in ['A','B']:
        d=leer(OUT/f'metricas_{n}.json');res[n]={}
        for dom in ds:
            p=RAIZ/d['pasos']['2400']['resultados'][dom]['analisis'];res[n][dom]={'final_1v':matrices(p.parent,ds[dom])}
        p=RAIZ/d['mejor']['desarrollo']['analisis'];res[n]['desarrollo']['mejor_1v']=matrices(p.parent,va);res[n]['desarrollo']['oportunidades_mejor_5v']=oportunidades(p.parent,va)
        print(n,'análisis adicional completo',flush=True)
    publicar(OUT/'analisis_adicional.json',{'resultados':res,'exposicion_organicos':exposiciones,'inferencia_repetida':False,'gt_solo_posthoc':True,'test':False})
if __name__=='__main__':main()
