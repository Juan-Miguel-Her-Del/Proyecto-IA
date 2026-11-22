"""Captura sin GT y medición posterior exclusivamente sobre los train autorizados."""
from collections import Counter
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from ecowaste.datasets import CLASES,letterbox,geometria_letterbox
from ecowaste.detector_mapas import cajas_desde_mapas
from ecowaste.traza_propuestas import trazar_logits,ETAPAS
from ecowaste.mosaico import ventanas
from ecowaste.cajas import iou_pares
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.ablacion_fondo import huella,metricas_semanticas,resumir_deteccion
from ecowaste.reanudacion_regional import publicar
from ecowaste.incorporacion import sha,leer
from ecowaste.validacion_anotaciones import mask_poligonos,celdas
from scripts.auditar_propuestas_internas import serializar_traza,cargar_traza


@torch.inference_mode()
def capturar(modelo,imagenes,metadatos,carpeta,raiz):
    if any(set(m)!={'id','ruta','ancho','alto','sha256_imagen'} for m in metadatos):raise ValueError('Inferencia no admite GT')
    carpeta,raiz=Path(carpeta),Path(raiz);cert=carpeta/'captura.json'
    clave={'pesos':huella(modelo.state_dict()),'metadatos':huella(metadatos),'entradas':huella(imagenes)}
    if cert.exists():
        c=leer(cert)
        if c['clave']!=clave:raise ValueError('Captura ajena')
        for p,h in c['archivos'].items():
            if sha(carpeta/p)!=h:raise ValueError('Captura alterada')
        return c
    carpeta.mkdir(parents=True,exist_ok=True);archivos={};costes={}
    for v in [1,5]:
        inicio=time.perf_counter();torch.cuda.reset_peak_memory_stats();ps=[]
        for k in range(0,len(metadatos),8 if v==1 else 1):
            p=carpeta/f'logits_{v}v/{k:04d}.pt'
            ms=metadatos[k:k+8] if v==1 else [metadatos[k]]
            if p.exists():
                dato=torch.load(p,map_location='cpu',weights_only=True)
                if dato['metadatos']!=ms or dato['clave']!=clave:raise ValueError('Logits parciales ajenos')
                raw=dato['logits'];gs=dato['geometrias'];zs=dato.get('ventanas')
            else:
                if v==1:
                    x=imagenes[k:k+8].cuda().float()/255
                    gs=[geometria_letterbox(m['ancho'],m['alto'],384) for m in ms];zs=None
                else:
                    m=ms[0]
                    if sha(raiz/m['ruta'])!=m['sha256_imagen']:raise ValueError('Foto alterada')
                    with Image.open(raiz/m['ruta']) as pil:im=pil.convert('RGB')
                    zs=ventanas(im.width,im.height);xgs=[letterbox(im.crop(z),384) for z in zs]
                    gs=[g for _,g in xgs];x=torch.stack([x for x,_ in xgs]).cuda()
                with torch.autocast('cuda',dtype=torch.float16):raw=modelo(x)
                raw=raw.cpu();publicar(p,{'clave':clave,'metadatos':ms,'logits':raw,'geometrias':gs,'ventanas':zs},tensor=True)
            archivos[p.relative_to(carpeta).as_posix()]=sha(p)
            # Conservar softmax en CUDA como en el evaluador histórico. El archivo
            # CPU preserva exactamente los logits FP16, no cambia dónde se procesan.
            if v==1:ps.extend(cajas_desde_mapas(raw.cuda(),.05,2,gs,politica='argmax7'))
            else:
                m=ms[0];q=carpeta/f'trazas_5v/{k:04d}.pt'
                if q.exists():t=cargar_traza(q)
                else:
                    t=trazar_logits(raw.cuda(),gs,zs,m['ancho'],m['alto'],m['id']);publicar(q,serializar_traza(t),tensor=True)
                ps.append(t['final']);archivos[q.relative_to(carpeta).as_posix()]=sha(q)
            if (k+len(ms))%80==0 or k+len(ms)==len(metadatos):print(f'{carpeta.parent.name}/{carpeta.name} {v}v: {k+len(ms)}/{len(metadatos)}',flush=True)
        pp=[{'id':m['id'],**{key:t.tolist() for key,t in pred.items()}} for m,pred in zip(metadatos,ps)]
        q=carpeta/f'predicciones_{v}v.json'
        if q.exists():
            if leer(q)!=pp:raise ValueError('Predicciones distintas en reanudación de captura')
        else:publicar(q,pp)
        archivos[q.name]=sha(q);torch.cuda.synchronize()
        costes[str(v)]={'segundos_captura_incluye_disco_y_observador':time.perf_counter()-inicio,
            'memoria_gpu_max_allocated_bytes':torch.cuda.max_memory_allocated(),
            'memoria_gpu_max_reserved_bytes':torch.cuda.max_memory_reserved()}
    if huella(modelo.state_dict())!=clave['pesos']:raise AssertionError('Inferencia alteró pesos')
    c={'clave':clave,'archivos':archivos,'costes':costes,'gt_utilizado':False,
        'observador_conserva_predicciones':True,'imagenes':len(metadatos)};publicar(cert,c);return c


def cargar_predicciones(carpeta,v):
    rs=leer(Path(carpeta)/f'predicciones_{v}v.json')
    return [{'boxes':torch.tensor(r['boxes'],dtype=torch.float32).reshape(-1,4),
        'labels':torch.tensor(r['labels'],dtype=torch.long),'scores':torch.tensor(r['scores'],dtype=torch.float32)} for r in rs]


def geometria_posthoc(preds,objetivos):
    """Indicadores de cajas; fusión/fragmentación no certifican identidad física."""
    por={c:Counter() for c in CLASES};filas=[]
    for pred,o in zip(preds,objetivos):
        b=pred['boxes'];gt=o['boxes'];ious=iou_pares(b,gt)
        inter=(torch.minimum(b[:,None,2:],gt[None,:,2:])-torch.maximum(b[:,None,:2],gt[None,:,:2])).clamp(min=0).prod(2)
        gtarea=(gt[:,2:]-gt[:,:2]).prod(1).clamp(min=1e-12)
        cob=inter/gtarea[None,:];usados=set()
        for j,c in enumerate(o['labels'].tolist()):
            mismo=pred['labels']==c;razonable=ious[:,j]>=.5
            libres=[i for i in torch.where(mismo&razonable)[0].tolist() if i not in usados]
            if libres:usados.add(max(libres,key=lambda i:float(ious[i,j])));causa='correcto'
            elif bool((~mismo&razonable).any()):causa='material_incorrecto_iou50'
            elif bool(mismo.any()) and float(ious[mismo,j].max())>0:causa='geometria_insuficiente'
            else:causa='sin_caja_correcta_solapada'
            fusion=any(float(cob[i,j])>=.5 and int((cob[i]>=.5).sum())>=2 for i in torch.where(mismo)[0].tolist())
            partes=torch.where(mismo&(cob[:,j]>=.1))[0];fragmentos=False
            if len(partes)>=2:
                bb=b[partes];union=torch.cat([bb[:,:2].min(0).values,bb[:,2:].max(0).values])[None]
                fragmentos=float(iou_pares(union,gt[j:j+1])[0,0])>=.5 and float(ious[partes,j].max())<.5
            por[CLASES[c]][causa]+=1;por[CLASES[c]]['indicador_fusion_bbox']+=int(fusion)
            por[CLASES[c]]['indicador_fragmentacion_bbox']+=int(fragmentos)
            filas.append({'imagen':o['id'],'indice_gt':j,'material':CLASES[c],'causa':causa,
                'max_iou_cualquier_material':float(ious[:,j].max()) if len(b) else 0.,
                'max_iou_material_correcto':float(ious[mismo,j].max()) if bool(mismo.any()) else 0.,
                'indicador_fusion_bbox':fusion,'indicador_fragmentacion_bbox':fragmentos})
    return {'por_material':{c:dict(x) for c,x in por.items()},'objetos':filas,
        'limitacion':'fusión y fragmentación son indicadores geométricos de cajas, no adjudicación visual'}


def analizar(carpeta,ds,regiones):
    """GT entra aquí, solo después de capturar todas las predicciones."""
    carpeta=Path(carpeta);sal={};cm=torch.zeros(7,7,dtype=torch.long);ci=cm.clone();cc=torch.zeros(6,6,dtype=torch.long)
    instancias=[]
    for k in range(0,len(ds.registros),8):
        raw=torch.load(carpeta/f'logits_1v/{k:04d}.pt',map_location='cpu',weights_only=True)['logits'].float()
        pred=raw.argmax(1);cond=raw[:,1:].argmax(1);y=ds.mascaras[k:k+8];ok=y!=255
        cm+=torch.bincount(7*y[ok]+pred[ok],minlength=49).reshape(7,7)
        for j in range(len(raw)):
            for etiqueta,mascara,ident in regiones[k+j]:
                val=torch.from_numpy(mascara)&(y[j]==etiqueta+1)
                if not bool(val.any()):instancias.append({'imagen':ds.registros[k+j]['id'],'id':ident,'clase':etiqueta,'tokens':0});continue
                ganador=int(torch.bincount(pred[j][val],minlength=7).argmax())
                material=int(torch.bincount(cond[j][val],minlength=6).argmax())
                ci[etiqueta+1,ganador]+=1;cc[etiqueta,material]+=1
                instancias.append({'imagen':ds.registros[k+j]['id'],'id':ident,'clase':etiqueta,'tokens':int(val.sum()),
                    'prediccion_con_fondo':ganador,'prediccion_condicional_material':material})
    sal['cabeza_principal']={'semantica':metricas_semanticas(cm),'matriz_instancias_con_fondo':ci.tolist(),
        'matriz_instancias_material_condicional':cc.tolist(),'instancias':instancias,
        'criterio':'voto mayoritario sobre tokens GT válidos de cada instancia; excluye ignore y mezclas de material'}
    for v in [1,5]:
        ps=cargar_predicciones(carpeta,v);m=evaluar_detecciones(ps,ds.objetivos)
        sal[str(v)]={'metricas':resumir_deteccion(m),'detalle_metricas':m,'geometria':geometria_posthoc(ps,ds.objetivos)}
    oportunidades={e:Counter() for e in ETAPAS};por_instancia=[]
    for j,o in enumerate(ds.objetivos):
        tr=cargar_traza(carpeta/f'trazas_5v/{j:04d}.pt');fila={'imagen':o['id'],'por_etapa':{}}
        for e in ETAPAS:
            ns=[tr['nodos'][i] for i in tr['etapas'][e]]
            bs=torch.tensor([n['caja_global'] for n in ns],dtype=torch.float32).reshape(-1,4)
            ls=torch.tensor([n['clase'] for n in ns],dtype=torch.long);iu=iou_pares(bs,o['boxes'])
            encontrados=[]
            for g,c in enumerate(o['labels'].tolist()):
                if bool(((ls==c)&(iu[:,g]>=.5)).any()):oportunidades[e][CLASES[c]]+=1;encontrados.append(g)
            fila['por_etapa'][e]=encontrados
        por_instancia.append(fila)
    sal['oportunidades_5v']={'por_etapa':{e:dict(x) for e,x in oportunidades.items()},'por_imagen':por_instancia,
        'criterio':'GT con al menos una caja de material correcto e IoU>=.5; no asignación uno a uno'}
    return sal
