"""Localización, material en los mismos recortes y detección conjunta. Solo desarrollo autorizado."""
import argparse,json,sys,time
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ));OUT=RAIZ/'resultados/integracion_d2'
import numpy as np,torch
from PIL import Image,ImageOps
from ecowaste.integracion_d2 import MotorD2,CLASES
from ecowaste.aplicacion_integrada import proteger_imagen
from ecowaste.vision_equipo.preprocesado import crop_box
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.ablacion_fondo import resumir_deteccion
from ecowaste.cajas import iou_pares
from ecowaste.incorporacion import sha
MODOS=['transformer','refinado','escena','combinado','historico']
def leer(p):return json.loads(p.read_text(encoding='utf8'))
def guardar(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def xyxy(b):x,y,w,h=b;return [x,y,x+w,y+h]
def objetivo(r,generico=False):
    positivos=[b for b in r['cajas'] if not b.get('iscrowd',0) and (generico or b['clase'] in CLASES)]
    ignorados=[b for b in r['cajas'] if b not in positivos]
    return {'id':r['id'],'boxes':torch.tensor([xyxy(b['bbox']) for b in positivos],dtype=torch.float32).reshape(-1,4),'labels':torch.tensor([0 if generico else CLASES.index(b['clase']) for b in positivos],dtype=torch.long),'ignore_boxes':torch.tensor([xyxy(b['bbox']) for b in ignorados],dtype=torch.float32).reshape(-1,4)}
def resumen_confusion(cm):
    cm=np.asarray(cm);gt=cm.sum(1);pred=cm.sum(0);tp=cm.diagonal();pre=tp/np.maximum(pred,1);rec=tp/np.maximum(gt,1);f1=2*pre*rec/np.maximum(pre+rec,1e-12);present=gt>0
    return {'objetos':int(cm.sum()),'aciertos':int(tp.sum()),'accuracy':float(tp.sum()/max(cm.sum(),1)),'macro_F1_clases_con_GT':float(f1[present].mean()) if present.any() else None,'matriz_confusion_GT_filas_pred_columnas':cm.tolist(),'clases':CLASES,'por_material':{c:{'gt':int(gt[i]),'correctos':int(tp[i]),'recall':float(rec[i]) if gt[i] else None,'precision':float(pre[i]) if pred[i] else None} for i,c in enumerate(CLASES)}}
def main():
    p=argparse.ArgumentParser();p.add_argument('--cohorte',choices=['d2_val_autorizada','ecowaste_val_conocida_por_d2'],required=True);a=p.parse_args();co=leer(OUT/'cohortes.json');rows=co[a.cohorte];folder=OUT/'evaluacion'/a.cohorte
    registros={}
    for sp in ['train','val']:
        for line in (RAIZ/f'datos/procesados/deteccion/{sp}.jsonl').read_text(encoding='utf8').splitlines():
            r=json.loads(line);registros[r['id']]=r
    m=MotorD2();[m.banco.cargar(n) for n in MODOS];pg=[];og=[];ps={n:[] for n in MODOS};os=[];cm={n:np.zeros((6,6),np.int64) for n in MODOS};cg={n:np.zeros((6,6),np.int64) for n in MODOS};costes=[];detalles=[]
    for k,meta in enumerate(rows):
        r=registros[meta['id']];path=RAIZ/meta['ruta'];proteger_imagen(path)
        if sha(path)!=meta['sha256_archivo']:raise ValueError('Foto alterada')
        capture=folder/'capturas'/f'{k:04d}.json';truth=objetivo(r);generic=objetivo(r,True)
        if capture.exists():
            q=leer(capture)
            if q['sha256_imagen']!=meta['sha256_archivo'] or q['id']!=r['id'] or q['pesos']!=m.metadatos['checkpoints_sha256']:raise ValueError('Captura ajena')
        else:
            with Image.open(path) as im:rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
            t=time.perf_counter();dets=m.detector.detect(rgb);det_ms=(time.perf_counter()-t)*1000
            boxes=[list(d.box) for d in dets];rois=[list(d.region.box) for d in dets];crops=[rgb[b[1]:b[3],b[0]:b[2]].copy() for b in rois];sc=[float(d.confidence) for d in dets]
            probs={};tiempos={}
            for modo in MODOS:
                t=time.perf_counter();probs[modo]=m.banco.probabilidades(crops,modo).tolist();tiempos[modo]=(time.perf_counter()-t)*1000
            q={'id':r['id'],'sha256_imagen':meta['sha256_archivo'],'pesos':m.metadatos['checkpoints_sha256'],'boxes':boxes,'scores_detector':sc,'rois':rois,'probabilidades':probs,'detector_ms':det_ms,'clasificadores_ms':tiempos,'GT_en_inferencia':False};guardar(capture,q)
        # Análisis empieza después de guardar la captura sin GT.
        b=torch.tensor(q['boxes'],dtype=torch.float32).reshape(-1,4);pg.append({'boxes':b,'scores':torch.tensor(q['scores_detector']),'labels':torch.zeros(len(b),dtype=torch.long)});og.append(generic);os.append(truth)
        for modo in MODOS:
            prob=np.asarray(q['probabilidades'][modo]).reshape(-1,6);labels=prob.argmax(1) if len(prob) else np.empty(0,int);scores=np.array(q['scores_detector'])*(prob.max(1) if len(prob) else np.empty(0))
            ps[modo].append({'boxes':b,'scores':torch.tensor(scores,dtype=torch.float32),'labels':torch.tensor(labels,dtype=torch.long)})
        # Matching genérico único por score detector; igual para todos los clasificadores.
        iu=iou_pares(b,generic['boxes']);remaining=set(range(len(generic['boxes'])));positive=[x for x in r['cajas'] if not x.get('iscrowd',0)];matches=[]
        for i in sorted(range(len(b)),key=lambda i:-q['scores_detector'][i]):
            best=max([(float(iu[i,j]),j) for j in remaining],default=(0,None))
            if best[0]<.5:continue
            j=best[1];remaining.remove(j);cl=positive[j]['clase']
            if cl not in CLASES:continue
            gt=CLASES.index(cl);matches.append({'indice_deteccion':i,'indice_GT_generico':j,'GT':cl,'IoU':best[0]})
            for modo in MODOS:cm[modo][gt,int(np.argmax(q['probabilidades'][modo][i]))]+=1
        detalles.append({'id':r['id'],'matches_material_IoU50':matches});costes.append({'id':r['id'],'detector_ms':q['detector_ms'],'clasificadores_ms':q['clasificadores_ms']})
        if (k+1)%20==0 or k+1==len(rows):print(a.cohorte,k+1,'/',len(rows),flush=True)
    generic=resumir_deteccion(evaluar_detecciones(pg,og));generic.pop('ap50_por_material',None);generic['AP50_clase_residuo']=generic['map50'];joint={n:resumir_deteccion(evaluar_detecciones(ps[n],os)) for n in MODOS}
    res={'cohorte':a.cohorte,'imagenes':len(rows),'GT_materiales':sum(len(o['boxes']) for o in os),'GT_genericos':sum(len(o['boxes']) for o in og),'localizacion_sin_material':generic,'clasificacion_sobre_recortes_D2_matcheados_IoU50':{n:resumen_confusion(cm[n]) for n in MODOS},'deteccion_conjunta_por_clasificador':joint,'costes_por_imagen':costes,'matches':detalles,'umbral_detector':.27,'AP_truncado_por_umbral_aplicacion':True,'score_conjunto':'score_D2 * prob_material; no calibrado; fijado antes de evaluar','independencia_fisica_demostrada':False,'test_evaluado':False,'seleccion_modelo_por_resultados':False,'limitaciones':co['limitaciones'],'nota_organico':'Sin GT orgánico en D2 val autorizada; no puede evaluarse allí su AP. Un GT en EcoWaste val es train de D2.'}
    guardar(folder/'resumen.json',res);print('mAP genérico',generic['map50'],'conjuntos',{n:v['map50'] for n,v in joint.items()},flush=True)
if __name__=='__main__':main()
