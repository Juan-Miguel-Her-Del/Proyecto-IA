"""Evaluación val 1/5 vistas y regiones train de A/B; nunca adopta modelos ni usa test."""
import argparse
import json
from pathlib import Path
import sys
import time
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from PIL import Image
import torch
from torch.utils.data import DataLoader
from ecowaste.mosaico import predecir_mosaico
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.ablacion_fondo import resumir_deteccion
from ecowaste.auditoria_deteccion import auditar_imagen,ESTADOS
from scripts.entrenar_detector_mapas import DatosMapas,evaluar
from ecowaste.poligonos import ContornosTACO
from ecowaste.detector_mapas import DetectorMapas
from scripts.entrenar_clasificador import escribir_json
from scripts.supervision_regional_materiales import (CONFIG,contexto,sha,evaluar_regiones,validar_protocolo,hashes_codigo,exigir_habilitacion)
from ecowaste.reanudacion_regional import DiarioRama


@torch.inference_mode()
def evaluar_vistas(m,val,base,vistas):
    inicio=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    if vistas==1:
        metricas,pred=evaluar(m,DataLoader(val,batch_size=8,generator=torch.Generator().manual_seed(0)),'cuda',True,base)
    elif vistas==5:
        pred=[]
        for r in val.base.origen.registros:
            with Image.open(RAIZ/r['ruta']) as im:
                pred.append(predecir_mosaico(m,im.convert('RGB'),base['umbral_evaluacion'],amp=True,politica_mapas='argmax7'))
        metricas=evaluar_detecciones(pred,val.base.objetivos)
    else:raise ValueError('Vistas no predefinidas')
    torch.cuda.synchronize()
    visual=[{k:v[p['scores']>=.35] for k,v in p.items()} for p in pred]
    carton={'gt':0,'estados':{s:0 for s in ESTADOS},'material_equivocado_a_plastico':0,'material_equivocado_a_metal':0}
    for pr,o in zip(pred,val.base.objetivos):
        for d in auditar_imagen(pr,o)['objetos']:
            if d['clase_real']!=2:continue
            carton['gt']+=1;carton['estados'][d['estado']]+=1
            if d['estado']=='material_equivocado':
                if d['material_testigo']==0:carton['material_equivocado_a_plastico']+=1
                if d['material_testigo']==4:carton['material_equivocado_a_metal']+=1
    return {'metricas':metricas,'resumen':resumir_deteccion(metricas),
        'auditoria_carton_val':carton,
        'resumen_umbral_visual035':resumir_deteccion(evaluar_detecciones(visual,val.base.objetivos)),
        'vistas':vistas,'segundos':time.perf_counter()-inicio,
        'memoria_gpu_max_allocated_bytes':torch.cuda.max_memory_allocated(),
        'memoria_gpu_max_reserved_bytes':torch.cuda.max_memory_reserved()},pred


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=CONFIG)
    a=p.parse_args();cfg=json.loads(a.config.read_text(encoding='utf-8'));validar_protocolo(cfg)
    salida=RAIZ/f"resultados/{cfg['nombre']}_evaluacion.json"
    if salida.exists():raise FileExistsError('Evaluación existente')
    paths={f'{r}_{s}':RAIZ/f"checkpoints/{cfg['nombre']}_{r}_{s}.pt" for r in cfg['ramas'] for s in ['ultimo','mejor']}
    if not all(p.is_file() for p in paths.values()):raise ValueError('A/B completo aún no disponible; no se evalúan pilotos como resultados finales')
    informe=json.loads((RAIZ/f"resultados/{cfg['nombre']}.json").read_text(encoding='utf-8'))
    if any(r['actualizaciones']!=1452 for r in informe['ramas'].values()):raise ValueError('Presupuesto incompleto')
    exigir_habilitacion(cfg)
    diag=json.loads((RAIZ/cfg['diagnostico']).read_text(encoding='utf-8'));diag['sha256_archivo']=sha(RAIZ/cfg['diagnostico'])
    for rama in cfg['ramas']:
        d=DiarioRama(RAIZ,cfg,rama,diag,hashes_codigo());final,registros,_=d.leer()
        if final['actualizaciones_desde_origen']!=1452 or len(registros)!=12:raise ValueError('Diario incompleto')
        conocidos={r['sha256_checkpoint'] for r in registros}
        if any(sha(paths[f'{rama}_{s}']) not in conocidos for s in ['ultimo','mejor']):raise ValueError('Aliases fuera del diario')
    ck,train,aux,regiones,w,estadistica,pm=contexto(cfg)
    val=DatosMapas('val',ContornosTACO(RAIZ,ck['config_entrenamiento']['cobertura_minima']))
    paths={'origen':RAIZ/cfg['checkpoint_origen'],**paths};resultados={}
    for nombre,path in paths.items():
        actual=torch.load(path,map_location='cpu',weights_only=True)
        if nombre!='origen' and (actual.get('protocolo_regional')!=cfg or actual.get('sha256_origen')!=cfg['sha256_checkpoint_origen']):
            raise ValueError('Checkpoint de otro experimento')
        m=DetectorMapas(**actual['configuracion']).cuda();m.load_state_dict(actual['pesos'],strict=True);m.eval()
        entrada={'sha256_checkpoint':sha(path),'epoca':actual['epoca'],'vistas':{},
            'confusion_regional_train':evaluar_regiones(m,train,regiones,w)}
        for vistas in cfg['vistas_evaluacion']:
            metricas,pred=evaluar_vistas(m,val,ck['config_entrenamiento'],vistas)
            entrada['vistas'][str(vistas)]=metricas
            escribir_json(RAIZ/f"resultados/{cfg['nombre']}_{nombre}_{vistas}v_predicciones_val.json",
                [{'id':o['id'],**{k:v.tolist() for k,v in pr.items()}} for o,pr in zip(val.base.objetivos,pred)])
        resultados[nombre]=entrada;del m;torch.cuda.empty_cache()
    deltas={}
    for tipo in ['ultimo','mejor']:
        aa=resultados['A_'+tipo];bb=resultados['B_'+tipo]
        deltas[tipo]={'vistas':{},'delta_carton_a_plastico_metal_cabeza_auxiliar_train':
            bb['confusion_regional_train']['carton']['tasa_a_plastico_metal']-aa['confusion_regional_train']['carton']['tasa_a_plastico_metal']}
        def tasa_mapas(x):
            cm=x['confusion_regional_train']['mapas_logits_promediados_por_region']['matriz_confusion']
            return (cm[3][1]+cm[3][5])/sum(cm[3])
        deltas[tipo]['delta_carton_a_plastico_metal_cabeza_mapas_train']=tasa_mapas(bb)-tasa_mapas(aa)
        for v in ['1','5']:
            x=aa['vistas'][v]['resumen'];y=bb['vistas'][v]['resumen']
            deltas[tipo]['vistas'][v]={k:y[k]-x[k] for k in x if isinstance(x[k],(int,float))}
    if sha(RAIZ/cfg['checkpoint_origen'])!=cfg['sha256_checkpoint_origen']:raise ValueError('Origen modificado')
    escribir_json(salida,{'estado':'evaluacion_ab_completa','configuracion':cfg,'resultados':resultados,
        'deltas_B_menos_A':deltas,'test_evaluado':False,'cambio_predeterminado':False,
        'adopcion':'Pendiente de revisión conjunta de AP seis clases, recall, precisión, FP, confusiones y coste; no automática.',
        'limitacion':'Validación de desarrollo reutilizada; AP propio de 101 puntos, no COCOeval oficial.'})


if __name__=='__main__':main()
