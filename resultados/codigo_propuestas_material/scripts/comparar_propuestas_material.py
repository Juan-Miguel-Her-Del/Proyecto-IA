"""A/B pareado de propuestas por material, con logits compartidos y pesos congelados."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader
from ecowaste.datasets import CLASES, collate_deteccion, letterbox
from ecowaste.ejecucion_detector import CacheDeteccion
from ecowaste.detector_mapas import cargar_detector_mapas,cajas_desde_mapas
from ecowaste.mosaico import ventanas,combinar_vistas
from ecowaste.evaluacion_detector import evaluar_detecciones
from scripts.entrenar_clasificador import escribir_json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def filtrar(predicciones,umbral):
    return [{k:v[p['scores']>=umbral] for k,v in p.items()} for p in predicciones]


def resumen(metricas):
    filas=[c['por_umbral'][0] for c in metricas['por_clase'].values()]
    tp=sum(c['tp'] for c in filas);fp=sum(c['fp'] for c in filas)
    ignoradas=sum(c['predicciones_ignoradas'] for c in filas)
    return {'map50':metricas['map50'],'map50_95':metricas['map50_95'],
        'recall50':metricas['recall50'],'precision50':metricas['precision50'],
        'tp50':tp,'fp50':fp,'fn50':metricas['objetos']-tp,
        'predicciones_ignoradas50':ignoradas,'cajas_totales':tp+fp+ignoradas}


def contar_admisiones(raw,geometrias,umbral,acumulado):
    probs=raw.float().softmax(1).cpu();ganador=probs.argmax(1)
    h,w=raw.shape[-2:]
    yy,xx=torch.meshgrid((torch.arange(h)+.5)/h,(torch.arange(w)+.5)/w,indexing='ij')
    for i,g in enumerate(geometrias):
        x0,y0=g['relleno_x']/g['lado'],g['relleno_y']/g['lado']
        x1=x0+g['ancho_original']*g['escala_x']/g['lado']
        y1=y0+g['alto_original']*g['escala_y']/g['lado']
        valida=(xx>=x0)&(xx<=x1)&(yy>=y0)&(yy<=y1)
        acumulado['celdas_validas']+=int(valida.sum())
        for c,nombre in enumerate(CLASES,1):
            adm=(probs[i,c]>=umbral)&valida
            for clave,mask in [('admisiones_sin_argmax',adm),('admisiones_argmax',adm&(ganador[i]==c)),
                ('excluidas_por_fondo_ganador',adm&(ganador[i]==0)),
                ('excluidas_por_otro_material_ganador',adm&(ganador[i]!=0)&(ganador[i]!=c))]:
                acumulado[nombre][clave]+=int(mask.sum())


def nuevo_conteo():
    return {'celdas_validas':0,**{c:defaultdict(int) for c in CLASES}}


def reproducir_base(predicciones,referencia,objetivos):
    registros=json.loads(referencia.read_text(encoding='utf-8'))
    if len(registros)!=len(predicciones):raise ValueError('Referencia incompleta')
    iguales=0;delta_max=0.
    for p,r,o in zip(predicciones,registros,objetivos):
        if r['id']!=o['id']:raise ValueError('Referencia de otra escena')
        cajas=torch.tensor(r['boxes'],dtype=torch.float32).reshape(-1,4)
        labels=torch.tensor(r['labels'],dtype=torch.long)
        scores=torch.tensor(r['scores'],dtype=torch.float32)
        coincide=p['boxes'].shape==cajas.shape and torch.equal(p['boxes'],cajas) and torch.equal(p['labels'],labels)
        if coincide:
            delta=float((p['scores']-scores).abs().max()) if len(scores) else 0.
            delta_max=max(delta_max,delta)
            if torch.allclose(p['scores'],scores,atol=1e-6,rtol=0):iguales+=1
    return {'imagenes_con_cajas_etiquetas_y_scores_coincidentes':iguales,'imagenes':len(registros),
            'maximo_delta_scores_con_cajas_coincidentes':delta_max,
            'referencia':referencia.relative_to(RAIZ).as_posix(),'sha256_referencia':sha(referencia)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=RAIZ/'configuracion/experimento_propuestas_material.json')
    a=p.parse_args();cfg=json.loads(a.config.read_text(encoding='utf-8'))
    if not re.fullmatch(r'[A-Za-z0-9_-]+',cfg['nombre']):raise ValueError('Nombre inválido')
    if cfg['entrenar'] or cfg['evaluar_test'] or cfg['cambiar_modelo_predeterminado']:
        raise ValueError('Este experimento solo compara validación con pesos congelados')
    esperado={'politicas':['argmax7','por_material'],'vistas':[1,5],'conectividad':8,'nms_iou':.5,'maximo_cajas':100}
    if any(cfg[k]!=v for k,v in esperado.items()):raise ValueError('Protocolo no implementado')
    salida=RAIZ/f"resultados/{cfg['nombre']}.json"
    claves=[f'{v}_vistas_{politica}' for v in cfg['vistas'] for politica in cfg['politicas']]
    rutas={k:RAIZ/f"resultados/{cfg['nombre']}_{k}_predicciones_val.json" for k in claves}
    if any(p.exists() for p in [salida,*rutas.values()]):raise FileExistsError('Resultados existentes; conservar evidencia con otro nombre')
    checkpoint=RAIZ/cfg['checkpoint'];hash_ck=sha(checkpoint)
    if hash_ck!=cfg['sha256_checkpoint']:raise ValueError('Checkpoint distinto del plan fijado')
    valid=json.loads((RAIZ/'datos/reportes/validacion.json').read_text(encoding='utf-8'))
    if not valid['validacion_estructural_correcta']:raise ValueError('Datos sin validar')
    for ruta,h in valid['sha256_manifiestos'].items():
        if sha(RAIZ/ruta)!=h:raise ValueError('Manifiesto alterado')
    hash_seleccion=sha(RAIZ/'configuracion/arquitectura.json')
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    amp=device=='cuda' and cfg['precision_mixta_cuda']
    m,ck=cargar_detector_mapas(checkpoint,device)
    if m.configuracion_mapas['formulacion']!='softmax7':raise ValueError('Se requieren mapas de siete clases')
    if ck['hashes_manifiestos']['val']!=valid['sha256_manifiestos']['datos/procesados/deteccion/val.jsonl']:
        raise ValueError('Checkpoint de otros datos')
    m.requires_grad_(False);m.eval()
    estado_inicial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    ds=CacheDeteccion(RAIZ,'val',.1)
    predicciones={k:[] for k in claves};conteos={v:nuevo_conteo() for v in cfg['vistas']}
    inicio=time.perf_counter()
    with torch.inference_mode():
        for x,lote in DataLoader(ds,batch_size=8,collate_fn=collate_deteccion):
            gs=[o['geometria'] for o in lote]
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):raw=m(x.to(device).float()/255)
            contar_admisiones(raw,gs,cfg['umbral_celda'],conteos[1])
            for politica in cfg['politicas']:
                predicciones[f'1_vistas_{politica}'].extend(cajas_desde_mapas(raw,umbral=cfg['umbral_celda'],
                    area_minima=cfg['area_minima_celdas'],geometrias=gs,politica=politica))
        print('Una vista: ambas políticas evaluadas sobre los mismos logits',flush=True)
        for i,r in enumerate(ds.origen.registros):
            with Image.open(RAIZ/r['ruta']) as im:imagen=ImageOps.exif_transpose(im).convert('RGB')
            zs=ventanas(imagen.width,imagen.height)
            vistas=[letterbox(imagen.crop(v),m.imagen) for v in zs]
            gs=[g for _,g in vistas];x=torch.stack([t for t,_ in vistas]).to(device)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):raw=m(x)
            contar_admisiones(raw,gs,cfg['umbral_celda'],conteos[5])
            for politica in cfg['politicas']:
                parciales=cajas_desde_mapas(raw,umbral=cfg['umbral_celda'],area_minima=cfg['area_minima_celdas'],
                    geometrias=gs,politica=politica)
                predicciones[f'5_vistas_{politica}'].append(combinar_vistas(parciales,gs,zs,imagen.width,
                    imagen.height,m.imagen,cfg['umbral_celda']))
            if (i+1)%30==0:print(f'Cinco vistas: comparación pareada {i+1}/{len(ds)}',flush=True)
    resultados={}
    for k,pred in predicciones.items():
        metricas={}
        for umbral in cfg['filtros_puntuacion_caja']:
            mm=evaluar_detecciones(filtrar(pred,umbral),ds.objetivos)
            metricas[str(umbral)]={'resumen':resumen(mm),'metricas':mm}
        escribir_json(rutas[k],[{'id':o['id'],**{n:t.tolist() for n,t in pp.items()}} for o,pp in zip(ds.objetivos,pred)])
        resultados[k]={'vistas':int(k[0]),'politica':k.split('_vistas_')[1],'por_filtro_caja':metricas,
            'predicciones_val':rutas[k].relative_to(RAIZ).as_posix(),'sha256_predicciones_val':sha(rutas[k])}
        print(k,json.dumps(metricas[str(cfg['umbral_celda'])]['resumen']),flush=True)
    referencias={1:'detector_mapas_v6_auxiliar_ext48',5:'detector_mapas_v6_mosaico'}
    regresion={str(v):reproducir_base(predicciones[f'{v}_vistas_argmax7'],
        RAIZ/f'resultados/{nombre}_predicciones_val.json',ds.objetivos) for v,nombre in referencias.items()}
    if any(r['imagenes_con_cajas_etiquetas_y_scores_coincidentes']!=len(ds) for r in regresion.values()):
        raise ValueError(f'La base no reproduce resultados previos: {regresion}')
    if any(not torch.equal(v.detach().cpu(),estado_inicial[k]) for k,v in m.state_dict().items()):
        raise ValueError('El estado del modelo cambió')
    if any(p.grad is not None or p.requires_grad for p in m.parameters()):raise ValueError('Modelo no congelado')
    if sha(checkpoint)!=hash_ck or sha(RAIZ/'configuracion/arquitectura.json')!=hash_seleccion:
        raise ValueError('Checkpoint o selección modificados')
    informe={'estado':'comparacion_val_completada','fecha_utc':datetime.now(timezone.utc).isoformat(),
        'configuracion':cfg,'sha256_configuracion':sha(a.config),'checkpoint':cfg['checkpoint'],
        'sha256_checkpoint_antes':hash_ck,'sha256_checkpoint_despues':sha(checkpoint),'epoca_checkpoint':ck['epoca'],
        'sha256_seleccion_antes':hash_seleccion,'sha256_seleccion_despues':sha(RAIZ/'configuracion/arquitectura.json'),
        'estado_modelo_inalterado':True,'parametros_congelados':True,'logits_compartidos_entre_politicas':True,
        'imagenes_val':len(ds),'objetos_val':sum(len(o['boxes']) for o in ds.objetivos),
        'hashes_manifiestos':ck['hashes_manifiestos'],'sha256_manifiestos_invariantes':valid['sha256_manifiestos'],
        'regresion_bases':regresion,'admisiones_celdas':conteos,'resultados':resultados,
        'dispositivo':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
        'torch':str(torch.__version__),'precision':'AMP FP16' if amp else 'FP32',
        'sha256_codigo':{p:sha(RAIZ/p) for p in ['ecowaste/detector_mapas.py','ecowaste/mosaico.py',
            'ecowaste/evaluacion_detector.py','scripts/comparar_propuestas_material.py']},
        'duracion_segundos':time.perf_counter()-inicio,'entrenamiento_realizado':False,
        'seleccion_modificada':False,'test_evaluado':False,
        'limite':'Comparación sobre validación usada en desarrollo. Sin cambio de pesos, umbral, puntuaciones ni NMS; no modifica la ponderación del fondo.'}
    escribir_json(salida,informe)
    print('Pesos y selección intactos; ambas bases reproducidas en las 181 imágenes',flush=True)


if __name__=='__main__':main()
