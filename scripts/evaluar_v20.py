"""Evaluación futura V20: rechaza presupuestos ausentes y nunca selecciona ni entrena."""
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from PIL import Image
import torch
from ecowaste.datasets import CLASES,ResiduosDataset,letterbox,objetivo_deteccion
from ecowaste.detector_mapas import DetectorMapas
from ecowaste.poligonos import ContornosTACO
from ecowaste.traza_propuestas import trazar_logits
from ecowaste.mosaico import ventanas
from ecowaste.auditoria_propuestas_internas import analizar_imagen,resumir_objetos
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.ablacion_fondo import resumir_deteccion
from ecowaste.reanudacion_regional import publicar,sha
from scripts.preparar_v20 import (leer_cfg,validar,protocolo,exigir_autorizacion,contexto_train,evaluar_principal_train,huellas_codigo)
from scripts.entrenar_detector_mapas import DatosMapas
from scripts.evaluar_supervision_regional import evaluar_vistas


def exigir_entrenamiento_completo(cfg):
    validar(cfg);path=RAIZ/f"resultados/{cfg['nombre']}.json"
    if not path.exists():raise ValueError('V20 A/B completo no ejecutado: validación permanece pendiente')
    r=json.loads(path.read_text(encoding='utf-8'))
    if any(r['ramas'][rama]['actualizaciones']!=1452 for rama in ['A','B']):raise ValueError('Presupuesto incompleto')
    exigir_autorizacion(cfg)
    return r


def main():
    cfg=leer_cfg();manifiesto=exigir_entrenamiento_completo(cfg)
    salida=RAIZ/f"resultados/{cfg['nombre']}_evaluacion.json"
    if salida.exists():raise FileExistsError('Evaluación existente')
    ck,train,aux,regiones,pm,est=contexto_train(cfg)
    val=DatosMapas('val',ContornosTACO(RAIZ,1/64));contornos=ContornosTACO(RAIZ,1/64)
    if sha(RAIZ/'datos/procesados/deteccion/val.jsonl')!=ck['hashes_manifiestos']['val']:raise ValueError('Val cambió')
    paths={'origen':RAIZ/cfg['checkpoint_origen'],**{f'{rama}_{tipo}':RAIZ/f"checkpoints/{cfg['nombre']}_{rama}_{tipo}.pt"
        for rama in ['A','B'] for tipo in ['ultimo','mejor']}}
    resultados={}
    for nombre,path in paths.items():
        actual=torch.load(path,map_location='cpu',weights_only=True)
        if nombre!='origen':
            if actual['protocolo_v20']!=protocolo(cfg) or actual['hashes_codigo_v20']!=huellas_codigo():raise ValueError('Checkpoint de otro protocolo')
            if actual['sha256_origen']!=cfg['sha256_checkpoint_origen']:raise ValueError('Otro origen')
            if nombre.endswith('ultimo') and actual['actualizaciones_desde_origen']!=1452:raise ValueError('No es final')
        m=DetectorMapas(**actual['configuracion']).cuda();m.load_state_dict(actual['pesos'],strict=True);m.eval()
        principal=evaluar_principal_train(m,train,regiones)
        una,_=evaluar_vistas(m,val,ck['config_entrenamiento'],1)
        pred=[];filas=[]
        # Las cinco vistas y el observador no reciben GT; se compara después.
        with torch.inference_mode():
            for registro in val.base.origen.registros:
                with Image.open(RAIZ/registro['ruta']) as im:imagen=im.convert('RGB')
                zs=ventanas(imagen.width,imagen.height);xs=[letterbox(imagen.crop(z),384) for z in zs]
                with torch.autocast('cuda',dtype=torch.float16):raw=m(torch.stack([x for x,g in xs]).cuda())
                t=trazar_logits(raw,[g for x,g in xs],zs,imagen.width,imagen.height,registro['id'])
                pred.append(t['final']);ff,_=analizar_imagen(t,registro,contornos);filas+=ff
        met=evaluar_detecciones(pred,val.base.objetivos)
        resultados[nombre]={'sha256_checkpoint':sha(path),'epoca':actual['epoca'],
            'principal_train':principal,'val_1_vista':una,
            'val_5_vistas':{'metricas':met,'resumen':resumir_deteccion(met)},
            'oportunidades_y_geometria_val':resumir_objetos(filas)}
        del m;torch.cuda.empty_cache()
    if sha(RAIZ/cfg['checkpoint_origen'])!=cfg['sha256_checkpoint_origen']:raise ValueError('Referencia alterada')
    publicar(salida,{'estado':'evaluacion_v20_completa','protocolo':protocolo(cfg),'resultados':resultados,
        'test_evaluado':False,'cambio_predeterminado':False,'adopcion_automatica':False,
        'criterio':'Revisar AP seis materiales, recall, precisión, FP, geometría, oportunidades, fondo y coste conjuntamente.'})


if __name__=='__main__':main()
