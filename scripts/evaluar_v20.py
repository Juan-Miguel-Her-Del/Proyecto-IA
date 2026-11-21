"""V20: captura sin GT, análisis posterior train/val y referencia histórica exacta."""
import copy
import json
from pathlib import Path
import sys
import time
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from PIL import Image
import torch
from ecowaste.detector_mapas import DetectorMapas,cajas_desde_mapas
from ecowaste.poligonos import ContornosTACO
from ecowaste.traza_propuestas import trazar_logits
from ecowaste.mosaico import ventanas
from ecowaste.datasets import letterbox,geometria_letterbox
from ecowaste.auditoria_propuestas_internas import analizar_imagen,resumir_objetos
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.ablacion_fondo import resumir_deteccion,huella
from ecowaste.reanudacion_regional import publicar,sha
from ecowaste.reanudacion_v20 import DiarioV20
from scripts.preparar_v20 import (leer_cfg,validar,protocolo,exigir_autorizacion,contexto_train,evaluar_principal_train,huellas_codigo)
from scripts.entrenar_v20 import cargar_diagnostico
from scripts.entrenar_detector_mapas import DatosMapas
from scripts.auditar_propuestas_internas import serializar_traza,cargar_traza


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def exigir_entrenamiento_completo(cfg):
    validar(cfg);path=RAIZ/f"resultados/{cfg['nombre']}.json"
    if not path.exists():raise ValueError('V20 A/B completo no ejecutado: validación permanece pendiente')
    r=leer(path)
    if r.get('estado')!='entrenamiento_v20_completo' or r['protocolo']!=protocolo(cfg):raise ValueError('Manifiesto ajeno')
    if any(r['ramas'][rama]['actualizaciones']!=1452 for rama in ['A','B']):raise ValueError('Presupuesto incompleto')
    exigir_autorizacion(cfg);diag=cargar_diagnostico(cfg);h=huellas_codigo()
    if r['hashes_codigo']!=h:raise ValueError('Código distinto del entrenamiento')
    for rama in cfg['ramas']:
        d=DiarioV20(RAIZ,cfg,rama,diag,h);c,rs,_=d.leer()
        if len(rs)!=12 or c['actualizaciones_desde_origen']!=1452:raise ValueError('Diario incompleto')
        mejor=max(rs,key=lambda e:e['historial_epoca']['map50_val'])
        for tipo,ev in [('ultimo',rs[-1]),('mejor',mejor)]:
            if sha(RAIZ/f'checkpoints/{d.nombre}_{tipo}.pt')!=ev['sha256_checkpoint']:raise ValueError('Alias incorrecto')
        if r['ramas'][rama]['historial']!=c['historial']:raise ValueError('Historial del manifiesto alterado')
    return r


@torch.inference_mode()
def inferir_sin_gt(modelo,imagenes,metadatos,carpeta):
    """Solo píxeles y procedencia. Una vista conserva batch=8 y cajas locales;
    cinco vistas usa el observador V19 con igualdad exacta sobre sus logits.
    """
    if any(set(m)!={'id','ruta','ancho','alto','sha256_imagen'} for m in metadatos):raise ValueError('Metadatos con campos no autorizados')
    carpeta=Path(carpeta)
    if carpeta.exists():raise FileExistsError('Captura existente; no reemplazar')
    carpeta.mkdir(parents=True);estado=huella(modelo.state_dict());pred1=[];pred5=[];indice=[];costes={}
    for vistas in [1,5]:
        inicio=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        if vistas==1:
            for k in range(0,len(metadatos),8):
                ms=metadatos[k:k+8];gs=[geometria_letterbox(m['ancho'],m['alto'],384) for m in ms]
                with torch.autocast('cuda',dtype=torch.float16):raw=modelo(imagenes[k:k+8].cuda().float()/255)
                pp=cajas_desde_mapas(raw,.05,2,gs,politica='argmax7');pred1.extend(pp)
                p=carpeta/'logits_1v'/f'{k:03d}.pt'
                publicar(p,{'logits':raw.cpu(),'geometrias':gs,'metadatos':ms,'vistas':1,'amp':True},tensor=True)
                indice.append({'archivo':p.relative_to(RAIZ).as_posix(),'sha256':sha(p)})
        else:
            for i,m in enumerate(metadatos):
                if sha(RAIZ/m['ruta'])!=m['sha256_imagen']:raise ValueError('Imagen distinta')
                with Image.open(RAIZ/m['ruta']) as im:imagen=im.convert('RGB')
                zs=ventanas(imagen.width,imagen.height);xs=[letterbox(imagen.crop(z),384) for z in zs];gs=[g for x,g in xs]
                with torch.autocast('cuda',dtype=torch.float16):raw=modelo(torch.stack([x for x,g in xs]).cuda())
                t=trazar_logits(raw,gs,zs,imagen.width,imagen.height,m['id']);pred5.append(t['final'])
                p=carpeta/'logits_5v'/f'{i:03d}.pt';q=carpeta/'trazas_5v'/f'{i:03d}.pt'
                publicar(p,{'logits':raw.cpu(),'geometrias':gs,'ventanas':zs,'metadatos':m,'vistas':5,'amp':True},tensor=True)
                publicar(q,serializar_traza(t),tensor=True)
                indice.extend({'archivo':f.relative_to(RAIZ).as_posix(),'sha256':sha(f)} for f in [p,q])
                if (i+1)%30==0 or i+1==len(metadatos):print(f'Captura {carpeta.name}: {i+1}/{len(metadatos)}',flush=True)
        torch.cuda.synchronize();costes[str(vistas)]={'segundos_captura_incluye_archivo_y_observador':time.perf_counter()-inicio,
            'memoria_gpu_max_allocated_bytes':torch.cuda.max_memory_allocated(),'memoria_gpu_max_reserved_bytes':torch.cuda.max_memory_reserved()}
    predicciones={}
    for v,ps in [('1',pred1),('5',pred5)]:
        pp=[{'id':m['id'],**{k:x.tolist() for k,x in p.items()}} for m,p in zip(metadatos,ps)]
        publicar(carpeta/f'predicciones_{v}v_val.json',pp);predicciones[v]=pp
    if huella(modelo.state_dict())!=estado or any(p.grad is not None for p in modelo.parameters()):raise AssertionError('Inferencia alteró el modelo')
    publicar(carpeta/'captura.json',{'imagenes':len(metadatos),'archivos':indice,'costes':costes,
        'gt_utilizado':False,'regresion_observador_5v_exacta':len(pred5),'hashes_codigo':huellas_codigo()})
    return pred1,pred5,costes,predicciones


def comparar_control_v18(manifiesto):
    anterior=leer(RAIZ/'resultados/supervision_regional_ep19_v1.json')['ramas']['A']['historial']
    nuevo=manifiesto['ramas']['A']['historial'];filas=[]
    for a,b in zip(anterior,nuevo):
        filas.append({'epoca':b['epoca'],'traza_exacta':a['huella_orden_y_aumentos']==b['huella_orden_y_aumentos'],
            'map50_v18':a['map50_val'],'map50_v20':b['map50_val'],'delta_map50':b['map50_val']-a['map50_val'],
            'loss_mapas_v18':a.get('loss_mapas'),'loss_original_v20':b['loss_original']})
    return {'epocas':filas,'entradas_iguales':all(f['traza_exacta'] for f in filas),
        'identidad_bit_a_bit_no_presupuesta':True,'fuente':'resultados/supervision_regional_ep19_v1.json'}


def main():
    cfg=leer_cfg();manifiesto=exigir_entrenamiento_completo(cfg)
    salida=RAIZ/f"resultados/{cfg['nombre']}_evaluacion.json";carpeta=salida.with_suffix('')
    if salida.exists() or carpeta.exists():raise FileExistsError('Evaluación existente; conservar')
    ck,train,aux,regiones,pm,est=contexto_train(cfg)
    val=DatosMapas('val',ContornosTACO(RAIZ,1/64));contornos=ContornosTACO(RAIZ,1/64)
    if len(val)!=181 or sha(RAIZ/'datos/procesados/deteccion/val.jsonl')!=ck['hashes_manifiestos']['val']:raise ValueError('Val cambió')
    metadatos=[{k:r[k] for k in ['id','ruta','ancho','alto']} for r in val.base.origen.registros]
    for m in metadatos:m['sha256_imagen']=sha(RAIZ/m['ruta'])
    paths={'origen':RAIZ/cfg['checkpoint_origen'],**{f'{r}_{t}':RAIZ/f"checkpoints/{cfg['nombre']}_{r}_{t}.pt"
        for r in ['A','B'] for t in ['ultimo','mejor']}}
    resultados={};vistos={};carpeta.mkdir(parents=True)
    for nombre,path in paths.items():
        h=sha(path)
        if h in vistos:
            entrada=copy.deepcopy(resultados[vistos[h]]);entrada['reutiliza_captura_de']=vistos[h]
            resultados[nombre]=entrada;publicar(carpeta/f'{nombre}.json',entrada);continue
        actual=torch.load(path,map_location='cpu',weights_only=True)
        m=DetectorMapas(**actual['configuracion']).cuda();m.load_state_dict(actual['pesos'],strict=True);m.eval()
        p=carpeta/nombre;una,cinco,costes,pp=inferir_sin_gt(m,val.base.imagenes,metadatos,p)
        regresion={}
        if nombre=='origen':
            for v,f in [('1','detector_mapas_v6_auxiliar_ext48_predicciones_val.json'),('5','detector_mapas_v6_mosaico_predicciones_val.json')]:
                historico=leer(RAIZ/'resultados'/f)
                if pp[v]!=historico:raise AssertionError(f'Referencia histórica {v}v diferente; detener V20')
                regresion[v]={'imagenes_exactas':181,'sha256_historico':sha(RAIZ/'resultados'/f),'archivo_historico':f}
        # GT se consulta desde aquí, después de terminar y archivar inferencia.
        filas=[]
        for i,r in enumerate(val.base.origen.registros):
            traza=cargar_traza(p/'trazas_5v'/f'{i:03d}.pt');ff,_=analizar_imagen(traza,r,contornos);filas+=ff
        publicar(p/'objetos_val_5v.json',filas)
        inicio=time.perf_counter();torch.cuda.reset_peak_memory_stats();principal=evaluar_principal_train(m,train,regiones)
        torch.cuda.synchronize();coste_train={'segundos':time.perf_counter()-inicio,'memoria_gpu_max_allocated_bytes':torch.cuda.max_memory_allocated()}
        publicar(p/'principal_train.json',principal)
        entrada={'sha256_checkpoint':h,'epoca':actual['epoca'],'principal_train':principal,'val_1_vista':{},'val_5_vistas':{},
            'oportunidades_y_geometria_val':resumir_objetos(filas),'regresion_historica':regresion,
            'captura':(p/'captura.json').relative_to(RAIZ).as_posix(),'coste_principal_train':coste_train}
        for v,preds,k in [('1',una,'val_1_vista'),('5',cinco,'val_5_vistas')]:
            met=evaluar_detecciones(preds,val.base.objetivos)
            entrada[k]={'metricas':met,'resumen':resumir_deteccion(met),'coste':costes[v]}
        resultados[nombre]=entrada;vistos[h]=nombre;publicar(carpeta/f'{nombre}.json',entrada)
        print(nombre,entrada['val_5_vistas']['resumen'],flush=True)
        del m;torch.cuda.empty_cache()
    if sha(RAIZ/cfg['checkpoint_origen'])!=cfg['sha256_checkpoint_origen']:raise ValueError('Referencia alterada')
    protegidos=leer(RAIZ/'resultados/v20_ejecucion/estado_previo.json')['historicos']
    if any(sha(RAIZ/p)!=h for p,h in protegidos.items()):raise ValueError('Históricos alterados')
    publicar(salida,{'estado':'evaluacion_v20_completa','protocolo':protocolo(cfg),'resultados':resultados,
        'control_A_frente_V18':comparar_control_v18(manifiesto),'hashes_codigo':huellas_codigo(),
        'test_evaluado':False,'cambio_predeterminado':False,'adopcion_automatica':False,
        'limitaciones':['Una semilla; validación reutilizada de desarrollo; AP propio de 101 puntos, no COCOeval oficial.',
            'Coste de captura incluye escritura de logits, trazas y observador; no es latencia pura de producción.'],
        'criterio':'Revisar AP seis materiales, recall, precisión, FP, geometría, oportunidades, fondo y coste conjuntamente.'})


if __name__=='__main__':main()
