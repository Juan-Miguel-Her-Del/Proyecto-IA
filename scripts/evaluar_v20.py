"""V20: captura sin GT, análisis posterior train/val y referencia histórica exacta."""
import copy
import argparse
from contextlib import nullcontext
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
from ecowaste.datasets import letterbox,geometria_letterbox,CLASES
from ecowaste.discriminacion_mapas import metricas_instancias
from ecowaste.auditoria_propuestas_internas import analizar_imagen,resumir_objetos
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.ablacion_fondo import resumir_deteccion,huella,metricas_semanticas
from ecowaste.reanudacion_regional import publicar,sha
from ecowaste.reanudacion_v20 import DiarioV20
from scripts.preparar_v20 import (leer_cfg,validar,protocolo,exigir_autorizacion,contexto_train,evaluar_principal_train,huellas_codigo)
from scripts.entrenar_v20 import cargar_diagnostico
from scripts.entrenar_detector_mapas import DatosMapas
from scripts.auditar_propuestas_internas import serializar_traza,cargar_traza
from ecowaste.backend_numerico import aplicar_backend,registro_efectivo,verificar_registro
from scripts.entrenar_detector_mapas import evaluar
from torch.utils.data import DataLoader


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def argumentos(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path)
    p.add_argument('--compatibilidad-evaluacion',type=Path);p.add_argument('--directorio-capturas',type=Path);p.add_argument('--reusar-origen',type=Path)
    g=p.add_mutually_exclusive_group();g.add_argument('--referencia-historica',action='store_true');g.add_argument('--verificar-etapa-20',action='store_true')
    return p.parse_args(argv)


def contexto_evaluacion(cfg):
    return aplicar_backend(cfg['backend_numerico']) if cfg.get('backend_numerico') else nullcontext()


def validar_compatibilidad_codigo(entrenamiento,actual,certificado,p):
    if not certificado['aprobada'] or certificado['hashes_entrenamiento']!=entrenamiento or certificado['hashes_evaluacion']!=actual or certificado['protocolo']!=p:
        raise ValueError('Compatibilidad no demostrada')
    if set(entrenamiento)!=set(actual) or [n for n in entrenamiento if entrenamiento[n]!=actual[n]]!=['scripts/evaluar_v20.py']:
        raise ValueError('Cambios fuera de evaluación')


def comprobar_checkpoint_backend(ck,cfg,efectivo,origen=False,hashes_entrenamiento=None):
    if origen:
        if ck['epoca']!=19:raise ValueError('La referencia debe ser época 19')
        return
    if ck.get('protocolo_v20')!=protocolo(cfg) or ck.get('hashes_codigo_v20')!=(hashes_entrenamiento or huellas_codigo()):raise ValueError('Checkpoint de otro protocolo o código')
    if ck.get('sha256_origen')!=cfg['sha256_checkpoint_origen']:raise ValueError('Otro origen')
    if cfg.get('backend_numerico'):
        p=cfg['backend_numerico']
        if ck.get('backend_numerico')!=p:raise ValueError('Checkpoint de otro backend')
        anterior=ck['backend_efectivo'];verificar_registro(p,anterior);verificar_registro(p,efectivo)
        if {k:v for k,v in anterior.items() if k!='adam'}!={k:v for k,v in efectivo.items() if k!='adam'}:
            raise ValueError('Entorno de inferencia distinto del entrenamiento')
        if any(g.get('foreach')!=p['adam_foreach'] or g.get('fused')!=p['adam_fused'] for g in ck['optimizador']['param_groups']):
            raise ValueError('Opciones AdamW del checkpoint incompatibles')


def comparar_predicciones_historicas(pred,ref,exigir_identidad=False):
    if len(pred)!=len(ref) or [x['id'] for x in pred]!=[x['id'] for x in ref]:raise ValueError('Imágenes de referencia distintas')
    diferentes=[a['id'] for a,b in zip(pred,ref) if a!=b]
    if exigir_identidad and diferentes:raise AssertionError('Predicción histórica diferente bajo backend histórico: '+diferentes[0])
    return {'imagenes_exactas':len(pred)-len(diferentes),'imagenes_diferentes':diferentes,
        'identidad_exigida':exigir_identidad,'ambito':'regresion_historica' if exigir_identidad else 'comparacion_entre_backends'}


def verificar_referencia_historica(cfg):
    validar(cfg);ef=registro_efectivo();historico=leer(RAIZ/'resultados/v22_reproducibilidad_v20/historico/A/C1/entorno_inicio.json')
    for k in ['torch','cuda','cudnn','gpu','CUBLAS_WORKSPACE_CONFIG','deterministic_algorithms','cudnn_deterministic',
        'cudnn_benchmark','tf32_matmul','tf32_cudnn','float32_matmul_precision','sdpa','threads','threads_interop']:
        if ef[k]!=historico[k]:raise ValueError('La referencia histórica requiere su entorno original: '+k)
    salida=RAIZ/cfg['referencia_historica'];carpeta=salida.with_suffix('')
    if salida.exists() or carpeta.exists():raise FileExistsError('Referencia histórica ya registrada')
    path=RAIZ/cfg['checkpoint_origen']
    if sha(path)!=cfg['sha256_checkpoint_origen']:raise ValueError('Origen diferente')
    ck=torch.load(path,map_location='cpu',weights_only=True);val=DatosMapas('val',ContornosTACO(RAIZ,1/64))
    if len(val)!=181 or sha(RAIZ/'datos/procesados/deteccion/val.jsonl')!=ck['hashes_manifiestos']['val']:raise ValueError('Val histórico distinto')
    metadatos=[{k:r[k] for k in ['id','ruta','ancho','alto']} for r in val.base.origen.registros]
    for m in metadatos:m['sha256_imagen']=sha(RAIZ/m['ruta'])
    m=DetectorMapas(**ck['configuracion']).cuda();m.load_state_dict(ck['pesos'],strict=True);m.eval()
    una,cinco,costes,pp=inferir_sin_gt(m,val.base.imagenes,metadatos,carpeta);regresion={};metricas={}
    for v,ps,f in [('1',una,'detector_mapas_v6_auxiliar_ext48_predicciones_val.json'),('5',cinco,'detector_mapas_v6_mosaico_predicciones_val.json')]:
        reg=comparar_predicciones_historicas(pp[v],leer(RAIZ/'resultados'/f),True)
        regresion[v]={**reg,'archivo':f,'sha256':sha(RAIZ/'resultados'/f)}
        metricas[v]=resumir_deteccion(evaluar_detecciones(ps,val.base.objetivos))
    publicar(salida,{'aprobada':True,'sha256_checkpoint':sha(path),'hashes_codigo':huellas_codigo(),
        'backend_historico':ef,'regresion':regresion,'metricas':metricas,'costes':costes,
        'sha256_val':sha(RAIZ/'datos/procesados/deteccion/val.jsonl'),'test_evaluado':False})
    print('Referencia histórica exacta: 181 imágenes, una y cinco vistas',flush=True)


def verificar_etapa_20(cfg):
    exigir_autorizacion(cfg);diag=cargar_diagnostico(cfg);h=huellas_codigo()
    ck,train,aux,regiones,pm,est=contexto_train(cfg);val=DatosMapas('val',ContornosTACO(RAIZ,1/64));ramas={}
    salida=RAIZ/cfg['control_etapa_20']
    if salida.exists():raise FileExistsError('Etapa ya verificada')
    for rama in ['A','B']:
        diario=DiarioV20(RAIZ,cfg,rama,diag,h);actual,rs,_=diario.leer()
        if len(rs)!=1 or actual['epoca']!=20 or actual['actualizaciones_desde_origen']!=121:raise ValueError('Etapa 20 incompleta')
        comprobar_checkpoint_backend(actual,cfg,registro_efectivo())
        m=DetectorMapas(**actual['configuracion']).cuda();m.load_state_dict(actual['pesos'],strict=True);m.eval()
        met,pr=evaluar(m,DataLoader(val,batch_size=8,generator=torch.Generator().manual_seed(0)),'cuda',True,ck['config_entrenamiento'])
        hist=actual['historial'][0]
        if met['map50']!=hist['map50_val'] or met['map50_95']!=hist['map50_95_val']:raise ValueError('Validación de época 20 no reproducida')
        p=salida.parent/f'etapa20_{rama}_predicciones_1v.json'
        publicar(p,[{'id':r['id'],**{k:v.tolist() for k,v in pred.items()}} for r,pred in zip(val.base.origen.registros,pr)])
        ramas[rama]={'epoca':20,'actualizaciones':121,'sha256_checkpoint':rs[0]['sha256_checkpoint'],
            'huellas_estado':actual['reanudacion']['huellas_estado'],'estado_inicial':actual['reanudacion']['estado_inicial_origen'],
            'traza':hist['huella_orden_y_aumentos'],'backend_efectivo':actual['backend_efectivo'],
            'validacion_reproducida_exactamente':True,'resumen':resumir_deteccion(met),'sha256_predicciones':sha(p)}
        del m;torch.cuda.empty_cache()
    if ramas['A']['traza']!=ramas['B']['traza'] or ramas['A']['estado_inicial']!=ramas['B']['estado_inicial']:raise ValueError('Ramas con entradas/estados diferentes')
    publicar(salida,{'aprobada':True,'ramas':ramas,'hashes_codigo':h,'backend_numerico':cfg['backend_numerico'],'test_evaluado':False})
    print('Etapa 20 A/B verificada: diarios, estados, trazas y validación exacta',flush=True)


def exigir_entrenamiento_completo(cfg,compatibilidad=None):
    validar(cfg);path=RAIZ/f"resultados/{cfg['nombre']}.json"
    if not path.exists():raise ValueError('V20 A/B completo no ejecutado: validación permanece pendiente')
    r=leer(path)
    if r.get('estado')!='entrenamiento_v20_completo' or r['protocolo']!=protocolo(cfg):raise ValueError('Manifiesto ajeno')
    if any(r['ramas'][rama]['actualizaciones']!=1452 for rama in ['A','B']):raise ValueError('Presupuesto incompleto')
    diag=cargar_diagnostico(cfg);h=r['hashes_codigo'];actuales=huellas_codigo()
    if h!=actuales:
        if compatibilidad is None:raise ValueError('Código de evaluación diferente: exigir certificado explícito')
        validar_compatibilidad_codigo(h,actuales,leer(compatibilidad),protocolo(cfg))
    if not diag['aprobada'] or diag['hashes_codigo']!=h or diag['protocolo']!=protocolo(cfg):raise ValueError('Diagnóstico de entrenamiento incompatible')
    if diag['torch']!=str(torch.__version__) or diag['gpu']!=torch.cuda.get_device_name():raise ValueError('Entorno distinto')
    for campo in ['prueba_corta','control_backend']:
        c=leer(RAIZ/cfg[campo])
        if not c['aprobada'] or c['hashes_codigo']!=h or c['backend_numerico']!=cfg['backend_numerico']:raise ValueError('Certificado de entrenamiento incompatible')
    for rama in cfg['ramas']:
        d=DiarioV20(RAIZ,cfg,rama,diag,h);c,rs,_=d.leer()
        if len(rs)!=12 or c['actualizaciones_desde_origen']!=1452:raise ValueError('Diario incompleto')
        mejor=max(rs,key=lambda e:e['historial_epoca']['map50_val'])
        for tipo,ev in [('ultimo',rs[-1]),('mejor',mejor)]:
            if sha(RAIZ/f'checkpoints/{d.nombre}_{tipo}.pt')!=ev['sha256_checkpoint']:raise ValueError('Alias incorrecto')
        if r['ramas'][rama]['historial']!=c['historial']:raise ValueError('Historial del manifiesto alterado')
    return r


def agregar_regiones_cpu(raw,regiones):
    """Solo análisis GT; el forward ya terminó bajo el backend GPU seleccionado."""
    return metricas_instancias(raw.detach().cpu(),regiones)


@torch.inference_mode()
def evaluar_principal_train_cpu(modelo,train,regiones):
    modelo.eval();mat=torch.zeros(7,7,dtype=torch.float64);cond=torch.zeros(6,6,dtype=torch.float64)
    mi=torch.zeros(7,7,dtype=torch.long);mci=torch.zeros(6,6,dtype=torch.long);pix=mi.clone();filas=[]
    for inicio in range(0,len(train),8):
        ids=list(range(inicio,min(inicio+8,len(train))));x=train.base.imagenes[ids].cuda().float()/255
        with torch.autocast('cuda',dtype=torch.float16):raw=modelo(x)
        rr=agregar_regiones_cpu(raw,[regiones[i] for i in ids])
        mat+=torch.tensor(rr['confusion_token_normalizada_por_instancia']);cond+=torch.tensor(rr['confusion_condicional_token_normalizada_por_instancia'])
        mi+=torch.tensor(rr['confusion_instancia_voto_principal']);mci+=torch.tensor(rr['confusion_instancia_voto_condicional'])
        for f in rr['filas']:f['imagen']=train.base.origen.registros[ids[f['imagen_lote']]]['id'];filas.append(f)
        y=train.mascaras[ids];pp=raw.argmax(1).cpu();ok=y!=255;pix+=torch.bincount(7*y[ok]+pp[ok],minlength=49).reshape(7,7)
    return {'fuente':'logits de modelo.cabezal en train; forward GPU V23, agregación GT CPU determinista',
        'instancias':len(filas),'orden_siete':['fondo',*CLASES],'orden_materiales':CLASES,
        'confusion_token_normalizada_por_instancia':mat.tolist(),'confusion_condicional_token_normalizada_por_instancia':cond.tolist(),
        'confusion_instancia_voto_principal':mi.tolist(),'confusion_instancia_voto_condicional':mci.tolist(),
        'semantica_pixel':metricas_semanticas(pix),'filas':filas,
        'por_material':{c:{'instancias':sum(f['clase']==i for f in filas),
            'recuperadas':sum(f['clase']==i and f['recuperada_mayoria_gt'] for f in filas),
            'condicionales_recuperadas':sum(f['clase']==i and f['condicional_correcta_mayoria'] for f in filas),
            'fraccion_fondo_media':sum(f['fraccion_fondo'] for f in filas if f['clase']==i)/max(1,sum(f['clase']==i for f in filas)),
            'fraccion_material_equivocado_media':sum(f['fraccion_material_equivocado'] for f in filas if f['clase']==i)/max(1,sum(f['clase']==i for f in filas))} for i,c in enumerate(CLASES)}}


@torch.inference_mode()
def reusar_captura_origen(modelo,val,carpeta,hashes_entrenamiento):
    c=leer(carpeta/'captura.json')
    if c['hashes_codigo']!=hashes_entrenamiento or c['backend_efectivo']!=registro_efectivo() or c['gt_utilizado'] or c['imagenes']!=181:raise ValueError('Captura previa incompatible')
    for f in c['archivos']:
        if sha(RAIZ/f['archivo'])!=f['sha256']:raise ValueError('Captura previa alterada')
    muestra=torch.load(carpeta/'logits_1v/000.pt',map_location='cpu',weights_only=True)
    with torch.autocast('cuda',dtype=torch.float16):raw=modelo(val.base.imagenes[:8].cuda().float()/255)
    if not torch.equal(raw.cpu(),muestra['logits']):raise ValueError('Captura no corresponde al origen/backend esperado')
    pp={v:leer(carpeta/f'predicciones_{v}v_val.json') for v in ['1','5']};preds={}
    for v,rs in pp.items():
        if [x['id'] for x in rs]!=[x['id'] for x in val.base.origen.registros]:raise ValueError('Orden de captura diferente')
        preds[v]=[{'boxes':torch.tensor(x['boxes'],dtype=torch.float32).reshape(-1,4),'scores':torch.tensor(x['scores'],dtype=torch.float32),
            'labels':torch.tensor(x['labels'],dtype=torch.long)} for x in rs]
    return preds['1'],preds['5'],c['costes'],pp


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
        'gt_utilizado':False,'regresion_observador_5v_exacta':len(pred5),'hashes_codigo':huellas_codigo(),
        'backend_efectivo':registro_efectivo()})
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


def evaluar_completo(cfg,compatibilidad=None,directorio=None,reusar_origen=None):
    manifiesto=exigir_entrenamiento_completo(cfg,compatibilidad)
    salida=RAIZ/f"resultados/{cfg['nombre']}_evaluacion.json";carpeta=directorio or salida.with_suffix('')
    if salida.exists() or carpeta.exists():raise FileExistsError('Evaluación existente; conservar')
    ck,train,aux,regiones,pm,est=contexto_train(cfg)
    val=DatosMapas('val',ContornosTACO(RAIZ,1/64));contornos=ContornosTACO(RAIZ,1/64)
    if len(val)!=181 or sha(RAIZ/'datos/procesados/deteccion/val.jsonl')!=ck['hashes_manifiestos']['val']:raise ValueError('Val cambió')
    metadatos=[{k:r[k] for k in ['id','ruta','ancho','alto']} for r in val.base.origen.registros]
    for m in metadatos:m['sha256_imagen']=sha(RAIZ/m['ruta'])
    paths={'origen':RAIZ/cfg['checkpoint_origen'],**{f'{r}_{t}':RAIZ/f"checkpoints/{cfg['nombre']}_{r}_{t}.pt"
        for r in ['A','B'] for t in ['ultimo','mejor']}}
    resultados={};vistos={};carpeta.mkdir(parents=True)
    referencia=leer(RAIZ/cfg['referencia_historica']) if cfg.get('referencia_historica') else None
    if referencia and (not referencia['aprobada'] or referencia['hashes_codigo']!=manifiesto['hashes_codigo'] or referencia['sha256_checkpoint']!=cfg['sha256_checkpoint_origen']):
        raise ValueError('Regresión histórica ausente o de otro protocolo')
    for nombre,path in paths.items():
        h=sha(path)
        if h in vistos:
            entrada=copy.deepcopy(resultados[vistos[h]]);entrada['reutiliza_captura_de']=vistos[h]
            resultados[nombre]=entrada;publicar(carpeta/f'{nombre}.json',entrada);continue
        actual=torch.load(path,map_location='cpu',weights_only=True)
        comprobar_checkpoint_backend(actual,cfg,registro_efectivo(),origen=nombre=='origen',hashes_entrenamiento=manifiesto['hashes_codigo'])
        m=DetectorMapas(**actual['configuracion']).cuda();m.load_state_dict(actual['pesos'],strict=True);m.eval()
        p=carpeta/nombre;cap=reusar_origen if nombre=='origen' and reusar_origen else p
        if cap!=p:
            p.mkdir(parents=True);una,cinco,costes,pp=reusar_captura_origen(m,val,cap,manifiesto['hashes_codigo'])
            publicar(p/'reuso.json',{'captura_origen':str(cap.relative_to(RAIZ)),'checkpoint_sha256':h,'muestra_logit_8_imagenes_exacta':True,
                'archivos_captura_verificados':True,'hashes_codigo_inferencia_original':manifiesto['hashes_codigo']})
        else:una,cinco,costes,pp=inferir_sin_gt(m,val.base.imagenes,metadatos,p)
        regresion={}
        if nombre=='origen':
            for v,f in [('1','detector_mapas_v6_auxiliar_ext48_predicciones_val.json'),('5','detector_mapas_v6_mosaico_predicciones_val.json')]:
                historico=leer(RAIZ/'resultados'/f)
                regresion[v]={**comparar_predicciones_historicas(pp[v],historico,not bool(cfg.get('backend_numerico'))),
                    'sha256_historico':sha(RAIZ/'resultados'/f),'archivo_historico':f}
        # GT se consulta desde aquí, después de terminar y archivar inferencia.
        filas=[]
        for i,r in enumerate(val.base.origen.registros):
            traza=cargar_traza(cap/'trazas_5v'/f'{i:03d}.pt');ff,_=analizar_imagen(traza,r,contornos);filas+=ff
        publicar(p/'objetos_val_5v.json',filas)
        inicio=time.perf_counter();torch.cuda.reset_peak_memory_stats();principal=evaluar_principal_train_cpu(m,train,regiones)
        torch.cuda.synchronize();coste_train={'segundos':time.perf_counter()-inicio,'memoria_gpu_max_allocated_bytes':torch.cuda.max_memory_allocated()}
        publicar(p/'principal_train.json',principal)
        entrada={'sha256_checkpoint':h,'epoca':actual['epoca'],'principal_train':principal,'val_1_vista':{},'val_5_vistas':{},
            'oportunidades_y_geometria_val':resumir_objetos(filas),'regresion_historica':regresion,
            'captura':(cap/'captura.json').relative_to(RAIZ).as_posix(),'coste_principal_train':coste_train}
        entrada['backend_efectivo']=registro_efectivo()
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
        'referencia_historica_backend_original':referencia,'backend_numerico_medicion':cfg.get('backend_numerico'),
        'control_A_frente_V18':comparar_control_v18(manifiesto),'hashes_codigo':huellas_codigo(),
        'hashes_codigo_entrenamiento':manifiesto['hashes_codigo'],'compatibilidad_evaluacion':leer(compatibilidad) if compatibilidad else None,
        'agregacion_regional':'CPU; inferencia V23 y determinismo estricto conservados',
        'test_evaluado':False,'cambio_predeterminado':False,'adopcion_automatica':False,
        'limitaciones':['Una semilla; validación reutilizada de desarrollo; AP propio de 101 puntos, no COCOeval oficial.',
            'Coste de captura incluye escritura de logits, trazas y observador; no es latencia pura de producción.'],
        'criterio':'Revisar AP seis materiales, recall, precisión, FP, geometría, oportunidades, fondo y coste conjuntamente.'})


def main():
    a=argumentos();cfg=leer(a.config) if a.config else leer_cfg();validar(cfg);torch.set_num_threads(4)
    if a.referencia_historica:verificar_referencia_historica(cfg);return
    with contexto_evaluacion(cfg):
        if a.verificar_etapa_20:verificar_etapa_20(cfg)
        else:evaluar_completo(cfg,a.compatibilidad_evaluacion,
            a.directorio_capturas.resolve() if a.directorio_capturas else None,a.reusar_origen.resolve() if a.reusar_origen else None)


if __name__=='__main__':main()
