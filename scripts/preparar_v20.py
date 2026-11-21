"""Protocolo V20 y operación de lote. Esta preparación no ejecuta entrenamiento A/B."""
import argparse
import ast
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from torch.utils.data import DataLoader
from ecowaste.datasets import CLASES
from ecowaste.poligonos import ContornosTACO
from ecowaste.auxiliar_clasificacion import DatosClasificacionAuxiliar
from ecowaste.detector_mapas import perdida_mapas
from ecowaste.discriminacion_mapas import perdida_discriminativa,metricas_instancias,PESOS
from ecowaste.supervision_regional import preparar_regiones,VERSION as POLITICA
from ecowaste.ablacion_fondo import metricas_semanticas
from ecowaste.reanudacion_regional import sha
from scripts.entrenar_detector_mapas import DatosMapas

CONFIG=RAIZ/'configuracion/aprendizaje_discriminativo_mapas.json'
CODIGOS=['ecowaste/discriminacion_mapas.py','ecowaste/detector_mapas.py','ecowaste/detector.py',
    'ecowaste/transformer.py','ecowaste/clasificador.py','ecowaste/supervision_regional.py',
    'ecowaste/poligonos.py','ecowaste/datasets.py','ecowaste/auxiliar_clasificacion.py',
    'ecowaste/reanudacion_regional.py','ecowaste/ablacion_fondo.py','ecowaste/mosaico.py','ecowaste/cajas.py',
    'scripts/entrenar_detector_mapas.py','scripts/supervision_regional_materiales.py','scripts/ablacion_fondo.py',
    'scripts/preparar_v20.py','scripts/diagnosticar_v20.py','scripts/evaluar_v20.py',
    'pruebas/test_discriminacion_mapas.py','ecowaste/reanudacion_v20.py',
    'scripts/entrenar_v20.py','scripts/probar_entrenamiento_v20.py','pruebas/test_entrenamiento_v20.py',
    'scripts/ejecutar_diagnostico_reproducibilidad_v20.py','pruebas/test_reproducibilidad_v20.py']


def leer_cfg():return json.loads(CONFIG.read_text(encoding='utf-8'))
def dependencias_locales(raices):
    pendientes=list(raices);visitados=set()
    while pendientes:
        p=pendientes.pop()
        if p in visitados:continue
        path=RAIZ/p
        if not path.is_file():raise ValueError('Dependencia local ausente: '+p)
        visitados.add(p);arbol=ast.parse(path.read_text(encoding='utf-8-sig'),filename=p)
        for nodo in ast.walk(arbol):
            nombres=[]
            if isinstance(nodo,ast.Import):nombres=[a.name for a in nodo.names]
            elif isinstance(nodo,ast.ImportFrom) and nodo.module:
                nombres=[nodo.module]+[nodo.module+'.'+a.name for a in nodo.names]
            for nombre in nombres:
                ruta=Path(*nombre.split('.')).with_suffix('.py')
                if (RAIZ/ruta).is_file():pendientes.append(ruta.as_posix())
    return sorted(visitados)

def huellas_codigo():return {p:sha(RAIZ/p) for p in dependencias_locales(CODIGOS)}
def protocolo(cfg):return {k:v for k,v in cfg.items() if k!='entrenamiento_completo_habilitado'}


def validar(cfg):
    esperado={'ramas':{'A':0.,'B':.25},'margen_logits':1.,'fraccion_dificiles':.25,'delta_huber':1.,
        'pesos_clase_fijos':list(PESOS),'politica_tokens':POLITICA,'coeficiente_fondo':3.,'coeficiente_auxiliar_global':.5,
        'epocas_adicionales':12,'actualizaciones_por_rama':1452,'parada_temprana':False,'reiniciar_warmup':False,
        'criterio_mejor':'map50_val_una_vista','desempate':'primera_epoca','politica_mapas':'argmax7',
        'vistas_evaluacion':[1,5],'umbral_evaluacion':.05,'umbral_visual':.35,'nms_iou':.5,'top100':100,
        'area_minima_celdas':2,'evaluar_test':False,'cambiar_predeterminado':False}
    if any(cfg.get(k)!=v for k,v in esperado.items()):raise ValueError('Protocolo V20 incompatible')
    if Path(cfg['nombre']).name!=cfg['nombre']:raise ValueError('Nombre inválido')


def exigir_autorizacion(cfg):
    validar(cfg)
    if not cfg['entrenamiento_completo_habilitado']:
        raise ValueError('V20 completo deshabilitado: entregar y revisar primero diagnóstico train-only')
    d=json.loads((RAIZ/cfg['diagnostico']).read_text(encoding='utf-8'))
    if not d['aprobada'] or d['hashes_codigo']!=huellas_codigo() or d['protocolo']!=protocolo(cfg):
        raise ValueError('Diagnóstico V20 no aprobado o desactualizado')
    if d['torch']!=str(torch.__version__) or not torch.cuda.is_available() or d['gpu']!=torch.cuda.get_device_name():
        raise ValueError('Entorno distinto del diagnóstico')
    corta=json.loads((RAIZ/cfg['prueba_corta']).read_text(encoding='utf-8'))
    if not corta['aprobada'] or corta['hashes_codigo']!=huellas_codigo() or corta['sha256_diagnostico']!=sha(RAIZ/cfg['diagnostico']):
        raise ValueError('Prueba corta real ausente o desactualizada')


def contexto_train(cfg):
    validar(cfg);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise ValueError('Se conserva el entorno CUDA original')
    if sha(RAIZ/cfg['checkpoint_origen'])!=cfg['sha256_checkpoint_origen']:raise ValueError('Checkpoint diferente')
    ck=torch.load(RAIZ/cfg['checkpoint_origen'],map_location='cpu',weights_only=True);h=ck['hashes_manifiestos']
    if ck['epoca']!=19 or ck['pesos_externos'] or ck['recortes_train'] is not None:raise ValueError('Origen incompatible')
    if ck['configuracion']['formulacion']!='softmax7' or not ck['configuracion']['auxiliar_clasificacion']:raise ValueError('Arquitectura incompatible')
    if sha(RAIZ/'datos/procesados/deteccion/train.jsonl')!=h['train']:raise ValueError('Train cambiado')
    base=ck['config_entrenamiento']
    if base['batch_size']!=8 or base['clasificacion_auxiliar']['batch_size']!=16 or base['learning_rate']!=.0005:
        raise ValueError('Presupuesto/base alterados')
    if base['clasificacion_auxiliar']['coeficiente_perdida']!=.5:raise ValueError('Auxiliar global alterada')
    c=ContornosTACO(RAIZ,1/64)
    if c.sha256!=h['anotaciones_poligonos'] or c.sha256_inventario!=h['inventario_procedencia']:raise ValueError('Polígonos distintos')
    train=DatosMapas('train',c);regiones,_,est=preparar_regiones(train.base.origen.registros,c)
    est.pop('pesos_clase');est.pop('formula_pesos');est['pesos_v20']=list(PESOS)
    aux=DatosClasificacionAuxiliar(RAIZ)
    if sha(aux.metadata_path)!=h['cache_clasificacion_auxiliar'] or aux.metadata['sha256_manifiesto']!=h['clasificacion_auxiliar_train']:
        raise ValueError('Clasificación global diferente')
    counts=torch.bincount(train.mascaras[train.mascaras!=255],minlength=7)[1:].float()
    pm=counts.clamp(min=1e-6).rsqrt();pm=(pm/pm.mean()).clamp(max=3).cuda()
    if len(train)!=964 or 121*12!=1452:raise ValueError('Presupuesto distinto')
    return ck,train,aux,regiones,pm,est


def calcular_lote(modelo,x,y,regiones,pesos_fg,coeficiente,flips=None):
    if coeficiente not in (0.,.25):raise ValueError('Coeficiente no prefijado')
    raw=modelo(x)
    nueva,d=perdida_discriminativa(raw,regiones,flips,con_gradiente=coeficiente>0)
    original=perdida_mapas(raw,y,pesos_fg,coeficiente_fondo=3.)
    return (original+coeficiente*nueva if coeficiente else original),raw,original,nueva,d


@torch.inference_mode()
def evaluar_principal_train(modelo,train,regiones,amp=True,indices=None):
    modelo.eval();mat=torch.zeros(7,7,dtype=torch.float64);cond=torch.zeros(6,6,dtype=torch.float64)
    mi=torch.zeros(7,7,dtype=torch.long);mci=torch.zeros(6,6,dtype=torch.long);pix=mi.clone();filas=[]
    indices=list(range(len(train))) if indices is None else list(indices)
    for inicio in range(0,len(indices),8):
        ids=indices[inicio:inicio+8];x=train.base.imagenes[ids].cuda().float()/255
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp):raw=modelo(x)
        rr=metricas_instancias(raw,[regiones[i] for i in ids])
        mat+=torch.tensor(rr['confusion_token_normalizada_por_instancia']);cond+=torch.tensor(rr['confusion_condicional_token_normalizada_por_instancia'])
        mi+=torch.tensor(rr['confusion_instancia_voto_principal']);mci+=torch.tensor(rr['confusion_instancia_voto_condicional'])
        for f in rr['filas']:
            f['imagen']=train.base.origen.registros[ids[f['imagen_lote']]]['id'];filas.append(f)
        y=train.mascaras[ids];pp=raw.argmax(1).cpu();ok=y!=255
        pix+=torch.bincount(7*y[ok]+pp[ok],minlength=49).reshape(7,7)
    return {'fuente':'exclusivamente logits de modelo.cabezal en escenas train sin aumentos',
        'instancias':len(filas),'orden_siete':['fondo',*CLASES],'orden_materiales':CLASES,
        'confusion_token_normalizada_por_instancia':mat.tolist(),'confusion_condicional_token_normalizada_por_instancia':cond.tolist(),
        'confusion_instancia_voto_principal':mi.tolist(),'confusion_instancia_voto_condicional':mci.tolist(),
        'semantica_pixel':metricas_semanticas(pix),'filas':filas,
        'por_material':{c:{'instancias':sum(f['clase']==i for f in filas),
            'recuperadas':sum(f['clase']==i and f['recuperada_mayoria_gt'] for f in filas),
            'condicionales_recuperadas':sum(f['clase']==i and f['condicional_correcta_mayoria'] for f in filas),
            'fraccion_fondo_media':sum(f['fraccion_fondo'] for f in filas if f['clase']==i)/max(1,sum(f['clase']==i for f in filas)),
            'fraccion_material_equivocado_media':sum(f['fraccion_material_equivocado'] for f in filas if f['clase']==i)/max(1,sum(f['clase']==i for f in filas))}
            for i,c in enumerate(CLASES)}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--entrenar',action='store_true');a=p.parse_args()
    cfg=leer_cfg();validar(cfg)
    if a.entrenar:
        exigir_autorizacion(cfg)
        from scripts.entrenar_v20 import entrenar
        entrenar(cfg)
        return
    print(json.dumps({'protocolo':protocolo(cfg),'entrenamiento_ab_lanzado':False,
        'operacion_de_lote':'calcular_lote, con una sola pasada TACO y auxiliar global separada sin cambios'},ensure_ascii=False))


if __name__=='__main__':main()
