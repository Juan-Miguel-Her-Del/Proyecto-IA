"""Estado reproducible y métricas semánticas para una ablación pareada."""
import hashlib
import json
import torch


def huella(valor):
    h=hashlib.sha256()
    def visitar(v):
        if torch.is_tensor(v):
            t=v.detach().cpu().contiguous()
            h.update(f'tensor:{t.dtype}:{tuple(t.shape)}:'.encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,dict):
            h.update(b'dict{')
            for k in sorted(v,key=lambda k:(type(k).__name__,repr(k))):visitar(k);visitar(v[k])
            h.update(b'}')
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode()+b'[')
            for x in v:visitar(x)
            h.update(b']')
        else:h.update((type(v).__name__+':'+repr(v)+';').encode())
    visitar(valor)
    return h.hexdigest()


def estado_aleatorio(rng,rng_aux):
    return {'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'rng_loader':rng.get_state(),'rng_auxiliar':rng_aux.get_state()}


def restaurar_aleatorio(estado,rng,rng_aux):
    torch.set_rng_state(estado['rng_torch'].cpu())
    rng.set_state(estado['rng_loader'].cpu());rng_aux.set_state(estado['rng_auxiliar'].cpu())
    if torch.cuda.is_available():torch.cuda.set_rng_state_all([r.cpu() for r in estado['rng_cuda']])


def pasos_adam(opt):
    pasos={int(s['step']) for s in opt.state.values() if 'step' in s}
    if len(pasos)!=1:raise ValueError(f'Contadores de AdamW no uniformes: {pasos}')
    return pasos.pop()


def metricas_semanticas(matriz):
    cm=torch.as_tensor(matriz,dtype=torch.int64).cpu()
    if cm.shape!=(7,7) or bool((cm<0).any()):raise ValueError('Matriz semántica inválida')
    from ecowaste.datasets import CLASES
    nombres=['fondo',*CLASES];detalle={}
    for i,n in enumerate(nombres):
        tp=int(cm[i,i]);union=int(cm[i].sum()+cm[:,i].sum()-tp)
        detalle[n]={'celdas_reales':int(cm[i].sum()),'celdas_predichas':int(cm[:,i].sum()),
                    'iou':tp/union if union else None}
    validos=[d['iou'] for d in detalle.values() if d['iou'] is not None]
    materiales=[detalle[n]['iou'] for n in CLASES if detalle[n]['iou'] is not None]
    return {'matriz_confusion':cm.tolist(),'orden':['fondo',*CLASES],'por_clase':detalle,
        'miou_siete_clases':sum(validos)/len(validos) if validos else None,
        'miou_materiales':sum(materiales)/len(materiales) if materiales else None,
        'material_predicho_fondo':int(cm[1:,0].sum()),'fondo_predicho_material':int(cm[0,1:].sum()),
        'confusion_binaria_fondo_material':[[int(cm[0,0]),int(cm[0,1:].sum())],
                                          [int(cm[1:,0].sum()),int(cm[1:,1:].sum())]],
        'celdas_validas':int(cm.sum()),'excluye_etiqueta_255':True}


def resumir_deteccion(m):
    filas=[x['por_umbral'][0] for x in m['por_clase'].values()]
    tp=sum(x['tp'] for x in filas);fp=sum(x['fp'] for x in filas)
    return {'map50':m['map50'],'map50_95':m['map50_95'],'recall50':m['recall50'],'precision50':m['precision50'],
            'tp50':tp,'fp50':fp,'fn50':m['objetos']-tp,
            'ap50_por_material':{c:x['ap50'] for c,x in m['por_clase'].items()}}


def parametros_experimento(ck,config):
    if ck['tipo']!='ecowaste_mapas_desde_cero_v1' or ck['epoca']!=19 or ck['pesos_externos']:
        raise ValueError('Se requiere el checkpoint propio de época 19')
    if config['epocas_adicionales']<=0 or config['ramas']!={'A':3.,'B':1.}:
        raise ValueError('Protocolo de ramas inválido')
    if config['evaluar_test'] or config['cambiar_predeterminado']:raise ValueError('No se permite modificar selección ni evaluar test')
    if ck['config_entrenamiento']['formulacion']!='softmax7' or ck['recortes_train'] is not None:
        raise ValueError('Origen incompatible con el protocolo de mapas completos')
    if any(g['lr']!=ck['config_entrenamiento']['learning_rate'] for g in ck['optimizador']['param_groups']):
        raise ValueError('El LR guardado no coincide con la fase posterior al warmup')
