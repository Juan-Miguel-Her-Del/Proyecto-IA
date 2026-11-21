"""Captura sin GT y análisis posterior de propuestas; conserva producción y test."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import numpy as np
from PIL import Image
import torch
from ecowaste.datasets import ResiduosDataset,letterbox,CLASES
from ecowaste.detector_mapas import cargar_detector_mapas
from ecowaste.mosaico import ventanas
from ecowaste.traza_propuestas import trazar_logits,ETAPAS,VERSION as TRAZA_VERSION
from ecowaste.auditoria_propuestas_internas import analizar_imagen,resumir_objetos,VERSION as ANALISIS_VERSION
from ecowaste.poligonos import ContornosTACO
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.ablacion_fondo import huella,resumir_deteccion
from ecowaste.reanudacion_regional import publicar,sha

CONFIG=RAIZ/'configuracion/auditoria_propuestas_internas.json'
CORE=['ecowaste/detector.py','ecowaste/detector_mapas.py','ecowaste/mosaico.py','ecowaste/cajas.py',
    'ecowaste/datasets.py','ecowaste/transformer.py','ecowaste/clasificador.py']
CODIGOS=CORE+['ecowaste/traza_propuestas.py','ecowaste/auditoria_propuestas_internas.py',
    'scripts/auditar_propuestas_internas.py','pruebas/test_propuestas_internas.py']


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def validar(cfg):
    if cfg!=leer(CONFIG) or cfg['split']!='val' or cfg['imagenes']!=181 or cfg['vistas']!=5:
        raise ValueError('Solo se audita el protocolo seleccionado en val')
    if cfg['entrenar'] or cfg['evaluar_test'] or cfg['modificar_inferencia']:raise ValueError('Ámbito alterado')
    esperado={'politica':'argmax7','umbral':.05,'umbral_visual':.35,'area_minima':2,'nms_iou':.5,'top':100}
    if any(cfg[k]!=v for k,v in esperado.items()):raise ValueError('Umbrales distintos del detector seleccionado')
    if cfg['analisis']!={'iou_razonable':.5,'visibilidad_bbox_minima':.8,'fraccion_gt_minima_por_fragmento':.1,
        'cobertura_gt_minima_union_fragmentos':.5,'cobertura_gt_minima_por_objeto_fusionado':.5}:
        raise ValueError('Criterios de análisis no registrados')
    if sha(RAIZ/cfg['checkpoint'])!=cfg['sha256_checkpoint']:raise ValueError('Origen distinto')
    seleccion=leer(RAIZ/'configuracion/arquitectura.json')['detector']
    if seleccion['checkpoint']!=cfg['checkpoint'] or seleccion['modo_inferencia']!='mosaico_cinco_vistas':raise ValueError('No es el detector seleccionado')


def serializar_traza(t):
    copia={**t,'vistas':[]}
    for v in t['vistas']:
        copia['vistas'].append({k:torch.from_numpy(x) if isinstance(x,np.ndarray) else x for k,x in v.items()})
    return copia


def cargar_traza(path):
    t=torch.load(path,map_location='cpu',weights_only=True)
    for v in t['vistas']:
        for k in ['validas','ganador','probabilidades','componentes']:v[k]=v[k].numpy()
    return t


def proteger(salida):
    archivos=[*list((RAIZ/'resultados').rglob('*')),*list((RAIZ/'configuracion').glob('*.json')),
        *list((RAIZ/'checkpoints').glob('*.pt')),*[RAIZ/p for p in CORE]]
    return {p.relative_to(RAIZ).as_posix():sha(p) for p in archivos if p.is_file() and not p.is_relative_to(salida)}


@torch.inference_mode()
def capturar(cfg,salida,metadatos):
    """Recibe solo ID/ruta/tamaño/hash de imagen; no acepta GT ni anotaciones."""
    clave={'checkpoint':cfg['sha256_checkpoint'],'val':sha(RAIZ/'datos/procesados/deteccion/val.jsonl'),
        'core':{p:sha(RAIZ/p) for p in CORE},'torch':str(torch.__version__),'amp':True,'vistas':5}
    cache=RAIZ/'datos/cache'/('logits_auditoria_'+huella(clave)[:20]);cache.mkdir(parents=True,exist_ok=True)
    modelo=None;estado=None;inferencias=0;reutilizados=0;geometrias=[];referencia=leer(RAIZ/'resultados/detector_mapas_v6_mosaico_predicciones_val.json')
    archivos_cache=[];inicio=time.perf_counter();predicciones=[];indices=[]
    if (cache/'indice.json').exists():
        indice=leer(cache/'indice.json')
        if indice['clave']!=clave:raise ValueError('Caché de otro protocolo')
        for f,h in indice['sha256_archivos'].items():
            if sha(cache/f)!=h:raise ValueError('Logits en caché alterados')
    with gzip.open(salida/'propuestas.jsonl.gz','xt',encoding='utf-8') as log:
        for i,m in enumerate(metadatos):
            path=cache/f'{i:03d}.pt'
            if path.exists():
                ck=torch.load(path,map_location='cpu',weights_only=True)
                if ck['clave']!=clave or ck['imagen']!=m:raise ValueError('Logits de otra imagen')
                raw=ck['logits'].cuda();gs,zs=ck['geometrias'],ck['zonas'];reutilizados+=1
            else:
                if modelo is None:
                    modelo,ck_modelo=cargar_detector_mapas(RAIZ/cfg['checkpoint'],'cuda');modelo.requires_grad_(False)
                    if ck_modelo['epoca']!=19:raise ValueError('Se requiere época 19')
                    estado=huella(modelo.state_dict())
                if sha(RAIZ/m['ruta'])!=m['sha256_archivo']:raise ValueError('Imagen alterada')
                with Image.open(RAIZ/m['ruta']) as im:imagen=im.convert('RGB')
                zs=ventanas(imagen.width,imagen.height);xs=[letterbox(imagen.crop(z),384) for z in zs]
                gs=[g for _,g in xs]
                with torch.autocast('cuda',dtype=torch.float16):raw=modelo(torch.stack([x for x,_ in xs]).cuda())
                publicar(path,{'clave':clave,'imagen':m,'logits':raw.cpu(),'geometrias':gs,'zonas':zs},tensor=True);inferencias+=1
            t=trazar_logits(raw,gs,zs,m['ancho'],m['alto'],m['id'])
            pred={'id':m['id'],**{k:v.tolist() for k,v in t['final'].items()}}
            if pred!=referencia[i]:raise AssertionError('Predicción final diferente de la referencia histórica: '+m['id'])
            publicar(salida/'trazas'/f'{i:03d}.pt',serializar_traza(t),tensor=True)
            for v in t['vistas']:
                for celda in v['celdas']:log.write(json.dumps(celda,ensure_ascii=False)+'\n')
            for nodo in t['nodos'].values():
                for e in nodo['etapas']:
                    log.write(json.dumps({**nodo,'etapa':e,'caja':nodo['caja_global'],
                        'coordenadas':'xyxy_normalizado_letterbox_imagen_completa',
                        'contrafactual_sin_limite':e in ['nms_vista','nms_global']},ensure_ascii=False)+'\n')
                if nodo['descartada_en']:
                    log.write(json.dumps({**nodo,'etapa':'descartada','caja':nodo['caja_global']},ensure_ascii=False)+'\n')
            geometrias.append({'imagen':m['id'],'vistas':[{'vista':j,'ventana':list(z),'geometria':g} for j,(g,z) in enumerate(zip(gs,zs))]})
            predicciones.append(pred);archivos_cache.append(path)
            indices.append({'imagen':m['id'],'ruta':f'trazas/{i:03d}.pt','sha256':sha(salida/'trazas'/f'{i:03d}.pt')})
            if (i+1)%20==0 or i+1==len(metadatos):print(f'Captura: {i+1}/181, predicciones exactas',flush=True)
    if modelo is not None and (huella(modelo.state_dict())!=estado or any(p.grad is not None for p in modelo.parameters())):
        raise AssertionError('Cambió el estado del modelo')
    if not (cache/'indice.json').exists():publicar(cache/'indice.json',{'clave':clave,'sha256_archivos':{p.name:sha(p) for p in archivos_cache}})
    publicar(salida/'predicciones_val.json',predicciones);publicar(salida/'vistas.json',geometrias)
    publicar(salida/'captura.json',{'estado':'captura_completa_sin_gt','configuracion':cfg,'cache':cache.relative_to(RAIZ).as_posix(),
        'sha256_cache_indice':sha(cache/'indice.json'),'imagenes':181,'vistas':905,'llamadas_modelo':inferencias,
        'imagenes_logits_reutilizados':reutilizados,'predicciones_exactas':181,'trazas':indices,
        'hashes_codigo':{p:sha(RAIZ/p) for p in CODIGOS},'sha256_predicciones':sha(salida/'predicciones_val.json'),
        'sha256_propuestas':sha(salida/'propuestas.jsonl.gz'),'segundos':time.perf_counter()-inicio,
        'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),'traza_version':TRAZA_VERSION,
        'entrenamiento':False,'test_evaluado':False,'gt_utilizado_en_inferencia':False})


def analizar(cfg,salida,registros):
    captura=leer(salida/'captura.json')
    if captura['hashes_codigo']!={p:sha(RAIZ/p) for p in CODIGOS}:raise ValueError('Captura de otro código; conservar con otro nombre')
    contornos=ContornosTACO(RAIZ,1/64);filas=[];preds=[];objetivos=[];conteos={e:{c:0 for c in CLASES} for e in ETAPAS}
    descartes={};inicio=time.perf_counter();de_escenas=[]
    for i,r in enumerate(registros):
        path=salida/f'trazas/{i:03d}.pt'
        if sha(path)!=captura['trazas'][i]['sha256']:raise ValueError('Traza modificada')
        t=cargar_traza(path);ff,aud=analizar_imagen(t,r,contornos);filas.extend(ff);preds.append(t['final'])
        from ecowaste.datasets import objetivo_deteccion
        objetivos.append(objetivo_deteccion(r))
        de_escenas.append({'imagen':r['id'],'tp':aud['tp'],'fp':aud['fp'],'fn':aud['fn']})
        for e,ids in t['etapas'].items():
            for ident in ids:conteos[e][t['nodos'][ident]['material']]+=1
        for nodo in t['nodos'].values():
            if nodo['descartada_en']:
                etapa=nodo['descartada_en'];descartes.setdefault(etapa,{c:0 for c in CLASES})[nodo['material']]+=1
        if (i+1)%40==0:print(f'Análisis GT posterior: {i+1}/181',flush=True)
    metricas=evaluar_detecciones(preds,objetivos);resumen=resumir_objetos(filas)
    if resumen['total']['causas']['correcto']!=sum(x['tp'] for x in de_escenas):raise AssertionError('TP mal asignados')
    publicar(salida/'objetos_gt.json',filas)
    with (salida/'objetos_gt.csv').open('x',encoding='utf-8-sig',newline='') as f:
        campos=['imagen','id_gt','material','grupo_tamano','lado_menor_384','max_lado_menor_vistas_utiles_px','causa_principal','vista_testigo']
        w=csv.DictWriter(f,fieldnames=campos);w.writeheader()
        for fila in filas:w.writerow({k:fila[k] for k in campos})
    with (salida/'etapas.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['etapa','material','propuestas','gt_con_caja_correcta_iou50'])
        for e in ETAPAS:
            for c in CLASES:w.writerow([e,c,conteos[e][c],resumen['por_material'][c]['oportunidades_por_etapa'][e]])
    # Selección determinista de ejemplos, limitada a evidencias ya auditadas.
    ejemplos=[];usados=set()
    for clase in ['carton','vidrio','plastico','papel','metal','organico']:
        causas=sorted({f['causa_principal'] for f in filas if f['material']==clase})
        for causa in causas:
            candidatos=sorted([f for f in filas if f['material']==clase and f['causa_principal']==causa],key=lambda f:(f['imagen'],f['id_gt']))
            if not candidatos:continue
            ejemplo=candidatos[0];key=(ejemplo['imagen'],ejemplo['id_gt'])
            if key in usados:continue
            usados.add(key);ejemplos.append(ejemplo)
            if len(ejemplos)>=18:break
        if len(ejemplos)>=18:break
    for causa in ['perdida_postprocesamiento','fragmentacion','fusion_objetos']:
        if any(e['causa_principal']==causa for e in ejemplos):continue
        candidatos=[f for f in filas if f['causa_principal']==causa]
        if candidatos:ejemplos.append(candidatos[0])
    datos={};fichas=[]
    for j,ej in enumerate(ejemplos):
        i=next(i for i,r in enumerate(registros) if r['id']==ej['imagen']);r=registros[i];t=cargar_traza(salida/f'trazas/{i:03d}.pt')
        vista=ej['vista_testigo']
        if ej['fragmentos']:vista=ej['fragmentos'][0]['vista']
        elif ej['fusiones']:vista=ej['fusiones'][0]['vista']
        elif ej['ids_correctos_pre_filtros']:vista=t['nodos'][ej['ids_correctos_pre_filtros'][0]]['vista']
        v=t['vistas'][vista];ann=next(a for a in r['cajas'] if a['id_interno']==ej['id_gt'])
        from ecowaste.auditoria_propuestas_internas import cobertura_poligono
        gt=cobertura_poligono(contornos.contornos(r,ann),v['geometria'],v['ventana'],48)
        datos[f'p{j}']=v['probabilidades'][ej['clase']+1];datos[f'argmax{j}']=v['ganador'];datos[f'gt{j}']=gt
        fichas.append({'indice':j,'objeto':ej,'ruta':r['ruta'],'vista':vista,'ventana':v['ventana'],
            'geometria':v['geometria'],'nodos':[n for n in t['nodos'].values() if n['vista']==vista],
            'poligonos_originales':contornos.contornos(r,ann)})
    np.savez_compressed(salida/'ejemplos_mapas.npz',**datos);publicar(salida/'ejemplos.json',fichas)
    previo=leer(salida/'protegidos.json')
    cambiados=[p for p,h in previo.items() if sha(RAIZ/p)!=h]
    if cambiados:raise AssertionError('Se modificaron históricos: '+str(cambiados))
    publicar(salida/'analisis.json',{'estado':'auditoria_completa','configuracion':cfg,'version':ANALISIS_VERSION,
        'resumen':resumen,'conteos_propuestas':conteos,'descartes_reales':descartes,'metricas_finales':metricas,
        'resumen_metricas_finales':resumir_deteccion(metricas),'escenas':de_escenas,'segundos_analisis':time.perf_counter()-inicio,
        'sha256_poligonos':contornos.sha256,'sha256_inventario':contornos.sha256_inventario,
        'historicos_intactos':len(previo),'test_evaluado':False,'entrenamiento':False,
        'gt_solo_analisis':True,'hashes_codigo':captura['hashes_codigo']})
    print(json.dumps({'metricas':resumir_deteccion(metricas),'causas':resumen['total']},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--salida',type=Path)
    p.add_argument('--fase',choices=['captura','analisis','todo'],default='todo');a=p.parse_args()
    cfg=leer(CONFIG);validar(cfg);torch.set_num_threads(4)
    registros=ResiduosDataset(RAIZ,'deteccion','val').registros
    if len(registros)!=181 or any(r['split']!='val' for r in registros):raise ValueError('Solo 181 escenas val')
    salida=a.salida or RAIZ/'resultados'/cfg['nombre']
    if a.fase!='analisis':
        if salida.exists():raise FileExistsError('Salida existente; conservar históricos usando una ruta nueva')
        suite=unittest.defaultTestLoader.discover(str(RAIZ/'pruebas'),pattern='test_*.py')
        r=unittest.TextTestRunner(verbosity=1).run(suite)
        if not r.wasSuccessful() or r.skipped:raise RuntimeError('Pruebas no aprobadas')
        salida.mkdir(parents=True);publicar(salida/'pruebas.json',{'ejecutadas':r.testsRun,'aprobadas':True,'fallos':0,'errores':0,'omitidas':0})
        publicar(salida/'protegidos.json',proteger(salida))
        capturar(cfg,salida,[{k:r[k] for k in ['id','ruta','ancho','alto','sha256_archivo']} for r in registros])
    if a.fase!='captura':
        if (salida/'analisis.json').exists():raise FileExistsError('Análisis existente')
        analizar(cfg,salida,registros)


if __name__=='__main__':main()
