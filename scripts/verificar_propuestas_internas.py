"""Regresión, procedencia, contabilidad GT y replay de logits sin ejecutar el modelo."""
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.reanudacion_regional import publicar,sha
from ecowaste.traza_propuestas import trazar_logits,iguales,ETAPAS
from ecowaste.datasets import ResiduosDataset
from ecowaste.poligonos import ContornosTACO
from ecowaste.auditoria_propuestas_internas import analizar_imagen,CAUSAS
from scripts.auditar_propuestas_internas import cargar_traza,CODIGOS,CONFIG


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(4);cfg=leer(CONFIG);base=RAIZ/'resultados'/cfg['nombre']
    destino=base/'verificacion.json'
    if destino.exists():raise FileExistsError('Verificación histórica existente')
    cap=leer(base/'captura.json');r=leer(base/'analisis.json');fig=leer(base/'informe.json');pruebas=leer(base/'pruebas.json')
    assert pruebas['aprobadas'] and pruebas['ejecutadas']>=149
    assert not cap['gt_utilizado_en_inferencia'] and not cap['entrenamiento'] and not cap['test_evaluado']
    assert not r['entrenamiento'] and not r['test_evaluado'] and r['gt_solo_analisis']
    assert cap['predicciones_exactas']==181 and cap['vistas']==905
    assert cap['hashes_codigo']=={p:sha(RAIZ/p) for p in CODIGOS}==r['hashes_codigo']
    assert sha(RAIZ/cfg['checkpoint'])==cfg['sha256_checkpoint']
    ck=torch.load(RAIZ/cfg['checkpoint'],map_location='cpu',weights_only=True)
    assert ck['epoca']==19 and ck['hashes_manifiestos']['val']==sha(RAIZ/'datos/procesados/deteccion/val.jsonl')
    assert leer(base/'predicciones_val.json')==leer(RAIZ/'resultados/detector_mapas_v6_mosaico_predicciones_val.json')
    assert cap['sha256_predicciones']==sha(base/'predicciones_val.json')
    assert cap['sha256_propuestas']==sha(base/'propuestas.jsonl.gz')
    previos=leer(base/'protegidos.json');assert all(sha(RAIZ/p)==h for p,h in previos.items())
    assert all(sha(RAIZ/p)==h for p,h in fig['figuras'].items())
    doc=RAIZ/'documentacion/19_auditoria_propuestas_internas.md'
    assert sha(doc)==fig['sha256_informe']
    assert sha(RAIZ/'scripts/informar_propuestas_internas.py')==fig['sha256_script_figuras']
    for enlace in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
        assert (doc.parent/enlace).resolve().exists(),enlace
    registros=ResiduosDataset(RAIZ,'deteccion','val').registros;assert len(registros)==181
    assert all(x['split']=='val' for x in registros)
    filas=leer(base/'objetos_gt.json');assert len(filas)==361 and len({(f['imagen'],f['id_gt']) for f in filas})==361
    assert sum(r['resumen']['total']['causas'].values())==361
    for c in CAUSAS:assert sum(f['causa_principal']==c for f in filas)==r['resumen']['total']['causas'][c]
    nodos={};conteos={e:Counter() for e in ETAPAS};descartes={};indices_replay=[0,35,70,105,140,180]
    cache=RAIZ/cap['cache'];indice_cache=leer(cache/'indice.json')
    assert sha(cache/'indice.json')==cap['sha256_cache_indice']
    assert indice_cache['clave']['val']==sha(RAIZ/'datos/procesados/deteccion/val.jsonl')
    for p,h in indice_cache['sha256_archivos'].items():assert sha(cache/p)==h
    contornos=ContornosTACO(RAIZ,1/64);replays=[]
    assert contornos.sha256==r['sha256_poligonos'] and contornos.sha256_inventario==r['sha256_inventario']
    for i,registro in enumerate(registros):
        path=base/f'trazas/{i:03d}.pt';assert sha(path)==cap['trazas'][i]['sha256']
        t=cargar_traza(path);assert t['imagen']==registro['id'];nodos.update(t['nodos'])
        for e,ids in t['etapas'].items():
            assert len(set(ids))==len(ids)
            for ident in ids:
                assert e in t['nodos'][ident]['etapas'];conteos[e][t['nodos'][ident]['material']]+=1
        for nodo in t['nodos'].values():
            if nodo['descartada_en']:
                descartes.setdefault(nodo['descartada_en'],Counter())[nodo['material']]+=1
        if i in indices_replay:
            ca=torch.load(cache/f'{i:03d}.pt',map_location='cpu',weights_only=True)
            assert set(ca['imagen'])=={'id','ruta','ancho','alto','sha256_archivo'}
            nuevo=trazar_logits(ca['logits'].cuda(),ca['geometrias'],ca['zonas'],registro['ancho'],registro['alto'],registro['id'])
            assert iguales(nuevo['final'],t['final']) and nuevo['nodos']==t['nodos'] and nuevo['etapas']==t['etapas']
            ff,_=analizar_imagen(t,registro,contornos)
            assert ff==[f for f in filas if f['imagen']==registro['id']]
            replays.append(registro['id'])
    for e in ETAPAS:
        for c,v in r['conteos_propuestas'][e].items():assert conteos[e][c]==v
    for e,cs in r['descartes_reales'].items():
        for c,v in cs.items():assert descartes[e][c]==v
    celdas=Counter();eventos=Counter();registros_log=0
    with gzip.open(base/'propuestas.jsonl.gz','rt',encoding='utf-8') as f:
        for l in f:
            ev=json.loads(l);registros_log+=1
            if ev['etapa']=='antes_componentes':
                assert ev['imagen']==nodos[ev['componente']]['imagen'] and ev['vista']==nodos[ev['componente']]['vista']
                assert ev['clase']==nodos[ev['componente']]['clase'];celdas[ev['componente']]+=1
            else:
                n=nodos[ev['id']];assert ev['puntuacion']==n['puntuacion'] and ev['caja']==n['caja_global']
                eventos[ev['etapa']]+=1
    for ident,n in nodos.items():assert celdas[ident]==n['area_celdas']
    for e in ETAPAS:assert eventos[e]==sum(r['conteos_propuestas'][e].values())
    assert eventos['descartada']==sum(sum(x.values()) for x in r['descartes_reales'].values())
    ignoradas=sum(v['por_umbral'][0]['predicciones_ignoradas'] for v in r['metricas_finales']['por_clase'].values())
    m=r['resumen_metricas_finales']
    assert sum(r['conteos_propuestas']['final'].values())==m['tp50']+m['fp50']+ignoradas
    assert m['tp50']==74 and m['fp50']==2374 and m['fn50']==287
    for c in ['carton','vidrio']:
        assert r['resumen']['por_material'][c]['oportunidades_por_etapa']['componentes']==0
    documentos=[CONFIG,doc,RAIZ/'README.md',*[RAIZ/p for p in CODIGOS],
        RAIZ/'scripts/informar_propuestas_internas.py',Path(__file__),*list(base.glob('*.json')),
        *list(base.glob('*.csv')),*list(base.glob('*.npz')),*list(base.glob('*.gz')),*list((base/'graficas').glob('*.png'))]
    publicar(destino,{'aprobada':True,'pruebas':pruebas,'imagenes_val':181,'gt':361,'regresion_final_exacta':181,
        'replay_logits_sin_modelo':replays,'forwards_modelo_durante_verificacion':0,
        'gt_reanalizados_exactos':replays,'registros_propuestas_jsonl':registros_log,'componentes':len(nodos),
        'celdas_antes_componentes':sum(celdas.values()),'predicciones_ignoradas_solo_en_evaluacion':ignoradas,
        'historicos_intactos':len(previos),'checkpoint_inalterado':True,'entrenamiento':False,'test_evaluado':False,
        'sha256_entregables':{p.relative_to(RAIZ).as_posix():sha(p) for p in documentos}})
    print(json.dumps({'aprobada':True,'historicos_intactos':len(previos),'componentes':len(nodos),'registros_log':registros_log,'replay_sin_modelo':len(replays)}))


if __name__=='__main__':main()
