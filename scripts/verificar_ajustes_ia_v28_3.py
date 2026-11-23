"""Regresión sobre la copia IA real: máscaras, COCO, procedencia y originales."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
import numpy as np
from ecowaste.incorporacion import leer,sha,publicar,digest
from ecowaste.auxiliares_publicos import Taller,rle_coco
from ecowaste.validacion_anotaciones import mask_poligonos
from ecowaste.ajustes_ia_v28_3 import CLASES,decodificar,caja
from scripts.ajustar_anotaciones_ia_v28_3 import comprobar_origen
from scripts.anotar_v27_2 import PROYECTO


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preparacion',type=Path,required=True)
    p.add_argument('--version',required=True);a=p.parse_args()
    prep=leer(a.preparacion);datos=RAIZ/prep['datos'];carpeta=datos/a.version;res=a.preparacion.parent/a.version
    m=leer(res/'manifiesto_integridad.json')
    for nombre,h in m['archivos'].items():
        if sha(RAIZ/nombre)!=h:raise ValueError('Publicación alterada: '+nombre)
    t=Taller(PROYECTO,RAIZ);comprobar_origen(datos,t)
    for nombre,h in prep['archivos_respaldo'].items():
        if sha(datos/'respaldo_humano'/nombre)!=h or sha(PROYECTO/nombre)!=h:raise ValueError('Original/respaldo alterado: '+nombre)
    coco=leer(carpeta/'anotaciones_coco_IA_NO_ENTRENAR.json');cs={v['id']:v['name'] for v in coco['categories']}
    index={(i['eco_imagen_id'],ann['eco_objeto_id']):ann for i in coco['images'] for ann in coco['annotations'] if ann['image_id']==i['id']}
    if len(index)!=len(coco['annotations']):raise ValueError('IDs COCO duplicados')
    clases=Counter();nuevos=Counter();acciones=Counter();filas=[];count=0;ignore_nuevos=0
    for f in prep['imagenes']:
        base=leer(RAIZ/f['archivo']);r=leer(carpeta/f"{f['indice']:02}.json");im=r['imagen'];w,h=im['width'],im['height'];iid=im['id']
        if r['actor_ajustes']!='asistente_IA' or r['aceptada_V26'] or r['entrenamiento_autorizado']:raise ValueError('Proveniencia o estado indebidos')
        if sha(RAIZ/im['archivo'])!=im['sha256_archivo']:raise ValueError('Fotografía alterada')
        ids=[o['id'] for o in r['anotaciones']]
        if len(ids)!=len(set(ids)):raise ValueError('Objetos repetidos')
        if not {o['id'] for o in base['anotaciones_base']}.issubset(ids):raise ValueError('Objeto humano eliminado')
        union_antes=np.zeros((h,w),bool);union_despues=np.zeros_like(union_antes)
        for obj in base['anotaciones_base']:
            if not obj.get('descartada') and obj.get('poligonos'):union_antes|=mask_poligonos(obj,w,h,paridad=True).astype(bool)
        for o in r['anotaciones']:
            if o.get('descartada') or 'segmentacion_canonica_RLE' not in o:continue
            mask=decodificar(o['segmentacion_canonica_RLE'])
            if mask.shape!=(h,w) or not mask.any() or o['bbox']!=caja(mask) or o['area_px']!=int(mask.sum()):raise ValueError('Geometría incoherente: '+o['id'])
            if not np.array_equal(mask,decodificar(rle_coco(mask))):raise ValueError('Roundtrip RLE inexacto')
            ca=index[(iid,o['id'])]
            if cs[ca['category_id']]!=o['material'] or ca['segmentation']!=o['segmentacion_canonica_RLE'] or ca['bbox']!=o['bbox'] or ca['area']!=o['area_px']:raise ValueError('COCO difiere de la corrección')
            if bool(ca['ignore'])!=(o['material'] not in CLASES):raise ValueError('Ignore incompatible')
            if o.get('ajustes_IA') and (o.get('revision_humana_del_ajuste') is not False or o.get('actor_ultimo_ajuste')!='asistente_IA'):raise ValueError('Ajuste atribuido a humano')
            union_despues|=mask;count+=1
            if o['material'] in CLASES:clases[o['material']]+=1
            if o['id'].startswith('asistente_'):nuevos[o['material']]+=1
            if o['id'].startswith('ia_ignore_'):ignore_nuevos+=1
            acciones.update(v['accion'] for v in o.get('ajustes_IA',[]))
        perdidos=int(np.count_nonzero(union_antes&~union_despues))
        if perdidos:raise ValueError('Píxeles de tus regiones desaparecidos: '+iid)
        filas.append({'imagen':iid,'pixeles_originales_perdidos':perdidos,'objetos':len(ids)})
    huella_original=sha(RAIZ/'checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt')
    if huella_original!='eb8fd00ee536ead21e3a2a0ecd5b06baad04acb2e437b22e0eac3686a7bf36b9':raise ValueError('Checkpoint original modificado')
    eventos=sum(len(t.diario(i)) for i in t.imagenes)
    if eventos!=prep['eventos_humanos_base']:raise ValueError('Cambió el historial humano durante esta preparación')
    pruebas=sorted((RAIZ/'resultados/v28_3_correcciones/pruebas').glob('*/pruebas.json'))[-1];suite=leer(pruebas)
    if not suite['aprobada'] or suite['cantidad_pruebas']<369 or sha(pruebas.parent/'suite.log')!=suite['sha256_log']:raise ValueError('Suite no verificada')
    publicar(res/'control_calidad_real_v2.json',{'fecha_UTC':datetime.now(timezone.utc).isoformat(),
        'aprobada_integridad_tecnica':True,'no_certifica_calidad_visual_exhaustiva':True,
        'imagenes':len(filas),'mascaras_RLE_verificadas':count,'COCO_consistente':True,
        'distribucion_positiva_IA_no_aceptada':dict(clases),'regiones_propuestas_SAM_y_corregidas_IA':dict(nuevos),
        'regiones_nuevas_ignore':ignore_nuevos,
        'regiones_ignore_por_solape':acciones['solape_ambiguo_a_ignore']//2,
        'regiones_ignore_por_componentes':acciones['componentes_dudosos_a_ignore'],
        'acciones':dict(acciones),
        'pixeles_originales_perdidos':0,'eventos_humanos_intactos':eventos,
        'archivos_respaldo_verificados':len(prep['archivos_respaldo']),
        'checkpoint_original_sha256':huella_original,'pruebas':suite['cantidad_pruebas'],
        'sha256_pruebas':sha(pruebas),'entrenamiento':False,'aceptadas_V26':0,'por_imagen':filas})
    print('Integridad aprobada:',len(filas),'imágenes,',count,'máscaras,',eventos,'eventos humanos,',suite['cantidad_pruebas'],'tests.',flush=True)


if __name__=='__main__':main()
