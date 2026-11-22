"""Lista de decisiones de calidad y COCO de auditoría, sin transformar diarios humanos."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,publicar,publicar_bytes,sha,digest,caja_derivada,CLASES
from ecowaste.auxiliares_publicos import Taller,rle_coco
from ecowaste.validacion_anotaciones import mask_poligonos
from ecowaste.preparacion_v28 import suficiencia
from scripts.respaldar_anotaciones_v28 import SALIDA


def main():
    rec=leer(SALIDA/'respaldo_verificado.json');t=Taller(RAIZ/rec['destino'],RAIZ)
    filas=leer(SALIDA/'imagenes.json');observaciones=[]
    def agregar(iid,motivo,objetos=None):
        observaciones.append({'imagen':iid,'objetos':objetos,'motivo':motivo,'requiere_decision':True,
                              'origen':'inspeccion_asistente_NO_revision_humana_independiente','contornos_modificados':False})
    hpi={
        '0':'Restos alargados visibles arriba/derecha del puré sin contorno: añadirlos o justificar ignore.',
        '114':'Numerosos granos/trozos visibles entre las máscaras verdes; revisar cobertura completa, además de diferencia de relleno de 7 píxeles.',
        '132':'Comida beige/filamentosa visible en mitad superior y derecha sin contorno.',
        '155':'Restos morados visibles a la izquierda y debajo del contorno verde sin anotación.',
        '173':'Granos/trozos visibles entre y debajo de las regiones verdes: comprobar que no queden como fondo.',
        '192':'Restos claros visibles encima de la región inferior anotada: completar cobertura.',
        '78':'Granos y mezcla de comida visibles alrededor de los contornos, especialmente centro/derecha.',
        '96':'Mezcla de comida central y derecha visible fuera de los dos polígonos.',
        '23':'Confirmar granularidad de instancia: un contorno agrupa varias rodajas visibles; no dividir automáticamente.'}
    for n,detalle in hpi.items():agregar('hpi:'+n,detalle)
    zero={
        '1001':'Papel impreso visible hacia centro/derecha sin contorno; comprobar imagen completa.',
        '1076':'Hoja/periódico impreso grande visible en el centro y zona inferior sin contorno de papel.',
        '1709':'Periódico visible en el centro/inferior sin anotación suficiente.',
        '2670':'Papel blanco/impreso visible en el centro de la pila sin contorno.',
        '2688':'Hojas impresas visibles atravesando el centro de la pila; revisar omisiones.',
        '2706':'Papel impreso central visible sin cobertura completa; escena correlacionada con 2709/2710.',
        '2709':'Papel impreso central visible sin cobertura completa; mantener junto a 2706/2710.',
        '2710':'Papel impreso central visible; el descarte «no se distingue» no justifica enseñar residuos ambiguos como fondo.',
        '2843':'Papel/periódico grande visible a la derecha, sin contorno; aclarar también descarte «no se distingue».',
        '331':'Hoja/periódico visible en el centro, además de avisos geométricos/solape.',
        '588':'Papel impreso visible en el centro sin contorno; comprobar omisiones.',
        '896':'Papel impreso visible entre las cajas centrales: comprobar cobertura completa.'}
    for n,detalle in zero.items():agregar('zerowaste:'+n,detalle)
    # Contradicciones verificables con nuestra taxonomía de compuestos, sin
    # sustituir las decisiones del usuario por etiquetas del proveedor.
    for iid,im in t.imagenes.items():
        es=[e for e in t.diario(iid) if e['modo']=='asistida']
        if not es:continue
        finals={a['id']:a for a in es[-1]['anotaciones']}
        for n,a in enumerate(im['anotaciones'],1):
            cat=a.get('anotacion_fuente_original',{}).get('categoria_original')
            f=finals.get(a['id'])
            if cat in ['milk-cardboard','juice-cardboard'] and f and not f.get('descartada') and f['material'] in CLASES:
                agregar(iid,f'Objeto {n}: fuente {cat}, guardado como {f["material"]}. Envase posiblemente multicapa: confirmar composición o marcar compuesto/ambiguo con motivo; no convertir automáticamente.',[a['id']])
        for n,a in enumerate(es[-1]['anotaciones'],1):
            if a.get('descartada') and a.get('motivo','').strip().lower() in ['error','no se distingue','no se distigue']:
                agregar(iid,f'Descarte {n}: «{a["motivo"]}». Aclarar si era propuesta falsa o residuo visible ambiguo; en este último caso conservar región ignore.',[a['id']])
    agregar('warp:POSAD_1_13-Sep_03-20-46','Objeto 7: originalmente bottle-green/plástico, ahora orgánico. Verificar botella frente a material vegetal visible; la procedencia no resuelve sola el material.',['warp_6'])
    agregar('warp:prepared_data_all_MGS_20-Oct_01-03-59','Objeto 7: pieza porosa anotada orgánico; confirmar resto de comida frente a espuma/esponja. No cambiar material por apariencia sin decisión.',['usuario_06759e0dd8d2439585f09c85fd5283c2'])
    publicar(SALIDA/'observaciones_visuales_y_taxonomia.json',observaciones)
    estado=suficiencia(filas,observaciones)
    publicar(SALIDA/'suficiencia_exploratoria.json',estado)
    # Esta exportación documenta únicamente guardados de imágenes declaradas
    # completas; NO rellena las cuatro sin evento con propuestas automáticas.
    destino=RAIZ/'datos/nuevos/v28/colecciones'/('auditoria_usuario_v28_1_'+sha(RAIZ/rec['destino']/'PROYECTO.json')[:12])
    if destino.exists():raise FileExistsError('Colección existente: no reemplazar exportación')
    cats=[{'id':n+1,'name':c} for n,c in enumerate([*CLASES,'ambiguo','compuesto','fuera_alcance'])]
    cmap={c['name']:c['id'] for c in cats}
    coco={'info':{'version':'auxiliar_humano_auditoria_v28_1','estado':'QA_pendiente_NO_ENTRENAR',
                  'entrenamiento_autorizado':False,'aceptadas_V26':0,'respaldo':rec['destino']},
          'images':[],'annotations':[],'categories':cats,'physical_objects':[]}
    fuentes=[];humanas=[];descartes=[]
    for f in filas:
        if f['estado']!='completa_usuario_unico':continue
        iid=f['imagen'];im=t.imagen(iid);eventos=t.diario(iid);ev=[e for e in eventos if e['modo']=='asistida'][-1]
        imageid=len(coco['images'])+1;path=t.archivo_imagen(iid)
        nombre='imagenes/'+im['sha256_archivo']+path.suffix.lower();publicar_bytes(destino/nombre,path.read_bytes())
        if sha(destino/nombre)!=im['sha256_archivo']:raise ValueError('Copia de foto alterada')
        warnings=[x for x in observaciones if x['imagen']==iid]
        coco['images'].append({'id':imageid,'file_name':nombre,'width':im['width'],'height':im['height'],
            'eco':{'imagen_id_auxiliar':iid,'estado_revision':'corregida_usuario_unico','persona_id':ev['persona_id'],
                   'revision_huella':ev['huella'],'sha256_archivo':im['sha256_archivo'],'licencia':im['licencia'],
                   'fuente':im['fuente'],'grupo_procedencia':'fuente:'+im['fuente'],
                   'objeto_fisico_id':None,'sitio_id':None,'sesion_id':None,'dispositivo_id':None,'capturado_en':None,
                   'cobertura_completa_declarada_usuario':True,'QA_exhaustividad_confirmada':False,
                   'incidencias_automaticas':f['errores'],'avisos_automaticos':f['avisos_visuales'],
                   'decisiones_visuales_pendientes':warnings,'aceptada_V26':False}})
        fuentes.append({'imagen':iid,'metadatos_fuente':im,'propuestas_originales':im['anotaciones']})
        humanas.append({'imagen':iid,'revision_final_original':ev})
        for a in ev['anotaciones']:
            if a.get('descartada'):
                descartes.append({'imagen':iid,'anotacion':a});continue
            if not a.get('poligonos'):raise ValueError('No exportar caja sin máscara como instancia humana')
            m=mask_poligonos(a,im['width'],im['height']);p=mask_poligonos(a,im['width'],im['height'],paridad=True)
            eco={'id_auxiliar':a['id'],'poligonos_usuario':a['poligonos'],'huecos_usuario':a.get('huecos',[]),
                 'persona_id':ev['persona_id'],'revision_huella':ev['huella'],'origen':a.get('origen'),
                 'propuesta_ia_id':a.get('propuesta_ia_id'),'ignorada':a['material'] not in CLASES,
                 'motivo':a.get('motivo'),'objeto_fisico_id':None,'aceptada_V26':False,
                 'interpretacion_relleno':'union_menos_huecos_del_exportador_V27_2',
                 'pixeles_discrepancia_con_editor':int((m!=p).sum())}
            if (m!=p).any():eco['rle_alternativo_visto_en_editor']=rle_coco(p)
            coco['annotations'].append({'id':len(coco['annotations'])+1,'image_id':imageid,
                'category_id':cmap[a['material']],'segmentation':rle_coco(m) if a.get('huecos') else a['poligonos'],
                'bbox':caja_derivada(a['poligonos']),'area':int(m.sum()),'iscrowd':0,'ignore':int(a['material'] not in CLASES),'eco':eco})
        # Se preservan los bytes de TODOS los eventos de esta imagen, no solo el último.
        folder=digest(iid)[:24]
        for p in (t.path/'revisiones'/folder).glob('*.json'):publicar_bytes(destino/'revisiones'/folder/p.name,p.read_bytes())
    publicar(destino/'annotations.json',coco);publicar(destino/'anotaciones_fuente_originales.json',fuentes)
    publicar(destino/'correcciones_humanas_finales.json',humanas);publicar(destino/'descartes_humanos.json',descartes)
    publicar(destino/'QA_PENDIENTE.json',{'aprobada_qa_para_entrenar':False,'coleccion_congelada_para_entrenar':False,
        'entrenamiento_autorizado':False,'suficiencia':estado,'incidencias_no_resueltas':True})
    hashes={p.relative_to(destino).as_posix():sha(p) for p in destino.rglob('*') if p.is_file()}
    publicar(destino/'MANIFIESTO.json',{'version':'auxiliar_humano_auditoria_v28_1','archivos':hashes,
        'respaldo_huella':sha(SALIDA/'respaldo_verificado.json'),'entrenamiento_autorizado':False,'aceptadas_V26':0})
    publicar(SALIDA/'exportacion_auxiliar_auditoria.json',{'destino':destino.relative_to(RAIZ).as_posix(),
        'imagenes':len(coco['images']),'instancias':len(coco['annotations']),'seis_materiales':sum(not a['ignore'] for a in coco['annotations']),
        'ignoradas':sum(a['ignore'] for a in coco['annotations']),'fotografias_copiadas_sin_modificar_bytes':True,
        'estado':'auditoria_NO_ENTRENAR','aprobada_qa_para_entrenar':False,'aceptadas_V26':0,
        'sha256_manifiesto':sha(destino/'MANIFIESTO.json')})
    print(estado,flush=True)


if __name__=='__main__':main()
