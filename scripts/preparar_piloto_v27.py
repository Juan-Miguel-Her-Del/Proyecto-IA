"""Verifica fotos/propuestas reales V27 y demuestra los bloqueos V26 sin falsificar revisiones."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from PIL import Image,ImageDraw,ImageFont
from ecowaste.incorporacion import publicar,publicar_bytes,sha,candidatos_duplicados,importar,VERSION
from ecowaste.propuestas_publicas import imagen_verificada,propuesta,caja_yolo,diagnostico_regional,mapear_material
from scripts.investigar_v27 import DATOS,SALIDA


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def historicos_train_val():
    refs={}
    for tarea in ['deteccion','clasificacion']:
        for split in ['train','val']:
            path=RAIZ/f'datos/procesados/{tarea}/{split}.jsonl'
            for r in (json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()):
                refs[r['id']]={k:r.get(k) for k in ['id','sha256_archivo','sha256_pixeles','phash']}
    return list(refs.values())


def ficha(fuente,rec,metadatos=None):
    path=RAIZ/rec['archivo'];im=imagen_verificada(path)
    if im['sha256_archivo']!=(rec.get('sha256') or rec.get('descarga',{}).get('sha256')):raise ValueError('Foto descargada alterada')
    iid=str(rec.get('imagen_id_original',rec.get('row_idx')))
    return {'id':fuente+':'+iid,'fuente':fuente,'id_original':iid,'archivo':rec['archivo'],**im,
        'grupo_nombre_fuente':rec.get('grupo_nombre'),'grupo_comida_documentado':rec.get('bonid'),
        'restriccion_no_separar_fuente':fuente,'objeto_fisico_id':None,'sesion_id':None,'sitio_id':None,
        'dispositivo_id':None,'capturado_en':None,'destino_recomendado':'solo_candidato_train_nunca_generalizacion',
        'estado':'pendiente_humanos_y_metadatos','propuestas':[],'metadatos_originales':metadatos}


def contactos(fuente,ims):
    tiles=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for r in ims:
        with Image.open(RAIZ/r['archivo']) as im:im=im.convert('RGB')
        sx,sy=360/im.width,230/im.height;esc=min(sx,sy);im=im.resize((round(im.width*esc),round(im.height*esc)))
        d=ImageDraw.Draw(im)
        for p in r['propuestas']:
            if p['bbox_original'] is None:continue
            x,y,w,h=p['bbox_original'];d.rectangle([x*esc,y*esc,(x+w)*esc,(y+h)*esc],outline='#ffbb33',width=1)
        tile=Image.new('RGB',(380,270),'white');tile.paste(im,(0,0));d=ImageDraw.Draw(tile)
        d.text((3,233),r['id'][:46],font=font,fill='black');d.text((3,251),str(len(r['propuestas']))+' propuestas originales / 0 aceptadas',font=font,fill='black');tiles.append(tile)
    for k in range(0,len(tiles),12):
        ts=tiles[k:k+12];sheet=Image.new('RGB',(380*4,270*((len(ts)+3)//4)),'#ddd')
        for j,t in enumerate(ts):sheet.paste(t,((j%4)*380,(j//4)*270))
        destino=DATOS/'contactos'/f'{fuente}_{k//12:02d}.jpg';destino.parent.mkdir(parents=True,exist_ok=True)
        if destino.exists():raise FileExistsError('Contacto existente')
        sheet.save(destino,quality=88)


def main():
    torch.set_num_threads(4)
    if (SALIDA/'piloto_preparado.json').exists():raise FileExistsError('No reemplazar un piloto ya preparado')
    fotos=[];resumen={}
    zr=leer(SALIDA/'piloto_zerowaste.json');zc=leer(DATOS/'zerowaste/piloto_coco_original.json');cats={c['id']:c['name'] for c in zc['categories']}
    por_im={i['id']:i for i in zc['images']}
    for rec in zr['imagenes']:
        im=ficha('zerowaste',rec,por_im[rec['imagen_id_original']])
        if (im['width'],im['height'])!=(por_im[rec['imagen_id_original']]['width'],por_im[rec['imagen_id_original']]['height']):raise ValueError('Dimensiones COCO incompatibles')
        for a in zc['annotations']:
            if a['image_id']==rec['imagen_id_original']:im['propuestas'].append(propuesta('zerowaste',im['id_original'],a['id'],cats[a['category_id']],
                (im['width'],im['height']),poligonos=a['segmentation'],bbox=a['bbox']))
        fotos.append(im)
    wr=leer(SALIDA/'piloto_warp.json');wp={x['imagen_id_original']:x for x in leer(DATOS/'warp/propuestas_yolo_originales.json')}
    for rec in wr['imagenes']:
        im=ficha('warp',rec)
        for i,a in enumerate(wp[rec['imagen_id_original']]['anotaciones_yolo']):
            try:box=caja_yolo(a['xywh_normalizado'],im['width'],im['height']);err=None
            except ValueError as e:box=None;err=str(e)
            p=propuesta('warp',im['id_original'],f'{im["id_original"]}:{i}',a['clase_original'],(im['width'],im['height']),bbox=box)
            p['yolo_original']=a;p['error_conversion_yolo']=err;im['propuestas'].append(p)
        fotos.append(im)
    hr=leer(SALIDA/'piloto_hpi.json')
    for rec in hr['imagenes']:fotos.append(ficha('hpi',rec,rec['etiquetas_nutricionales']))
    dups=candidatos_duplicados(fotos,historicos_train_val());publicar(SALIDA/'duplicados_train_val_y_pilotos.json',dups)
    for fuente in ['zerowaste','warp','hpi']:
        ims=[i for i in fotos if i['fuente']==fuente];ps=[p for im in ims for p in im['propuestas']]
        regional=[diagnostico_regional(im,im['propuestas']) for im in ims]
        resumen[fuente]={'imagenes_verificadas':len(ims),'anotaciones_espaciales_originales':len(ps),
            'materiales_propuestos':dict(Counter(p['material_propuesto'] or 'revision_material' for p in ps)),
            'poligonos_validos_v26':sum(p['poligonos_normalizados_v26'] is not None for p in ps),
            'anotaciones_con_incidencias':sum(bool(p['errores_geometricos']) or bool(p.get('error_conversion_yolo')) for p in ps),
            'cajas_originales_no_coinciden_con_poligono_v26':sum(p.get('bbox_original_coincide_v26') is False for p in ps),
            'regiones_organicas_anotadas_utilizables_48':sum(r.get('organicos_utilizables') or 0 for r in regional),
            'requieren_poligono_manual':sum(p['poligonos_originales'] is None for p in ps),
            'identidades_fisicas_verificadas':0,'aceptadas_humanas':0,'paquetes_V26_importados':0,
            'bloqueos':['faltan_identidades_fisicas_y_metadatos_no_publicados','no_hay_revisores_reales_ni_adjudicacion'],
            'diagnostico_regional_fuente':regional}
        contactos(fuente,ims)
    # Probar el comportamiento real de V26 sin completar metadatos desconocidos con ficción.
    primera=fotos[0];cats6=[{'id':i+1,'name':c} for i,c in enumerate(['plastico','papel','carton','vidrio','metal','organico'])]
    intento={'eco':{'version':VERSION,'coleccion_id':'v27_publico_pendiente','tipo_datos':'real'},'categories':cats6,
        'physical_objects':[],'annotations':[],'images':[{'id':'0','file_name':Path(primera['archivo']).name,
            'width':primera['width'],'height':primera['height'],'eco':{'sha256_archivo':primera['sha256_archivo'],'sitio_id':None,'sesion_id':None,
                'grupo_captura_id':None,'escena_id':None,'dispositivo_id':None,'capturado_en':None,'objetos_fisicos_ids':None,
                'procedencia':{'tipo':'captura_real','autor':'Autores ZeroWaste','referencia':'https://ai.bu.edu/zerowaste/',
                    'licencia':'CC-BY-NC-4.0_segun_autores','permiso_ref':'README_original_licencia_dataset','permiso_confirmado':True}}}]}
    with tempfile.TemporaryDirectory(prefix='ecowaste_v27_preflight_') as td:
        td=Path(td);entrada=td/'annotations.json';entrada.write_text(json.dumps(intento),encoding='utf-8');rechazo=None
        try:importar(entrada,(RAIZ/primera['archivo']).parent,td/'importado',[],ensayo=False,historicos=historicos_train_val())
        except ValueError as e:rechazo=str(e)
        if not rechazo or (td/'importado/IMPORTADO.json').exists():raise AssertionError('Se aceptaron metadatos inventados o desconocidos')
    publicar(SALIDA/'preflight_v26.json',{'rechazo_correcto_sin_inventar_metadata':True,'error_real_V26':rechazo,
        'paquetes_importados':0,'revisiones_humanas_simuladas':0,'GT_aceptado':0,'entrenamiento':False,'test_evaluado':False})
    publicar(DATOS/'propuestas_pendientes.json',{'version':'v27_fuentes_publicas_propuestas_1','imagenes':fotos,
        'inventario_V26_acreditado':0,'no_es_IMPORTADO_V26':True,'generalizacion_bloqueada':True})
    publicar(DATOS/'tareas_humanas_sin_propuestas.json',{'estado':'pendiente_equipo_humano_y_registro_de_metadatos','revisores':[],
        'imagenes':[{'id':im['id'],'archivo':im['archivo'],'sha256':im['sha256_archivo'],'objeto_fisico_id':None,
            'instancias':[],'cobertura_completa':False} for im in fotos],
        'instruccion':'Completar registro físico/origen sin inferirlo de IDs de anotación; anotación independiente y posterior adjudicación. No registrar estas tareas como revisadas.'})
    publicar(SALIDA/'piloto_preparado.json',{'fuentes':resumen,'imagenes_reales_descargadas':len(fotos),
        'propuestas_espaciales':sum(len(x['propuestas']) for x in fotos),'aceptadas_humanas':0,
        'duplicados_exactos_historicos':sum(x['exacto'] for x in dups['historicos']),
        'candidatos_perceptuales_historicos':sum(not x['exacto'] for x in dups['historicos']),
        'candidatos_duplicados_entre_nuevas':len(dups['nuevos']),
        'regiones_organicas_utilizables_verificadas':0,'generalizacion_desbloqueada':False,'entrenamiento':False,'test_evaluado':False})
    print(json.dumps({k:{a:b for a,b in v.items() if a!='diagnostico_regional_fuente'} for k,v in resumen.items()},ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':main()
