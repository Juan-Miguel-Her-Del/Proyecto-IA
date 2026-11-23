"""Publica una copia anotada por IA y comparaciones; no escribe diarios humanos."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import html
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from ecowaste.incorporacion import leer,publicar,sha,digest
from ecowaste.auxiliares_publicos import Taller
from ecowaste.ajustes_ia_v28_3 import (VERSION,CLASES,canonizar,ambiguo,quitar_solape,
    nueva_propuesta,decodificar,guardar_mask,elegibilidad48,componentes_dudosos_a_ignore,anotar_cambio)
from scripts.ajustar_anotaciones_ia_v28_3 import comprobar_origen
from scripts.anotar_v27_2 import PROYECTO


def mayor_componente(m):
    n,lab,stats,_=cv2.connectedComponentsWithStats(np.uint8(m),8)
    if n<=1:raise ValueError('Propuesta vacía')
    return lab==(1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])))


def visual(im,anns,path):
    foto=Image.open(RAIZ/im['archivo']).convert('RGB');foto.thumbnail((1100,780))
    a=np.array(foto);lineas=Image.new('RGBA',foto.size);d=ImageDraw.Draw(lineas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    colores={'plastico':(35,190,255),'papel':(250,220,50),'carton':(255,150,55),
             'metal':(190,160,255),'vidrio':(60,240,200),'organico':(120,240,45),
             'ambiguo':(255,65,90),'compuesto':(255,80,200),'fuera_alcance':(200,200,200)}
    for n,obj in enumerate(anns,1):
        if obj.get('descartada') or 'segmentacion_canonica_RLE' not in obj:continue
        m=decodificar(obj['segmentacion_canonica_RLE']);p=Image.fromarray(np.uint8(m)*255).resize(foto.size,Image.Resampling.NEAREST)
        m=np.asarray(p)>0;c=np.array(colores.get(obj['material'],(255,255,255)))
        alpha=.22 if obj['material'] in CLASES else .5
        a[m]=(a[m]*(1-alpha)+c*alpha).astype(np.uint8)
        cs,_=cv2.findContours(np.uint8(m),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in cs:
            pts=contour[:,0,:].tolist()
            if len(pts)>1:d.line([tuple(q) for q in pts]+[tuple(pts[0])],fill=tuple(c),width=1)
        x,y,_,_=obj['bbox'];d.text((x*foto.width/im['width'],y*foto.height/im['height']),str(n),fill=tuple(c),stroke_width=1,stroke_fill='black',font=font)
    salida=Image.alpha_composite(Image.fromarray(a).convert('RGBA'),lineas).convert('RGB');salida.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preparacion',type=Path,required=True)
    p.add_argument('--decisiones',type=Path,required=True);p.add_argument('--version',required=True);a=p.parse_args()
    if not a.version.replace('_','').replace('-','').isalnum():raise ValueError('Versión no válida')
    prep=leer(a.preparacion);datos=RAIZ/prep['datos'];dec=leer(a.decisiones)
    salida=datos/a.version;res=a.preparacion.parent/a.version
    if salida.exists() or res.exists():raise FileExistsError('Publicación histórica; utilizar otra versión')
    actual=Taller(PROYECTO,RAIZ);comprobar_origen(datos,actual)
    for nombre,h in prep['archivos_respaldo'].items():
        if sha(datos/'respaldo_humano'/nombre)!=h:raise ValueError('Respaldo alterado: '+nombre)
    salida.mkdir(parents=True);res.mkdir(parents=True)
    specs=[leer(a.preparacion.parent/n)['propuestas'] for n in dec['criterios']]
    resumen=[];cambios=[];coco={'info':{'version':VERSION,'estado':'NO_ENTRENAR_IA_PENDIENTE_CONFIRMACION',
        'revision_humana_nueva':False,'aceptadas_V26':0},'images':[],'annotations':[],
        'categories':[{'id':n+1,'name':v} for n,v in enumerate([*CLASES,'ambiguo','compuesto','fuera_alcance','sin_asignar'])]}
    categorias={r['name']:r['id'] for r in coco['categories']};total_nuevos=total_cambios=0
    for fila in prep['imagenes']:
        n=fila['indice'];base=leer(RAIZ/fila['archivo']);im=base['imagen'];iid=im['id'];w,h=im['width'],im['height']
        anns=[canonizar(o,w,h) for o in base['anotaciones_base']];originales=deepcopy(anns)
        porid={o['id']:o for o in anns};observaciones=[]
        for regla in dec['materiales_a_ignore']:
            if regla['indice']==n:ambiguo(porid[regla['objeto']],regla['motivo'])
        for regla in dec.get('componentes_a_ignore',[]):
            if regla['indice']!=n:continue
            otro=componentes_dudosos_a_ignore(porid[regla['objeto']],regla['motivo'])
            if otro:anns.append(otro)
        padre=None;mask_padre=None
        for regla in dec.get('desagrupar',[]):
            if regla['indice']!=n:continue
            if padre is not None:raise ValueError('Sólo una desagrupación explícita por imagen')
            padre=porid[regla['objeto']];mask_padre=decodificar(padre['segmentacion_canonica_RLE'])
            ambiguo(padre,regla['motivo'])
        pares=set()
        for inc in base['incidencias']:
            if inc['motivo']!='solape_visible_requiere_decision_no_prueba_de_occlusion':continue
            ids=tuple(sorted(inc['objetos']))
            if ids in pares:continue
            pares.add(ids)
            try:
                quitar_solape(anns,ids,'No se asigna una identidad inventada al área de intersección. Se conserva como región ignore; las partes no solapadas siguen disponibles.',
                    frente='wsc_158' if n==40 else None)
            except ValueError as ex:observaciones.append(str(ex))
        ocupada=np.zeros((h,w),bool)
        for obj in anns:
            if obj is not padre and not obj.get('descartada') and 'segmentacion_canonica_RLE' in obj:ocupada|=decodificar(obj['segmentacion_canonica_RLE'])
        nuevos=0
        for sel in dec['incorporar']:
            s=specs[sel['criterio']][sel['propuesta']]
            if s['indice']!=n:continue
            clave=digest(s)[:24];sam=leer(datos/'sam_IA'/(clave+'.json'));raw=decodificar(sam['resultado']['rle_ia_original'])
            m=mayor_componente(raw) if sel.get('mayor_componente') else raw.copy()
            if sel.get('poligono_visual'):
                trazado=Image.new('1',(w,h));dr=ImageDraw.Draw(trazado)
                dr.polygon([(x*w,y*h) for x,y in sel['poligono_visual']],fill=1);m=np.asarray(trazado).astype(bool)
            if mask_padre is not None:m &= mask_padre
            # Estas propuestas se añaden detrás de las regiones ya guardadas, sin erosionar el trabajo humano.
            m &= ~ocupada
            if not m.any():observaciones.append('Propuesta ya cubierta: '+sam['id']);continue
            if np.count_nonzero(m)<.5*np.count_nonzero(raw):
                observaciones.append('Propuesta no incorporada: más de la mitad ya cubierta: '+sam['id']);continue
            obj=nueva_propuesta(sam,s['material'],sel.get('detalle',s['detalle']),m)
            obj['ajustes_IA']=[{'accion':'objeto_omitido_anadido','criterio':sel,
                'pixeles_SAM_originales':int(raw.sum()),'pixeles_finales':int(m.sum()),
                'respeta_regiones_existentes':True,'inspeccion_visual':'asistente_IA_NO_humano'}]
            if sel.get('poligono_visual'):
                obj['origen']='SAM2_contorno_incompleto_corregido_visualmente_por_asistente_IA'
            anns.append(obj);ocupada|=m;nuevos+=1
        if padre is not None:
            restante=mask_padre&~ocupada
            if restante.any():guardar_mask(padre,restante)
            else:padre['descartada']=True
            anotar_cambio(padre,'grupo_separado_instancias_visibles',
                'Las rodajas con límite visible se separan; el resto queda ignore, sin inventar superficies ocultas.',
                pixeles_residuales_ignore=int(restante.sum()))
        for inc in base['incidencias']:
            if 'fragmentacion' in inc['motivo']:
                observaciones.append('Identidad de componentes por confirmar; no se conectan regiones invisibles: '+', '.join(inc['objetos']))
        observaciones.extend(dec.get('observaciones',{}).get(str(n),[]))
        positivos=[o for o in anns if not o.get('descartada') and o['material'] in CLASES and 'segmentacion_canonica_RLE' in o]
        modificados=[o for o in anns if o.get('ajustes_IA')]
        documento={'version':VERSION,'imagen':im,'revision_humana_base':base['revision_humana_base'],
            'actor_ajustes':'asistente_IA','anotaciones':anns,'anotaciones_base_sha256':sha(RAIZ/fila['archivo']),
            'estado':'PROPUESTA_CORREGIDA_POR_IA_NO_REVISION_HUMANA','cobertura_completa_confirmada_IA':False,
            'aceptada_V26':False,'entrenamiento_autorizado':False,'observaciones':observaciones,
            'advertencia':'RLE canónico. No usar directamente con adaptadores históricos basados sólo en polígonos.'}
        publicar(salida/f'{n:02}.json',documento)
        visual(im,originales,salida/f'{n:02}_antes.png');visual(im,anns,salida/f'{n:02}_despues.png')
        coco['images'].append({'id':n+1,'file_name':im['archivo'],'width':w,'height':h,
            'eco_imagen_id':iid,'sha256':im['sha256_archivo'],'licencia':im['licencia'],
            'provenance':im,'revision_humana_base':base['revision_humana_base'],
            'cobertura_exhaustiva_confirmada':False,'physical_identity_known':False})
        for obj in anns:
            if obj.get('descartada') or 'segmentacion_canonica_RLE' not in obj:continue
            coco['annotations'].append({'id':len(coco['annotations'])+1,'image_id':n+1,
                'category_id':categorias[obj['material']],'segmentation':obj['segmentacion_canonica_RLE'],
                'bbox':obj['bbox'],'area':obj['area_px'],'iscrowd':0,'ignore':int(obj['material'] not in CLASES),
                'eco_objeto_id':obj['id'],'actor_ajuste':obj.get('actor_ultimo_ajuste'),
                'nueva_revision_humana':False,'aceptada_V26':False,'motivo':obj.get('motivo','')})
        r={'indice':n,'imagen':iid,'nuevas_regiones':nuevos,'objetos_con_ajustes':len(modificados),
            'positivos_por_material':dict(Counter(o['material'] for o in positivos)),
            'organicos_48_con_al_menos_2_celdas':sum(o['material']=='organico' and elegibilidad48(decodificar(o['segmentacion_canonica_RLE']),w,h)>=2 for o in positivos),
            'observaciones':observaciones,'revisada_previamente_por_usuario':base['revision_humana_base'] is not None,
            'cobertura_exhaustiva_confirmada':False}
        resumen.append(r);total_nuevos+=nuevos;total_cambios+=len(modificados)
        cambios.extend({'imagen':iid,'objeto':o['id'],'material':o['material'],'cambios':o['ajustes_IA']} for o in modificados)
    publicar(salida/'anotaciones_coco_IA_NO_ENTRENAR.json',coco)
    publicar(salida/'resumen.json',{'version':VERSION,'imagenes_inspeccionadas':len(resumen),
        'imagenes_con_cambios':sum(r['objetos_con_ajustes']>0 for r in resumen),'regiones_anadidas':total_nuevos,
        'objetos_con_ajustes':total_cambios,'eventos_humanos_conservados':prep['eventos_humanos_base'],
        'eventos_humanos_creados_por_asistente':0,'aceptadas_V26':0,'entrenamiento':False,'imagenes':resumen})
    publicar(res/'cambios.json',cambios)
    filas=[]
    for r in resumen:
        n=r['indice'];obs=' '.join(r['observaciones']) or 'Queda confirmar exhaustividad de la escena; no equivale a aceptación humana.'
        antes='Antes: guardado humano' if r['revisada_previamente_por_usuario'] else 'Antes: propuestas originales SIN guardado humano'
        filas.append(f'<section id="im{n}"><h2>{html.escape(r["imagen"])}</h2><p>{r["nuevas_regiones"]} regiones añadidas · {r["objetos_con_ajustes"]} objetos ajustados. <a href="{n:02}.json">Anotaciones y procedencia</a></p><div class="par"><figure><figcaption>{antes}</figcaption><img loading="lazy" src="{n:02}_antes.png"></figure><figure><figcaption>Después: ajustes IA pendientes de confirmación</figcaption><img loading="lazy" src="{n:02}_despues.png"></figure></div><p>{html.escape(obs)}</p></section>')
    pagina='''<!doctype html><html lang="es"><meta charset="utf-8"><title>EcoWasteFormer · Ajustes IA</title><style>body{font:17px system-ui;background:#15202b;color:#eee;margin:24px}a{color:#78d6ff}h1{font-size:26px}.par{display:flex;gap:12px}figure{margin:0;width:50%}img{width:100%}section{border-top:1px solid #567;padding:20px 0}nav a{display:inline-block;margin:6px}.aviso{background:#40341f;padding:16px}@media(max-width:800px){.par{display:block}figure{width:100%}}</style><h1>Correcciones hechas por el asistente IA</h1><p class="aviso">Tus guardados originales se conservan. Rojo: regiones ambiguas/ignore. Estas correcciones no son revisiones humanas ni aceptación V26. No se ha entrenado. La exhaustividad sigue sin confirmar y no se certifican identidades físicas nuevas.</p>'''
    pagina+=f'<p>{len(resumen)} imágenes inspeccionadas · {total_nuevos} regiones añadidas · {total_cambios} objetos con ajustes. <a href="resumen.json">Resumen verificable</a> · <a href="anotaciones_coco_IA_NO_ENTRENAR.json">COCO RLE auxiliar bloqueado</a></p><nav>'
    pagina+=' '.join(f'<a href="#im{r["indice"]}">{html.escape(r["imagen"])}</a>' for r in resumen)+'</nav>'+''.join(filas)
    (salida/'index.html').write_text(pagina,encoding='utf8')
    # Verificación de todos los archivos originales, no sólo del checkpoint seleccionado.
    protegidos=leer(RAIZ/'resultados/v28_3_correcciones/historicos_protegidos.json')
    for nombre,h in protegidos.items():
        if sha(RAIZ/nombre)!=h:raise ValueError('Histórico alterado: '+nombre)
    for nombre,h in prep['archivos_respaldo'].items():
        if sha(PROYECTO/nombre)!=h:raise ValueError('Archivo del proyecto humano cambió durante la corrección: '+nombre)
    publicar(res/'verificacion.json',{'fecha_UTC':datetime.now(timezone.utc).isoformat(),
        'archivos_humanos_intactos':len(prep['archivos_respaldo']),'historicos_intactos':len(protegidos),
        'eventos_humanos':sum(len(actual.diario(i)) for i in actual.imagenes),
        'nuevos_eventos_humanos':0,'SAM_CPU':True,'GPU_utilizada':False,'entrenamiento':False,
        'test_evaluado':False,'generalizacion_desbloqueada':False})
    publicar(res/'manifiesto_integridad.json',{'version':VERSION,'origen':str(a.preparacion),
        'preparacion_sha256':sha(a.preparacion),'decisiones_sha256':sha(a.decisiones),
        'archivos':{p.relative_to(RAIZ).as_posix():sha(p) for p in sorted(salida.iterdir()) if p.is_file()}})
    print(json.dumps({'publicacion':str(salida),'regiones_anadidas':total_nuevos,'objetos_ajustados':total_cambios,'imagenes':len(resumen)}),flush=True)


if __name__=='__main__':main()
