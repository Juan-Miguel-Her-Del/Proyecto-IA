"""Incorporación V26: paquetes inmutables, QA, revisión humana y partición física.

No contiene inferencia, entrenamiento ni escritura en los datasets históricos.
"""
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path,PureWindowsPath
import re
import tempfile

import cv2
import numpy as np
from PIL import Image,ImageDraw
from ecowaste.datasets import CLASES,objetivo_deteccion
from ecowaste.poligonos import rasterizar_contornos
from ecowaste.supervision_regional import regiones_poligonales
from ecowaste.reanudacion_regional import publicar,bloqueo

RAIZ=Path(__file__).resolve().parents[1]
VERSION='ecowaste_datos_reales_v26_1'
POLITICA=RAIZ/'configuracion/incorporacion_v26.json'
IGNORADAS={'ambiguo','compuesto','fuera_alcance'}
DESTINOS={'sin_asignar','train_nuevo','generalizacion_bloqueada'}


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def politica():
    p=leer(POLITICA)
    if p['version']!=VERSION or p['clases']!=CLASES or p['entrenamiento_autorizado'] or p['evaluacion_autorizada']:
        raise ValueError('Política V26 incompatible')
    propuesta=leer(RAIZ/p['referencia_v25'])
    if p['cuotas_train']!=propuesta['instancias_nuevas_train'] or sum(p['cuotas_train'].values())!=1550:
        raise ValueError('Cuotas distintas de V25')
    return p


def identificador(x):
    if isinstance(x,bool) or not isinstance(x,(int,str)) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9:_-]{0,127}',str(x)):
        raise ValueError('Identificador inválido')
    return str(x)


def texto(x,campo):
    if not isinstance(x,str) or not x.strip():raise ValueError('Falta '+campo)
    return x.strip()


def ruta_segura(raiz,nombre):
    if not isinstance(nombre,str) or not nombre or '\\' in nombre or PureWindowsPath(nombre).drive:
        raise ValueError('Ruta de imagen no relativa')
    rel=Path(nombre)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Ruta fuera del paquete')
    ruta=(Path(raiz)/rel).resolve()
    if not ruta.is_relative_to(Path(raiz).resolve()):raise ValueError('Enlace fuera del paquete')
    return ruta


def destino_nuevo(path,*,existente=False):
    path=Path(path).resolve()
    protegidos=['.git','datos','checkpoints','configuracion','ecowaste','pruebas','documentacion','resultados']
    permitidos=[RAIZ/'datos/nuevos/v26',RAIZ/'resultados/v26_pipeline_datos']
    if path==RAIZ or any(path.is_relative_to((RAIZ/p).resolve()) for p in protegidos) and not any(path.is_relative_to(p.resolve()) for p in permitidos):
        raise ValueError('Destino dentro de datos, código o resultados históricos')
    if not existente and path.exists():raise FileExistsError('Destino existente; usar una versión nueva')
    return path


def publicar_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.pendiente')
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)


def copiar_imagen_verificada(origen,destino,esperado):
    blob=Path(origen).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=esperado:raise ValueError('Imagen cambió antes de copiarla')
    publicar_bytes(destino,blob)


def poligonos_validos(ps,w,h):
    if not isinstance(ps,list) or not ps:raise ValueError('Se requieren polígonos visibles; no se derivan de bbox/etiquetas globales')
    salida=[]
    def orient(a,b,c):
        u,v=b-a,c-a
        return u[...,0]*v[...,1]-u[...,1]*v[...,0]
    def entre(a,b,c):return np.all(c>=np.minimum(a,b)-1e-10,axis=-1)&np.all(c<=np.maximum(a,b)+1e-10,axis=-1)
    for p in ps:
        if not isinstance(p,list) or len(p)<6 or len(p)%2:raise ValueError('Polígono mal formado; RLE/huecos no admitidos')
        if any(isinstance(n,bool) or not isinstance(n,(int,float)) for n in p):raise ValueError('Coordenadas no numéricas')
        q=np.asarray(p,dtype=float).reshape(-1,2)
        if not np.isfinite(q).all() or (q<0).any() or (q[:,0]>w).any() or (q[:,1]>h).any():raise ValueError('Polígono no finito o fuera de imagen')
        if np.array_equal(q[0],q[-1]):q=q[:-1]
        if len(q)<3 or len(np.unique(q,axis=0))!=len(q):raise ValueError('Vértices repetidos')
        n=len(q)
        for i in range(n):
            a,b=q[i],q[(i+1)%n]
            js=np.arange(i+2,n)
            if i==0:js=js[js!=n-1]
            if not len(js):continue
            c,d=q[js],q[(js+1)%n];v=[orient(a,b,c),orient(a,b,d),orient(c,d,a),orient(c,d,b)]
            cruce=(v[0]*v[1]<0)&(v[2]*v[3]<0)
            for k,t in enumerate([(a,b,c),(a,b,d),(c,d,a),(c,d,b)]):cruce|=(abs(v[k])<1e-10)&entre(*t)
            if cruce.any():raise ValueError('Polígono con cruce o autocontacto')
        if abs(float(cv2.contourArea(q.astype(np.float32))))<=1e-10:raise ValueError('Polígono degenerado')
        salida.append(q.flatten().tolist())
    return salida


def caja_derivada(ps):
    q=np.concatenate([np.array(p).reshape(-1,2) for p in ps]);a=q.min(0);b=q.max(0)
    return [float(a[0]),float(a[1]),float(b[0]-a[0]),float(b[1]-a[1])]


def firma_geometria(a):
    formas=[]
    for pol in a['poligonos']:
        pts=list(zip(pol[::2],pol[1::2]));inicio=min(range(len(pts)),key=lambda i:pts[i])
        normal=pts[inicio:]+pts[:inicio];reversa=[normal[0],*reversed(normal[1:])]
        formas.append(min(tuple(normal),tuple(reversa)))
    return digest({'material':a['material'],'formas':sorted(formas)})


def mascara(ps,w,h):
    im=Image.new('1',(w,h));d=ImageDraw.Draw(im)
    for p in ps:d.polygon(list(zip(p[::2],p[1::2])),fill=1)
    return np.asarray(im,dtype=bool)


def normalizar_instancia(a,im,categorias,objetos):
    eco=a.get('eco',{});oid=identificador(eco.get('objeto_fisico_id'))
    if oid not in objetos:raise ValueError('Objeto físico sin registro')
    if oid not in im['objetos_fisicos_ids']:raise ValueError('Objeto no registrado en la captura de esta imagen')
    material=categorias.get(str(a.get('category_id')))
    if material not in CLASES and material not in IGNORADAS:raise ValueError('Categoría/material desconocido')
    if eco.get('visible_only') is not True or eco.get('holes'):raise ValueError('Solo contorno visible; huecos requieren otro esquema')
    if eco.get('origen') not in ['humana','propuesta_automatica']:raise ValueError('Origen de anotación no trazable')
    if eco.get('oclusion') not in ['ninguna','parcial','alta']:raise ValueError('Oclusión no declarada')
    if any(not isinstance(eco.get(k),bool) for k in ['truncado','transparente']):raise ValueError('Faltan indicadores de visibilidad')
    if material in IGNORADAS:texto(eco.get('motivo_ambiguo'),'motivo de categoría ambigua')
    solapes=eco.get('solapa_con')
    if not isinstance(solapes,list) or any(identificador(x) not in im['objetos_fisicos_ids'] or str(x)==oid for x in solapes):raise ValueError('Solapes declarados inválidos')
    ps=poligonos_validos(a.get('segmentation'),im['width'],im['height']);bbox=caja_derivada(ps)
    if 'bbox' in a:
        try:coincide=np.asarray(a['bbox']).shape==(4,) and all(type(x) in [int,float] for x in a['bbox']) and bool(np.allclose(a['bbox'],bbox,rtol=0,atol=1e-6))
        except (ValueError,TypeError):coincide=False
        if not coincide:raise ValueError('Caja declarada no deriva del polígono visible')
    if a.get('iscrowd',0)!=0:raise ValueError('Se requiere identidad por instancia, no crowd')
    area=int(mascara(ps,im['width'],im['height']).sum())
    if not area:raise ValueError('Máscara visible vacía')
    if 'area' in a and (isinstance(a['area'],bool) or not isinstance(a['area'],(int,float)) or not math.isfinite(a['area']) or a['area']<=0):
        raise ValueError('Área declarada no finita/positiva')
    return {'id':identificador(a['id']),'objeto_fisico_id':oid,'material':material,'poligonos':ps,'bbox':bbox,
        'area_mask_px':area,'area_declarada':a.get('area'),'origen':eco['origen'],'oclusion':eco['oclusion'],
        'truncado':eco['truncado'],'transparente':eco['transparente'],'visible_only':True,
        'motivo_ambiguo':eco.get('motivo_ambiguo'),'solapa_con':sorted(set(map(str,solapes)))}


def normalizar_revisores(revisores,tipo):
    salida={}
    for r in revisores:
        rid=identificador(r['id']);persona=identificador(r['persona_id'])
        if rid in salida or r.get('humano') is not True or r.get('tipo')!=tipo:raise ValueError('Registro de revisor incompatible')
        if not r.get('roles') or not set(r['roles'])<= {'anotador','adjudicador'}:raise ValueError('Roles inválidos')
        salida[rid]={'id':rid,'persona_id':persona,'humano':True,'tipo':tipo,'roles':sorted(r['roles'])}
    return salida


def normalizar_paquete(data,raiz_imagenes,revisores,*,ensayo=False):
    if not isinstance(data,dict) or any(not isinstance(data.get(k),list) for k in ['images','annotations','categories','physical_objects']):
        raise ValueError('Se requieren arrays TACO y registro de objetos físicos')
    p=politica();meta=data.get('eco',{});tipo='fixture' if ensayo else 'real'
    if meta.get('version')!=VERSION or meta.get('tipo_datos')!=tipo:raise ValueError('No mezclar fixtures con datos reales')
    coleccion=identificador(meta.get('coleccion_id'));categorias={}
    if not data['categories']:raise ValueError('Declarar el catálogo de materiales incluso para imágenes negativas')
    mapa=leer(RAIZ/'configuracion/datasets.json');mapeo={**mapa['taco'],**{k:k for k in CLASES},**{k:k for k in IGNORADAS},
        **{k:'ambiguo' for k in mapa['categorias_taco_excluidas']}}
    for c in data['categories']:
        cid=identificador(c['id'])
        if cid in categorias or c['name'] not in mapeo:raise ValueError('Categoría TACO no reconocida o repetida')
        categorias[cid]=mapeo[c['name']]
    objetos={}
    for obj in data['physical_objects']:
        oid=identificador(obj['id'])
        if oid in objetos:raise ValueError('Objeto físico duplicado')
        if obj.get('material_declarado') not in [*CLASES,*IGNORADAS,'por_verificar']:raise ValueError('Material físico no reconocido')
        objetos[oid]={**obj,'id':oid,'evidencia_identidad':texto(obj.get('evidencia_identidad'),'evidencia de identidad física')}
    images={};paths={};hashes={}
    for raw in data['images']:
        iid=identificador(raw['id']);eco=raw.get('eco',{});archivo=ruta_segura(raiz_imagenes,raw['file_name'])
        if iid in images:raise ValueError('Imagen duplicada por ID')
        blob=archivo.read_bytes()
        with Image.open(io.BytesIO(blob)) as im:
            im.load()
            if archivo.suffix.lower() not in ['.png','.jpg','.jpeg','.webp'] or getattr(im,'n_frames',1)!=1:raise ValueError('Importar imágenes estáticas PNG/JPEG/WebP, no secuencias sin extraer')
            if im.getexif().get(274,1)!=1:raise ValueError('Normalizar EXIF antes de anotar, no durante importación')
            rgb=im.convert('RGB');w,h=rgb.size
            if type(raw['width']) is not int or type(raw['height']) is not int or (raw['width'],raw['height'])!=(w,h):raise ValueError('Dimensiones declaradas incorrectas')
            px=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            gris=np.asarray(rgb.resize((32,32),Image.Resampling.LANCZOS).convert('L'),dtype=np.float32);dct=cv2.dct(gris)[:8,:8].flatten();bits=dct>np.median(dct[1:]);bits[0]=False
            ph=f'{sum(int(b)<<i for i,b in enumerate(bits)):016x}'
        hf=hashlib.sha256(blob).hexdigest()
        if eco.get('sha256_archivo')!=hf:raise ValueError('SHA de imagen ausente o incompatible')
        if px in hashes:raise ValueError('Imágenes exactamente duplicadas por píxeles: '+hashes[px]+' / '+iid)
        hashes[px]=iid
        procedencia=eco.get('procedencia',{})
        if procedencia.get('tipo') not in (['captura_real'] if not ensayo else ['fixture','reuso_historico']):raise ValueError('Procedencia no válida para esta colección')
        for k in ['autor','referencia','licencia','permiso_ref']:texto(procedencia.get(k),k)
        if not ensayo and procedencia['licencia'].strip().lower() in ['desconocida','pendiente','sin_licencia','no_especificada','no_especificada_en_metadatos']:
            raise ValueError('Licencia/permiso de datos reales pendiente de aclarar')
        if procedencia.get('permiso_confirmado') is not True:raise ValueError('Falta permiso de incorporación')
        fecha=datetime.fromisoformat(texto(eco.get('capturado_en'),'fecha').replace('Z','+00:00'))
        if fecha.tzinfo is None:raise ValueError('Fecha sin zona horaria')
        campos={k:identificador(eco.get(k)) for k in ['escena_id','sitio_id','sesion_id','grupo_captura_id','dispositivo_id']}
        sec=eco.get('secuencia_id');campos['secuencia_id']=identificador(sec) if sec is not None else None
        if sec is not None and (not isinstance(eco.get('fotograma'),int) or isinstance(eco.get('fotograma'),bool) or eco['fotograma']<0):raise ValueError('Secuencia sin número de fotograma')
        destino=eco.get('destino_previsto','sin_asignar')
        if destino not in DESTINOS:raise ValueError('Destino previo inválido')
        presentes=eco.get('objetos_fisicos_ids')
        if not isinstance(presentes,list) or len(set(map(str,presentes)))!=len(presentes) or any(identificador(oid) not in objetos for oid in presentes):
            raise ValueError('Registrar los objetos físicos de la captura, incluso antes de anotar')
        images[iid]={'id':iid,'width':w,'height':h,'file_name':'imagenes/'+hf+archivo.suffix.lower(),
            'sha256_archivo':hf,'sha256_pixeles':px,'phash':ph,'procedencia':procedencia,
            'capturado_en':eco['capturado_en'],**campos,'fotograma':eco.get('fotograma'),'destino_previsto':destino,
            'objetos_fisicos_ids':sorted(map(str,presentes)),'instancias':[]}
        paths[iid]=archivo
    vistos=set()
    for a in data['annotations']:
        aid=identificador(a['id']);iid=identificador(a['image_id'])
        if aid in vistos or iid not in images:raise ValueError('ID de anotación duplicado o imagen inexistente')
        vistos.add(aid);inst=normalizar_instancia(a,images[iid],categorias,objetos)
        if any(x['objeto_fisico_id']==inst['objeto_fisico_id'] or firma_geometria(x)==firma_geometria(inst) for x in images[iid]['instancias']):
            raise ValueError('Objeto físico o anotación material/polígono duplicado en la misma imagen')
        images[iid]['instancias'].append(inst)
    if not images:raise ValueError('Paquete vacío: usar la plantilla para planificar, no importar')
    return {'version':VERSION,'coleccion_id':coleccion,'tipo_datos':tipo,'objetos_fisicos':objetos,
        'revisores':normalizar_revisores(revisores,tipo),'imagenes':[dict(im,instancias=sorted(im['instancias'],key=lambda a:a['id'])) for im in sorted(images.values(),key=lambda im:im['id'])],
        'sha256_politica':sha(POLITICA),'evaluacion_autorizada':False,'entrenamiento_autorizado':False},paths


def referencias_historicas():
    f=RAIZ/'datos/reportes/inventario_imagenes.jsonl'
    return [{'id':r['id'],'sha256_pixeles':r.get('sha256_pixeles'),'sha256_archivo':r.get('sha256_archivo'),'phash':r.get('phash')}
        for r in (json.loads(l) for l in f.read_text(encoding='utf-8').splitlines() if l.strip())]


def candidatos_duplicados(imagenes,historicos):
    nuevos=[];previos=[];limite=politica()['qa']['distancia_phash_candidata']
    for i,im in enumerate(imagenes):
        for otro in imagenes[:i]:
            dist=(int(im['phash'],16)^int(otro['phash'],16)).bit_count()
            if dist<=limite:nuevos.append({'a':otro['id'],'b':im['id'],'distancia':dist})
        for h in historicos:
            exacto=(im['sha256_pixeles']==h.get('sha256_pixeles') or im['sha256_archivo']==h.get('sha256_archivo'))
            dist=(int(im['phash'],16)^int(h['phash'],16)).bit_count() if h.get('phash') else 65
            if exacto or dist<=limite:previos.append({'imagen':im['id'],'historico':h['id'],'sha256_historico':h.get('sha256_archivo'),
                'exacto':exacto,'distancia':dist})
    return {'nuevos':nuevos,'historicos':previos}


def qa_geometria(im):
    aa=im['instancias'];reg={'id':im['id'],'ancho':im['width'],'alto':im['height'],
        'cajas':[{'bbox':a['bbox'],'clase':a['material'] if a['material'] in CLASES else None} for a in aa]}
    o=objetivo_deteccion(reg);ss=[{'id':a['id'],'clase':CLASES.index(a['material']),'poligonos':a['poligonos']} for a in aa if a['material'] in CLASES]
    sem=rasterizar_contornos(o,[(s['poligonos'],s['clase']) for s in ss],48,1/64)
    regional=regiones_poligonales(o,ss,sem);elegibles={}
    from ecowaste.auditoria_instancias import analizar_componentes
    instancias_geometricas=[(s['id'],s['clase'],(rasterizar_contornos(o,[(s['poligonos'],s['clase'])],48,1/64)==s['clase']+1).numpy()) for s in ss]
    fusiones=analizar_componentes(sem.numpy(),instancias_geometricas)
    for i,ident in enumerate(regional['ids']):
        m=(regional['pesos'][i].numpy()>0).astype(np.uint8);n,_,stats,_=cv2.connectedComponentsWithStats(m,connectivity=8)
        elegibles[ident]={'tokens':int(m.sum()),'componente_max':int(stats[1:,cv2.CC_STAT_AREA].max()) if n>1 else 0}
    resultados=[];masks={a['id']:mascara(a['poligonos'],im['width'],im['height']) for a in aa};solapes=[]
    for i,a in enumerate(aa):
        datos=elegibles.get(a['id'],{'tokens':0,'componente_max':0});menor=min(a['bbox'][2]*o['geometria']['escala_x'],a['bbox'][3]*o['geometria']['escala_y'])
        resultados.append({'id':a['id'],'objeto_fisico_id':a['objeto_fisico_id'],'material':a['material'],**datos,
            'lado_menor_384':menor,'pequeno_menos_8px':menor<8,
            'componente_compartido_48':a['id'] in fusiones['instancias_en_componentes_compartidos'],
            'diferencia_area_declarada_mask':abs(a['area_declarada']-a['area_mask_px']) if a.get('area_declarada') is not None else None,
            'organico_utilizable_48':a['material']=='organico' and datos['componente_max']>=2})
        for b in aa[:i]:
            inter=int((masks[a['id']]&masks[b['id']]).sum())
            if inter:
                fr=inter/max(1,min(a['area_mask_px'],b['area_mask_px']))
                solapes.append({'a':a['id'],'b':b['id'],'pixeles':inter,'fraccion_menor':fr,'materiales_distintos':a['material']!=b['material'],
                    'declarado':b['objeto_fisico_id'] in a['solapa_con'] and a['objeto_fisico_id'] in b['solapa_con']})
    return {'imagen':im['id'],'instancias':resultados,'solapes':solapes,'representacion_instancias_48':fusiones,
        'politica_regional':'original_384_48_cobertura_1_64_sin_padding_ignore_mezclas'}


def importar(entrada,raiz_imagenes,destino,revisores,*,ensayo=False,historicos=None):
    destino=destino_nuevo(destino)
    original=Path(entrada).read_bytes()
    canon,paths=normalizar_paquete(json.loads(original),raiz_imagenes,revisores,ensayo=ensayo)
    refs=referencias_historicas() if historicos is None else historicos;dups=candidatos_duplicados(canon['imagenes'],refs)
    if not ensayo and any(d['exacto'] for d in dups['historicos']):raise ValueError('La colección real contiene imágenes históricas; no son adquisición nueva')
    destino.mkdir(parents=True)
    publicar_bytes(destino/'entrada_annotations.json',original)
    for im in canon['imagenes']:copiar_imagen_verificada(paths[im['id']],destino/im['file_name'],im['sha256_archivo'])
    publicar(destino/'coleccion.json',canon);publicar(destino/'duplicados.json',dups)
    publicar(destino/'qa_importacion.json',[qa_geometria(im) for im in canon['imagenes']])
    publicar(destino/'IMPORTADO.json',{'version':VERSION,'tipo_datos':canon['tipo_datos'],'sha256_coleccion':digest(canon),
        'sha256_politica':sha(POLITICA),'sha256_entrada':hashlib.sha256(original).hexdigest(),'sha256_historicos':digest(refs),
        'archivos':{f.relative_to(destino).as_posix():sha(f) for f in destino.rglob('*') if f.is_file()},'estado':'cuarentena_sin_GT_aceptado'})
    return canon


def cargar(paquete):
    paquete=destino_nuevo(paquete,existente=True);commit=leer(paquete/'IMPORTADO.json');c=leer(paquete/'coleccion.json')
    if commit.get('version')!=VERSION or c.get('version')!=VERSION:raise ValueError('Paquete de otro protocolo')
    for f,h in commit['archivos'].items():
        if sha(ruta_segura(paquete,f))!=h:raise ValueError('Paquete alterado: '+f)
    if commit['sha256_coleccion']!=digest(c) or c['sha256_politica']!=sha(POLITICA):raise ValueError('Versión/política distinta de la importada')
    return c
