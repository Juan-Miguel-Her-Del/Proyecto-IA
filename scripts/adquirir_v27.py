"""Pilotos públicos acotados: miembros train de ZIP por rangos y fotos HPI con CC BY.

Sin modelos ni credenciales. Nunca extrae ZIP completos ni abre miembros test.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from collections import Counter,defaultdict
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import publicar,publicar_bytes,sha
from scripts.investigar_v27 import DATOS,SALIDA,obtener


class ZipPorRangos(io.RawIOBase):
    def __init__(self,url,*,size=None,presupuesto=160*1024*1024):
        self.url=url;self.pos=0;self.bytes=0;self.presupuesto=presupuesto;self.rangos=[];self.cache={}
        data,headers=self._solicitar(0,1)
        self.size=int(headers['Content-Range'].split('/')[-1])
        if size is not None and self.size!=size:raise ValueError('Tamaño remoto cambió respecto al registro')

    def _solicitar(self,start,n):
        if n<1 or self.bytes+n>self.presupuesto:raise ValueError('Presupuesto de descarga excedido')
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{start+n-1}','User-Agent':'EcoWasteFormer-public-data-pilot/27'})
        with urllib.request.urlopen(req,timeout=60) as r:
            if r.status!=206:raise ValueError('Servidor no respeta Range; no descargar el archivo completo')
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''))
            if not match or int(match[1])!=start or int(match[2])!=start+n-1:raise ValueError('Content-Range incorrecto')
            data=r.read(n+1)
            if len(data)!=n:raise ValueError('Respuesta de rango incompleta')
            self.bytes+=n;self.rangos.append({'inicio':start,'bytes':n,'sha256':hashlib.sha256(data).hexdigest()})
            return data,{'Content-Range':r.headers.get('Content-Range')}

    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if not 0<=self.pos<=self.size:raise ValueError('Seek fuera del archivo')
        return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if not n:return b''
        bloque=self.pos//(1024*1024)*(1024*1024)
        if bloque in self.cache and self.pos+n<=bloque+len(self.cache[bloque]):
            data=self.cache[bloque][self.pos-bloque:self.pos-bloque+n]
        elif n<4096 and self.pos+n<=min(bloque+1024*1024,self.size):
            data,_=self._solicitar(bloque,min(1024*1024,self.size-bloque))
            if len(self.cache)>=16:self.cache.pop(next(iter(self.cache)))
            self.cache[bloque]=data;data=data[self.pos-bloque:self.pos-bloque+n]
        else:data,_=self._solicitar(self.pos,n)
        self.pos+=n;return data


def solo_train(nombre):
    p=PurePosixPath(nombre.replace('\\','/'));partes=[x.lower() for x in p.parts]
    return not p.is_absolute() and '..' not in partes and not any(':' in x or 'test' in x or 'val' in x for x in partes) and any(x=='train' or x.startswith('train.') or x.startswith('train_') for x in partes)


def miembro(z,info):
    if not solo_train(info.filename):raise ValueError('No abrir miembros de evaluación ni rutas inseguras')
    if info.file_size>64*1024*1024 or info.compress_size>48*1024*1024:raise ValueError('Miembro excesivo para piloto')
    return z.read(info,pwd=b'UP#1VuX409z4')  # Contraseña publicada en el registro oficial ZeroWaste; no es una credencial.


def abrir_zip(nombre):
    if nombre=='zerowaste':
        meta=json.loads((DATOS/'fuentes/zerowaste_zenodo.bin').read_bytes())
        f=next(x for x in meta['files'] if x['key']=='zerowaste-f-final.zip')
        return ZipPorRangos(f['links']['self'],size=f['size'])
    if nombre=='warp':return ZipPorRangos('https://www.kaggle.com/api/v1/datasets/download/parohod/warp-waste-recycling-plant-dataset')
    raise ValueError('Fuente no autorizada')


def indices():
    for nombre in ['zerowaste','warp']:
        destino=DATOS/nombre;recibo=SALIDA/f'indice_{nombre}.json'
        if recibo.exists():
            if json.loads(recibo.read_bytes()).get('disponible'):continue
            recibo=SALIDA/f'indice_{nombre}_revision02.json'
            if recibo.exists():continue
        try:
            f=abrir_zip(nombre)
            with zipfile.ZipFile(f) as z:
                lista=[{'nombre':i.filename,'bytes':i.file_size,'comprimidos':i.compress_size,'crc32':f'{i.CRC:08x}',
                    'train':solo_train(i.filename)} for i in z.infolist()]
                publicar(destino/'indice_zip.json',lista)
                candidatos=[i for i in z.infolist() if solo_train(i.filename) and i.filename.lower().endswith(('.json','.txt'))]
                for i in sorted(candidatos,key=lambda i:(0 if 'class' in i.filename.lower() else 1,i.filename))[:2]:
                    data=miembro(z,i);path=destino/'anotaciones_muestra'/PurePosixPath(i.filename).name
                    publicar_bytes(path,data);print(nombre,'anotacion muestra',i.filename,len(data),flush=True)
            publicar(recibo,{'disponible':True,'miembros':len(lista),'miembros_train':sum(i['train'] for i in lista),
                'url':f.url,'bytes_remotos_archivo':f.size,'bytes_transferidos':f.bytes,'rangos':f.rangos,
                'archivo_completo_descargado':False,'checksum_archivo_completo_verificado':False})
            print(nombre, 'train primeros', [i['nombre'] for i in lista if i['train']][:12],flush=True)
        except Exception as e:publicar(recibo,{'disponible':False,'error':str(e)});print(nombre,e,flush=True)


def hpi():
    destino=DATOS/'hpi';recibo=SALIDA/'piloto_hpi.json'
    if recibo.exists():raise FileExistsError('Piloto HPI existente')
    # El permiso de fotos y datos está explícito en la ficha original, no en un espejo.
    card=(DATOS/'fuentes/hpi_readme.bin').read_text(encoding='utf-8')
    if 'cc-by-4.0' not in card.lower() or 'photographs' not in card:raise ValueError('Licencia original no comprobada')
    filas=[];consultas=[]
    for offset,n in [(0,100),(100,100),(200,15)]:
        q=urllib.parse.urlencode({'dataset':'AI-ServicesBB/food-waste-dataset','config':'default','split':'train','offset':offset,'length':n})
        data,r=obtener('https://datasets-server.huggingface.co/rows?'+q)
        j=json.loads(data);publicar_bytes(destino/'metadatos'/f'rows_train_{offset}.json',data);consultas.append(r)
        filas.extend(j['rows'])
    print('HPI filas train recibidas',len(filas),'campos',list(filas[0]['row']),flush=True)
    recibos=[];bonids=set()
    for item in filas:
        row=item['row'];bon=str(row['bonid'])
        if bon in bonids:continue
        bonids.add(bon);img=row['image'];url=img['src'];data,receipt=obtener(url,limite=16*1024*1024)
        path=destino/'imagenes'/f'train_{item["row_idx"]:03d}.jpg';publicar_bytes(path,data)
        recibos.append({'row_idx':item['row_idx'],'bonid':bon,'archivo':path.relative_to(RAIZ).as_posix(),
            'anotaciones_espaciales_originales':False,'etiquetas_nutricionales':row,'descarga':receipt})
    publicar(recibo,{'fuente':'AI-ServicesBB/food-waste-dataset','split':'train','licencia':'CC-BY-4.0',
        'imagenes':recibos,'filas_train_verificadas':len(filas),'recibos_meal_distintos':len(bonids),
        'instancias_anotadas_originales':0,'propuestas_espaciales_creadas':0,'aceptadas':0,'consultas':consultas})
    print('HPI fotografías reales descargadas',len(recibos),'cajas/máscaras originales: 0',flush=True)


def seleccionar(candidatos,clases,limites,*,semilla='v27_piloto_1'):
    """Diversifica prefijos de secuencia disponibles; no afirma identidad física."""
    elegidos=[];usados=set();grupos=Counter()
    for clase,limite in limites.items():
        disponibles=[c for c in candidatos if clase in clases[c['id']]]
        for _ in range(limite):
            libres=[c for c in disponibles if c['id'] not in usados]
            if not libres:break
            c=min(libres,key=lambda c:(grupos[c['grupo_nombre']],hashlib.sha256((semilla+c['nombre']).encode()).hexdigest()))
            usados.add(c['id']);elegidos.append(c);grupos[c['grupo_nombre']]+=1
    return elegidos


def guardar_miembro(destino,nombre,data,info):
    p=destino/nombre
    if p.exists():
        if sha(p)!=hashlib.sha256(data).hexdigest():raise ValueError('No reemplazar otro miembro')
    else:publicar_bytes(p,data)
    return {'archivo':p.relative_to(RAIZ).as_posix(),'miembro_original':info.filename,'bytes':len(data),
        'sha256':sha(p),'crc32_zip':f'{info.CRC:08x}','crc32_verificado_por_zipfile':True}


def piloto_zero():
    recibo=SALIDA/'piloto_zerowaste.json'
    if recibo.exists():return
    destino=DATOS/'zerowaste';j=json.loads((destino/'anotaciones_muestra/labels.json').read_bytes())
    cats={a['id']:a['name'] for a in j['categories']};por_im=defaultdict(list)
    for a in j['annotations']:por_im[a['image_id']].append(a)
    cs={i['id']:{cats[a['category_id']] for a in por_im[i['id']]} for i in j['images']}
    cand=[{'id':i['id'],'nombre':i['file_name'],'grupo_nombre':i['file_name'].split('_frame_')[0]} for i in j['images']]
    ss=seleccionar(cand,cs,{'cardboard':12,'rigid_plastic':4,'soft_plastic':4,'metal':4})
    f=abrir_zip('zerowaste');archivos=[]
    with zipfile.ZipFile(f) as z:
        por_nombre={PurePosixPath(i.filename).name:i for i in z.infolist() if solo_train(i.filename) and '/data/' in i.filename and not i.is_dir()}
        for c in ss:
            info=por_nombre[c['nombre']];data=miembro(z,info);r=guardar_miembro(destino/'imagenes',c['nombre'],data,info)
            archivos.append({'imagen_id_original':c['id'],'grupo_nombre':c['grupo_nombre'],**r});print('ZeroWaste imagen',len(archivos),'/',len(ss),flush=True)
    publicar(destino/'piloto_coco_original.json',{**j,'images':[i for i in j['images'] if i['id'] in {s['id'] for s in ss}],
        'annotations':[a for a in j['annotations'] if a['image_id'] in {s['id'] for s in ss}]})
    publicar(recibo,{'fuente':'ZeroWaste-f-final','split':'train','imagenes':archivos,
        'permiso_aplicado':'uso_academico_no_comercial_con_atribucion_segun_autores_CC_BY_NC_4_0',
        'conflicto_licencia':'Zenodo declara CC BY 4.0; autores NC. No se presume permiso comercial ni se usa ZeroWasteAug.',
        'train_imagenes_en_json':len(j['images']),'train_instancias_en_json':len(j['annotations']),
        'train_categorias_instancias':dict(Counter(cats[a['category_id']] for a in j['annotations'])),
        'piloto_instancias':sum(len(por_im[s['id']]) for s in ss),'bytes_transferidos':f.bytes,'rangos':f.rangos,
        'anotaciones_originales_sin_cambios':True,'test_abierto':False,'aceptadas':0})


def piloto_warp():
    recibo=SALIDA/'piloto_warp.json'
    if recibo.exists():return
    lic=json.loads((DATOS/'fuentes/kaggle_warp_search.bin').read_bytes())
    item=next(x for x in lic if x['ref']=='parohod/warp-waste-recycling-plant-dataset')
    if item['licenseName']!='CC0: Public Domain':raise ValueError('Licencia del distribuidor oficial no confirmada')
    destino=DATOS/'warp';f=abrir_zip('warp');archivos=[];propuestas=[];labels_guardadas=[]
    with zipfile.ZipFile(f) as z:
        # Este catálogo es metadato compartido de clases, no un miembro test.
        catinfo=z.getinfo('Warp-D/classes.txt');raw=z.read(catinfo);guardar_miembro(destino,'classes_original.txt',raw,catinfo)
        cats=[x.strip() for x in raw.decode('utf-8').splitlines() if x.strip()];counts=Counter();clases={};cand=[];labels={}
        infos=[i for i in z.infolist() if i.filename.startswith('Warp-D/train/labels/') and i.filename.endswith('.txt')]
        for k,info in enumerate(infos):
            data=miembro(z,info);lineas=[x.split() for x in data.decode('utf-8').splitlines() if x.strip()];parsed=[]
            for fila in lineas:
                if len(fila)!=5:raise ValueError('Anotación YOLO de formato inesperado')
                ci=int(fila[0]);valores=list(map(float,fila[1:]));parsed.append({'clase_id':ci,'clase_original':cats[ci],'xywh_normalizado':valores})
                counts[cats[ci]]+=1
            stem=PurePosixPath(info.filename).stem;labels[stem]=(info,data,parsed)
            grupos=stem.split('_');grupo='_'.join(grupos[:2])+'_'+grupos[2].split('-')[0] if len(grupos)>2 else stem
            cand.append({'id':stem,'nombre':stem,'grupo_nombre':grupo});clases[stem]={a['clase_original'] for a in parsed}
            if (k+1)%500==0:print('WaRP etiquetas train',k+1,'/',len(infos),'bytes red',f.bytes,flush=True)
        ss=seleccionar(cand,clases,{'glass-dark':4,'glass-green':4,'glass-transp':4,'juice-cardboard':2,'milk-cardboard':2,'cans':4,'bottle-transp':4})
        imgs={PurePosixPath(i.filename).stem:i for i in z.infolist() if i.filename.startswith('Warp-D/train/images/') and i.filename.lower().endswith(('.jpg','.png','.jpeg'))}
        for s in ss:
            info,data,parsed=labels[s['id']];labels_guardadas.append(guardar_miembro(destino/'labels_piloto',PurePosixPath(info.filename).name,data,info))
            pic=imgs[s['id']];blob=miembro(z,pic);r=guardar_miembro(destino/'imagenes',PurePosixPath(pic.filename).name,blob,pic)
            archivos.append({'imagen_id_original':s['id'],'grupo_nombre':s['grupo_nombre'],**r})
            propuestas.append({'imagen_id_original':s['id'],'anotaciones_yolo':parsed})
            print('WaRP imagen',len(archivos),'/',len(ss),flush=True)
        # Conservar las etiquetas train leídas, sin tocar imágenes/cajas de otros splits.
        for stem,(info,data,_) in labels.items():guardar_miembro(destino/'labels_train_originales',PurePosixPath(info.filename).name,data,info)
    publicar(destino/'propuestas_yolo_originales.json',propuestas)
    publicar(recibo,{'fuente':'WaRP-D distribuido por autor enlazado desde AIRI','split':'train','licencia':'CC0: Public Domain',
        'categorias_originales':cats,'train_imagenes_en_indice':len(imgs),'train_archivos_labels_leidos':len(infos),
        'train_instancias_verificadas':sum(counts.values()),'train_categorias_instancias':dict(counts),
        'imagenes':archivos,'labels_piloto':labels_guardadas,'piloto_instancias':sum(len(x['anotaciones_yolo']) for x in propuestas),
        'bytes_transferidos':f.bytes,'rangos':f.rangos,'poligonos_originales':False,'poligonos_fabricados':0,
        'test_abierto':False,'aceptadas':0})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--fase',choices=['indices','hpi','pilotos'],required=True);a=ap.parse_args()
    if a.fase=='indices':indices()
    elif a.fase=='hpi':hpi()
    else:piloto_zero();piloto_warp()


if __name__=='__main__':main()
