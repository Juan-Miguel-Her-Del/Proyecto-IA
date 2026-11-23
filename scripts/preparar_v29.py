"""QA técnica y colección experimental de etiquetas COCO publicadas por autores.

No escribe en históricos/revisiones, no crea aceptación humana/V26 y no abre test.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse,copy,json,re,shutil,sys,uuid,importlib.metadata
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RAIZ))
import numpy as np
import cv2
from PIL import Image
import torch
from ecowaste.datos_v29 import validar_imagen,conflicto_duplicado,resumen_materiales,VERSION
from ecowaste.datasets import CLASES,objetivo_deteccion
from ecowaste.incorporacion import leer,sha,digest
from ecowaste.reanudacion_regional import publicar
from ecowaste.backend_numerico import politica

SALIDA=RAIZ/'resultados/v29_packwise_click'
BASE=RAIZ/'datos/nuevos/v29'
MAPEO={
 'click':{'Bottle Cap':None,'Cardboard':'carton','Crumbled Paper':'papel','Match Box':None,
    'Metal Can':'metal','Others':None,'Paper Cup':None,'Paper Pack':None,'Plastic Bag':'plastico',
    'Plastic Bottle':'plastico','Plastic Packet':'plastico','Silver Foil':'metal','Slippers':None},
 'packwise':{'undefined':None,'plastic-foil/bag':'plastico','aluminium-coated-foil':None,'paper-bag':'papel',
    'beverage-carton':None,'plastic-other':'plastico','plastic-lid':'plastico','plastic-cup/pot':'plastico',
    'blister':None,'metal-can':'metal','metal-foil':'metal','net':None,'metal-lid':'metal','tissue':'papel',
    'wood':None,'paper-other':None,'foamed-plastic':'plastico','textile':None,'plastic-bottle':'plastico',
    'ribbon/tape':None,'metal-other':'metal','tube':None,'paper-cup/pot':None}}


def tareas():
    for fuente in ['packwise','click']:
        base=BASE/'externos'/fuente/'extraidos'
        for split in ['train','val']:
            f=base/(f'data/{split}.json' if fuente=='packwise' else f'multi class/annotations/instances_{split}.json')
            d=leer(f);cats={c['id']:c['name'] for c in d['categories']}
            if set(cats.values())!=set(MAPEO[fuente]):raise ValueError('Taxonomía fuente cambió: '+fuente)
            mp={c:MAPEO[fuente][n] for c,n in cats.items()};anns=defaultdict(list);vistos=set();iids={i['id'] for i in d['images']}
            if len(iids)!=len(d['images']):raise ValueError('Imágenes duplicadas por ID en COCO')
            for a in d['annotations']:
                if a['id'] in vistos or a['image_id'] not in iids:raise ValueError('ID duplicado o referencia COCO inexistente')
                vistos.add(a['id']);anns[a['image_id']].append(a)
            for im in d['images']:
                if not any(mp[a['category_id']] in CLASES for a in anns[im['id']]):continue
                im=copy.deepcopy(im);im['eco_id']=fuente+':'+str(im['id'])
                path=base/(f'data/{split}/' if fuente=='packwise' else f'images/{split}/')/im['file_name']
                yield fuente,split,f,im,anns[im['id']],mp,path


def grupo(fuente,nombre):
    if fuente=='packwise':
        m=re.fullmatch(r'(Vollstrom\d+)_frame\d+_Probe\d+(?:_\d+)*\.jpg',nombre)
        if not m:raise ValueError('Nombre PackWISE sin secuencia conocida')
        g='packwise:secuencia:'+m[1]
        return g,'val' if m[1]=='Vollstrom6' else 'train','secuencia_codificada_en_nombre; identidad física individual desconocida'
    m=re.fullmatch(r'trash \((\d+)\)\.jpg',nombre)
    if not m:raise ValueError('Nombre CLICK sin índice de captura')
    bloque=(int(m[1])-1)//20;g='click:bloque_consecutivo:'+str(bloque)
    split='val' if int(digest({'grupo':g,'semilla':2901})[:8],16)%5==0 else 'train'
    return g,split,'heurística conservadora de 20 nombres; no demuestra sesión/identidad física'


def cachear():
    torch.set_num_threads(4)
    clave=digest({'codigo_lector':sha(RAIZ/'ecowaste/datos_v29.py'),'mapeo':MAPEO})[:16]
    folder=BASE/'cache'/clave;ok=[];fallos=[];pendientes=[]
    for k,(fuente,split,src,im,anns,mp,path) in enumerate(tareas()):
        if not path.exists():pendientes.append({'id':im['eco_id'],'archivo':path.relative_to(RAIZ).as_posix()});continue
        entrada=digest({'imagen':sha(path),'anotaciones':anns,'mapeo':mp,'id':im['eco_id'],'version_lector':clave})
        p=folder/(im['eco_id'].replace(':','_')+'.pt')
        if p.exists():
            r=torch.load(p,map_location='cpu',weights_only=True)
            if r['entrada']!=entrada:raise ValueError('Caché de otro contenido')
        else:
            try:
                f,x,y,rs=validar_imagen(path,im,anns,mp)
                g,sp,metodo=grupo(fuente,im['file_name'])
                r={'entrada':entrada,'fila':{**f,'fuente':fuente,'split_original':split,'split_propuesto':sp,
                    'grupo':g,'agrupacion_metodo':metodo,'ruta_original':path.relative_to(RAIZ).as_posix(),
                    'anotaciones_originales':src.relative_to(RAIZ).as_posix(),'sha256_anotaciones_originales':sha(src),
                    'objeto_fisico':None,'sesion':None,'dispositivo':None,'licencia':'CC BY 4.0'},
                    'imagen':x,'mascara':y,'regiones':[(c,torch.from_numpy(m.copy()),aid) for c,m,aid in rs]}
                publicar(p,r,tensor=True)
            except (ValueError,TypeError,KeyError,OSError) as e:
                fallos.append({'id':im['eco_id'],'fuente':fuente,'ruta':path.relative_to(RAIZ).as_posix(),'error':str(e)});continue
        ok.append(p)
        if len(ok)%25==0:print('QA',len(ok),'válidas',len(fallos),'cuarentena',len(pendientes),'pendientes',flush=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
    out=SALIDA/'qa'/stamp
    publicar(out/'cache.json',{'version':VERSION,'clave':clave,'archivos':[p.relative_to(RAIZ).as_posix() for p in ok],
        'fallos':fallos,'descargas_pendientes':pendientes,'mapeo':MAPEO,'test_abierto':False,'aceptadas_V26':0})
    return out/'cache.json'


class Grupos:
    def __init__(self,keys):self.padre={x:x for x in keys}
    def raiz(self,x):
        while self.padre[x]!=x:self.padre[x]=self.padre[self.padre[x]];x=self.padre[x]
        return x
    def unir(self,a,b):
        a,b=self.raiz(a),self.raiz(b)
        if a!=b:self.padre[max(a,b)]=min(a,b)


def congelar(qa):
    qa=Path(qa).resolve();q=leer(qa);dest=BASE/'coleccion'
    if dest.exists():raise FileExistsError('Colección existente: conservar versión congelada')
    if q['descargas_pendientes']:raise ValueError('Descarga incompleta: finalizar/reintentar antes de congelar')
    datos=[torch.load(RAIZ/p,map_location='cpu',weights_only=True) for p in q['archivos']]
    filas=[x['fila'] for x in datos];refs=[]
    for tarea in ['deteccion','clasificacion']:
        for split in ['train','val']:
            p=RAIZ/'datos/procesados'/tarea/(split+'.jsonl')
            refs += [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]
    # Solo hashes ya publicados de train/val. No abrir/evaluar test o generalización.
    grupos=Grupos({f['grupo'] for f in filas});conf=[];vetados=set();duplicados=set()
    for j,f in enumerate(filas):
        a={**f,'id':f['registro']['id']}
        for h in refs:
            t=conflicto_duplicado(a,h)
            if t:conf.append({'a':a['id'],'b':h['id'],'tipo':t,'con_historico':True});vetados.add(f['grupo'])
        for b in filas[:j]:
            t=conflicto_duplicado(a,b)
            if t:
                grupos.unir(f['grupo'],b['grupo']);conf.append({'a':a['id'],'b':b['registro']['id'],'tipo':t,'con_historico':False})
                if t=='exacto':duplicados.add(a['id'])
    prohibidos={grupos.raiz(g) for g in vetados};componentes=defaultdict(list)
    for f in filas:componentes[grupos.raiz(f['grupo'])].append(f['split_propuesto'])
    # Si una similitud enlaza cohortes previstas, todo el componente pasa a train;
    # ninguna imagen cercana permanece en evaluación. Regla anterior a métricas.
    destinos={g:('train' if 'train' in ss else 'val') for g,ss in componentes.items()}
    finales={'train':[],'val':[]};excluidos=[]
    for dato in datos:
        f=dato['fila'];g=grupos.raiz(f['grupo']);iid=f['registro']['id']
        with Image.open(RAIZ/f['ruta_original']) as pil:f['orientacion_EXIF']=int(pil.getexif().get(274,1))
        if f['orientacion_EXIF']!=1:
            excluidos.append({'id':iid,'motivo':'orientacion_EXIF_no_neutra_requiere_verificar_marco_de_anotacion'});continue
        if g in prohibidos or iid in duplicados:
            excluidos.append({'id':iid,'motivo':'grupo_coincide_con_historico' if g in prohibidos else 'duplicado_exacto_nuevo'});continue
        f['grupo_componente']=g;f['split_final']=destinos[g];finales[destinos[g]].append(dato)
    if not finales['train'] or not finales['val']:raise ValueError('Separación no permite train y evaluación nuevos')
    if {d['fila']['fuente'] for d in finales['train']}!={'packwise','click'}:raise ValueError('Falta fuente nueva en train')
    archivos={};resumen={}
    for split,dd in finales.items():
        registros=[];objetivos=[];regiones=[];ims=[];ms=[];ci=[];ca=[];gid=0
        for i,d in enumerate(dd,1):
            f=d['fila'];r=f['registro'];original=RAIZ/f['ruta_original'];rel='imagenes/'+f['fuente']+'/'+original.name;foto=dest/rel
            foto.parent.mkdir(parents=True,exist_ok=True)
            if foto.exists():raise FileExistsError('Colisión de fotografías; no sobrescribir '+rel)
            with original.open('rb') as entrada,foto.open('xb') as salida:shutil.copyfileobj(entrada,salida)
            if sha(foto)!=f['sha256_archivo']:raise ValueError('Copia alterada')
            archivos[rel]=f['sha256_archivo'];registros.append({**r,'ruta':foto.relative_to(RAIZ).as_posix(),
                'sha256_archivo':f['sha256_archivo'],'fuente':f['fuente'],'grupo':f['grupo_componente']})
            objetivos.append(objetivo_deteccion(r));regiones.append(d['regiones']);ims.append(d['imagen']);ms.append(d['mascara'])
            ci.append({'id':i,'file_name':rel,'width':r['ancho'],'height':r['alto'],'license':1,
                'eco':{'id':r['id'],**{k:v for k,v in f.items() if k not in ['anotaciones','registro','objetos_48','avisos']}}})
            for a in f['anotaciones']:
                gid+=1;cl=CLASES.index(a['material'])+1 if a['material'] in CLASES else 7
                ca.append({'id':gid,'image_id':i,'category_id':cl,'bbox':a['bbox'],'area':a['area'],
                    'segmentation':a['segmentation'],'iscrowd':0,'ignore':int(cl==7),
                    'eco':{k:v for k,v in a.items() if k not in ['bbox','segmentation','area']}})
        cache={'imagenes':torch.stack(ims),'mascaras':torch.stack(ms),'registros':registros,'objetivos':objetivos,'regiones':regiones}
        publicar(dest/f'{split}.pt',cache,tensor=True);archivos[f'{split}.pt']=sha(dest/f'{split}.pt')
        publicar(dest/f'{split}_coco.json',{'info':{'version':VERSION,'description':'Etiquetas de autores; QA técnica; sin revisión independiente EcoWasteFormer'},
            'images':ci,'annotations':ca,'categories':[{'id':i+1,'name':c} for i,c in enumerate(CLASES+['ignore'])],
            'licenses':[{'id':1,'name':'CC BY 4.0','url':'https://creativecommons.org/licenses/by/4.0/'}]})
        archivos[f'{split}_coco.json']=sha(dest/f'{split}_coco.json')
        resumen[split]={'imagenes':len(dd),'por_fuente':dict(Counter(d['fila']['fuente'] for d in dd)),
            'por_material':resumen_materiales([d['fila'] for d in dd]),
            'regiones_sin_token_48':sum(a['tokens_48']==0 for d in dd for a in d['fila']['objetos_48'] if a['material'] in CLASES)}
        minimas=Counter()
        for dato in dd:
            for c,m,aid in dato['regiones']:
                n,_,stats,_=cv2.connectedComponentsWithStats(m.numpy().astype(np.uint8),connectivity=8)
                if n<2 or int(stats[1:,cv2.CC_STAT_AREA].max())<2:minimas[CLASES[c]]+=1
        resumen[split]['sin_componente_2_celdas_48_por_material']=dict(minimas)
    publicar(dest/'mapeo_y_agrupacion.json',{'mapeo':MAPEO,'conflictos_duplicados':conf,'excluidos':excluidos,
        'fallos_qa':q['fallos'],'grupos':{g:{'split':destinos[g],'fuera':g in prohibidos} for g in componentes},
        'sin_garantia':'No hay identificadores físicos ni sesiones verificadas en CLICK. Los bloques de nombres son una precaución, no independencia demostrada.'})
    archivos['mapeo_y_agrupacion.json']=sha(dest/'mapeo_y_agrupacion.json')
    publicar(dest/'MANIFIESTO.json',{'version':VERSION,'estado':'EXPERIMENTAL_ETIQUETAS_AUTORES','qa_tecnica_aprobada':True,
        'archivos':archivos,'resumen':resumen,'qa':qa.relative_to(RAIZ).as_posix(),'sha256_qa':sha(qa),'aceptadas_V26':0,
        'entorno_preparacion':{'python':sys.version,'ejecutable':sys.executable,'torch':str(torch.__version__),
            'paquetes':{p:importlib.metadata.version(p) for p in ['numpy','pillow','pycocotools','opencv-python']}},
        'revisiones_humanas_proyecto_creadas':0,'fuentes':['https://fordatis.fraunhofer.de/handle/fordatis/463.2',
        'https://data.mendeley.com/datasets/htcvvz6fm2/2'],'nota':'Los autores proveen etiquetas. QA automática no certifica exhaustividad, material químico o adjudicación independiente.'})
    config={'version':'v29_datos_preserva_historico_1','checkpoint_origen':'checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt',
        'sha256_origen':'eb8fd00ee536ead21e3a2a0ecd5b06baad04acb2e437b22e0eac3686a7bf36b9',
        'coleccion':dest.relative_to(RAIZ).as_posix(),'sha256_coleccion':sha(dest/'MANIFIESTO.json'),
        'actualizaciones':600,'historico_batch':8,'nuevo_batch':4,'coeficiente_nuevos':.5,'auxiliar_batch':16,
        'coeficiente_auxiliar_global':.5,'learning_rate':.0005,'weight_decay':.01,'clipping':1.,'coeficiente_fondo':3.,
        'backend_numerico':politica('completo'),'warmup_nuevo':False,'parada_temprana':False,'semilla_nuevos':2901,
        'checkpoints':[0,100,300,600],'seleccion_por_validacion':False,'test':False,'generalizacion':False,
        'adopcion_automatica':False,'vistas':[1,5],'umbral':.05,'politica':'argmax7','nms':.5,'top':100,'area_minima':2,
        'justificacion_presupuesto':'600 pasos: 4800 exposiciones históricas por rama; 2400 nuevas B, 1200 por fuente, aproximadamente 2-4 recorridos de la colección. Piloto fijo sin ampliación tras métricas.',
        'hipotesis':'Añadir datos de autores con exposición histórica intacta mejora ajuste y detección en la cohorte nueva sin el reemplazo 25% de V28.2. Vidrio/orgánico no reciben supervisión nueva de estas fuentes.'}
    publicar(RAIZ/'configuracion/entrenamiento_packwise_click_v29.json',config)
    publicar(SALIDA/'coleccion_congelada.json',{'resumen':resumen,'sha256_manifiesto':sha(dest/'MANIFIESTO.json'),'excluidos':excluidos,'fallos_qa':q['fallos']})
    print(json.dumps(resumen,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--congelar',type=Path);a=p.parse_args()
    if a.congelar:congelar(a.congelar)
    else:print('QA',cachear(),flush=True)


if __name__=='__main__':main()
