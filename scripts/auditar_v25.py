"""V25: censo exhaustivo y analisis posterior de logits ya archivados (train/val)."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from ecowaste.datasets import CLASES,objetivo_deteccion
from ecowaste.poligonos import ContornosTACO
from ecowaste.supervision_regional import regiones_poligonales
from ecowaste.discriminacion_mapas import metricas_instancias
from ecowaste.auditoria_datos_representacion import censar_escena,resumir_censo,VERSION
from ecowaste.auditoria_propuestas_internas import analizar_imagen,resumir_objetos
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.ablacion_fondo import resumir_deteccion
from ecowaste.reanudacion_regional import publicar,sha
from scripts.auditar_propuestas_internas import cargar_traza
from scripts.capturar_v25 import SALIDA,leer


def registros(split):
    if split not in ['train','val']:raise ValueError('Test reservado')
    return [json.loads(l) for l in (RAIZ/f'datos/procesados/deteccion/{split}.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]


def font(n=12):
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)


def fichas(r,filas,contornos,tiles):
    with Image.open(RAIZ/r['ruta']) as im:im=im.convert('RGB')
    for f in filas:
        if not f['clase']:continue
        a=next(a for a in r['cajas'] if a['id_interno']==f['id']);ps=contornos.contornos(r,a)
        x,y,w,h=f['bbox'];b=max(w,h)*.18
        region=(max(0,int(x-b)),max(0,int(y-b)),min(im.width,int(x+w+b+1)),min(im.height,int(y+h+b+1)))
        crop=im.crop(region);d=ImageDraw.Draw(crop)
        for p in ps:
            xy=[(px-region[0],py-region[1]) for px,py in zip(p[::2],p[1::2])];d.line(xy+[xy[0]],fill='#00ff44',width=max(2,round(max(crop.size)/160)))
        crop.thumbnail((174,150));tile=Image.new('RGB',(190,205),'#fafafa');tile.paste(crop,((190-crop.width)//2,0));d=ImageDraw.Draw(tile)
        d.text((3,151),f"{r['id']} / {f['id']}",fill='black',font=font())
        d.text((3,167),f['categoria'][:29],fill='black',font=font(11))
        d.text((3,184),f"{f['lado_menor_384']:.1f}px / {f['region_elegible']['tokens']} tok",fill='black',font=font())
        tiles[f['clase']].append((f['id'],tile))


def censo():
    path=SALIDA/'censo';path.mkdir(exist_ok=False);cont=ContornosTACO(RAIZ,1/64);inicio=time.perf_counter();salidas={}
    for split in ['train','val']:
        filas=[];oraculos=[];imagenes=[];solapes=[];tiles={c:[] for c in CLASES};casos=[]
        rr=registros(split)
        for i,r in enumerate(rr):
            with Image.open(RAIZ/r['ruta']) as im:im=im.convert('RGB');pixels=np.asarray(im)
            hp=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest();hf=sha(RAIZ/r['ruta'])
            if im.size!=(r['ancho'],r['alto']) or hp!=r['sha256_pixeles'] or hf!=r['sha256_archivo']:raise AssertionError('Integridad de imagen diferente: '+r['id'])
            small=cv2.resize(pixels,(192,192),interpolation=cv2.INTER_AREA);gris=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
            hsv=cv2.cvtColor(small,cv2.COLOR_RGB2HSV)
            imagenes.append({'id':r['id'],'ruta':r['ruta'],'ancho':im.width,'alto':im.height,'pixeles_sha256':hp,'archivo_sha256':hf,
                'lote':r['grupo_adquisicion'],'grupo':r['grupo'],'licencia':r['licencia'],'url_fuente':r.get('url_fuente'),
                'brillo_medio':float(gris.mean()),'saturacion_media':float(hsv[:,:,1].mean()),
                'varianza_laplaciano_192':float(cv2.Laplacian(gris,cv2.CV_64F).var()),
                'hist_rgb':np.concatenate([np.histogram(small[:,:,c],bins=16,range=(0,256),density=True)[0]*16 for c in range(3)]).tolist()})
            ff,oo,ss,cc=censar_escena(r,cont);filas.extend(ff);solapes.extend(ss);casos.append({'imagen':r['id'],**cc})
            oraculos.append({'imagen':r['id'],'rejillas':oo});fichas(r,ff,cont,tiles)
            if (i+1)%50==0:print(f'Censo {split}: {i+1}/{len(rr)}',flush=True)
        hojas=[]
        for c,tt in tiles.items():
            for k in range(0,len(tt),64):
                actual=tt[k:k+64];sheet=Image.new('RGB',(190*8,205*((len(actual)+7)//8)),'#dddddd')
                for j,(_,tile) in enumerate(actual):sheet.paste(tile,((j%8)*190,(j//8)*205))
                p=path/'contactos'/f'{split}_{c}_{k//64:02d}.jpg';p.parent.mkdir(exist_ok=True);sheet.save(p,quality=88)
                hojas.append({'archivo':p.relative_to(SALIDA).as_posix(),'ids':[x[0] for x in actual],'clase':c})
        publicar(path/f'instancias_{split}.json',filas);publicar(path/f'oraculos_{split}.json',oraculos)
        publicar(path/f'imagenes_{split}.json',imagenes);publicar(path/f'solapes_{split}.json',solapes)
        publicar(path/f'politica_tokens_{split}.json',casos);publicar(path/f'contactos_{split}.json',hojas)
        salidas[split]={'escenas':len(rr),'censo':resumir_censo(filas),'tokens_positivos_sobre_ignore_o_padding':sum(x['tokens_positivos_en_zonas_previamente_ignoradas'] for x in casos),
            'solapes_bbox_iou50':len(solapes),'mismo_material':sum(s['mismo_material'] for s in solapes),
            'integridad_imagenes_verificada':len(imagenes),'fuente':'TACO poligonos originales; ninguna caja RealWaste'}
    publicar(path/'resumen.json',{'version':VERSION,'splits':salidas,'segundos':time.perf_counter()-inicio,
        'sha256_anotaciones':cont.sha256,'sha256_inventario':cont.sha256_inventario,'test_abierto':False,
        'limite_revision':'Geometria/procedencia exhaustivas; etiquetas de material requieren adjudicacion humana. Contactos no certifican correccion semantica.'})


def principales(split,carpeta,rr,cont):
    """Reutiliza logits fullframe 1v; no vuelve a ejecutar forward ni auxiliar."""
    cm=np.zeros((7,7));cc=np.zeros((6,6));ci=np.zeros((7,7),dtype=int);cci=np.zeros((6,6),dtype=int);filas=[]
    for k in range(0,len(rr),8):
        raw=torch.load(carpeta/'logits_1v'/f'{k:03d}.pt',map_location='cpu',weights_only=True)['logits']
        rs=[]
        for r in rr[k:k+8]:
            ss=[{'id':a['id_interno'],'clase':CLASES.index(a['clase']),'poligonos':cont.contornos(r,a)} for a in r['cajas'] if a['clase']]
            rs.append(regiones_poligonales(objetivo_deteccion(r),ss,cont.rasterizar(r)))
        met=metricas_instancias(raw,rs)
        cm+=np.array(met['confusion_token_normalizada_por_instancia']);cc+=np.array(met['confusion_condicional_token_normalizada_por_instancia'])
        ci+=np.array(met['confusion_instancia_voto_principal']);cci+=np.array(met['confusion_instancia_voto_condicional'])
        for f in met['filas']:f['imagen']=rr[k+f['imagen_lote']]['id'];filas.append(f)
    return {'split':split,'instancias_elegibles':len(filas),'orden_siete':['fondo',*CLASES],'orden_materiales':CLASES,
        'confusion_token_normalizada_por_instancia':cm.tolist(),'confusion_condicional_token_normalizada_por_instancia':cc.tolist(),
        'confusion_instancia_voto_principal':ci.tolist(),'confusion_instancia_voto_condicional':cci.tolist(),'filas':filas,
        'por_material':{c:{'elegibles':sum(f['clase']==j for f in filas),
            'recuperadas':sum(f['clase']==j and f['recuperada_mayoria_gt'] for f in filas),
            'condicionales':sum(f['clase']==j and f['condicional_correcta_mayoria'] for f in filas),
            'pred_fondo':int(ci[j+1,0]),'pred_material_erroneo':int(ci[j+1,1:].sum()-ci[j+1,j+1])} for j,c in enumerate(CLASES)},
        'gt_solo_analisis':True,'cabeza_auxiliar_utilizada':False,'vistas':1}


def analisis():
    path=SALIDA/'analisis';path.mkdir(exist_ok=False);cont=ContornosTACO(RAIZ,1/64);res={}
    captura=leer(SALIDA/'captura_train/captura.json')
    for p,h in captura['archivos'].items():
        if sha(SALIDA/p)!=h:raise AssertionError('Captura train alterada')
    fuenteval=RAIZ/'resultados/v24_v20_evaluacion/referencia_historica'
    valcap=leer(fuenteval/'captura.json')
    for f in valcap['archivos']:
        if sha(RAIZ/f['archivo'])!=f['sha256']:raise AssertionError('Captura val historica alterada')
    for split,carpeta in [('train',SALIDA/'captura_train'),('val',fuenteval)]:
        rr=registros(split);objs=[objetivo_deteccion(r) for r in rr];metricas={}
        for v in [1,5]:
            ps=leer(carpeta/f'predicciones_{v}v_{split}.json')
            if [p['id'] for p in ps]!=[r['id'] for r in rr]:raise AssertionError('Orden de imagenes diferente')
            pp=[{'boxes':torch.tensor(p['boxes']).reshape(-1,4),'scores':torch.tensor(p['scores']),'labels':torch.tensor(p['labels'],dtype=torch.long)} for p in ps]
            met=evaluar_detecciones(pp,objs);metricas[str(v)]={'resumen':resumir_deteccion(met),'completas':met}
            if split=='val':
                nombre='detector_mapas_v6_mosaico_predicciones_val.json' if v==5 else 'detector_mapas_v6_auxiliar_ext48_predicciones_val.json'
                if ps!=leer(RAIZ/'resultados'/nombre):raise AssertionError('Val no reproduce historico exacto')
        if split=='train':
            filas=[]
            for i,r in enumerate(rr):
                t=cargar_traza(carpeta/'trazas_5v'/f'{i:03d}.pt');ff,_=analizar_imagen(t,r,cont);filas.extend(ff)
                if (i+1)%60==0:print(f'Analisis posterior train: {i+1}/{len(rr)}',flush=True)
        else:
            filas=leer(RAIZ/'resultados/auditoria_propuestas_internas_v1/objetos_gt.json')
            # V19 observo exactamente las mismas predicciones. Evitar repetir su auditoria.
            if leer(RAIZ/'resultados/auditoria_propuestas_internas_v1/predicciones_val.json')!=leer(carpeta/'predicciones_5v_val.json'):raise AssertionError('Trazas val de otra inferencia')
        principal=principales(split,carpeta,rr,cont)
        publicar(path/f'objetos_{split}.json',filas);publicar(path/f'principal_{split}.json',principal)
        res[split]={'metricas':metricas,'propuestas':resumir_objetos(filas),'principal':{k:v for k,v in principal.items() if k!='filas'}}
        publicar(path/f'resumen_{split}.json',res[split])
    publicar(path/'resumen.json',{'splits':res,'val_reutilizada_sin_nueva_inferencia':True,
        'captura_val':fuenteval.relative_to(RAIZ).as_posix(),'sha256_captura_val':sha(fuenteval/'captura.json'),
        'test_evaluado':False,'entrenamiento':False,'checkpoint':'original_epoca_19_backend_historico'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fase',choices=['censo','analisis'],required=True);a=p.parse_args()
    torch.set_num_threads(4)
    if a.fase=='censo':censo()
    else:analisis()


if __name__=='__main__':main()
