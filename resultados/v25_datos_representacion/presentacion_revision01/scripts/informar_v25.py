"""Tablas, figuras y sintesis V25 desde evidencias offline inmutables."""
import csv
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
from ecowaste.datasets import CLASES,objetivo_deteccion
from ecowaste.auditoria_datos_representacion import mascara_instancia
from ecowaste.poligonos import ContornosTACO
from ecowaste.reanudacion_regional import publicar
from scripts.capturar_v25 import SALIDA,leer
from scripts.auditar_v25 import registros,font

COLORES=['#4b9b70','#a886ce','#b07b3a','#3aa8ba','#718495','#db9146']


def tabla(nombre,filas):
    p=SALIDA/'tablas'/nombre;p.parent.mkdir(exist_ok=True)
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(filas[0]));w.writeheader();w.writerows(filas)


def resumir_oraculos():
    resumen={};filas_csv=[]
    for split in ['train','val']:
        datos=leer(SALIDA/f'censo/oraculos_{split}.json');rr=[]
        for escena in datos:
            for rejilla in escena['rejillas']:
                for f in rejilla['filas']:rr.append({'imagen':escena['imagen'],'lado':rejilla['lado'],**f})
        resumen[split]={}
        for lado in [48,96,384]:
            resumen[split][str(lado)]={}
            for clase in ['total',*CLASES]:
                ff=[f for f in rr if f['lado']==lado and (clase=='total' or CLASES[f['clase']]==clase)]
                x={'instancias':len(ff),'caja_semantica_iou50':sum(f['iou_semantica_max']>=.5 for f in ff),
                    'caja_por_instancia_iou50':sum(f['iou_instancia_separada_max']>=.5 for f in ff),
                    'fusion_mayoritaria':sum(f['fusion_mayoritaria'] for f in ff),'fusion_toca':sum(f['fusion_toca'] for f in ff),
                    'sin_token_correcto':sum(f['tokens_semanticos_correctos']==0 for f in ff),
                    'fragmentacion_componentes':sum(f['componentes_tocados']>1 for f in ff)}
                resumen[split][str(lado)][clase]=x;filas_csv.append({'split':split,'rejilla':lado,'clase':clase,**x})
        por_id={n:{(f['imagen'],f['id']):f for f in rr if f['lado']==n} for n in [48,96,384]}
        trans={}
        for a,b in [(48,96),(48,384)]:
            trans[f'{a}_a_{b}']={'fusiones_que_desaparecen':sum(x['fusion_mayoritaria'] and not por_id[b][k]['fusion_mayoritaria'] for k,x in por_id[a].items()),
                'cajas_semanticas_recuperadas_iou50':sum(x['iou_semantica_max']<.5<=por_id[b][k]['iou_semantica_max'] for k,x in por_id[a].items()),
                'cajas_semanticas_perdidas_iou50':sum(x['iou_semantica_max']>=.5>por_id[b][k]['iou_semantica_max'] for k,x in por_id[a].items())}
        resumen[split]['transiciones']=trans
    tabla('contrafactuales_geometria.csv',filas_csv);publicar(SALIDA/'contrafactuales_resumen.json',resumen)
    return resumen


def comparar_distribuciones():
    datos={s:leer(SALIDA/f'censo/instancias_{s}.json') for s in ['train','val']}
    imagenes={s:leer(SALIDA/f'censo/imagenes_{s}.json') for s in ['train','val']}
    lotes={s:sorted({r['lote'] for r in rr}) for s,rr in imagenes.items()}
    if set(lotes['train'])&set(lotes['val']):raise AssertionError('Particiones no independientes por lote')
    sal={'lotes':lotes,'lotes_compartidos':[],'por_material':{},'limite':'Descriptivo, no prueba causal de domain shift. RGB/brillo global incluye fondo; tamanos por instancia y categorias confunden lotes.'}
    for c in CLASES:
        f={s:[x for x in ff if x['clase']==c] for s,ff in datos.items()}
        categorias={s:sorted({x['categoria'] for x in ff}) for s,ff in f.items()}
        ids={s:{x['imagen'] for x in ff} for s,ff in f.items()}
        estad={s:{k:float(np.median([x[k] for x in rr if x['id'] in ids[s]])) for k in ['brillo_medio','saturacion_media','varianza_laplaciano_192']} for s,rr in imagenes.items()}
        sal['por_material'][c]={'categorias':categorias,'categorias_val_no_train':sorted(set(categorias['val'])-set(categorias['train'])),
            'mediana_lado_menor_384':{s:float(np.median([x['lado_menor_384'] for x in ff])) for s,ff in f.items()},
            'escenas_estadisticas_globales':estad,'escenas':{s:len(ids[s]) for s in ids}}
    cfg=leer(RAIZ/'configuracion/datasets.json');mapping={}
    for s,ff in datos.items():
        malos=[(f['imagen'],f['id']) for f in ff if cfg['taco'].get(f['categoria'])!=f['clase']]
        mapping[s]={'anotaciones_verificadas':len(ff),'incompatibles_con_mapeo_actual':malos}
        if malos:raise AssertionError('Mapeo de clase incompatible')
    sal['correspondencia_mapeo']=mapping;publicar(SALIDA/'distribuciones.json',sal)
    return sal


def organicos():
    cont=ContornosTACO(RAIZ,1/64);resultados=[]
    for r in registros('train'):
        aa=[a for a in r['cajas'] if a['clase']]
        if not any(a['clase']=='organico' for a in aa):continue
        o=objetivo_deteccion(r);g=o['geometria'];ms=[mascara_instancia(cont.contornos(r,a),g) for a in aa]
        cob=[cv2.resize(m.astype(np.float32),(48,48),interpolation=cv2.INTER_AREA) for m in ms]
        for i,a in enumerate(aa):
            if a['clase']!='organico':continue
            distintos=[j for j in range(len(ms)) if aa[j]['clase']!='organico']
            otros=np.maximum.reduce([ms[j] for j in distintos]) if distintos else np.zeros((384,384),dtype=np.uint8)
            otros48=(np.maximum.reduce([cob[j] for j in distintos])>0) if distintos else np.zeros((48,48),dtype=bool)
            propios=cob[i]>0;total=int(propios.sum())
            resultados.append({'imagen':r['id'],'id':a['id_interno'],'tokens_con_poligono':total,
                'tokens_mezcla_material':int((propios&otros48).sum()),
                'fraccion_pixel_poligono_solapada_otro_material':float((ms[i]*otros).sum())/max(1,int(ms[i].sum())),
                'otros_materiales':sorted({x['clase'] for x in aa if x['clase']!='organico'})})
    publicar(SALIDA/'organicos_solapes.json',resultados)


def ejemplos():
    cont=ContornosTACO(RAIZ,1/64);seleccion=[]
    for split in ['train','val']:
        rr={r['id']:r for r in registros(split)};filas=leer(SALIDA/f'analisis/objetos_{split}.json')
        for clase,causa in [('carton','material_incorrecto'),('carton','correcto'),('vidrio','ausencia_respuesta_fg'),
            ('vidrio','material_incorrecto'),('plastico','fusion_objetos'),('plastico','geometria_insuficiente')]:
            ff=[f for f in filas if f['material']==clase and f['causa_principal']==causa]
            if not ff:continue
            f=sorted(ff,key=lambda x:(-x['lado_menor_384'],x['imagen'],x['id_gt']))[0];r=rr[f['imagen']]
            a=next(a for a in r['cajas'] if a['id_interno']==f['id_gt']);predpath=(SALIDA/'captura_train' if split=='train' else RAIZ/'resultados/v24_v20_evaluacion/referencia_historica')
            ps=leer(predpath/f'predicciones_5v_{split}.json');p=next(p for p in ps if p['id']==r['id'])
            with Image.open(RAIZ/r['ruta']) as im:im=im.convert('RGB')
            # Un panel de contexto; GT verde y predicciones >= .05 del entorno en naranja.
            im.thumbnail((620,470));sx,sy=im.width/r['ancho'],im.height/r['alto'];d=ImageDraw.Draw(im)
            for pol in cont.contornos(r,a):
                xy=[(x*sx,y*sy) for x,y in zip(pol[::2],pol[1::2])];d.line(xy+[xy[0]],fill='#00ff44',width=3)
            g=objetivo_deteccion(r)['geometria']
            for box,lab,score in zip(p['boxes'],p['labels'],p['scores']):
                x0,y0,x1,y1=box
                xy=[(x0*384-g['relleno_x'])/g['escala_x']*sx,(y0*384-g['relleno_y'])/g['escala_y']*sy,
                    (x1*384-g['relleno_x'])/g['escala_x']*sx,(y1*384-g['relleno_y'])/g['escala_y']*sy]
                if xy[2]<=xy[0] or xy[3]<=xy[1]:continue
                d.rectangle(xy,outline='#ff9933',width=1)
                if score>=.35:d.text(xy[:2],CLASES[lab],fill='#ffff44',font=font(11))
            tile=Image.new('RGB',(650,525),'white');tile.paste(im,(0,0));d=ImageDraw.Draw(tile)
            d.text((5,475),f"{split}: {r['id']} / GT {a['id_interno']} {clase}",fill='black',font=font(16))
            d.text((5,497),causa,fill='black',font=font(16));pout=SALIDA/'ejemplos'/f'{split}_{clase}_{causa}.jpg';pout.parent.mkdir(exist_ok=True);tile.save(pout,quality=90)
            seleccion.append({'split':split,'imagen':r['id'],'id':a['id_interno'],'clase':clase,'causa':causa,'archivo':pout.relative_to(SALIDA).as_posix()})
    publicar(SALIDA/'ejemplos/indice.json',seleccion)




def main():
    torch.set_num_threads(4)
    ana=leer(SALIDA/'analisis/resumen.json');censo=leer(SALIDA/'censo/resumen.json')
    # Continuar una presentacion incompleta no repite inferencia ni reemplaza evidencia.
    oracle=leer(SALIDA/'contrafactuales_resumen.json') if (SALIDA/'contrafactuales_resumen.json').exists() else resumir_oraculos()
    if not (SALIDA/'distribuciones.json').exists():comparar_distribuciones()
    if not (SALIDA/'organicos_solapes.json').exists():organicos()
    if not (SALIDA/'ejemplos/indice.json').exists():ejemplos()
    met=[];causas=[];datos=[]
    for s in ['train','val']:
        for v in ['1','5']:
            m=ana['splits'][s]['metricas'][v]['resumen'];met.append({'split':s,'vistas':v,**{k:val for k,val in m.items() if k!='ap50_por_material'},**{'ap50_'+c:m['ap50_por_material'][c] for c in CLASES}})
        for c in CLASES:
            p=ana['splits'][s]['propuestas']['por_material'][c]
            causas.append({'split':s,'clase':c,'gt':p['gt'],**p['causas'],'oportunidades_antes':p['oportunidades_por_etapa']['componentes'],'oportunidades_final':p['oportunidades_por_etapa']['final']})
            q=censo['splits'][s]['censo']['por_material'][c]
            datos.append({'split':s,'clase':c,**{k:q[k] for k in ['instancias','escenas','tocan_borde','cruces_estrictos','bbox_poligono_iou_menor_090','fuera_imagen','degenerados','elegibles']}})
    tabla('metricas_train_val.csv',met);tabla('causas_train_val.csv',causas);tabla('censo_materiales.csv',datos)
    print('V25 tablas, ejemplos y figuras guardadas sin nueva inferencia',flush=True)


if __name__=='__main__':main()
