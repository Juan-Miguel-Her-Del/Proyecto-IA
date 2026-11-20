"""Audita contornos train/val y el límite de reconstruir cajas desde mapas ideales."""
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.datasets import CLASES, objetivo_deteccion, letterbox
from ecowaste.poligonos import ContornosTACO, rasterizar_contornos
from ecowaste.detector_mapas import cajas_desde_mapas
from ecowaste.evaluacion_detector import evaluar_detecciones


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cobertura-minima',type=float,default=.25)
    parser.add_argument('--nombre',default='poligonos')
    args=parser.parse_args()
    if Path(args.nombre).name!=args.nombre:raise ValueError('Nombre inválido')
    torch.set_num_threads(4)
    c=ContornosTACO(RAIZ,args.cobertura_minima)
    informe={'politica':c.politica,'cobertura_minima':args.cobertura_minima,'sha256_anotaciones':c.sha256,
        'sha256_inventario':c.sha256_inventario,'test_evaluado':False,'particiones':{}}
    for split in ['train','val']:
        rows=[json.loads(l) for l in (RAIZ/f'datos/procesados/deteccion/{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        conteos=torch.zeros(7,dtype=torch.long)
        objetos=fuera=fusionados=sin_celda=0
        pred,objetivos=[],[]
        for r in rows:
            o=objetivo_deteccion(r)
            for a in r['cajas']:
                if not a['clase']:continue
                p=c.contornos(r,a);objetos+=1
                fusionados+=c.anotaciones[a['id_interno']-1]['image_id']!=r['id_original']
                fuera+=sum(x<0 or y<0 or x>r['ancho'] or y>r['alto'] for s in p for x,y in zip(s[::2],s[1::2]))
                # Objeto individual: permite contar pequeños sin ocultar su GT.
                propio=rasterizar_contornos(o,[(p,CLASES.index(a['clase']))],cobertura_minima=args.cobertura_minima)
                sin_celda+=not bool(((propio>0)&(propio<7)).any())
            y=c.rasterizar(r)
            conteos+=torch.bincount(y[y!=255],minlength=7)
            if split=='val':
                logits=F.one_hot(y.clamp(max=6),7).permute(2,0,1).float().unsqueeze(0)*40
                logits[:,0][y.unsqueeze(0)==255]=50
                pred.extend(cajas_desde_mapas(logits,area_minima=2,geometrias=[o['geometria']]))
                objetivos.append(o)
        informe['particiones'][split]={'escenas':len(rows),'objetos_validados':objetos,
            'vertices_fuera_de_lienzo_recortados':fuera,'objetos_fusionados_hash_identico':fusionados,
            'objetos_sin_celda_positiva':sin_celda,'celdas_fondo_y_clases':conteos.tolist()}
        if split=='val':informe['oracle_mapas_val']=evaluar_detecciones(pred,objetivos)
    informe['limite_oracle']='Diagnóstico de representación usando GT val, no predicciones de un modelo ni rendimiento independiente. El evaluador conserva todos los objetos.'
    (RAIZ/f'datos/reportes/{args.nombre}.json').write_text(json.dumps(informe,indent=2,ensure_ascii=False),encoding='utf-8')
    # Una evidencia visual fija de entrenamiento, sin elegir una predicción favorable.
    r=json.loads((RAIZ/'datos/procesados/deteccion/train.jsonl').read_text(encoding='utf-8').splitlines()[0])
    with Image.open(RAIZ/r['ruta']) as im:
        x,g=letterbox(im.convert('RGB'),384)
    base=Image.fromarray((x.permute(1,2,0).numpy()*255).astype('uint8'))
    colores=[(130,130,130),(255,80,80),(80,170,255),(200,150,60),(80,230,150),(210,120,255),(255,200,60)]
    capa=Image.new('RGB',(48,48));capa.putdata([colores[v] if v!=255 else (20,20,20) for v in c.rasterizar(r).flatten().tolist()])
    panel=Image.new('RGB',(768,410),'white');panel.paste(base,(0,26))
    panel.paste(Image.blend(base,capa.resize((384,384),Image.Resampling.NEAREST),.45),(384,26))
    ImageDraw.Draw(panel).text((8,5),'Train taco:0 | imagen y contorno original cuantizado',fill='black')
    panel.save(RAIZ/f'datos/reportes/{args.nombre}_train.png')
    print(json.dumps({'particiones':informe['particiones'],'oracle_map50_val':informe['oracle_mapas_val']['map50']}),flush=True)


if __name__=='__main__':main()
