"""Cuantifica pérdidas de separación con GT train; no evalúa modelos ni toca test."""
import json
from pathlib import Path
import sys

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.datasets import ResiduosDataset,objetivo_deteccion,CLASES
from ecowaste.poligonos import ContornosTACO,rasterizar_contornos
from ecowaste.auditoria_instancias import analizar_componentes
from scripts.entrenar_clasificador import escribir_json


def main():
    salida=RAIZ/'datos/reportes/representacion_instancias_train.json'
    if salida.exists():raise FileExistsError('Auditoría existente')
    torch.set_num_threads(4);ds=ResiduosDataset(RAIZ,'deteccion','train');contornos=ContornosTACO(RAIZ,1/64)
    casos=[];totales={k:0 for k in ['objetos','instancias_en_componentes_compartidos','instancias_fragmentadas','instancias_sin_componente_minimo','fusiones']}
    maximo=0
    for r in ds.registros:
        o=objetivo_deteccion(r);y=contornos.rasterizar(r).numpy();instancias=[]
        for a in r['cajas']:
            if not a['clase']:continue
            clase=CLASES.index(a['clase'])
            propio=rasterizar_contornos(o,[(contornos.contornos(r,a),clase)],cobertura_minima=1/64)
            instancias.append((a['id_interno'],clase,(propio==clase+1).numpy()))
        # Igual exclusión de padding por centro de celda que en la inferencia.
        g=o['geometria'];yy,xx=torch.meshgrid((torch.arange(48)+.5)/48,(torch.arange(48)+.5)/48,indexing='ij')
        x0,y0=g['relleno_x']/384,g['relleno_y']/384
        x1=x0+g['ancho_original']*g['escala_x']/384;y1=y0+g['alto_original']*g['escala_y']/384
        y[((xx<x0)|(xx>x1)|(yy<y0)|(yy>y1)).numpy()]=255
        d=analizar_componentes(y,instancias);maximo=max(maximo,d['objetos'])
        for k in totales:totales[k]+=d[k] if k=='objetos' else len(d[k])
        if any(d[k] for k in d if k!='objetos'):casos.append({'id':r['id'],**d})
    informe={'particion':'train','escenas':len(ds),'totales':totales,'maximo_objetos_por_escena':maximo,
        'politica':contornos.politica,'sha256_anotaciones':contornos.sha256,'casos':casos,
        'test_evaluado':False,'limite':'Diagnóstico geométrico con anotaciones conocidas; no es rendimiento del modelo ni techo absoluto de AP.'}
    escribir_json(salida,informe);print(json.dumps({'totales':totales,'maximo_objetos_por_escena':maximo}),flush=True)


if __name__=='__main__':main()
