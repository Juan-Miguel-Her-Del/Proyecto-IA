"""Complemento de evidencia legal conservada, sin reemplazar la exportación humana."""
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,publicar,publicar_bytes,sha
from scripts.respaldar_anotaciones_v28 import SALIDA


def main():
    rec=leer(SALIDA/'exportacion_auxiliar_auditoria.json');dest=RAIZ/rec['destino']
    fuentes={
      'zerowaste':{'url':'https://github.com/dbash/zerowaste','licencia':'CC BY-NC 4.0 autores; discrepancia CC BY de Zenodo conservada; solo alcance académico NC',
           'ficheros':['datos/nuevos/v27/externos/fuentes/zerowaste_readme.bin','datos/nuevos/v27/externos/fuentes/zerowaste_zenodo.bin']},
      'warp':{'url':'https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset','licencia':'CC0: Public Domain declarada en distribuidor original enlazado',
           'ficheros':['datos/nuevos/v27/externos/fuentes/warp_readme.bin','datos/nuevos/v27/externos/fuentes_adicionales/warp_kaggle_metadata.bin']},
      'hpi':{'url':'https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset','licencia':'CC BY 4.0 fotos y registros; atribuir autores originales',
           'ficheros':['datos/nuevos/v27/externos/fuentes/hpi_readme.bin','datos/nuevos/v27/externos/fuentes_adicionales/hpi_cache_readme.bin']},
      'wsc':{'url':'https://data.mendeley.com/datasets/3wnf3n242s/4','licencia':'CC BY 4.0 Mendeley y README de exportación; preservar cadena Roboflow',
           'ficheros':['datos/nuevos/v27_2/externos/fuentes/wsc_pagina.bin','datos/nuevos/v27_2/externos/wsc/anotaciones_originales/README.dataset.txt',
                      'datos/nuevos/v27_2/externos/wsc/anotaciones_originales/README.roboflow.txt']}}
    archivos={}
    for n,f in fuentes.items():
        for i,ruta in enumerate(f['ficheros']):
            p=RAIZ/ruta;copia=dest/'licencias'/n/(str(i)+'_'+p.name)
            publicar_bytes(copia,p.read_bytes());archivos[copia.relative_to(dest).as_posix()]=sha(copia)
    fuentes['nota']='Los originales se copian con bytes intactos; solo se añaden exportación COCO, metadatos de auditoría y QA. No hay permiso comercial homogéneo para el conjunto con ZeroWaste.'
    publicar(dest/'LICENCIAS_Y_PROCEDENCIA.json',fuentes)
    archivos['LICENCIAS_Y_PROCEDENCIA.json']=sha(dest/'LICENCIAS_Y_PROCEDENCIA.json')
    publicar(dest/'MANIFIESTO_COMPLEMENTO_LICENCIAS.json',{'manifiesto_principal_sha256':sha(dest/'MANIFIESTO.json'),'archivos':archivos,'no_autoriza_entrenamiento':True})
    publicar(SALIDA/'licencias_conservadas.json',{'destino':rec['destino'],'archivos_copiados':len(archivos),
             'sha256_complemento':sha(dest/'MANIFIESTO_COMPLEMENTO_LICENCIAS.json'),'fotografias_redistribuidas_en_Git':False})


if __name__=='__main__':main()
