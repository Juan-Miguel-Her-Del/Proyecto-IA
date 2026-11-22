"""Evidencia adicional de licencias/revisiones; solo documentos públicos, sin fotos."""
import json
from pathlib import Path
import re
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from scripts.investigar_v27 import DATOS,SALIDA,obtener
from ecowaste.incorporacion import publicar,publicar_bytes


def main():
    h=json.loads((SALIDA/'piloto_hpi.json').read_bytes());url=h['imagenes'][0]['descarga']['url_solicitada']
    revision=re.search(r'/--/([0-9a-f]{40})/--/',url)[1]
    actuales=json.loads((DATOS/'fuentes/hpi_hf_api.bin').read_bytes())['sha']
    urls={
        'hpi_cache_readme':f'https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset/resolve/{revision}/README.md',
        'hpi_cache_tree':f'https://huggingface.co/api/datasets/AI-ServicesBB/food-waste-dataset/tree/{revision}/data',
        'hpi_actual_tree':f'https://huggingface.co/api/datasets/AI-ServicesBB/food-waste-dataset/tree/{actuales}/data',
        'warp_kaggle_metadata':'https://www.kaggle.com/api/v1/datasets/view/parohod/warp-waste-recycling-plant-dataset',
        'cc0':'https://creativecommons.org/publicdomain/zero/1.0/deed.es',
        'trashcan_autores':'https://irvlab.cs.umn.edu/resources/trashcan',
        'trashcan_condiciones_lab':'https://irvlab.cs.umn.edu/resources',
        'uc_repo_metadata':'https://api.github.com/repos/CollaborativeRoboticsLab/UC-Food-Waste',
        'rf_cep_card':'https://universe.roboflow.com/cep-owycw/garbage-classification-3-i5yna',
        'rf_warp_card':'https://universe.roboflow.com/trash-recycling-data/warp-d-jsdws',
        'rf_wast_card':'https://universe.roboflow.com/myfast/wast-6srsc',
        'spectral_categories':'https://huggingface.co/datasets/ferpb/spectralwaste-segmentation/resolve/65542cc35c28c2f4157c1a6f5afc95030d2ab502/categories.json',
        'voxel51_card':'https://huggingface.co/datasets/Voxel51/food-waste-dataset/resolve/main/README.md'
    }
    recs={}
    for nombre,url in urls.items():
        try:
            data,rec=obtener(url);path=DATOS/'fuentes_adicionales'/f'{nombre}.bin';publicar_bytes(path,data)
            rec['archivo_local']=path.relative_to(RAIZ).as_posix();recs[nombre]=rec
            print(nombre,len(data),flush=True)
        except Exception as e:recs[nombre]={'url_solicitada':url,'error':str(e)};print(nombre,str(e),flush=True)
    info={'revision_API_HPI':actuales,'revision_assets_HPI':revision,'consultas':recs}
    a=DATOS/'fuentes_adicionales/hpi_actual_tree.bin';b=DATOS/'fuentes_adicionales/hpi_cache_tree.bin'
    if a.exists() and b.exists():
        def train(p):return {x['path']:x.get('lfs',{}).get('oid',x.get('oid')) for x in json.loads(p.read_bytes()) if '/train-' in x['path']}
        info['parquet_train_misma_huella_entre_revisiones']=train(a)==train(b);info['train_actual']=train(a);info['train_cache']=train(b)
    publicar(SALIDA/'fuentes_adicionales.json',info)


if __name__=='__main__':main()
