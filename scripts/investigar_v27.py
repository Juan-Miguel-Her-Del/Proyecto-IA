"""Consulta fuentes públicas y conserva evidencia local; no descarga fotos sin licencia."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import publicar,publicar_bytes,sha

SALIDA=RAIZ/'resultados/v27_adquisicion'
DATOS=RAIZ/'datos/nuevos/v27/externos'
FUENTES={
    'zerowaste_zenodo':'https://zenodo.org/api/records/4899926',
    'spectral_zenodo':'https://zenodo.org/api/records/10880544',
    'spectral_hf_api':'https://huggingface.co/api/datasets/ferpb/spectralwaste-segmentation',
    'spectral_hf_tree':'https://huggingface.co/api/datasets/ferpb/spectralwaste-segmentation/tree/main/data?limit=1000',
    'spectral_hf_readme':'https://huggingface.co/datasets/ferpb/spectralwaste-segmentation/resolve/main/README.md',
    'spectral_hf_license':'https://huggingface.co/datasets/ferpb/spectralwaste-segmentation/resolve/main/LICENSE',
    'foodinsseg_readme':'https://raw.githubusercontent.com/jamesjg/FoodInsSeg/main/README.md',
    'warp_readme':'https://raw.githubusercontent.com/AIRI-Institute/WaRP/main/README.md',
    'hpi_hf_api':'https://huggingface.co/api/datasets/AI-ServicesBB/food-waste-dataset',
    'hpi_readme':'https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset/resolve/main/README.md',
    'hf_foodinsseg_search':'https://huggingface.co/api/datasets?search=FoodInsSeg&limit=30',
    'kaggle_warp_search':'https://www.kaggle.com/api/v1/datasets/list?search=warp%20waste',
    'zerowaste_readme':'https://raw.githubusercontent.com/dbash/zerowaste/main/README.md',
    'uc_food_readme':'https://raw.githubusercontent.com/CollaborativeRoboticsLab/UC-Food-Waste/main/README.md',
    'cc_by':'https://creativecommons.org/licenses/by/4.0/deed.es',
    'cc_by_nc':'https://creativecommons.org/licenses/by-nc/4.0/deed.es',
    'cc_by_nc_sa':'https://creativecommons.org/licenses/by-nc-sa/4.0/deed.es'
}


def obtener(url,limite=16*1024*1024):
    req=urllib.request.Request(url,headers={'User-Agent':'EcoWasteFormer-academic-data-audit/27'})
    with urllib.request.urlopen(req,timeout=45) as r:
        data=r.read(limite+1)
        if len(data)>limite:raise ValueError('Respuesta supera límite de metadatos')
        return data,{'url_solicitada':url,'url_final':r.url,'status':r.status,'content_type':r.headers.get('Content-Type'),
            'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'consultada_en_UTC':datetime.now(timezone.utc).isoformat()}


def preparar():
    if SALIDA.exists():raise FileExistsError('Conservar resultados anteriores')
    SALIDA.mkdir(parents=True)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=RAIZ).decode().split('\0')
    paths=[RAIZ/f for f in tracked if f]
    paths.extend([*list((RAIZ/'datos/procesados').rglob('*.jsonl')),RAIZ/'checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt',
        RAIZ/'datos/originales/taco/annotations.json',RAIZ/'datos/reportes/inventario_imagenes.jsonl'])
    publicar(SALIDA/'historicos_protegidos.json',{p.relative_to(RAIZ).as_posix():sha(p) for p in paths if p.is_file()})
    publicar(SALIDA/'checkpoints_previos.json',{p.relative_to(RAIZ).as_posix():[p.stat().st_size,p.stat().st_mtime_ns] for p in (RAIZ/'checkpoints').rglob('*.pt')})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--preparar',action='store_true');a=ap.parse_args()
    if a.preparar:preparar()
    registros={}
    for nombre,url in FUENTES.items():
        p=DATOS/'fuentes'/f'{nombre}.bin';receipt=SALIDA/'consultas'/f'{nombre}.json'
        if receipt.exists():
            registros[nombre]=json.loads(receipt.read_text(encoding='utf-8'));continue
        try:
            data,rec=obtener(url);publicar_bytes(p,data);rec['archivo_local']=p.relative_to(RAIZ).as_posix();registros[nombre]=rec
            publicar(receipt,rec);print(f'{nombre}: {len(data)} bytes',flush=True)
        except Exception as e:
            rec={'url_solicitada':url,'error':str(e),'consultada_en_UTC':datetime.now(timezone.utc).isoformat()}
            publicar(receipt,rec);registros[nombre]=rec;print(f'{nombre}: {e}',flush=True)
    publicar(SALIDA/'fuentes_consultadas.json',registros)
    for n in ['zerowaste_zenodo','spectral_zenodo']:
        p=DATOS/'fuentes'/f'{n}.bin'
        if p.exists():
            try:
                j=json.loads(p.read_bytes());print(n,json.dumps({'license':j.get('metadata',{}).get('license'),
                    'files':[{k:f.get(k) for k in ['key','size','checksum','links']} for f in j.get('files',[])]}),flush=True)
            except (ValueError,KeyError):pass


if __name__=='__main__':main()
