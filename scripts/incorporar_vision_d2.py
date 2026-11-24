"""Copia verificable y extracción mínima; nunca importa ni escribe en Vision-IA."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
RAIZ=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guardar(p,b):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_bytes()!=b:raise ValueError('Destino existente diferente: '+str(p))
    else:
        with p.open('xb') as f:f.write(b)
def main():
    p=argparse.ArgumentParser();p.add_argument('--origen',type=Path,required=True);a=p.parse_args();src=a.origen.resolve();out=RAIZ/'resultados/integracion_d2';dest=RAIZ/'ecowaste/vision_equipo'
    if not (src/'training/detector_own.py').is_file():raise ValueError('Raíz Vision-IA incorrecta')
    files=['training/detector_own.py','training/detection_data.py','training/detection_metrics.py','app/detector.py','app/localization.py','app/vision.py','app/stability.py','training/model.py','training/compact.py','training/runtime.py','training/data.py','training/prediction.py','utils/config.py','config/app.json','models/runtime/manifest.json']
    manifest=json.loads((src/'models/runtime/manifest.json').read_text(encoding='utf8'));weights=['detector_D2.pth','classifier_materials.pth','classifier_scene.pth','classifier_B.pth']
    for n in weights:
        m=manifest[n]
        if m.get('trained_in_project') is not True or m.get('foreign_pretrained_weights') is not False or sha(src/'models/runtime'/n)!=m['sha256']:raise ValueError('Procedencia/hash no válido: '+n)
    initial=out/'estado_inicial.json'
    if not initial.exists():
        protected={}
        for name in ['checkpoints','configuracion','documentacion','ecowaste','scripts','pruebas','datos/procesados','datos/nuevos/v27_2/proyecto/seleccion_72']:
            for q in (RAIZ/name).rglob('*'):
                if q.is_file() and '__pycache__' not in q.parts and 'vision_equipo' not in q.parts:protected[q.relative_to(RAIZ).as_posix()]=sha(q)
        for q in (RAIZ/'resultados').glob('*/integridad*.json'):protected[q.relative_to(RAIZ).as_posix()]=sha(q)
        state={'protegidos':protected,'git_status_previo':'V31 local sin commit; conservar resultados/prueba_camara_01','fuente':str(src),'origen_archivos':{n:sha(src/n) for n in files+['models/runtime/'+n for n in weights]}}
        guardar(initial,json.dumps(state,ensure_ascii=False,indent=2).encode())
    for n in files:guardar(out/'fuentes'/n,(src/n).read_bytes())
    for n in weights:guardar(RAIZ/'checkpoints/vision_equipo'/n,(src/'models/runtime'/n).read_bytes())
    guardar(RAIZ/'checkpoints/vision_equipo/manifest.json',(src/'models/runtime/manifest.json').read_bytes())
    def extraer(file,names,imports):
        s=(src/file).read_text(encoding='utf8');tree=ast.parse(s);lines=s.splitlines();found={n.name:'\n'.join(lines[min([n.lineno]+[d.lineno for d in n.decorator_list])-1:n.end_lineno]) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        return '# Aporte del equipo Vision-IA; extracción por scripts/incorporar_vision_d2.py.\n'+imports+'\n\n'+'\n\n'.join(found[n] for n in names)+'\n'
    guardar(dest/'__init__.py',b'"""Modelos propios aportados por el equipo Vision-IA, sin pesos externos."""\n')
    guardar(dest/'detector.py',extraer('training/detector_own.py',['conv','LocalizationHead','VisionLocaliza','decode_head','decode_predictions'],"import math\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\nARCHITECTURE='VisionLocaliza-D2'\nSTRIDES=(8,16,32)").encode())
    guardar(dest/'metricas.py',extraer('training/detection_metrics.py',['box_iou','nms','fuse_detection_views'],'import numpy as np').encode())
    guardar(dest/'modelo.py',(src/'training/model.py').read_bytes())
    guardar(dest/'compacto.py',extraer('training/compact.py',['ResidualBlock','CompactCNN','CompactTransform'],'import random\nimport numpy as np\nimport torch\nfrom torch import nn\nfrom PIL import Image,ImageEnhance,ImageFilter,ImageOps\nARCHITECTURE="VisionResidual-v2"').encode())
    pre=extraer('training/detection_data.py',['letterbox','image_tensor'],'import random\nimport cv2\nimport numpy as np\nimport torch\nfrom dataclasses import dataclass\nfrom PIL import Image,ImageEnhance,ImageOps')
    pre+='\n'+extraer('app/localization.py',['Region','square_box'],'')+'\n'+extraer('app/detector.py',['crop_box'],'')+'\n'+extraer('training/data.py',['ImageTransform'],'')
    guardar(dest/'preprocesado.py',pre.encode())
    neural=extraer('app/detector.py',['Detection','NeuralDetector'],'from dataclasses import dataclass\nfrom pathlib import Path\nimport threading,time\nimport cv2,numpy as np,torch\nfrom .preprocesado import Region,crop_box,letterbox,image_tensor\nfrom .detector import ARCHITECTURE,VisionLocaliza,decode_predictions\nfrom .metricas import nms,fuse_detection_views')
    neural=neural.replace('from utils.dataset import sha256','from ecowaste.incorporacion import sha as sha256').replace('nms_iou=.45,views=()):',"nms_iou=.45,views=(),device='cpu'):").replace('self.model.eval().float()',"self.device=torch.device(device)\n        self.model.eval().float().to(self.device)").replace('tensor=image_tensor(canvas).unsqueeze(0)','tensor=image_tensor(canvas).unsqueeze(0).to(self.device)')
    guardar(dest/'localizador.py',neural.encode())
    record={'equipo':'Código y pesos Vision-IA aportados por un integrante del mismo equipo académico; sin modelos externos','archivos_originales':{n:sha(src/n) for n in files},'pesos':{n:sha(RAIZ/'checkpoints/vision_equipo'/n) for n in weights},'adaptaciones':['namespace ecowaste.vision_equipo','extracción solo inferencia; sin entrenamiento ni selector de objeto principal','import sha local','opción device explícita; float32; CPU predeterminada fiel al original'],'archivos_extraidos':{q.relative_to(RAIZ).as_posix():sha(q) for q in dest.glob('*.py')}}
    guardar(out/'incorporacion_final.json',json.dumps(record,ensure_ascii=False,indent=2).encode());print('D2 y cuatro checkpoints del equipo incorporados y verificados',flush=True)
if __name__=='__main__':main()
