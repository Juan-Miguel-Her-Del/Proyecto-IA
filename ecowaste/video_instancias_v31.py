"""Motor explícitamente experimental V31; nunca sustituye la selección histórica."""
from pathlib import Path
import cv2
from PIL import Image
import torch
from ecowaste.video import MotorVideo,LIMITE
from ecowaste.detector_instancias_v31 import cargar
from ecowaste.detector_hibrido import mascara_letterbox
from ecowaste.detector_consultas import postprocesar_consultas
from ecowaste.datasets import letterbox,CLASES
from ecowaste.mosaico import ventanas,combinar_vistas
from ecowaste.cajas import deshacer_letterbox
from ecowaste.backend_numerico import aplicar_backend,politica
from ecowaste.incorporacion import sha


class MotorInstanciasV31(MotorVideo):
    def __init__(self,raiz,checkpoint,vistas=1,umbral=.35,dispositivo=None):
        if vistas not in [1,5] or not 0<=umbral<=1:raise ValueError('Vistas/umbral inválidos')
        torch.set_num_threads(4);self.raiz=Path(raiz);self.device=dispositivo or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint=Path(checkpoint);self.modelo,self.ck=cargar(self.checkpoint,self.device);self.modelo.requires_grad_(False)
        self.vistas,self.umbral=vistas,umbral
        self.metadatos={'modelo':'EcoWasteFormer-Instancias-V31','checkpoint':str(self.checkpoint),'sha256_checkpoint':sha(self.checkpoint),
            'paso':self.ck['paso'],'rama':self.ck['rama'],'vistas':vistas,'estado_detector':'experimental_no_adoptado',
            'limite':LIMITE,'seguimiento_de_objetos':False,'umbral_puntuacion':umbral,'umbral_propuestas':.05}

    @torch.inference_mode()
    def detectar(self,frame_bgr):
        im=Image.fromarray(cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB))
        zs=ventanas(im.width,im.height) if self.vistas==5 else [(0,0,im.width,im.height)]
        xgs=[letterbox(im.crop(z),384) for z in zs];gs=[g for x,g in xgs];x=torch.stack([x for x,g in xgs]).to(self.device)
        with aplicar_backend(politica('completo')),torch.autocast(self.device,dtype=torch.float16,enabled=self.device=='cuda'):
            raw=self.modelo(x,mascara_letterbox(gs,self.device))
        ps=postprocesar_consultas(raw,gs,.05,True)
        if self.vistas==5:p=combinar_vistas(ps,gs,zs,im.width,im.height,384,self.umbral)
        else:
            p=ps[0];ok=p['scores']>=self.umbral;p={k:v[ok] for k,v in p.items()}
        _,g=letterbox(im,384);bs=deshacer_letterbox(p['boxes'],g)
        ds=[{'material':CLASES[int(c)],'categoria_operativa':'organico' if int(c)==5 else 'inorganico',
             'puntuacion':float(s),'bbox_xyxy':b.tolist()} for b,s,c in zip(bs,p['scores'],p['labels'])]
        return {'detecciones':ds,'estado_detector':'experimental_no_adoptado','limite':LIMITE,
                'coordenadas':'xyxy_pixeles_frame','ancho':im.width,'alto':im.height}
