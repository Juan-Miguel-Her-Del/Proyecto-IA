# Aporte del equipo Vision-IA; extracción por scripts/incorporar_vision_d2.py.
from dataclasses import dataclass
from pathlib import Path
import threading,time
import cv2,numpy as np,torch
from .preprocesado import Region,crop_box,letterbox,image_tensor
from .detector import ARCHITECTURE,VisionLocaliza,decode_predictions
from .metricas import nms,fuse_detection_views

@dataclass(frozen=True)
class Detection:
    box: tuple
    confidence: float
    region: Region
    inference_ms: float = 0.

class NeuralDetector:
    def __init__(self,checkpoint,cpu_threads=2,confidence=.35,nms_iou=.45,views=(),device='cpu'):
        path=Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError('Falta el detector propio D2 entrenado. Puedes usar el modo manual.')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        from ecowaste.incorporacion import sha as sha256
        self.checkpoint_sha256=sha256(path)
        provenance=payload.get('provenance',{})
        if provenance.get('pretrained') is not False or provenance.get('initialization')!='random':
            raise ValueError('Detector sin procedencia de entrenamiento desde cero.')
        if payload.get('architecture')!=ARCHITECTURE or provenance.get('official_detector_components') is not False:
            raise ValueError('Se exige arquitectura propia VisionLocaliza-D2. D1/YOLO no es válido como detector final.')
        torch.set_num_threads(cpu_threads)
        self.cpu_threads=cpu_threads
        self.model=VisionLocaliza()
        self.model.load_state_dict(payload['state_dict'],strict=True)
        self.device=torch.device(device)
        self.model.eval().float().to(self.device)
        self.parameter_count=sum(p.numel() for p in self.model.parameters())
        self.size=int(payload['image_size'])
        self.confidence,self.nms_iou=confidence,nms_iou
        self.views=tuple((size, flip) for size, flip in views)
        if len(self.views) == 1 or any(type(size) is not int or size % 32 or not 128 <= size <= 1024
                                       or type(flip) is not bool for size, flip in self.views):
            raise ValueError('Las vistas deben tener tamaño múltiplo de 32 y un indicador de espejo.')
        self.provenance=provenance
        self._lock=threading.Lock()

    def detect(self,rgb,confidence=None):
        # Streamlit may invoke a cached model from a new script/callback thread.
        threads=getattr(self,'cpu_threads',2)
        if torch.get_num_threads()!=threads:torch.set_num_threads(threads)
        if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
            raise ValueError('Se requiere imagen RGB uint8.')
        started=time.perf_counter()
        height,width=rgb.shape[:2]
        threshold=self.confidence if confidence is None else confidence
        views=getattr(self,'views',())
        if views:
            predictions=fuse_detection_views([self._detect_view(rgb,size,flip,.001) for size,flip in views],
                                             final_nms=self.nms_iou,min_confidence=threshold)
        else:
            predictions=self._detect_view(rgb,self.size,False,threshold)
        elapsed=(time.perf_counter()-started)*1000
        return [Detection(tuple(p['box']),p['score'],
                          Region(crop_box(p['box'],width,height),'neuronal-VisionLocaliza-D2'),elapsed)
                for p in predictions]

    def _detect_view(self,rgb,size,flip,confidence):
        height,width=rgb.shape[:2]
        canvas,_,(scale_x,scale_y,left,top)=letterbox(rgb,size)
        if flip:
            # Mirror the resized image, keeping the letterbox offsets and avoiding a full-frame copy.
            width_scaled=round(width*scale_x)
            picture=canvas[:,left:left+width_scaled]
            picture[:]=picture[:,::-1].copy()
        tensor=image_tensor(canvas).unsqueeze(0).to(self.device)
        with self._lock,torch.inference_mode():
            raw=self.model(tensor)
            decoded=decode_predictions(raw,confidence)[0]
            boxes,scores=decoded[0].cpu().numpy(),decoded[1].cpu().numpy()
        if len(scores)==0:
            return []
        boxes[:,[0,2]]=(boxes[:,[0,2]]-left)/scale_x
        boxes[:,[1,3]]=(boxes[:,[1,3]]-top)/scale_y
        boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,width)
        boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,height)
        valid=np.isfinite(boxes).all(axis=1)&np.isfinite(scores)&((boxes[:,2]-boxes[:,0])>=2)&((boxes[:,3]-boxes[:,1])>=2)
        boxes,scores=boxes[valid],scores[valid]
        keep=nms(boxes,scores,self.nms_iou,max_detections=100)
        predictions=[]
        for i in keep:
            x1,y1,x2,y2=map(float,boxes[i])
            box=(width-x2,y1,width-x1,y2) if flip else (x1,y1,x2,y2)
            predictions.append({'box':box,'score':float(scores[i])})
        return predictions
