# Aporte del equipo Vision-IA; extracción por scripts/incorporar_vision_d2.py.
import random
import cv2
import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image,ImageEnhance,ImageOps

def letterbox(rgb,size,boxes=None,training=False,config=None):
    config=config or {}
    height,width=rgb.shape[:2]
    scale=min(size/width,size/height)
    if training:scale*=random.uniform(*config.get('scale_range',[.65,1.25]))
    nw,nh=max(1,round(width*scale)),max(1,round(height*scale))
    if training:
        left=random.randint(min(0,size-nw),max(0,size-nw))
        top=random.randint(min(0,size-nh),max(0,size-nh))
    else:left,top=(size-nw)//2,(size-nh)//2
    scaled=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    canvas=np.full((size,size,3),114,dtype=np.uint8)
    x1,y1=max(0,left),max(0,top);x2,y2=min(size,left+nw),min(size,top+nh)
    canvas[y1:y2,x1:x2]=scaled[y1-top:y2-top,x1-left:x2-left]
    transformed=None
    if boxes is not None:
        transformed=np.asarray(boxes,dtype=np.float32).reshape(-1,4).copy()
        transformed[:,[0,2]]=transformed[:,[0,2]]*(nw/width)+left
        transformed[:,[1,3]]=transformed[:,[1,3]]*(nh/height)+top
        transformed=np.clip(transformed,0,size)
        keep=((transformed[:,2]-transformed[:,0])>=2)&((transformed[:,3]-transformed[:,1])>=2)
        if training:transformed=transformed[keep]
        if training and random.random()<config.get('horizontal_flip',.5):
            canvas=canvas[:,::-1].copy()
            transformed[:,[0,2]]=size-transformed[:,[2,0]]
        if training:
            image=Image.fromarray(canvas)
            for key,enhancer in [('brightness',ImageEnhance.Brightness),('contrast',ImageEnhance.Contrast)]:
                amount=config.get(key,.25)
                image=enhancer(image).enhance(random.uniform(1-amount,1+amount))
            canvas=np.asarray(image).copy()
    return canvas,transformed,(nw/width,nh/height,left,top)

def image_tensor(rgb):
    return torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/127.5-1.

# Aporte del equipo Vision-IA; extracción por scripts/incorporar_vision_d2.py.


@dataclass(frozen=True)
class Region:
    box: tuple
    method: str

def square_box(box, width, height):
    """Amplía la caja sin recortar; un ROI cuadrado evita deformar botellas al redimensionar."""
    left,top,right,bottom=box
    side=max(right-left,bottom-top)
    if side>min(width,height):
        return box  # No cabe un cuadrado: conservar todo el objeto tiene prioridad.
    x=min(max(0,round((left+right-side)/2)),width-side)
    y=min(max(0,round((top+bottom-side)/2)),height-side)
    return (x,y,x+side,y+side)

# Aporte del equipo Vision-IA; extracción por scripts/incorporar_vision_d2.py.


def crop_box(box,width,height,margin=.12,*,square=True):
    x1,y1,x2,y2=box
    dx,dy=max((x2-x1)*margin,(16-(x2-x1))/2),max((y2-y1)*margin,(16-(y2-y1))/2)
    expanded=(max(0,int(np.floor(x1-dx))),max(0,int(np.floor(y1-dy))),
              min(width,int(np.ceil(x2+dx))),min(height,int(np.ceil(y2+dy))))
    if expanded[2]-expanded[0]<16 or expanded[3]-expanded[1]<16:
        left=min(max(0,expanded[0]),width-16)
        top=min(max(0,expanded[1]),height-16)
        expanded=(left,top,min(width,max(left+16,expanded[2])),min(height,max(top+16,expanded[3])))
    return square_box(expanded,width,height) if square else expanded

# Aporte del equipo Vision-IA; extracción por scripts/incorporar_vision_d2.py.


class ImageTransform:
    def __init__(self, training=False, normalization=None, augmentation=None):
        self.training = training
        self.normalization = normalization or {"mean": [.5] * 3, "std": [.5] * 3}
        self.augmentation = augmentation or {}
        self.mean = torch.tensor(self.normalization["mean"], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(self.normalization["std"], dtype=torch.float32)[:, None, None]

    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        if self.training:
            options = self.augmentation
            if random.random() < options.get("horizontal_flip_probability", .5):
                image = ImageOps.mirror(image)
            angle = options.get("rotation_degrees", 10)
            image = image.rotate(random.uniform(-angle, angle), resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
            for option, enhancer in (("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast)):
                magnitude = options.get(option, .1)
                image = enhancer(image).enhance(random.uniform(1 - magnitude, 1 + magnitude))
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
        return (torch.from_numpy(pixels) - self.mean) / self.std
