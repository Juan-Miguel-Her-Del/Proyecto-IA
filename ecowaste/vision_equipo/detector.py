# Aporte del equipo Vision-IA; extracción por scripts/incorporar_vision_d2.py.
import math
import torch
from torch import nn
from torch.nn import functional as F
ARCHITECTURE='VisionLocaliza-D2'
STRIDES=(8,16,32)

def conv(incoming, outgoing, kernel=3, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(incoming,outgoing,kernel,stride,
                                  dilation*(kernel//2),dilation=dilation,bias=False),
                         nn.BatchNorm2d(outgoing),nn.ReLU(inplace=True))

class LocalizationHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local=conv(channels,48,kernel=1)
        self.mix=nn.Sequential(conv(64,48),conv(48,32,dilation=2))
        self.presence=nn.Conv2d(32,1,1)
        self.box=nn.Sequential(conv(32,24),nn.Conv2d(24,4,1))
        self.quality=nn.Conv2d(32,1,1)

    def forward(self,features,context):
        local=self.local(features)
        context=context.expand(-1,-1,*local.shape[-2:])
        mixed=self.mix(torch.cat((local,context),dim=1))
        return torch.cat((self.presence(mixed),self.box(mixed),self.quality(mixed)),dim=1)

class VisionLocaliza(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=conv(3,16,kernel=5,stride=2)
        self.detail=nn.Sequential(conv(16,32,stride=2),conv(32,32))
        self.small=nn.Sequential(conv(32,64,stride=2),conv(64,64))
        self.medium=nn.Sequential(conv(64,96,stride=2),conv(96,96))
        self.large=nn.Sequential(conv(96,128,stride=2),conv(128,128))
        self.context=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(128,16,1),nn.ReLU(inplace=True))
        self.heads=nn.ModuleList(LocalizationHead(c) for c in (64,96,128))
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
                if layer.bias is not None:nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.BatchNorm2d):
                nn.init.ones_(layer.weight);nn.init.zeros_(layer.bias)
        for head in self.heads:
            nn.init.constant_(head.presence.bias,math.log(.01/.99))
            nn.init.normal_(head.presence.weight,std=.01)
            nn.init.normal_(head.box[-1].weight,std=.01)
            nn.init.normal_(head.quality.weight,std=.01)

    def forward(self,images):
        detail=self.detail(self.stem(images))
        small=self.small(detail)
        medium=self.medium(small)
        large=self.large(medium)
        context=self.context(large)
        return [head(features,context) for head,features in zip(self.heads,(small,medium,large))]

def decode_head(raw,stride):
    """Cada posición predice centro dentro de su celda y anchura/altura positivas."""
    _,_,height,width=raw.shape
    y,x=torch.meshgrid(torch.arange(height,device=raw.device),torch.arange(width,device=raw.device),indexing='ij')
    grid=torch.stack((x,y)).to(raw.dtype).unsqueeze(0)
    center=(raw[:,1:3].sigmoid()+grid)*stride
    size=F.softplus(raw[:,3:5])*stride*4
    return torch.cat((center-size/2,center+size/2),dim=1)

def decode_predictions(outputs,confidence=.05,max_candidates=1500):
    """Devuelve cajas y puntuaciones por imagen; NMS se aplica fuera de la red."""
    results=[]
    decoded=[decode_head(o.float(),s).permute(0,2,3,1) for o,s in zip(outputs,STRIDES)]
    for index in range(outputs[0].shape[0]):
        all_boxes=[];all_scores=[]
        for raw,boxes in zip(outputs,decoded):
            # Presencia de centro y calidad de localización son objetivos distintos.
            score=raw[index,0].float().sigmoid()*raw[index,5].float().sigmoid()
            mask=score>=confidence
            all_boxes.append(boxes[index][mask]);all_scores.append(score[mask])
        boxes=torch.cat(all_boxes);scores=torch.cat(all_scores)
        if len(scores)>max_candidates:
            scores,chosen=scores.topk(max_candidates);boxes=boxes[chosen]
        results.append((boxes,scores))
    return results
