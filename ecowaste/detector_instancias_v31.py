"""Control de extractor V31: misma cabeza propia uno-a-uno, sin pesos externos."""
import torch
from torch import nn
from ecowaste.detector_hibrido_instancias import DetectorHibridoInstancias
from ecowaste.datasets import CLASES

TIPO='ecowaste_instancias_v31_desde_cero_1'


class DetectorInstanciasV31(DetectorHibridoInstancias):
    def __init__(self,cnn=True,consultas=100,capas=2,dimension=96):
        super().__init__(consultas=consultas,capas=capas,dimension=dimension)
        if not cnn:
            self.backbone=None
            self.tokenizador_detalle=nn.Linear(3*8*8,dimension)
            self.tokenizador_contexto=nn.Linear(3*16*16,dimension)
        self.configuracion_hibrido['cnn']=cnn
        self.configuracion_v31=dict(cnn=cnn,consultas=consultas,capas=capas,dimension=dimension)


def pareja(semilla):
    torch.manual_seed(semilla);a=DetectorInstanciasV31(cnn=False)
    torch.manual_seed(semilla);b=DetectorInstanciasV31(cnn=True)
    s=b.state_dict()
    for n,t in a.state_dict().items():
        if n in s and s[n].shape==t.shape:s[n].copy_(t)
    b.load_state_dict(s)
    return a,b


def cargar(path,device='cpu'):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('tipo')!=TIPO or ck.get('clases')!=CLASES or ck.get('pesos_externos') is not False:
        raise ValueError('Checkpoint V31 incompatible')
    m=DetectorInstanciasV31(**ck['arquitectura']).to(device)
    m.load_state_dict(ck['pesos'],strict=True)
    return m.eval(),ck
