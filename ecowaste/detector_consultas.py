"""Detector propio de instancias: consultas espaciales y asignación uno a uno."""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from ecowaste.detector_mapas import DetectorMapas
from ecowaste.transformer import Atencion
from ecowaste.datasets import CLASES
from ecowaste.asignacion_optima import asignacion_optima
from ecowaste.cajas import giou_alineado,cubiertas_por_ignorar

VERSION_CONSULTAS='consultas_espaciales_hardneg_sin_nms_v2'
VERSION_PONDERADA='consultas_centros_clasificacion_por_densidad_v3'


def pesos_materiales(conteos):
    c=torch.as_tensor(conteos,dtype=torch.float32)
    if c.shape!=(6,) or float(c.sum())<=0:raise ValueError('Conteos de materiales inválidos')
    w=c.clamp(min=1).rsqrt();w=w/((w*c).sum()/c.sum())
    w=w.clamp(max=10)
    return w/((w*c).sum()/c.sum())


def auto_atencion_posicional(modulo,x,posicion):
    """Posición en Q/K; V conserva el contenido, con proyecciones propias existentes."""
    b,n,d=x.shape;h=modulo.cabezas
    def proy(t,a,z):return F.linear(t,modulo.qkv.weight[a:z],modulo.qkv.bias[a:z]).reshape(b,n,h,d//h).transpose(1,2)
    q=proy(x+posicion,0,d);k=proy(x+posicion,d,2*d);v=proy(x,2*d,3*d)
    y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
    return modulo.proyeccion(y.transpose(1,2).reshape(b,n,d))


class AtencionCruzada(nn.Module):
    def __init__(self,d,cabezas):
        super().__init__();self.cabezas=cabezas
        self.q=nn.Linear(d,d);self.kv=nn.Linear(d,2*d);self.salida=nn.Linear(d,d)
    def forward(self,x,memoria,sesgo,coordenadas=None,posicion_consultas=None,posicion_memoria=None):
        b,n,d=x.shape;h=self.cabezas
        q=self.q(x if posicion_consultas is None else x+posicion_consultas).reshape(b,n,h,d//h).transpose(1,2)
        if posicion_memoria is None:
            kv=self.kv(memoria).reshape(b,memoria.shape[1],2,h,d//h).permute(2,0,3,1,4);k,v=kv.unbind(0)
        else:
            k=F.linear(memoria+posicion_memoria,self.kv.weight[:d],self.kv.bias[:d]).reshape(b,-1,h,d//h).transpose(1,2)
            v=F.linear(memoria,self.kv.weight[d:],self.kv.bias[d:]).reshape(b,-1,h,d//h).transpose(1,2)
        centro=None
        if coordenadas is None:
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=sesgo[None,None].to(q.dtype),dropout_p=0.)
        else:
            with torch.autocast(device_type=x.device.type,enabled=False):
                pesos=(q.float()@k.float().transpose(-1,-2)/math.sqrt(d//h)+sesgo[None,None]).softmax(-1)
                centro=pesos.mean(1)@coordenadas.float()
            y=pesos.to(v.dtype)@v
        salida=self.salida(y.transpose(1,2).reshape(b,n,d))
        return salida if centro is None else (salida,centro)


class BloqueConsultas(nn.Module):
    def __init__(self,d=96):
        super().__init__();self.normas=nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.propia=Atencion(d,4);self.cruzada=AtencionCruzada(d,4)
        self.mlp=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
    def forward(self,x,memoria,sesgo,coordenadas=None,posicion_consultas=None,posicion_memoria=None):
        propia=self.propia(self.normas[0](x)) if posicion_consultas is None else auto_atencion_posicional(self.propia,self.normas[0](x),posicion_consultas)
        x=x+propia
        cruzada=self.cruzada(self.normas[1](x),memoria,sesgo,coordenadas,posicion_consultas,posicion_memoria)
        centro=None
        if coordenadas is not None:cruzada,centro=cruzada
        x=x+cruzada;x=x+self.mlp(self.normas[2](x))
        return x if centro is None else (x,centro)


def xyxy_a_cxcywh(b):return torch.cat([(b[...,:2]+b[...,2:])/2,b[...,2:]-b[...,:2]],-1)
def cxcywh_a_xyxy(b):return torch.cat([b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2],-1)


class DetectorConsultas(DetectorMapas):
    def __init__(self,consultas=100,capas=2,dimension=96,radio_atencion=.15,centros_atencion=False,posicion_por_capa=False):
        if int(math.isqrt(consultas))**2!=consultas:raise ValueError('Las consultas deben formar una cuadrícula')
        if not math.isfinite(radio_atencion) or radio_atencion<=0 or capas<1:raise ValueError('Configuración de atención inválida')
        super().__init__(dimension=dimension,auxiliar_clasificacion=True)
        self.cabezal=nn.Identity()
        self.configuracion_consultas=dict(consultas=consultas,capas=capas,dimension=dimension,radio_atencion=radio_atencion,centros_atencion=centros_atencion,posicion_por_capa=posicion_por_capa)
        self.consultas=nn.Parameter(torch.randn(consultas,dimension)*.02)
        self.decodificador=nn.ModuleList([BloqueConsultas(dimension) for _ in range(capas)])
        self.norma_memoria=nn.LayerNorm(dimension);self.norma_salida=nn.LayerNorm(dimension)
        self.material=nn.Linear(dimension,7)
        self.caja=nn.Sequential(nn.Linear(dimension,dimension),nn.GELU(),nn.Linear(dimension,4))
        lado=math.isqrt(consultas);y,x=torch.meshgrid((torch.arange(lado)+.5)/lado,(torch.arange(lado)+.5)/lado,indexing='ij')
        referencia=torch.stack([x,y],-1).reshape(-1,2)
        yy,xx=torch.meshgrid((torch.arange(48)+.5)/48,(torch.arange(48)+.5)/48,indexing='ij')
        posiciones=torch.stack([xx,yy],-1).reshape(-1,2)
        self.register_buffer('coordenadas_memoria',posiciones,persistent=False)
        sesgo=-(referencia[:,None]-posiciones[None]).square().sum(-1)/(2*radio_atencion**2)
        self.register_buffer('referencia',referencia);self.register_buffer('sesgo_espacial',sesgo)
        self.register_buffer('prior_wh',torch.tensor([.1,.1]))
        with torch.no_grad():
            self.material.bias.zero_();self.material.bias[6]=math.log(54)
            self.caja[-1].weight.zero_();self.caja[-1].bias.zero_()
    @torch.no_grad()
    def configurar_priores(self,wh):
        wh=torch.as_tensor(wh,device=self.prior_wh.device)
        if wh.shape!=(2,) or not bool(((wh>0)&(wh<1)).all()):raise ValueError('Priores inválidos')
        self.prior_wh.copy_(wh)
    def forward(self,imagenes):
        memoria=self.norma_memoria(self.caracteristicas(imagenes).flatten(1,2))
        x=self.consultas[None].expand(len(imagenes),-1,-1)
        posicion_consultas=posicion_memoria=None
        if self.configuracion_consultas['posicion_por_capa']:
            posicion_consultas=F.layer_norm(self.consultas,(self.consultas.shape[-1],))[None]
            posicion_memoria=self.posicion_detalle*self.escala_posicion
            x=torch.zeros_like(x)
        centro_base=self.referencia[None]
        for i,bloque in enumerate(self.decodificador):
            if self.configuracion_consultas['centros_atencion'] and i==len(self.decodificador)-1:
                x,centro_base=bloque(x,memoria,self.sesgo_espacial,self.coordenadas_memoria,posicion_consultas,posicion_memoria)
            else:x=bloque(x,memoria,self.sesgo_espacial,posicion_consultas=posicion_consultas,posicion_memoria=posicion_memoria)
        x=self.norma_salida(x);raw=self.caja(x)
        centro=(raw[...,:2]+torch.logit(centro_base.clamp(.0001,.9999))).sigmoid()
        wh=(raw[...,2:]+torch.logit(self.prior_wh)[None,None]).sigmoid()
        return {'logits':self.material(x),'boxes':cxcywh_a_xyxy(torch.cat([centro,wh],-1))}


def perdida_consultas(salida,objetivos,objetivo='hardneg',pesos_clase=None):
    if objetivo not in ['hardneg','densidad_ponderada']:raise ValueError('Objetivo desconocido')
    logits=salida['logits'].float();boxes=salida['boxes'].float()
    cero=logits.sum()*0+boxes.sum()*0
    positivos=[];negativos=[];l1=[];giou=[];asignadas=0
    for i,o in enumerate(objetivos):
        gt=o['boxes'].to(boxes.device);labels=o['labels'].to(boxes.device)
        libres=torch.ones(len(boxes[i]),dtype=torch.bool,device=boxes.device)
        if len(gt):
            # El matching solo decide parejas; los términos emparejados conservan gradiente.
            with torch.no_grad():
                b=boxes[i].detach();g=gt.detach()
                coste_clase=-logits[i].softmax(-1)[:,labels].T
                coste_l1=torch.cdist(xyxy_a_cxcywh(g),xyxy_a_cxcywh(b),p=1)
                coste_giou=giou_alineado(g[:,None].expand(-1,len(b),-1).reshape(-1,4),b[None].expand(len(g),-1,-1).reshape(-1,4)).reshape(len(g),len(b))
                indices=asignacion_optima((coste_clase+5*coste_l1-2*coste_giou).cpu().numpy())
            q=torch.as_tensor(indices,device=boxes.device);libres[q]=False;asignadas+=len(q)
            ce_pos=F.cross_entropy(logits[i,q],labels,reduction='none')
            if objetivo=='densidad_ponderada' and pesos_clase is not None:ce_pos=ce_pos*pesos_clase.to(logits.device)[labels]
            positivos.append(ce_pos)
            l1.append(F.l1_loss(xyxy_a_cxcywh(boxes[i,q]),xyxy_a_cxcywh(gt),reduction='none').sum(-1))
            giou.append(1-giou_alineado(boxes[i,q],gt))
        regiones=o['ignore_boxes'].to(boxes.device)
        ignoradas=cubiertas_por_ignorar(boxes[i].detach(),regiones)
        libres&=~ignoradas
        if bool(libres.any()):
            ce_neg=F.cross_entropy(logits[i,libres],torch.full((int(libres.sum()),),6,device=boxes.device),reduction='none')
            negativos.append(ce_neg if objetivo=='densidad_ponderada' else ce_neg.topk(min(len(ce_neg),max(16,5*len(gt)))).values)
    media=lambda partes:torch.cat(partes).mean() if partes else cero
    if objetivo=='densidad_ponderada':
        loss_clase=(torch.cat(positivos).sum() if positivos else cero)/max(asignadas,1)
        loss_fondo=.1*(torch.cat(negativos).sum() if negativos else cero)/max(asignadas,1)
    else:loss_clase=media(positivos);loss_fondo=.5*media(negativos)
    total=loss_clase+loss_fondo+5*media(l1)+2*media(giou)
    return total,{'objetos_asignados':asignadas,'loss_clase':float(loss_clase.detach()),
        'loss_fondo':float(loss_fondo.detach()),'loss_l1':float(media(l1).detach()),'loss_giou':float(media(giou).detach())}


@torch.inference_mode()
def postprocesar_consultas(salida,geometrias,umbral=.05,descartar_nula=True):
    prob=salida['logits'].float().softmax(-1).cpu();boxes=salida['boxes'].float().cpu();resultados=[]
    for i,g in enumerate(geometrias):
        scores,labels=prob[i,:,:6].max(-1)
        b=boxes[i].clone();centros=(b[:,:2]+b[:,2:])/2
        x0,y0=g['relleno_x']/g['lado'],g['relleno_y']/g['lado']
        x1=x0+g['ancho_original']*g['escala_x']/g['lado'];y1=y0+g['alto_original']*g['escala_y']/g['lado']
        keep=(scores>=umbral)&(centros[:,0]>=x0)&(centros[:,0]<=x1)&(centros[:,1]>=y0)&(centros[:,1]<=y1)
        if descartar_nula:keep&=prob[i].argmax(-1)!=6
        b[:,[0,2]]=b[:,[0,2]].clamp(x0,x1);b[:,[1,3]]=b[:,[1,3]].clamp(y0,y1)
        keep&=(b[:,2]>b[:,0])&(b[:,3]>b[:,1])
        b,scores,labels=b[keep],scores[keep],labels[keep]
        # La asignación uno a uno enseña a rechazar duplicados; no se fusionan
        # instancias distintas de igual clase mediante NMS al inferir.
        indices=torch.argsort(scores,descending=True)[:100]
        resultados.append({'boxes':b[indices],'scores':scores[indices],'labels':labels[indices]})
    return resultados


def cargar_detector_consultas(path,dispositivo='cpu'):
    datos=torch.load(Path(path),map_location=dispositivo,weights_only=True)
    if (datos.get('tipo')!='ecowaste_consultas_desde_cero_v1' or datos.get('clases')!=CLASES
            or datos.get('version') not in [VERSION_CONSULTAS,VERSION_PONDERADA]):raise ValueError('Checkpoint de consultas incompatible')
    m=DetectorConsultas(**datos['configuracion']).to(dispositivo);m.load_state_dict(datos['pesos'])
    return m.eval(),datos
