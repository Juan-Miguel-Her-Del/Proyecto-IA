"""Transformer de mapas de materiales; cajas obtenidas por componentes conexos."""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ecowaste.detector import DetectorResiduos
from ecowaste.datasets import CLASES
from ecowaste.cajas import nms_por_clase


class DetectorMapas(DetectorResiduos):
    def __init__(self, imagen=384, dimension=96, escala_posicion=.1, formulacion='softmax7',auxiliar_clasificacion=False,ventanas_desplazadas=False):
        if formulacion not in ['softmax7','presencia_y_material']:raise ValueError('Formulación desconocida')
        super().__init__(imagen=imagen, dimension=dimension, mlp=dimension * 2, escala_posicion=escala_posicion,ventanas_desplazadas=ventanas_desplazadas)
        self.configuracion_mapas = dict(imagen=imagen, dimension=dimension, escala_posicion=escala_posicion,formulacion=formulacion,auxiliar_clasificacion=auxiliar_clasificacion,ventanas_desplazadas=ventanas_desplazadas)
        self.cabezal = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, 7))
        with torch.no_grad():
            self.cabezal[-1].bias.zero_()
            self.cabezal[-1].bias[0] = float(np.log(54)) * (1 if formulacion=='softmax7' else -1)
        self.clasificador_auxiliar=nn.Linear(dimension,6) if auxiliar_clasificacion else None

    def forward(self, imagenes):
        return self.cabezal(self.caracteristicas(imagenes)).permute(0, 3, 1, 2)

    def clasificar(self,imagenes):
        if self.clasificador_auxiliar is None:raise ValueError('El modelo no tiene tarea auxiliar')
        return self.clasificador_auxiliar(self.caracteristicas(imagenes).mean((1,2)))


def rasterizar(objetivo, lado=48):
    """Etiquetas débiles por cobertura del bbox; solapes entre materiales se ignoran."""
    resultado = torch.zeros(lado, lado, dtype=torch.long)
    g = objetivo.get("geometria")
    if g:
        y, x = torch.meshgrid((torch.arange(lado) + .5) / lado, (torch.arange(lado) + .5) / lado, indexing="ij")
        x0, y0 = g["relleno_x"] / g["lado"], g["relleno_y"] / g["lado"]
        x1 = x0 + g["ancho_original"] * g["escala_x"] / g["lado"]
        y1 = y0 + g["alto_original"] * g["escala_y"] / g["lado"]
        resultado[(x < x0) | (x > x1) | (y < y0) | (y > y1)] = 255
    def region(box):
        x0, y0 = torch.floor(box[:2] * lado).long().clamp(0, lado - 1).tolist()
        x1, y1 = torch.ceil(box[2:] * lado).long().clamp(1, lado).tolist()
        mascara = torch.zeros(lado, lado, dtype=torch.bool)
        mascara[y0:y1, x0:x1] = True
        return mascara
    for box in objetivo["ignore_boxes"]:
        resultado[region(box)] = 255
    coberturas = torch.zeros(6, lado, lado, dtype=torch.bool)
    for box, etiqueta in zip(objetivo["boxes"], objetivo["labels"]):
        coberturas[int(etiqueta)] |= region(box)
    total = coberturas.sum(0)
    for c in range(6):
        resultado[coberturas[c] & (total == 1)] = c + 1
    resultado[total > 1] = 255
    return resultado


def perdida_mapas(logits, etiquetas, pesos_clase=None, formulacion='softmax7'):
    if formulacion=='presencia_y_material':return perdida_presencia_material(logits,etiquetas,pesos_clase)
    if formulacion!='softmax7':raise ValueError('Formulación desconocida')
    raw = logits.float()
    valida = etiquetas != 255
    fg, bg = (etiquetas > 0) & valida, (etiquetas == 0) & valida
    ce = F.cross_entropy(raw, etiquetas, ignore_index=255, reduction="none")
    cero = raw.sum() * 0
    if bool(fg.any()):
        pesos = torch.ones_like(ce[fg]) if pesos_clase is None else pesos_clase[etiquetas[fg] - 1]
        loss_fg = (ce[fg] * pesos).sum() / pesos.sum()
    else:
        loss_fg = cero
    loss_bg = ce[bg].mean() if bool(bg.any()) else cero
    probs = raw.softmax(1)[:, 1:]
    gt = F.one_hot(etiquetas.clamp(max=6), 7)[..., 1:].permute(0, 3, 1, 2).float() * valida[:, None]
    probs = probs * valida[:, None]
    presentes = gt.sum((0, 2, 3)) > 0
    dice = 1 - (2 * (probs * gt).sum((0, 2, 3)) + 1e-6) / (probs.square().sum((0, 2, 3)) + gt.square().sum((0, 2, 3)) + 1e-6)
    loss_dice = dice[presentes].mean() if bool(presentes.any()) else cero
    return loss_fg + 3 * loss_bg + loss_dice


def perdida_presencia_material(logits,etiquetas,pesos_clase=None):
    """BCE balanceada por regiones y CE de material condicionada a presencia GT."""
    raw=logits.float();valida=etiquetas!=255
    fg=(etiquetas>0)&valida;bg=(etiquetas==0)&valida
    cero=raw.sum()*0
    bce=F.binary_cross_entropy_with_logits(raw[:,0],fg.float(),reduction='none')
    loss_pos=bce[fg].mean() if bool(fg.any()) else cero
    loss_neg=bce[bg].mean() if bool(bg.any()) else cero
    if bool(fg.any()):
        materiales=raw[:,1:].permute(0,2,3,1)[fg]
        loss_mat=F.cross_entropy(materiales,etiquetas[fg]-1,weight=pesos_clase)
        p=raw[:,0].sigmoid()*valida
        dice=1-(2*(p*fg).sum()+1e-6)/(p.square().sum()+fg.sum()+1e-6)
    else:loss_mat=dice=cero
    return loss_pos+loss_neg+loss_mat+dice


def etiquetas_mapas(logits,formulacion='softmax7',umbral_presencia=.5):
    if formulacion=='softmax7':return logits.argmax(1)
    if formulacion!='presencia_y_material':raise ValueError('Formulación desconocida')
    materiales=logits[:,1:].argmax(1)+1
    return torch.where(logits[:,0].float().sigmoid()>=umbral_presencia,materiales,0)


@torch.inference_mode()
def cajas_desde_mapas(logits, umbral=.35, area_minima=2, geometrias=None,formulacion='softmax7',umbral_presencia=.5):
    raw=logits.float()
    if formulacion=='presencia_y_material':
        presencia=raw[:,0:1].sigmoid()
        probs=torch.cat([1-presencia,presencia*raw[:,1:].softmax(1)],1).cpu()
    elif formulacion=='softmax7':probs=raw.softmax(1).cpu()
    else:raise ValueError('Formulación desconocida')
    etiquetas = etiquetas_mapas(raw,formulacion,umbral_presencia).cpu()
    resultados = []
    for b in range(len(probs)):
        boxes, scores, labels = [], [], []
        h, w = etiquetas.shape[-2:]
        valida = torch.ones(h,w,dtype=torch.bool)
        if geometrias:
            g = geometrias[b]
            minimo = [g["relleno_x"] / g["lado"], g["relleno_y"] / g["lado"]]
            maximo = [minimo[0] + g["ancho_original"] * g["escala_x"] / g["lado"], minimo[1] + g["alto_original"] * g["escala_y"] / g["lado"]]
            yy,xx=torch.meshgrid((torch.arange(h)+.5)/h,(torch.arange(w)+.5)/w,indexing='ij')
            valida=(xx>=minimo[0])&(xx<=maximo[0])&(yy>=minimo[1])&(yy<=maximo[1])
        for c in range(1, 7):
            mascara = ((etiquetas[b] == c) & (probs[b, c] >= umbral) & valida).numpy().astype(np.uint8)
            n, componentes, estadisticas, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
            for j in range(1, n):
                x, y, ancho, alto, area = estadisticas[j].tolist()
                if area < area_minima:
                    continue
                box = [x/w, y/h, (x+ancho)/w, (y+alto)/h]
                if geometrias:
                    g = geometrias[b]
                    minimo = [g["relleno_x"] / g["lado"], g["relleno_y"] / g["lado"]]
                    maximo = [minimo[0] + g["ancho_original"] * g["escala_x"] / g["lado"], minimo[1] + g["alto_original"] * g["escala_y"] / g["lado"]]
                    box = [max(box[0], minimo[0]), max(box[1], minimo[1]), min(box[2], maximo[0]), min(box[3], maximo[1])]
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                boxes.append(box)
                scores.append(float(probs[b, c][torch.from_numpy(componentes == j)].mean()))
                labels.append(c - 1)
        box = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        score = torch.tensor(scores)
        label = torch.tensor(labels, dtype=torch.long)
        keep = nms_por_clase(box, score, label, .5, 100)
        resultados.append({"boxes": box[keep], "scores": score[keep], "labels": label[keep]})
    return resultados


def cargar_detector_mapas(path, dispositivo="cpu"):
    datos = torch.load(Path(path), map_location=dispositivo, weights_only=True)
    if datos.get("tipo") != "ecowaste_mapas_desde_cero_v1" or datos.get("clases") != CLASES:
        raise ValueError("Checkpoint de mapas no compatible")
    modelo = DetectorMapas(**datos["configuracion"]).to(dispositivo)
    modelo.load_state_dict(datos["pesos"])
    return modelo.eval(), datos
