"""Transformer propio: picos por material/escala y regresión de cajas individuales."""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from ecowaste.detector_mapas import DetectorMapas
from ecowaste.datasets import CLASES
from ecowaste.cajas import giou_alineado, nms_por_clase

VERSION = 'centros_focal_gaussiano_dos_escalas_v1'


class DetectorCentros(DetectorMapas):
    def __init__(self, imagen=384, dimension=96):
        super().__init__(imagen=imagen, dimension=dimension, auxiliar_clasificacion=True)
        self.configuracion_centros = dict(imagen=imagen, dimension=dimension)
        self.cabezal = nn.Identity()
        self.calor = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, 12))
        self.regresion = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, 48))
        self.register_buffer('prior_wh', torch.full((2, 2), .1))
        with torch.no_grad():
            self.calor[-1].bias.fill_(math.log(.01 / .99))
            nn.init.normal_(self.calor[-1].weight, std=.001)
            self.regresion[-1].weight.zero_()
            self.regresion[-1].bias.zero_()

    @torch.no_grad()
    def configurar_centros(self, priors, corte):
        p = torch.as_tensor(priors, device=self.prior_wh.device)
        if p.shape != (2, 2) or not bool(((p > 0) & (p < 1)).all()) or not 0 < corte < 1:
            raise ValueError('Priores de centros inválidos')
        self.prior_wh.copy_(p)
        self.umbral_escala.fill_(corte)

    def forward(self, imagenes):
        f = self.caracteristicas(imagenes)
        b, h, w, _ = f.shape
        raw = self.regresion(f).reshape(b, h, w, 6, 2, 4).float()
        yy, xx = torch.meshgrid(torch.arange(h, device=f.device), torch.arange(w, device=f.device), indexing='ij')
        grid = torch.stack([xx, yy], -1)[None, :, :, None, None]
        centro = (grid + raw[..., :2].sigmoid()) / raw.new_tensor([w, h])
        wh = (raw[..., 2:] + torch.logit(self.prior_wh)[None, None, None, None]).sigmoid()
        return {'logits': self.calor(f).reshape(b, h, w, 6, 2),
                'boxes': torch.cat([centro - wh / 2, centro + wh / 2], -1)}


def objetivos_centros(objetivo, corte, lado=48):
    """Dos candidatos de escala por material: reserva centros sin perder colisiones."""
    boxes = objetivo['boxes'].cpu()
    labels = objetivo['labels'].cpu()
    wh = boxes[:, 2:] - boxes[:, :2]
    xy = ((boxes[:, :2] + boxes[:, 2:]) / 2 * lado).long().clamp(0, lado - 1)
    calor = torch.zeros(lado, lado, 6, 2)
    yy, xx = torch.meshgrid((torch.arange(lado) + .5) / lado, (torch.arange(lado) + .5) / lado, indexing='ij')
    valida = torch.ones(lado, lado, dtype=torch.bool)
    g = objetivo.get('geometria')
    if g:
        x0, y0 = g['relleno_x'] / g['lado'], g['relleno_y'] / g['lado']
        x1 = x0 + g['ancho_original'] * g['escala_x'] / g['lado']
        y1 = y0 + g['alto_original'] * g['escala_y'] / g['lado']
        valida &= (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    for r in objetivo['ignore_boxes'].cpu():
        valida &= ~((xx >= r[0]) & (xx <= r[2]) & (yy >= r[1]) & (yy <= r[3]))
    ocupados = set()
    indices, usados = [], []
    for i in wh.prod(-1).argsort(stable=True).tolist():
        x, y = xy[i].tolist()
        c = int(labels[i])
        preferido = int(float(wh[i].prod().sqrt()) > corte)
        slot = next((s for s in [preferido, 1 - preferido] if (y, x, c, s) not in ocupados), None)
        if slot is None:
            raise ValueError(f'Más de dos centros de igual material/celda: {objetivo.get("id")}')
        ocupados.add((y, x, c, slot))
        # Anchura del suavizado proviene del bbox train; el pico exacto es uno.
        sigma = (wh[i] * lado / 6).clamp(1, 4)
        gauss = torch.exp(-.5 * (((xx * lado - .5 - x) / sigma[0]).square()
                                 + ((yy * lado - .5 - y) / sigma[1]).square()))
        calor[:, :, c, slot] = torch.maximum(calor[:, :, c, slot], gauss)
        indices.append(((y * lado + x) * 6 + c) * 2 + slot)
        usados.append(i)
    return {'calor': calor, 'valida': valida, 'indices': torch.tensor(indices, dtype=torch.long),
            'objetos': torch.tensor(usados, dtype=torch.long)}


def cxwh(box):
    return torch.cat([(box[..., :2] + box[..., 2:]) / 2, box[..., 2:] - box[..., :2]], -1)


def perdida_centros(salida, objetivos, corte):
    raw = salida['logits'].float()
    boxes = salida['boxes'].float()
    cero = raw.sum() * 0 + boxes.sum() * 0
    loss_calor, loss_l1, loss_giou = cero, cero, cero
    n = 0
    for i, o in enumerate(objetivos):
        t = objetivos_centros(o, corte, raw.shape[1])
        y = t['calor'].to(raw.device)
        pos = y == 1
        neg = (y < 1) & t['valida'].to(raw.device)[:, :, None, None]
        p = raw[i].sigmoid()
        loss_calor = loss_calor - (F.logsigmoid(raw[i]) * (1 - p).square())[pos].sum()
        loss_calor = loss_calor - (F.logsigmoid(-raw[i]) * p.square() * (1 - y).pow(4))[neg].sum()
        ix = t['indices'].to(raw.device)
        if len(ix):
            gt = o['boxes'][t['objetos']].to(raw.device)
            b = boxes[i].reshape(-1, 4)[ix]
            loss_l1 = loss_l1 + (cxwh(b) - cxwh(gt)).abs().sum()
            loss_giou = loss_giou + (1 - giou_alineado(b, gt)).sum()
            n += len(ix)
    den = max(n, 1)
    total = (loss_calor + 5 * loss_l1 + 2 * loss_giou) / den
    return total, {'objetos_asignados': n, 'loss_calor': float(loss_calor.detach()) / den,
                   'loss_l1': float(loss_l1.detach()) / den, 'loss_giou': float(loss_giou.detach()) / den}


@torch.inference_mode()
def postprocesar_centros(salida, geometrias, umbral=.01, nms_iou=.5, maximo=100):
    scores = salida['logits'].float().sigmoid().permute(0, 3, 4, 1, 2)
    b, c, s, h, w = scores.shape
    planos = scores.reshape(b, c * s, h, w)
    picos = planos == F.max_pool2d(planos, 3, stride=1, padding=1)
    scores = scores.permute(0, 3, 4, 1, 2).cpu()
    picos = picos.reshape(b, c, s, h, w).permute(0, 3, 4, 1, 2).cpu()
    boxes = salida['boxes'].float().cpu()
    resultados = []
    for i, g in enumerate(geometrias):
        bb = boxes[i].reshape(-1, 4).clone()
        ss = scores[i].reshape(-1)
        labels = torch.arange(6).view(1, 1, 6, 1).expand(h, w, 6, 2).reshape(-1)
        centros = (bb[:, :2] + bb[:, 2:]) / 2
        x0, y0 = g['relleno_x'] / g['lado'], g['relleno_y'] / g['lado']
        x1 = x0 + g['ancho_original'] * g['escala_x'] / g['lado']
        y1 = y0 + g['alto_original'] * g['escala_y'] / g['lado']
        keep = picos[i].reshape(-1) & (ss >= umbral) & (centros[:, 0] >= x0) & (centros[:, 0] <= x1)
        keep &= (centros[:, 1] >= y0) & (centros[:, 1] <= y1)
        bb[:, [0, 2]] = bb[:, [0, 2]].clamp(x0, x1)
        bb[:, [1, 3]] = bb[:, [1, 3]].clamp(y0, y1)
        keep &= (bb[:, 2] > bb[:, 0]) & (bb[:, 3] > bb[:, 1])
        bb, ss, labels = bb[keep], ss[keep], labels[keep]
        top = ss.argsort(descending=True)[:1000]
        bb, ss, labels = bb[top], ss[top], labels[top]
        ix = nms_por_clase(bb, ss, labels, nms_iou, maximo)
        resultados.append({'boxes': bb[ix], 'scores': ss[ix], 'labels': labels[ix]})
    return resultados


def cargar_detector_centros(checkpoint, dispositivo='cpu'):
    ck = torch.load(Path(checkpoint), map_location=dispositivo, weights_only=True)
    if ck.get('tipo') != 'ecowaste_centros_desde_cero_v1' or ck.get('version') != VERSION or ck.get('clases') != CLASES:
        raise ValueError('Checkpoint de centros incompatible')
    m = DetectorCentros(**ck['configuracion']).to(dispositivo)
    m.load_state_dict(ck['pesos'])
    return m.eval(), ck
