"""Geometría diferenciable, NMS por clase y recuperación de coordenadas."""
import torch


def area(cajas):
    return (cajas[..., 2:] - cajas[..., :2]).clamp(min=0).prod(-1)


def intersecciones(a, b):
    return (torch.minimum(a[:, None, 2:], b[None, :, 2:]) - torch.maximum(a[:, None, :2], b[None, :, :2])).clamp(min=0).prod(-1)


def iou_pares(a, b):
    inter = intersecciones(a, b)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter).clamp(min=1e-9)


def giou_alineado(a, b):
    inter = (torch.minimum(a[:, 2:], b[:, 2:]) - torch.maximum(a[:, :2], b[:, :2])).clamp(min=0).prod(-1)
    union = (area(a) + area(b) - inter).clamp(min=1e-9)
    envolvente = (torch.maximum(a[:, 2:], b[:, 2:]) - torch.minimum(a[:, :2], b[:, :2])).clamp(min=0).prod(-1).clamp(min=1e-9)
    return inter / union - (envolvente - union) / envolvente


def cubiertas_por_ignorar(cajas, regiones, fraccion=0.5):
    if len(regiones) == 0:
        return torch.zeros(len(cajas), dtype=torch.bool, device=cajas.device)
    inter = intersecciones(cajas, regiones)
    centros = (cajas[:, :2] + cajas[:, 2:]) / 2
    dentro = ((centros[:, None] >= regiones[None, :, :2]) & (centros[:, None] <= regiones[None, :, 2:])).all(-1).any(-1)
    return dentro | ((inter / area(cajas).clamp(min=1e-9)[:, None]).amax(1) >= fraccion)


def nms_por_clase(cajas, puntuaciones, etiquetas, umbral=0.5, maximo=100):
    orden = puntuaciones.argsort(descending=True, stable=True)
    elegidos = []
    while len(orden) and len(elegidos) < maximo:
        i = orden[0]
        elegidos.append(i)
        restantes = orden[1:]
        solapamiento = iou_pares(cajas[i].unsqueeze(0), cajas[restantes]).squeeze(0)
        orden = restantes[(etiquetas[restantes] != etiquetas[i]) | (solapamiento <= umbral)]
    return torch.stack(elegidos) if elegidos else torch.empty(0, dtype=torch.long, device=cajas.device)


def deshacer_letterbox(cajas, g):
    salida = cajas.clone()
    salida[:, [0, 2]] = (salida[:, [0, 2]] * g["lado"] - g["relleno_x"]) / g["escala_x"]
    salida[:, [1, 3]] = (salida[:, [1, 3]] * g["lado"] - g["relleno_y"]) / g["escala_y"]
    salida[:, [0, 2]] = salida[:, [0, 2]].clamp(0, g["ancho_original"])
    salida[:, [1, 3]] = salida[:, [1, 3]].clamp(0, g["alto_original"])
    return salida
