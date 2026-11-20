"""Filtrado y NMS; las puntuaciones del detector no están calibradas."""
import torch
from ecowaste.cajas import area, nms_por_clase
from ecowaste.detector import decodificar


@torch.no_grad()
def postprocesar(salida, umbral=0.01, umbral_nms=0.5, maximo=100, pre_nms=200, geometrias=None, radio_centros=0):
    cajas, presencia, clases = decodificar(salida, radio_centros)
    probabilidades, etiquetas = clases.softmax(-1).max(-1)
    puntuaciones = presencia.sigmoid() * probabilidades
    resultados = []
    for b in range(len(salida)):
        box = cajas[b].reshape(-1, 4).clamp(0, 1)
        score, label = puntuaciones[b].flatten(), etiquetas[b].flatten()
        validos = score >= umbral
        if geometrias is not None:
            g = geometrias[b]
            inicio = box.new_tensor([g["relleno_x"], g["relleno_y"]]) / g["lado"]
            fin = inicio + box.new_tensor([g["ancho_original"] * g["escala_x"], g["alto_original"] * g["escala_y"]]) / g["lado"]
            centro = cajas[b].reshape(-1, 4)
            centro = (centro[:, :2] + centro[:, 2:]) / 2
            validos &= ((centro >= inicio) & (centro <= fin)).all(-1)
            box = torch.cat([torch.maximum(box[:, :2], inicio), torch.minimum(box[:, 2:], fin)], -1)
        filtro = (validos & (area(box) > 1e-9)).nonzero().flatten()
        filtro = filtro[score[filtro].argsort(descending=True, stable=True)[:pre_nms]]
        # NMS sobre los candidatos filtrados en CPU evita cientos de sincronizaciones GPU.
        box_cpu, score_cpu, label_cpu = box[filtro].cpu(), score[filtro].cpu(), label[filtro].cpu()
        elegidos = nms_por_clase(box_cpu, score_cpu, label_cpu, umbral_nms, maximo)
        resultados.append({"boxes": box_cpu[elegidos], "scores": score_cpu[elegidos], "labels": label_cpu[elegidos]})
    return resultados
