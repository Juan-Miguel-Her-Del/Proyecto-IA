"""Inferencia en imagen completa y cuatro zonas, sin consultar anotaciones."""
import math
import torch
from ecowaste.datasets import letterbox, geometria_letterbox
from ecowaste.cajas import deshacer_letterbox, nms_por_clase
from ecowaste.detector_mapas import cajas_desde_mapas
from ecowaste.detector_centros import postprocesar_centros, DetectorCentros

VERSION = 'completa_cuatro_zonas065_bordes_nms050_v1'


def ventanas(ancho, alto):
    if min(ancho, alto) <= 0: raise ValueError('Imagen vacía')
    w, h = max(1, math.ceil(ancho*.65)), max(1, math.ceil(alto*.65))
    zonas = [(0, 0, ancho, alto)]
    for y in [0, alto-h]:
        for x in [0, ancho-w]:
            t = (x, y, x+w, y+h)
            if t not in zonas: zonas.append(t)
    return zonas


def proyectar_al_lienzo(boxes, g, ventana, ancho, alto, lado=384):
    b = deshacer_letterbox(boxes, g)
    b[:, [0, 2]] += ventana[0]
    b[:, [1, 3]] += ventana[1]
    total = geometria_letterbox(ancho, alto, lado)
    b[:, [0, 2]] = (b[:, [0, 2]]*total['escala_x']+total['relleno_x'])/lado
    b[:, [1, 3]] = (b[:, [1, 3]]*total['escala_y']+total['relleno_y'])/lado
    return b


@torch.inference_mode()
def predecir_mosaico(modelo, imagen, umbral=.05, amp=False, umbral_propuestas=None,
                    politica_mapas='argmax7'):
    device = next(modelo.parameters()).device
    zonas = ventanas(imagen.width, imagen.height)
    if umbral_propuestas is None:umbral_propuestas=umbral
    entradas, geometrias = [], []
    for v in zonas:
        x, g = letterbox(imagen.crop(v), modelo.imagen)
        entradas.append(x)
        geometrias.append(g)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        raw = modelo(torch.stack(entradas).to(device))
    if isinstance(modelo, DetectorCentros):
        if politica_mapas!='argmax7':raise ValueError('La política por material requiere mapas softmax7')
        predicciones = postprocesar_centros(raw, geometrias, umbral_propuestas)
    else:
        predicciones = cajas_desde_mapas(raw, umbral=umbral_propuestas, area_minima=2, geometrias=geometrias,
                                       formulacion=modelo.configuracion_mapas['formulacion'],politica=politica_mapas)
    return combinar_vistas(predicciones,geometrias,zonas,imagen.width,imagen.height,modelo.imagen,umbral)


def combinar_vistas(predicciones,geometrias,zonas,ancho,alto,lado=384,umbral=.05):
    """Combina cajas ya propuestas, sin acceso a logits ni anotaciones."""
    if not (len(predicciones)==len(geometrias)==len(zonas)) or not zonas:
        raise ValueError('Vistas y predicciones incompatibles')
    cajas, scores, labels = [], [], []
    for p, g, v in zip(predicciones, geometrias, zonas):
        # Fragmentos que tocan un corte interno se omiten; las otras vistas
        # solapadas y la imagen completa siguen disponibles.
        originales = deshacer_letterbox(p['boxes'], g)
        keep = torch.ones(len(originales), dtype=torch.bool)
        if v[0] > 0: keep &= originales[:, 0] > 1
        if v[1] > 0: keep &= originales[:, 1] > 1
        if v[2] < ancho: keep &= originales[:, 2] < (v[2]-v[0]-1)
        if v[3] < alto: keep &= originales[:, 3] < (v[3]-v[1]-1)
        cajas.append(proyectar_al_lienzo(p['boxes'][keep], g, v, ancho, alto, lado))
        scores.append(p['scores'][keep])
        labels.append(p['labels'][keep])
    boxes = torch.cat(cajas)
    scores = torch.cat(scores)
    labels = torch.cat(labels)
    ix = nms_por_clase(boxes, scores, labels, .5, 100)
    ix = ix[scores[ix]>=umbral]
    return {'boxes': boxes[ix], 'scores': scores[ix], 'labels': labels[ix]}
