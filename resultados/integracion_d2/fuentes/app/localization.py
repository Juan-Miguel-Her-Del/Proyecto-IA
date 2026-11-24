"""Localización geométrica en fondos controlados, sin reconocimiento aprendido."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Region:
    box: tuple
    method: str


@dataclass(frozen=True)
class Localization:
    region: object
    message: str
    candidates: int = 0


class LocalizationError(ValueError):
    pass


def iou(first, second):
    a, b = first.box, second.box
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0.


def manual_region(rgb, bounds=(.20, .20, .80, .80)):
    height, width = rgb.shape[:2]
    if len(bounds) != 4 or not all(np.isfinite(v) and 0 <= v <= 1 for v in bounds):
        raise ValueError("Coordenadas manuales inválidas.")
    left, top, right, bottom = bounds
    box = (round(left * width), round(top * height), round(right * width), round(bottom * height))
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        raise ValueError("Amplía la región manual: debe medir al menos 16 × 16 píxeles.")
    return Region(box, "manual")


def square_box(box, width, height):
    """Amplía la caja sin recortar; un ROI cuadrado evita deformar botellas al redimensionar."""
    left,top,right,bottom=box
    side=max(right-left,bottom-top)
    if side>min(width,height):
        return box  # No cabe un cuadrado: conservar todo el objeto tiene prioridad.
    x=min(max(0,round((left+right-side)/2)),width-side)
    y=min(max(0,round((top+bottom-side)/2)),height-side)
    return (x,y,x+side,y+side)


class Localizer:
    """Estima fondo desde los bordes y busca en TODO el fotograma.

    La estabilidad temporal solo amplía el margen: nunca excluye el candidato
    actual. Una pérdida borra el historial, sin inventar un objeto en el centro.
    """
    def __init__(self, mode="auto", manual_bounds=(.20, .20, .80, .80), margin=.12):
        if mode not in ("auto", "manual", "full"):
            raise ValueError("Método de localización desconocido.")
        self.mode, self.manual_bounds, self.margin = mode, manual_bounds, margin
        self.previous = None
        self.background = None

    def set_background(self, rgb):
        self.background = rgb.copy()
        self.previous = None

    def clear_background(self):
        self.background = None
        self.previous = None

    def locate(self, rgb):
        if self.mode == "full":
            return Localization(Region((0, 0, rgb.shape[1], rgb.shape[0]), "imagen-completa"), "Imagen completa: sin localización.")
        if self.mode == "manual":
            return Localization(manual_region(rgb, self.manual_bounds), "Región elegida manualmente.")
        result = self._automatic(rgb)
        if result.region is None:
            self.previous = None
            return result
        current = result.region
        if self.previous and iou(current, self.previous) > .15:
            blended = [round(.45*a + .55*b) for a, b in zip(self.previous.box, current.box)]
            # Conservar todo el candidato actual y su margen; suavizar solo hacia fuera.
            box = (min(blended[0], current.box[0]), min(blended[1], current.box[1]),
                   max(blended[2], current.box[2]), max(blended[3], current.box[3]))
            current = Region(box, current.method)
        current=Region(square_box(current.box,rgb.shape[1],rgb.shape[0]),current.method)
        self.previous = current
        return Localization(current, result.message, result.candidates)

    def _automatic(self, rgb):
        height, width = rgb.shape[:2]
        scale = min(1., 640 / max(height, width))
        small = cv2.resize(rgb, (round(width*scale), round(height*scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
        sh, sw = small.shape[:2]
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
        calibrated = self.background is not None
        if calibrated:
            if self.background.shape != rgb.shape:
                return Localization(None, "Cambió la resolución: vuelve a capturar el fondo sin residuo.")
            background_rgb = cv2.resize(self.background, (sw, sh), interpolation=cv2.INTER_AREA)
            background_lab = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            difference = lab-background_lab
            # Compensar pequeños cambios globales de luz antes de buscar el objeto.
            illumination = np.median(difference.reshape(-1, 3), axis=0)
            distance = np.linalg.norm(difference-illumination, axis=2)
            foreground = (distance > 18).astype(np.uint8)*255
            if np.count_nonzero(foreground) > .45*sw*sh:
                return Localization(None, "La cámara o el fondo cambiaron demasiado. Captura el fondo nuevamente, sin residuo.")
        else:
            foreground, failure = self._border_foreground(lab)
            if failure:
                return Localization(None, failure)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        foreground_contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in foreground_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) > .015*sw*sh and (x <= 1 or y <= 1 or x+w >= sw-1 or y+h >= sh-1):
                return Localization(None, "Hay una región importante cortada por el borde. Retira la mano o aleja el objeto del borde.")
        candidates = []
        masks = [(foreground, "auto-fondo-calibrado" if calibrated else "auto-segmentación")]
        if not calibrated:
            gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)
            edges = cv2.morphologyEx(cv2.Canny(gray, 35, 100), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
            masks.append((edges, "auto-contorno"))
        for mask, method in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                if not (.004*sw*sh <= area <= .65*sw*sh and min(w, h) >= 12 and area/(w*h) >= .22):
                    continue
                if x <= 1 or y <= 1 or x+w >= sw-1 or y+h >= sh-1:
                    continue
                raw = Region((round(x/scale), round(y/scale), round((x+w)/scale), round((y+h)/scale)), method)
                if not any(iou(raw, item[1]) > .55 for item in candidates):
                    candidates.append((area, raw))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            return Localization(None, "Sin región fiable: no se localizó un residuo. Coloca un objeto entero o ajusta la región manual.")
        if len(candidates) > 1 and candidates[1][0] > .60*candidates[0][0]:
            return Localization(None, "Hay varias regiones parecidas. Deja un solo residuo y retira la mano.", len(candidates))
        _, region = candidates[0]
        left, top, right, bottom = region.box
        padding = max(6, round(max(right-left, bottom-top)*self.margin))
        region = Region((max(0, left-padding), max(0, top-padding), min(width, right+padding), min(height, bottom+padding)), region.method)
        return Localization(region, "Región por diferencia con el fondo." if calibrated else "Región por contraste: comprueba el recorte.", len(candidates))

    @staticmethod
    def _border_foreground(lab):
        sh, sw = lab.shape[:2]
        band = max(2, min(sh, sw)//35)
        border = np.concatenate((lab[:band].reshape(-1, 3), lab[-band:].reshape(-1, 3),
                                 lab[:, :band].reshape(-1, 3), lab[:, -band:].reshape(-1, 3)))
        background = np.median(border, axis=0)
        border_distance = np.linalg.norm(border-background, axis=1)
        median = float(np.median(border_distance))
        spread = float(np.median(np.abs(border_distance-median)))
        if median > 30 or spread > 18:
            return None, "Fondo no uniforme: captura el fondo sin residuo o utiliza región manual."
        distance = np.linalg.norm(lab-background, axis=2)
        foreground = (distance > max(20., median+4*spread+8)).astype(np.uint8)*255
        return foreground, ""
