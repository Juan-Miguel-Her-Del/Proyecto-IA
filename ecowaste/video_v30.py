"""Motor híbrido seleccionable solo tras superar el criterio de integración V30."""
from pathlib import Path
import cv2
from PIL import Image
import torch
from ecowaste.video import MotorVideo, LIMITE
from ecowaste.detector_hibrido import cargar_hibrido, mascara_letterbox
from ecowaste.datasets import CLASES, letterbox
from ecowaste.detector_mapas import cajas_desde_mapas
from ecowaste.mosaico import ventanas, combinar_vistas
from ecowaste.cajas import deshacer_letterbox
from ecowaste.backend_numerico import aplicar_backend, politica
from ecowaste.incorporacion import leer, sha


class MotorHibrido(MotorVideo):
    def __init__(self, raiz, checkpoint, vistas=5, umbral=.35, dispositivo='cuda'):
        if vistas not in [1, 5] or not 0 <= umbral <= 1:
            raise ValueError('Vistas/umbral inválidos')
        self.raiz = Path(raiz)
        met = leer(self.raiz/'resultados/v30_hibrido_revision02/metricas.json')
        if not met['puerta_integracion_superada']:
            raise ValueError('Híbrido no supera el criterio prefijado; conservar motor anterior')
        self.checkpoint = Path(checkpoint)
        self.modelo, self.ck = cargar_hibrido(self.checkpoint, dispositivo)
        if self.ck['rama'] != 'B' or self.ck['paso'] != 2400:
            raise ValueError('Solo B final evaluada es seleccionable')
        commit = leer(self.raiz/'resultados/v30_hibrido_revision02/entrenamiento/B/confirmaciones/paso_2400.json')
        if sha(self.checkpoint) != commit['sha256']:
            raise ValueError('Checkpoint distinto del evaluado')
        self.modelo.requires_grad_(False)
        self.vistas, self.umbral, self.device = vistas, umbral, dispositivo
        self.metadatos = {'modelo': 'EcoWasteFormer-CNN-Transformer-V30', 'sha256_checkpoint': sha(self.checkpoint),
            'checkpoint': str(self.checkpoint), 'vistas': vistas, 'estado_detector': 'experimental', 'limite': LIMITE,
            'seguimiento_de_objetos': False, 'umbral_puntuacion': umbral}

    @torch.inference_mode()
    def detectar(self, frame_bgr):
        im = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        zs = ventanas(im.width, im.height) if self.vistas == 5 else [(0, 0, im.width, im.height)]
        xg = [letterbox(im.crop(z), 384) for z in zs]
        x = torch.stack([x for x, _ in xg]).to(self.device); gs = [g for _, g in xg]
        with aplicar_backend(politica('completo')), torch.autocast(self.device, dtype=torch.float16, enabled=self.device == 'cuda'):
            raw = self.modelo(x, mascara_letterbox(gs, self.device))
        ps = cajas_desde_mapas(raw, .05, 2, gs)
        if self.vistas == 5: p = combinar_vistas(ps, gs, zs, im.width, im.height, 384, self.umbral)
        else:
            p = ps[0]; keep = p['scores'] >= self.umbral; p = {k: v[keep] for k, v in p.items()}
        _, g = letterbox(im, 384); boxes = deshacer_letterbox(p['boxes'], g)
        ds = [{'material': CLASES[int(c)], 'grupo': 'organico' if int(c) == 5 else 'inorganico',
               'puntuacion': float(s), 'bbox_xyxy': b.tolist()} for b, s, c in zip(boxes, p['scores'], p['labels'])]
        return {'detecciones': ds, 'estado_detector': 'experimental', 'limite': LIMITE,
                'coordenadas': 'xyxy_pixeles_frame', 'ancho': im.width, 'alto': im.height}
