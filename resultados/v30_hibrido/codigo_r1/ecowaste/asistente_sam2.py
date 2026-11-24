"""Inferencia de SAM 2.1 Tiny en CPU, a demanda y sin entrenamiento."""
import threading
import time
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from ecowaste.auxiliares_publicos import contornos_mask
from ecowaste.incorporacion import leer, sha, poligonos_validos


class AsistenteSAM2:
    def __init__(self, raiz):
        self.raiz = Path(raiz)
        self.predictor = None
        self.imagen_actual = None
        self.lock = threading.Lock()
        self.modelos = self.raiz / 'datos/nuevos/v27_2/modelos'

    def cargar(self):
        if self.predictor is not None:
            return
        import torch
        if torch.version.cuda is not None:
            raise ValueError('Usar el entorno EcoWasteAnotacion CPU separado; no el entorno histórico')
        torch.set_num_threads(4)
        fuente = leer(self.raiz / 'resultados/v27_2_anotacion/sam2_archivos_fuente.json')
        for nombre, esperado in fuente['archivos'].items():
            if sha(self.modelos / 'fuente' / nombre) != esperado:
                raise ValueError('Código/configuración SAM 2 alterado: ' + nombre)
        recibo = leer(self.raiz / 'resultados/v27_2_anotacion/sam2_pesos.json')
        checkpoint = self.modelos / 'sam2.1_hiera_tiny.pt'
        if sha(checkpoint) != recibo['sha256']:
            raise ValueError('Pesos SAM 2 alterados')
        sys.path.insert(0, str(self.modelos / 'fuente'))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        modelo = build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml', str(checkpoint), device='cpu',
                            apply_postprocessing=False)
        modelo.eval()
        self.predictor = SAM2ImagePredictor(modelo, max_hole_area=0, max_sprinkle_area=0)

    def proponer(self, taller, iid, indicaciones):
        import torch
        im = taller.imagen(iid)
        box = indicaciones.get('caja')
        pts = indicaciones.get('puntos', [])
        labels = indicaciones.get('etiquetas', [])
        if box is not None:
            if len(box) != 4 or not np.isfinite(box).all():
                raise ValueError('Caja XYXY no finita')
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= im['width'] and 0 <= y1 < y2 <= im['height']):
                raise ValueError('Caja fuera de imagen o vacía')
        if len(pts) != len(labels) or any(v not in [0, 1] for v in labels):
            raise ValueError('Puntos y etiquetas incompatibles')
        if len(pts) > 100:
            raise ValueError('Máximo cien indicaciones por solicitud')
        for p in pts:
            if len(p) != 2 or not np.isfinite(p).all() or not (0 <= p[0] < im['width'] and 0 <= p[1] < im['height']):
                raise ValueError('Punto fuera de imagen')
        if box is None and not pts:
            raise ValueError('Indicar una caja o al menos un punto')
        path = taller.archivo_imagen(iid)
        with self.lock, torch.inference_mode():
            t = time.perf_counter()
            self.cargar()
            inicio_embedding = time.perf_counter()
            cache = self.imagen_actual == im['sha256_archivo']
            if not cache:
                with Image.open(path) as pil:
                    self.predictor.set_image(np.asarray(pil.convert('RGB')).copy())
                self.imagen_actual = im['sha256_archivo']
            embedding_seg = time.perf_counter() - inicio_embedding
            masks, scores, _ = self.predictor.predict(
                point_coords=np.asarray(pts, dtype=np.float32) if pts else None,
                point_labels=np.asarray(labels, dtype=np.int32) if pts else None,
                box=np.asarray(box, dtype=np.float32) if box is not None else None,
                multimask_output=True)
            k = int(np.argmax(scores))
            resultado = {**contornos_mask(masks[k].astype(np.uint8)), 'score_SAM_no_probabilidad_material': float(scores[k]),
                         'segundos': time.perf_counter() - t, 'segundos_embedding': embedding_seg,
                         'embedding_reutilizado': cache, 'dispositivo': 'cpu', 'memoria_GPU_bytes': 0,
                         'seleccion_multimascara': 'max_score_SAM_prefijado_no_GT', 'no_predice_material': True}
            resultado['errores_geometricos_V26'] = []
            try:
                poligonos_validos(resultado['poligonos'], im['width'], im['height'])
                if resultado['huecos']:
                    poligonos_validos(resultado['huecos'], im['width'], im['height'])
            except ValueError as e:
                resultado['errores_geometricos_V26'].append(str(e))
            modelo = {'nombre': 'SAM 2.1 Hiera Tiny', 'revision_codigo': '2b90b9f5ceec907a1c18123530e92e794ad901a4',
                      'checkpoint_sha256': leer(self.raiz / 'resultados/v27_2_anotacion/sam2_pesos.json')['sha256'],
                      'licencia': 'Apache-2.0', 'torch': torch.__version__, 'device': 'cpu',
                      'apply_postprocessing': False, 'max_hole_area': 0, 'max_sprinkle_area': 0}
            return taller.registrar_ia(iid, resultado, indicaciones, modelo)

    def liberar(self):
        with self.lock:
            self.predictor = None
            self.imagen_actual = None
        import gc
        gc.collect()
