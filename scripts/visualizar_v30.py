"""Paneles con IDs prefijados; GT aparte, nunca inyectado en inferencia."""
from pathlib import Path
import sys
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import torch
from PIL import Image, ImageDraw
from ecowaste.experimento_v30 import SALIDA, contexto
from ecowaste.incorporacion import leer
from ecowaste.reanudacion_regional import publicar
from ecowaste.datos_v29 import cargar_coleccion
from ecowaste.datasets import CLASES
from ecowaste.cajas import deshacer_letterbox
from ecowaste.evaluacion_v28_2 import cargar_predicciones
from scripts.entrenar_detector_mapas import DatosMapas
from ecowaste.poligonos import ContornosTACO
from types import SimpleNamespace


def main():
    cfg, ht, nu, gl, pm = contexto(); del gl, pm
    v = DatosMapas('val', ContornosTACO(RAIZ, 1/64))
    datasets = {'historico_train': ht, 'nuevo_train': nu, 'desarrollo': SimpleNamespace(registros=v.base.origen.registros, objetivos=v.base.objetivos),
                'externo_retenido': cargar_coleccion(RAIZ, RAIZ/cfg['coleccion'], 'val')}
    met = leer(SALIDA/'metricas.json'); ejemplos = leer(SALIDA/'ejemplos_prefijados.json')
    cache = {}; index = []
    colores = ['#2288bb', '#995599', '#bd7711', '#00a099', '#aa4444', '#228833']
    for k, e in enumerate(ejemplos):
        ds = datasets[e['dominio']]; i = e['indice']; r = ds.registros[i]; o = ds.objetivos[i]
        with Image.open(RAIZ/r['ruta']) as im: im = im.convert('RGB')
        w = 480; h = max(120, round(im.height*w/im.width)); h = min(h, 620)
        thumb = im.resize((w, h)); panel = Image.new('RGB', (w*4, (h+50)*2), 'white')
        for fila, vistas in enumerate([1, 5]):
            for col, nombre in enumerate(['GT separado', 'Original', 'A', 'B']):
                pieza = thumb.copy(); draw = ImageDraw.Draw(pieza)
                if col == 0: boxes, labels, scores = o['boxes'], o['labels'], None
                else:
                    key = (nombre, e['dominio'], vistas)
                    if key not in cache:
                        folder = RAIZ/met['resultados'][nombre][e['dominio']]['captura']
                        cache[key] = cargar_predicciones(folder, vistas)
                    p = cache[key][i]; boxes, labels, scores = p['boxes'], p['labels'], p['scores']
                pix = deshacer_letterbox(boxes, o['geometria'])
                n = 0
                for j, (b, c) in enumerate(zip(pix, labels)):
                    if scores is not None and float(scores[j]) < .35: continue
                    xy = [float(b[0])*w/im.width, float(b[1])*h/im.height, float(b[2])*w/im.width, float(b[3])*h/im.height]
                    color = colores[int(c)]
                    draw.rectangle(xy, outline=color, width=2)
                    text = CLASES[int(c)] + (f' {scores[j]:.2f}' if scores is not None else '')
                    draw.text((xy[0], max(0, xy[1]-12)), text, fill=color, stroke_width=1, stroke_fill='white'); n += 1
                ox, oy = col*w, fila*(h+50)
                panel.paste(pieza, (ox, oy+50))
                ImageDraw.Draw(panel).text((ox+5, oy+5), f'{nombre} | {vistas} vista(s) | {n} cajas visibles\n{r["id"]} | score>=0.35; metricas>=0.05', fill='black')
        path = RAIZ/f'datos/nuevos/v30/privado/visuales/{k+1:02d}_{e["dominio"]}_{CLASES[e["clase"]]}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f: panel.save(f, format='JPEG', quality=88)
        index.append({**e, 'archivo': path.relative_to(RAIZ).as_posix(), 'regla': 'menor ID con GT por material/dominio; definida antes de inferencia'})
    publicar(SALIDA/'visuales.json', index)
    print('Ejemplos locales:', len(index), flush=True)


if __name__ == '__main__': main()
