"""Prepara 80 tareas reales, sin revisores ni anotaciones humanas inventadas."""
from collections import Counter
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import sys
import zipfile

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ecowaste.incorporacion import leer, publicar, publicar_bytes, digest, sha, poligonos_validos, caja_derivada, candidatos_duplicados
from ecowaste.auxiliares_publicos import VERSION, DESCONOCIDOS
from ecowaste.propuestas_publicas import imagen_verificada
from scripts.fuentes_v27_2 import SALIDA, DATOS
from scripts.preparar_piloto_v27 import historicos_train_val

PROYECTO = RAIZ / 'datos/nuevos/v27_2/proyecto'


def extraer(z, miembro, destino):
    data = z.read(miembro)  # zipfile verifica el CRC del miembro.
    publicar_bytes(destino, data)
    return {'miembro': miembro, 'archivo': destino.relative_to(RAIZ).as_posix(), 'sha256': sha(destino),
            'crc32_verificado': True, 'bytes': len(data)}


def anotacion(aid, material, ps, bbox, origen, original, motivo=''):
    return {'id': str(aid), 'material': material, 'poligonos': ps, 'huecos': [], 'bbox': bbox,
            'origen': origen, 'anotacion_fuente_original': original, 'motivo': motivo,
            'descartada': False, 'aceptada_V26': False, 'propuesta_ia_id': None}


def imagen(iid, fuente, archivo, licencia, **kwargs):
    meta = imagen_verificada(RAIZ / archivo)
    return {'id': iid, 'fuente': fuente, 'archivo': archivo, 'licencia': licencia,
            **meta, **{k: None for k in DESCONOCIDOS}, 'anotaciones': [], 'nuevo_objeto_fisico_acreditado': False,
            'restriccion_particion': 'toda_la_fuente_unida_solo_candidato_auxiliar_train', **kwargs}


def originales_v27():
    prev = leer(RAIZ / 'datos/nuevos/v27/externos/propuestas_pendientes.json')
    fotos = []
    for im in prev['imagenes']:
        licencia = {'zerowaste': 'CC BY-NC 4.0 según autores; discrepancia Zenodo preservada',
                    'warp': 'CC0: Public Domain según distribuidor original', 'hpi': 'CC BY 4.0'}[im['fuente']]
        r = imagen(im['id'], im['fuente'], im['archivo'], licencia, split_original='train',
                   reutilizada_V27=True, grupo_nombre_fuente=im.get('grupo_nombre_fuente'),
                   grupo_comida_documentado=im.get('grupo_comida_documentado'),
                   fuente_original_snapshot_sha256=digest(im))
        for n, p in enumerate(im['propuestas']):
            r['anotaciones'].append(anotacion(f'{im["fuente"]}_{n}', p['material_propuesto'] or 'sin_asignar',
                    p['poligonos_normalizados_v26'] or [], p['bbox_v26_derivada'] or p['bbox_original'],
                    'anotacion_publica_pendiente_revision', deepcopy(p)))
        fotos.append(r)
    return fotos


def wsc():
    with zipfile.ZipFile(DATOS / 'wsc/original.zip') as z:
        train = leer(DATOS / 'wsc/anotaciones_originales/train/_annotations.coco.json')
        # El train contiene exactamente cinco variantes por nombre fuente. No usarlas
        # como fotos independientes: se prepara la partición pública valid SIN aumentos
        # de train. Es una fuente auxiliar; NO es val/test de EcoWasteFormer.
        raw = z.read('valid/_annotations.coco.json')
        publicar_bytes(DATOS / 'wsc/anotaciones_originales/valid/_annotations.coco.json', raw)
        import json
        valid = json.loads(raw)
        cats = {x['id']: x['name'] for x in train['categories']}
        byim = {}
        for a in valid['annotations']:
            byim.setdefault(a['image_id'], []).append(a)
        elegidas, usados = [], set()
        for cat in ['Paper', 'Organic', 'Plastic']:
            candidatos = [i for i in valid['images'] if any(cats[a['category_id']] == cat for a in byim.get(i['id'], []))]
            candidatos.sort(key=lambda i: hashlib.sha256(('v27_2_1' + i['file_name']).encode()).hexdigest())
            nuevos = [i for i in candidatos if i['id'] not in usados][:4]
            elegidas.extend(nuevos)
            usados.update(i['id'] for i in nuevos)
        if len(elegidas) != 12:
            raise ValueError('No hay doce candidatos diversos bajo criterio prefijado')
        fotos, recibos = [], []
        for i in elegidas:
            miembro = 'valid/' + i['file_name']
            rec = extraer(z, miembro, DATOS / 'wsc/imagenes' / i['file_name'])
            recibos.append(rec)
            im = imagen('wsc:' + str(i['id']), 'wsc', rec['archivo'], 'CC BY 4.0', split_original='valid',
                         nombre_fuente=i.get('extra', {}).get('name'), reutilizada_V27=False,
                         transformaciones_fuente='auto-orientación y resize fit-within 640; aumentos declarados en train excluidos',
                         fecha_exportacion_NO_captura=i.get('date_captured'))
            if (im['width'], im['height']) != (i['width'], i['height']):
                raise ValueError('Dimensiones COCO WSC incompatibles')
            for a in byim.get(i['id'], []):
                cat = cats[a['category_id']]
                material = {'Paper': 'papel', 'Plastic': 'plastico', 'Composite_Rubber_Textile': 'compuesto'}.get(cat, 'sin_asignar')
                motivo = 'Etiqueta original compuesta: no corresponde a material simple' if material == 'compuesto' else ''
                try:
                    ps = poligonos_validos(a['segmentation'], im['width'], im['height'])
                except (ValueError, TypeError):
                    ps = []  # Original preservado; no corregir en nombre de un humano.
                obj = anotacion('wsc_' + str(a['id']), material, ps, caja_derivada(ps) if ps else a['bbox'],
                                 'poligono_publico_no_revisado', deepcopy(a), motivo)
                obj['categoria_original'] = cat
                obj['aviso'] = 'Organic no equivale automáticamente a restos de comida; comprobar vegetación viva/residuo y taxonomía' if cat == 'Organic' else ''
                im['anotaciones'].append(obj)
            fotos.append(im)
        grupos = Counter(i.get('extra', {}).get('name') for i in train['images'])
        auditoria = {'fotos_declaradas': 1947, 'instancias_declaradas_aproximadas': 3900,
                     'train_imagenes_en_COCO': len(train['images']), 'train_anotaciones': len(train['annotations']),
                     'train_categorias': dict(Counter(cats[a['category_id']] for a in train['annotations'])),
                     'train_nombres_fuente_distintos_NO_objetos': len(grupos), 'multiplicidad_variantes': dict(Counter(grupos.values())),
                     'valid_imagenes': len(valid['images']), 'valid_anotaciones': len(valid['annotations']),
                     'train_valid_nombres_fuente_solapados': len(set(grupos) & {i.get('extra', {}).get('name') for i in valid['images']}),
                     'fotos_piloto': 12, 'anotaciones_piloto': sum(len(i['anotaciones']) for i in fotos),
                     'organic_mapeado_automaticamente': False, 'test_anotaciones_abiertas': False,
                     'originales_raw_camara_disponibles': False, 'fotos_publicadas_redimensionadas': True,
                     'recibos': recibos}
        publicar(SALIDA / 'auditoria_wsc.json', auditoria)
        return fotos


def dwsd():
    with zipfile.ZipFile(DATOS / 'dwsd/original.zip') as z:
        masks = sorted(n for n in z.namelist() if n.startswith('DSWD/Train/Mask/') and n.endswith('.png'))
        histogram = Counter()
        por_im = []
        modos = Counter()
        for n in masks:
            with Image.open(io.BytesIO(z.read(n))) as im:
                a = np.asarray(im)
                modos[im.mode] += 1
                vals, counts = np.unique(a, return_counts=True)
                histogram.update(dict(zip(map(int, vals), map(int, counts))))
                por_im.append({'mascara': n, 'valores': [int(v) for v in vals], 'size': list(im.size)})
        # Diversificar combinaciones de valores; no asignarles nombres sin leyenda.
        elegidas, usados = [], set()
        for r in sorted(por_im, key=lambda r: hashlib.sha256(('v27_2_1' + r['mascara']).encode()).hexdigest()):
            key = tuple(r['valores'])
            if key not in usados:
                elegidas.append(r)
                usados.add(key)
            if len(elegidas) == 8:
                break
        fotos, recibos = [], []
        for r in elegidas:
            nombre = Path(r['mascara']).name
            foto = 'DSWD/Train/Image/' + nombre.replace('mask_', 'img_')
            rec = extraer(z, foto, DATOS / 'dwsd/imagenes' / Path(foto).name)
            rm = extraer(z, r['mascara'], DATOS / 'dwsd/mascaras_semanticas' / nombre)
            recibos.extend([rec, rm])
            im = imagen('dwsd:' + Path(foto).stem, 'dwsd', rec['archivo'], 'CC BY 4.0',
                         split_original='train', reutilizada_V27=False, mascara_semantica=rm['archivo'],
                         sha256_mascara_semantica=rm['sha256'], valores_semanticos_sin_leyenda=r['valores'],
                         aviso_fuente='Máscara de referencia sin IDs de instancia/leyenda verificables; no es GT por objeto')
            if [im['width'], im['height']] != r['size']:
                raise ValueError('Máscara DWSD no coincide con imagen')
            fotos.append(im)
        publicar(SALIDA / 'auditoria_dwsd.json', {'imagenes_declaradas': 784, 'segmentos_declarados_NO_verificados_como_instancias': 2350,
            'pares_train': len(masks), 'archivos_Labelme_JSON_en_ZIP': 0, 'modos_mascara_train': dict(modos),
            'histograma_valores_train': dict(histogram), 'leyenda_valor_material_incluida': False,
            'identidad_instancia_verificable': False, 'interpretacion_operativa': 'solo_referencia_semantica_sin_GT_de_instancia',
            'fotos_piloto': len(fotos), 'anotaciones_instancia_importadas': 0, 'test_original_evaluado': False, 'recibos': recibos})
        return fotos


def contactos(fuente, fotos):
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13)
    sheet = Image.new('RGB', (1200, 300 * ((len(fotos) + 3) // 4)), 'white')
    for n, im in enumerate(fotos):
        with Image.open(RAIZ / im['archivo']) as p:
            p = p.convert('RGB')
            p.thumbnail((290, 240))
        tile = Image.new('RGB', (300, 300), 'white')
        tile.paste(p, (0, 0))
        d = ImageDraw.Draw(tile)
        d.text((2, 242), im['id'], fill='black', font=font)
        d.text((2, 260), ','.join(sorted({a.get('categoria_original', a['material']) for a in im['anotaciones']}))[:39], fill='black', font=font)
        d.text((2, 279), 'PROPUESTAS; sin revisión humana', fill='black', font=font)
        sheet.paste(tile, ((n % 4) * 300, (n // 4) * 300))
    path = DATOS / 'contactos' / (fuente + '.jpg')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('No sobrescribir contacto')
    sheet.save(path, quality=90)


def main():
    if (PROYECTO / 'PROYECTO.json').exists():
        raise FileExistsError('Proyecto existente; conservar correcciones y originales')
    prev = originales_v27()
    nuevas = [*wsc(), *dwsd()]
    for fuente in ['wsc', 'dwsd']:
        contactos(fuente, [i for i in nuevas if i['fuente'] == fuente])
    duplicados = candidatos_duplicados(nuevas, [*historicos_train_val(), *prev])
    publicar(SALIDA / 'duplicados_nuevas_con_train_val_y_V27.json', duplicados)
    fotos = [*sorted(prev, key=lambda i: (0 if i['fuente'] == 'warp' else 1, i['id'])), *nuevas]
    proyecto = {'version': VERSION, 'coleccion_id': 'v27_2_piloto_80', 'imagenes': fotos,
                'propuestas_publicas_pendientes': True, 'aceptadas_V26': 0, 'cuotas_identidad_acreditadas': 0,
                'entrenamiento_autorizado': False, 'generalizacion_desbloqueada': False, 'test_original_evaluado': False}
    proyecto['huella'] = digest(proyecto)
    publicar(PROYECTO / 'PROYECTO.json', proyecto)
    realwaste = {}
    for split in ['train', 'val']:
        # No consultar la partición test.
        import json
        rs = [json.loads(l) for l in (RAIZ / f'datos/procesados/clasificacion/{split}.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        realwaste[split] = dict(Counter(r['categoria_original'] for r in rs if r['fuente'] == 'realwaste'))
    resumen = {'imagenes_preparadas': len(fotos), 'reutilizadas_V27': len(prev), 'nuevas_fotografias_publicadas_piloto': len(nuevas),
               'fuentes': dict(Counter(i['fuente'] for i in fotos)), 'anotaciones_publicas_pendientes': sum(len(i['anotaciones']) for i in fotos),
               'anotaciones_humanas': 0, 'revisiones_independientes': 0, 'aceptadas_V26': 0,
               'realwaste': {'ya_es_historico': True, 'conteos_train_val_historicos': realwaste,
                             'fotografias_nuevas_acreditadas': 0, 'mascaras_desde_etiquetas_globales': 0, 'test_consultado': False},
               'proyecto_sha256': sha(PROYECTO / 'PROYECTO.json'), 'entrenamiento': False, 'test_original_evaluado': False}
    publicar(SALIDA / 'piloto_preparado.json', resumen)
    print(resumen, flush=True)


if __name__ == '__main__':
    main()
