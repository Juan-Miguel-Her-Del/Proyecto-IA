"""Contrato auxiliar V27.2: propuestas y correcciones, nunca aceptación V26.

Los IDs de instancia NO acreditan objetos físicos. Los diarios son inmutables,
con concurrencia optimista. Ninguna función congela datos ni autoriza entrenar.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ecowaste.incorporacion import (CLASES, IGNORADAS, identificador, poligonos_validos,
    caja_derivada, leer, publicar, digest, sha, bloqueo)

VERSION = 'ecowaste_auxiliares_publicos_v27_2_1'
MATERIALES = [*CLASES, 'ambiguo', 'compuesto', 'fuera_alcance', 'sin_asignar']
DESCONOCIDOS = ['objeto_fisico_id', 'sitio_id', 'sesion_id', 'dispositivo_id', 'capturado_en']


def rle_coco(mask):
    """RLE COCO sin comprimir, orden Fortran: conserva huecos y componentes."""
    m = np.asarray(mask, dtype=np.uint8)
    if m.ndim != 2 or not np.isin(m, [0, 1]).all():
        raise ValueError('Máscara binaria 2D requerida')
    a = m.ravel(order='F')
    cortes = np.flatnonzero(np.diff(a.astype(np.int8))) + 1
    counts = np.diff(np.r_[0, cortes, a.size]).astype(int).tolist()
    if a.size and a[0]:
        counts.insert(0, 0)
    return {'size': list(m.shape), 'counts': counts}


def contornos_mask(mask):
    m = np.asarray(mask, dtype=np.uint8)
    if m.ndim != 2 or not m.any():
        raise ValueError('SAM no produjo una región; añadir/cambiar indicaciones')
    cs, h = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    exteriores, huecos = [], []
    for i, c in enumerate(cs):
        p = cv2.approxPolyDP(c, .5, True).reshape(-1, 2)
        if len(p) < 3 or cv2.contourArea(p) <= 0:
            continue
        depth, parent = 0, int(h[0, i, 3])
        while parent != -1:
            depth += 1
            parent = int(h[0, parent, 3])
        (exteriores if depth % 2 == 0 else huecos).append(p.astype(float).ravel().tolist())
    if not exteriores:
        raise ValueError('Región demasiado pequeña para polígono; no fabricar geometría')
    return {'poligonos': exteriores, 'huecos': huecos, 'rle_ia_original': rle_coco(m),
            'componentes_ia': int(cv2.connectedComponents(m, connectivity=8)[0] - 1),
            'area_mask_ia': int(m.sum())}


def mascara_poligonos(a, width, height):
    im = Image.new('1', (width, height))
    d = ImageDraw.Draw(im)
    for p in a['poligonos']:
        d.polygon(list(zip(p[::2], p[1::2])), fill=1)
    for p in a.get('huecos', []):
        d.polygon(list(zip(p[::2], p[1::2])), fill=0)
    return np.asarray(im, dtype=np.uint8)


def normalizar(a, im, *, completa=False):
    if not isinstance(a, dict) or a.get('aceptada_V26') or a.get('aceptada'):
        raise ValueError('No se puede declarar aceptación automática ni V26')
    aid = identificador(a.get('id'))
    material = a.get('material', 'sin_asignar')
    if material not in MATERIALES:
        raise ValueError('Material fuera de taxonomía')
    motivo = str(a.get('motivo', '')).strip()
    descartada = a.get('descartada') is True
    if (descartada or material in IGNORADAS) and len(motivo) < 5:
        raise ValueError('Justificar descarte o ambigüedad')
    ps = a.get('poligonos', [])
    huecos = a.get('huecos', [])
    if not descartada:
        if ps:
            ps = poligonos_validos(ps, im['width'], im['height'])
        if huecos:
            huecos = poligonos_validos(huecos, im['width'], im['height'])
            if not ps:
                raise ValueError('Huecos sin contorno exterior')
            exterior = mascara_poligonos({'poligonos': ps}, im['width'], im['height'])
            agujeros = mascara_poligonos({'poligonos': huecos}, im['width'], im['height'])
            if (agujeros & (1 - exterior)).any():
                raise ValueError('Hueco fuera de región exterior')
        if completa and (not ps or material == 'sin_asignar'):
            raise ValueError('Falta polígono o material: guardar borrador o completar revisión')
    if ps and not descartada:
        area = int(mascara_poligonos({'poligonos': ps, 'huecos': huecos}, im['width'], im['height']).sum())
        if not area:
            raise ValueError('Máscara vacía tras excluir huecos')
    return {'id': aid, 'material': material, 'poligonos': ps, 'huecos': huecos,
            'bbox': caja_derivada(ps) if ps else a.get('bbox'), 'motivo': motivo,
            'descartada': descartada, 'propuesta_ia_id': a.get('propuesta_ia_id'),
            'origen': a.get('origen', 'edicion_usuario'), 'aceptada_V26': False}


class Taller:
    def __init__(self, proyecto, raiz):
        self.path, self.raiz = Path(proyecto).resolve(), Path(raiz).resolve()
        self.base = leer(self.path / 'PROYECTO.json')
        if self.base.get('version') != VERSION or self.base.get('huella') != digest({k: v for k, v in self.base.items() if k != 'huella'}):
            raise ValueError('Proyecto auxiliar incompatible o alterado')
        if self.base['entrenamiento_autorizado'] or self.base['aceptadas_V26'] or self.base['generalizacion_desbloqueada']:
            raise ValueError('Proyecto fuera de alcance')
        self.imagenes = {identificador(i['id']): i for i in self.base['imagenes']}
        if len(self.imagenes) != len(self.base['imagenes']):
            raise ValueError('IDs de imagen repetidos')
        for im in self.imagenes.values():
            if any(im.get(k) is not None for k in DESCONOCIDOS):
                raise ValueError('Este contrato conserva metadatos físicos como desconocidos')

    def imagen(self, iid):
        return self.imagenes[identificador(iid)]

    def archivo_imagen(self, iid, *, semantica=False):
        im = self.imagen(iid)
        rel = im.get('mascara_semantica') if semantica else im['archivo']
        if not rel:
            raise ValueError('No hay máscara semántica de referencia')
        p = (self.raiz / rel).resolve()
        if not p.is_relative_to(self.raiz) or not p.is_file():
            raise ValueError('Imagen fuera de raíz o ausente')
        esperado = im['sha256_mascara_semantica'] if semantica else im['sha256_archivo']
        if sha(p) != esperado:
            raise ValueError('Archivo de imagen alterado')
        return p

    def diario(self, iid):
        self.imagen(iid)
        anteriores, prev = [], None
        folder = self.path / 'revisiones' / digest(iid)[:24]
        for n, f in enumerate(sorted(folder.glob('*.json')), 1):
            ev = leer(f)
            if f.name != f'{n:06d}.json' or ev['numero'] != n or ev['anterior'] != prev or ev['imagen_id'] != iid:
                raise ValueError('Diario incompleto/reordenado')
            if ev['proyecto_sha256'] != self.base['huella'] or ev['imagen_sha256'] != self.imagen(iid)['sha256_archivo']:
                raise ValueError('Diario de otra imagen/proyecto')
            if ev['huella'] != digest({k: v for k, v in ev.items() if k != 'huella'}):
                raise ValueError('Corrección alterada')
            anteriores.append(ev)
            prev = sha(f)
        return anteriores

    def tarea(self, iid, *, independiente=False):
        im = self.imagen(iid)
        evs = self.diario(iid)
        normales = [e for e in evs if e['modo'] == 'asistida']
        ultimo = normales[-1] if normales else None
        completas = [e for e in normales if e['cobertura_completa']]
        base = completas[-1]['huella'] if completas else None
        instancias = [] if independiente else deepcopy(ultimo['anotaciones'] if ultimo else im['anotaciones'])
        indice_ia = self.path / 'PROPUESTAS_IA_INICIALES.json'
        if not independiente and not ultimo and indice_ia.exists():
            indice = leer(indice_ia)
            if indice['proyecto_sha256'] != self.base['huella']:
                raise ValueError('Índice IA de otro proyecto')
            for a in instancias:
                pid = indice['por_imagen'].get(iid, {}).get(a['id'])
                if pid:
                    p = leer(self.path / 'propuestas_ia' / (identificador(pid) + '.json'))
                    if p['imagen_id'] != iid or p['proyecto_sha256'] != self.base['huella'] or p['huella'] != digest({k: v for k, v in p.items() if k != 'huella'}):
                        raise ValueError('Propuesta IA alterada')
                    a.update({'poligonos': p['resultado']['poligonos'], 'huecos': p['resultado']['huecos'],
                              'bbox': caja_derivada(p['resultado']['poligonos']), 'propuesta_ia_id': pid,
                              'origen': 'bbox_publica_mas_SAM2_no_revisada'})
        estado = 'propuestas_pendientes' if not ultimo else 'corregida_usuario_unico' if ultimo['cobertura_completa'] else 'borrador_usuario'
        if ultimo and ultimo['cobertura_completa'] and any(e['modo'] == 'independiente' and e['cobertura_completa']
                and e['revision_base'] == ultimo['huella'] and e['persona_id'] != ultimo['persona_id'] for e in evs):
            estado = 'con_revision_independiente_pendiente_adjudicacion'
        return {'imagen': {k: v for k, v in im.items() if k != 'anotaciones'}, 'anotaciones': instancias,
                'estado': estado, 'revision': len(evs), 'revision_base': base,
                'modo': 'independiente' if independiente else 'asistida', 'aceptada_V26': False}

    def guardar(self, iid, datos, *, modo='asistida', revision_base=None):
        if modo not in ['asistida', 'independiente']:
            raise ValueError('Modo de revisión desconocido')
        im = self.imagen(iid)
        actor = identificador(datos.get('persona_id'))
        completa = datos.get('cobertura_completa') is True
        if datos.get('aceptada_V26') or datos.get('aceptada'):
            raise ValueError('La revisión auxiliar no acepta V26')
        self.archivo_imagen(iid)
        with bloqueo(self.path / 'registro.lock'):
            evs = self.diario(iid)
            if datos.get('revision') != len(evs):
                raise ValueError('Conflicto de versión; recargar sin sobrescribir correcciones')
            if modo == 'independiente':
                prev = [e for e in evs if e['modo'] == 'asistida' and e['cobertura_completa']]
                if not prev or revision_base != prev[-1]['huella'] or actor in {e['persona_id'] for e in prev}:
                    raise ValueError('Revisión independiente requiere otra persona y una base completa vigente')
            if not isinstance(datos.get('anotaciones'), list):
                raise ValueError('Lista de anotaciones requerida')
            anns = [normalizar(a, im, completa=completa) for a in datos['anotaciones']]
            if len({a['id'] for a in anns}) != len(anns):
                raise ValueError('IDs de instancia repetidos')
            previo = self.tarea(iid, independiente=modo == 'independiente')['anotaciones']
            ids = {a['id'] for a in anns}
            if any(a['id'] not in ids for a in previo):
                raise ValueError('Conservar instancia descartada con motivo; no borrarla sin rastro')
            for a in anns:
                if a['propuesta_ia_id'] is not None:
                    pid = identificador(a['propuesta_ia_id'])
                    p = leer(self.path / 'propuestas_ia' / (pid + '.json'))
                    if p['imagen_id'] != iid or p['proyecto_sha256'] != self.base['huella'] or p['huella'] != digest({k: v for k, v in p.items() if k != 'huella'}):
                        raise ValueError('Propuesta IA de otra imagen/proyecto')
            folder = self.path / 'revisiones' / digest(iid)[:24]
            anterior = sha(folder / f'{len(evs):06d}.json') if evs else None
            antes = {a['id']: a for a in previo}
            cambios = [{'id': a['id'], 'antes': antes.get(a['id']), 'despues': a} for a in anns if antes.get(a['id']) != a]
            evento = {'version': VERSION, 'numero': len(evs) + 1, 'anterior': anterior,
                      'proyecto_sha256': self.base['huella'], 'imagen_id': iid, 'imagen_sha256': im['sha256_archivo'],
                      'persona_id': actor, 'identidad_autodeclarada_no_verificada': True, 'modo': modo,
                      'revision_base': revision_base, 'cobertura_completa': completa, 'anotaciones': anns,
                      'cambios': cambios, 'registrado_UTC': datetime.now(timezone.utc).isoformat(),
                      'aceptada_V26': False, 'cuotas_identidad_acreditadas': 0, 'entrenamiento_autorizado': False}
            evento['huella'] = digest(evento)
            publicar(folder / f'{len(evs)+1:06d}.json', evento)
        return self.tarea(iid)

    def registrar_ia(self, iid, resultado, indicaciones, modelo):
        self.imagen(iid)
        pid = 'sam2_' + uuid.uuid4().hex
        r = {'id': pid, 'imagen_id': iid, 'proyecto_sha256': self.base['huella'],
             'modelo': modelo, 'indicaciones': indicaciones, 'resultado': resultado,
             'origen': 'propuesta_automatica_no_revision_humana', 'aceptada_V26': False,
             'registrado_UTC': datetime.now(timezone.utc).isoformat()}
        r['huella'] = digest(r)
        publicar(self.path / 'propuestas_ia' / (pid + '.json'), r)
        return r

    def exportar_coco(self):
        cats = [{'id': i + 1, 'name': m} for i, m in enumerate(MATERIALES)]
        cmap = {c['name']: c['id'] for c in cats}
        salida = {'info': {'version': VERSION, 'proyecto_sha256': self.base['huella'],
                          'estado': 'auxiliar_no_aceptado_V26_no_entrenar', 'entrenamiento_autorizado': False},
                  'images': [], 'annotations': [], 'categories': cats, 'physical_objects': []}
        for n, iid in enumerate(self.imagenes, 1):
            t = self.tarea(iid)
            im = t['imagen']
            salida['images'].append({'id': n, 'file_name': im['archivo'], 'width': im['width'], 'height': im['height'],
                    'eco': {**{k: im[k] for k in DESCONOCIDOS}, 'imagen_id_auxiliar': iid,
                            'sha256_archivo': im['sha256_archivo'], 'fuente': im['fuente'], 'licencia': im['licencia'],
                            'estado_revision_auxiliar': t['estado'], 'aceptada_V26': False,
                            'bloqueos_V26': ['metadatos_fisicos_desconocidos', 'sin_adjudicacion_formal'],
                            'revisiones_huellas': [e['huella'] for e in self.diario(iid)]}})
            for a in t['anotaciones']:
                if a.get('descartada') or not a.get('poligonos'):
                    continue
                mask = mascara_poligonos(a, im['width'], im['height'])
                salida['annotations'].append({'id': len(salida['annotations']) + 1, 'image_id': n,
                    'category_id': cmap[a['material']], 'segmentation': rle_coco(mask) if a.get('huecos') else a['poligonos'],
                    'bbox': caja_derivada(a['poligonos']), 'area': int(mask.sum()), 'iscrowd': 0,
                    'eco': {'id_auxiliar': a['id'], 'origen': a.get('origen'), 'propuesta_ia_id': a.get('propuesta_ia_id'),
                            'motivo_ambiguo': a.get('motivo'), 'objeto_fisico_id': None, 'aceptada_V26': False,
                            'requiere_revision_huecos_V26': bool(a.get('huecos'))}})
        return salida
