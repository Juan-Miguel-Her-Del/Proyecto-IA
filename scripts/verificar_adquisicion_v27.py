"""Certifica V27 sin red, entrenamiento, inferencia ni lectura de ejemplos test.

--publicar ejecuta la suite y el verificador V26 y escribe certificados nuevos.
Sin argumentos comprueba los certificados y sus archivos, sin repetir pruebas.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from ecowaste.incorporacion import leer, publicar, publicar_bytes, sha
from ecowaste.propuestas_publicas import imagen_verificada, propuesta, caja_yolo
from scripts.adquirir_v27 import solo_train
from scripts.investigar_v27 import DATOS, SALIDA


def exigir(condicion, mensaje):
    if not condicion:
        raise ValueError(mensaje)


def comprobar(raiz, hashes):
    for nombre, esperado in hashes.items():
        path = raiz / nombre
        exigir(path.is_file() and sha(path) == esperado, 'Archivo alterado/ausente: ' + nombre)


def controles():
    historicos = leer(SALIDA / 'historicos_protegidos.json')
    comprobar(RAIZ, historicos)  # Los manifiestos test se hashean, no se interpretan.
    checkpoints = leer(SALIDA / 'checkpoints_previos.json')
    actuales = {p.relative_to(RAIZ).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
                for p in (RAIZ / 'checkpoints').rglob('*.pt')}
    exigir(actuales == checkpoints, 'Checkpoints históricos alterados')
    cfg = leer(RAIZ / 'configuracion/adquisicion_publica_v27.json')
    for campo in ['aceptacion_humana_automatica', 'registro_de_cuotas_aceptadas',
                  'generalizacion_desbloqueada', 'entrenamiento_autorizado', 'test_evaluado']:
        exigir(cfg[campo] is False, 'Opción fuera de alcance: ' + campo)

    paquete = leer(DATOS / 'propuestas_pendientes.json')
    fotos = paquete['imagenes']
    exigir(len(fotos) == 60 and len({i['id'] for i in fotos}) == 60, 'Inventario de fotos incorrecto')
    exigir(paquete['inventario_V26_acreditado'] == 0 and paquete['no_es_IMPORTADO_V26']
           and paquete['generalizacion_bloqueada'], 'Se acreditaron datos sin revisión')
    recibos = {}
    for fuente, cantidad in [('zerowaste', 24), ('warp', 24), ('hpi', 12)]:
        recibo = leer(SALIDA / ('piloto_' + fuente + '.json'))
        exigir(recibo['split'] == 'train' and len(recibo['imagenes']) == cantidad
               and recibo['aceptadas'] == 0, 'Recibo fuera de alcance: ' + fuente)
        for r in recibo['imagenes']:
            exigir(r['archivo'] not in recibos, 'Foto repetida en recibos')
            recibos[r['archivo']] = r
            if fuente != 'hpi':
                exigir(solo_train(r['miembro_original']) and r['crc32_verificado_por_zipfile'],
                       'Miembro ZIP no train o CRC sin verificar')
    desconocidos = ['objeto_fisico_id', 'sesion_id', 'sitio_id', 'dispositivo_id', 'capturado_en']
    coco = leer(DATOS / 'zerowaste/piloto_coco_original.json')
    anns = {str(a['id']): a for a in coco['annotations']}
    categorias = {a['id']: a['name'] for a in coco['categories']}
    propuestas = []
    for im in fotos:
        rec = recibos[im['archivo']]
        meta = imagen_verificada(RAIZ / im['archivo'])
        exigir(all(im[k] == v for k, v in meta.items()), 'Imagen o metadatos alterados: ' + im['id'])
        exigir(meta['sha256_archivo'] == (rec.get('sha256') or rec['descarga']['sha256']),
               'Imagen no corresponde al recibo: ' + im['id'])
        exigir(all(im[k] is None for k in desconocidos), 'Identidades/capturas inventadas')
        exigir(im['estado'] == 'pendiente_humanos_y_metadatos', 'Aceptación no autorizada')
        if im['fuente'] == 'hpi':
            exigir(not im['propuestas'], 'GT orgánico fabricado')
            original = rec['etiquetas_nutricionales']['image']
            exigir((im['width'], im['height']) == (original['width'], original['height']),
                   'Dimensiones HPI incompatibles')
        for p in im['propuestas']:
            propuestas.append(p)
            exigir(p['aceptada'] is False and all(p[k] is None for k in desconocidos),
                   'Propuesta aceptada o identidad inventada')
            if im['fuente'] == 'zerowaste':
                a = anns[p['id_fuente']]
                exigir(str(a['image_id']) == im['id_original'] and a['segmentation'] == p['poligonos_originales']
                       and a['bbox'] == p['bbox_original'], 'Anotación COCO original alterada')
                esperado = propuesta('zerowaste', im['id_original'], a['id'], categorias[a['category_id']],
                                     (im['width'], im['height']), poligonos=a['segmentation'], bbox=a['bbox'])
                exigir(esperado == p, 'Derivación V26 no reproducible')
            elif im['fuente'] == 'warp':
                exigir(p['poligonos_originales'] is None and p['poligonos_normalizados_v26'] is None,
                       'Máscara fabricada desde caja WaRP')
                exigir(p['bbox_original'] == caja_yolo(p['yolo_original']['xywh_normalizado'], im['width'], im['height']),
                       'Caja YOLO no reproducible')
    exigir(len(propuestas) == 225, 'Conteo de propuestas incorrecto')
    exigir(dict(Counter(p['material_propuesto'] or 'revision_material' for p in propuestas)) ==
           {'plastico': 98, 'carton': 80, 'metal': 8, 'vidrio': 12, 'revision_material': 27}, 'Mapeo cambiado')
    tareas = leer(DATOS / 'tareas_humanas_sin_propuestas.json')
    exigir(not tareas['revisores'] and len(tareas['imagenes']) == 60, 'Revisores o tareas incompatibles')
    for tarea, im in zip(tareas['imagenes'], fotos):
        exigir(tarea['id'] == im['id'] and not tarea['instancias'] and not tarea['cobertura_completa']
               and tarea['objeto_fisico_id'] is None, 'Se simularon anotaciones/revisión humana')
    pre = leer(SALIDA / 'preflight_v26.json')
    exigir(pre['rechazo_correcto_sin_inventar_metadata'] and pre['error_real_V26'] and
           all(pre[k] == 0 for k in ['paquetes_importados', 'revisiones_humanas_simuladas', 'GT_aceptado']),
           'Preflight no conserva el bloqueo V26')
    plan = leer(SALIDA / 'plan_cuotas_pendientes.json')
    exigir(plan['train_total_pendiente'] == 1550 and plan['generalizacion_pendiente'] == 600
           and plan['organicos_utilizables_pendientes'] == 300 and not plan['V28_habilitada']
           and not plan['suficiente_revisada_congelada'], 'Cuotas/bloqueo incorrectos')
    exigir(all(v['aceptadas'] == 0 and v['pendientes'] == v['objetivo'] for v in plan['cuotas_train'].values()),
           'Propuestas descontadas como aceptadas')
    exigir(leer(SALIDA / 'duplicados_train_val_y_pilotos.json') == {'nuevos': [], 'historicos': []},
           'Resultado de duplicados incompatible con informe')
    # Verificación de Git independiente de las reglas declaradas en .gitignore.
    privadas = [im['archivo'] for im in fotos] + [p.relative_to(RAIZ).as_posix() for p in (DATOS / 'contactos').glob('*.jpg')]
    ignoradas = subprocess.run(['git', 'check-ignore', '-z', '--stdin'], cwd=RAIZ,
                               input=('\0'.join(privadas) + '\0').encode('utf-8'), capture_output=True)
    exigir(ignoradas.returncode == 0 and set(ignoradas.stdout.decode('utf-8').rstrip('\0').split('\0')) == set(privadas),
           'Fotografías/montajes sin exclusión Git')
    tracked = subprocess.check_output(['git', 'ls-files', '-z', '--', 'datos/nuevos/v27/externos'], cwd=RAIZ)
    exigir(not tracked, 'Hay archivos externos de terceros incluidos en Git')
    doc = RAIZ / 'documentacion/27_adquisicion_datasets_reales.md'
    for target in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
        if not target.startswith('http'):
            exigir((doc.parent / target).exists(), 'Enlace local roto: ' + target)
    return {'aprobada': True, 'historicos_sha256_intactos': len(historicos), 'checkpoints_estadisticas_intactas': len(checkpoints),
            'imagenes_reales_verificadas': 60, 'propuestas_originales': 225, 'GT_aceptado': 0, 'importadas_V26': 0,
            'fotografias_y_montajes_ignorados_Git': len(privadas), 'train_pendiente': 1550,
            'generalizacion_pendiente': 600, 'organicos_utilizables_pendientes': 300,
            'entrenamiento': False, 'inferencia': False, 'test_evaluado': False, 'V28_habilitada': False}


def codigo():
    paths = [RAIZ / 'ecowaste/propuestas_publicas.py', RAIZ / 'pruebas/test_adquisicion_v27.py',
             RAIZ / 'configuracion/adquisicion_publica_v27.json', RAIZ / 'documentacion/27_adquisicion_datasets_reales.md',
             RAIZ / 'datos/nuevos/v27/.gitignore']
    paths.extend((RAIZ / 'scripts').glob('*v27.py'))
    return {p.relative_to(RAIZ).as_posix(): sha(p) for p in paths}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--publicar', action='store_true')
    args = ap.parse_args()
    mp, vp = SALIDA / 'manifiesto_integridad.json', SALIDA / 'verificacion.json'
    resultado = controles()
    if args.publicar:
        exigir(not mp.exists() and not vp.exists(), 'Certificado existente: verificar sin --publicar')
        prueba = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'pruebas', '-v'], cwd=RAIZ,
                                text=True, encoding='utf-8', errors='replace', capture_output=True)
        log = prueba.stdout + '\n' + prueba.stderr
        publicar_bytes(SALIDA / 'pruebas_finales.log', log.encode('utf-8'))
        n = re.search(r'Ran (\d+) tests? in', log)
        aprobadas = prueba.returncode == 0 and n is not None and int(n[1]) == 261 and not re.search(r'OK \(skipped=|FAILED', log)
        publicar(SALIDA / 'pruebas_finales.json', {'aprobadas': aprobadas, 'ejecutadas': int(n[1]) if n else None,
                 'existentes_conservadas': 246, 'nuevas': 15, 'returncode': prueba.returncode,
                 'sha256_log': sha(SALIDA / 'pruebas_finales.log'), 'hashes_codigo_bajo_prueba': codigo()})
        exigir(aprobadas, 'Suite incompleta/no aprobada: consultar pruebas_finales.log')
        previo = subprocess.run([sys.executable, 'scripts/verificar_v26.py'], cwd=RAIZ,
                                text=True, encoding='utf-8', errors='replace', capture_output=True)
        publicar(SALIDA / 'verificacion_v26_conservada.json', {'aprobada': previo.returncode == 0,
                 'comando': 'scripts/verificar_v26.py', 'salida': previo.stdout, 'errores': previo.stderr,
                 'entrenamiento': False, 'test_evaluado': False})
        exigir(previo.returncode == 0, 'V26 no conserva integridad: ' + previo.stderr)
        resultado = controles()  # Revalidar históricos después de las pruebas.
        publicar(mp, {'version': 'v27_adquisicion_publica_1', 'codigo_y_documentacion': codigo(),
                 'archivos_locales_descargados_y_propuestas': {p.relative_to(DATOS).as_posix(): sha(p) for p in DATOS.rglob('*') if p.is_file()},
                 'resultados': {p.relative_to(SALIDA).as_posix(): sha(p) for p in SALIDA.rglob('*') if p.is_file() and p not in [mp, vp]},
                 'entrenamiento_autorizado': False, 'generalizacion_desbloqueada': False})
        resultado.update({'pruebas_aprobadas': 261, 'V26_integridad_aprobada': True, 'sha256_manifiesto': sha(mp)})
        publicar(vp, resultado)
    else:
        m = leer(mp)
        comprobar(RAIZ, m['codigo_y_documentacion'])
        comprobar(DATOS, m['archivos_locales_descargados_y_propuestas'])
        comprobar(SALIDA, m['resultados'])
        certificado = leer(vp)
        exigir(certificado['sha256_manifiesto'] == sha(mp), 'Manifiesto alterado')
        exigir(leer(SALIDA / 'pruebas_finales.json')['aprobadas'] and leer(SALIDA / 'verificacion_v26_conservada.json')['aprobada'],
               'Pruebas o integridad previa no aprobadas')
        resultado.update({'pruebas_aprobadas': 261, 'V26_integridad_aprobada': True, 'sha256_manifiesto': sha(mp)})
        exigir(resultado == certificado, 'Certificado no corresponde a controles actuales')
    print(json.dumps(resultado, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
