"""Evaluación prefijada V30: GT separado de inferencia, sin test ni ajuste."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse
import sys
import time
from types import SimpleNamespace
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import torch
from PIL import Image
from ecowaste.experimento_v30 import SALIDA, contexto, contrato, Diario
from ecowaste.detector_hibrido import DetectorHibrido, mascara_letterbox
from ecowaste.detector_mapas import DetectorMapas, cajas_desde_mapas
from ecowaste.datasets import letterbox, geometria_letterbox
from ecowaste.mosaico import ventanas, combinar_vistas
from ecowaste.incorporacion import leer, sha
from ecowaste.reanudacion_regional import publicar
from ecowaste.ablacion_fondo import huella, resumir_deteccion, metricas_semanticas
from ecowaste.backend_numerico import aplicar_backend, politica, registro_efectivo
from ecowaste.datos_v29 import cargar_coleccion
from ecowaste.evaluacion_detector import evaluar_detecciones
from ecowaste.evaluacion_v28_2 import geometria_posthoc, cargar_predicciones
from scripts.entrenar_detector_mapas import DatosMapas
from ecowaste.poligonos import ContornosTACO
from scripts.evaluar_v28_2 import preparar_regiones


@torch.inference_mode()
def inferir(m, x, gs):
    with torch.autocast('cuda', dtype=torch.float16):
        if isinstance(m, DetectorHibrido):
            return m(x, mascara_letterbox(gs, 'cuda'))
        return m(x)


@torch.inference_mode()
def capturar(m, ds, folder):
    meta = [{k: r[k] for k in ['id', 'ruta', 'ancho', 'alto']} for r in ds.registros]
    # Recibe fotografías/geometrías; nunca pasa objetivos GT al modelo.
    key = {'pesos': huella(m.state_dict()), 'imagenes': huella(ds.imagenes), 'metadatos': huella(meta), 'backend': registro_efectivo()}
    if (folder/'captura.json').exists():
        cert = leer(folder/'captura.json')
        if cert['clave'] != key: raise ValueError('Captura ajena')
        for p, h in cert['archivos'].items():
            if sha(folder/p) != h: raise ValueError('Captura alterada')
        return cert
    archivos = {}; costes = {}
    for v in [1, 5]:
        preds = []; inicio = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        for i in range(0, len(meta), 8 if v == 1 else 1):
            ms = meta[i:i+8] if v == 1 else [meta[i]]
            p = folder/f'logits_{v}v/{i:04d}.pt'
            if p.exists():
                d = torch.load(p, weights_only=True)
                if d['clave'] != key or d['metadatos'] != ms: raise ValueError('Parcial ajeno')
                raw, gs, zs = d['logits'].cuda(), d['geometrias'], d['zonas']
            else:
                if v == 1:
                    gs = [geometria_letterbox(r['ancho'], r['alto'], 384) for r in ms]
                    zs = None; x = ds.imagenes[i:i+8].cuda().float()/255
                else:
                    r = ms[0]
                    with Image.open(RAIZ/r['ruta']) as im: im = im.convert('RGB')
                    zs = ventanas(im.width, im.height)
                    xg = [letterbox(im.crop(z), 384) for z in zs]
                    x = torch.stack([a for a, _ in xg]).cuda(); gs = [g for _, g in xg]
                raw = inferir(m, x, gs)
                publicar(p, {'clave': key, 'metadatos': ms, 'logits': raw.cpu(), 'geometrias': gs, 'zonas': zs}, tensor=True)
            archivos[p.relative_to(folder).as_posix()] = sha(p)
            ps = cajas_desde_mapas(raw, .05, 2, gs, politica='argmax7')
            if v == 1: preds.extend(ps)
            else: preds.append(combinar_vistas(ps, gs, zs, ms[0]['ancho'], ms[0]['alto'], 384, .05))
            if (i+len(ms)) % 100 == 0 or i+len(ms) == len(meta):
                print(folder.parent.name, folder.name, v, i+len(ms), '/', len(meta), flush=True)
        serial = [{'id': r['id'], **{k: t.tolist() for k, t in p.items()}} for r, p in zip(meta, preds)]
        q = folder/f'predicciones_{v}v.json'
        if q.exists():
            if leer(q) != serial: raise ValueError('Predicciones de reanudación distintas')
        else: publicar(q, serial)
        archivos[q.name] = sha(q); torch.cuda.synchronize()
        costes[v] = {'segundos_incluye_disco_y_postprocesamiento': time.perf_counter()-inicio,
                     'pico_gpu_bytes': torch.cuda.max_memory_allocated(), 'imagenes': len(meta)}
    cert = {'clave': key, 'archivos': archivos, 'costes': costes, 'gt_utilizado': False}
    publicar(folder/'captura.json', cert)
    return cert


def analizar(folder, ds, regiones):
    cm = torch.zeros(7, 7, dtype=torch.long); inst = cm.clone(); cond = torch.zeros(6, 6, dtype=torch.long)
    for k in range(0, len(ds.registros), 8):
        raw = torch.load(folder/f'logits_1v/{k:04d}.pt', weights_only=True)['logits'].float()
        y = ds.mascaras[k:k+8]; pred = raw.argmax(1); ok = y != 255
        cm += torch.bincount(7*y[ok]+pred[ok], minlength=49).reshape(7, 7)
        for j in range(len(raw)):
            for cl, mask, ident in regiones[k+j]:
                val = torch.as_tensor(mask, dtype=torch.bool) & (y[j] == cl+1)
                if val.any():
                    inst[cl+1, int(torch.bincount(pred[j][val], minlength=7).argmax())] += 1
                    cond[cl, int(torch.bincount(raw[j, 1:].argmax(0)[val], minlength=6).argmax())] += 1
    out = {'cabeza_principal': {'semantica': metricas_semanticas(cm), 'matriz_instancias_con_fondo': inst.tolist(),
        'matriz_instancias_condicional_material': cond.tolist()}}
    for v in [1, 5]:
        ps = cargar_predicciones(folder, v)
        out[str(v)] = {'metricas': resumir_deteccion(evaluar_detecciones(ps, ds.objetivos)), 'geometria': geometria_posthoc(ps, ds.objetivos)}
    return out


@torch.inference_mode()
def benchmark(m, ds):
    # Primeras 12 imágenes por ID: selección ajena a las predicciones.
    ids = sorted(range(len(ds.registros)), key=lambda i: ds.registros[i]['id'])[:12]
    out = {}
    for v in [1, 5]:
        tiempos = []; torch.cuda.reset_peak_memory_stats()
        for rep, i in enumerate(ids+ids[:4]):
            r = ds.registros[i]
            with Image.open(RAIZ/r['ruta']) as im: im = im.convert('RGB')
            zs = [(0, 0, im.width, im.height)] if v == 1 else ventanas(im.width, im.height)
            xgs = [letterbox(im.crop(z), 384) for z in zs]
            x = torch.stack([x for x, _ in xgs]).cuda(); gs = [g for _, g in xgs]
            torch.cuda.synchronize(); ti = time.perf_counter()
            raw = inferir(m, x, gs)
            ps = cajas_desde_mapas(raw, .05, 2, gs)
            if v == 5: combinar_vistas(ps, gs, zs, im.width, im.height, 384, .05)
            torch.cuda.synchronize()
            # Cuatro iniciales de calentamiento, 12 mediciones; excluye lectura y letterbox.
            if rep >= 4: tiempos.append((time.perf_counter()-ti)*1000)
        out[v] = {'ms_media_modelo_y_postproceso': sum(tiempos)/len(tiempos), 'mediciones': len(tiempos),
                  'pico_gpu_bytes': torch.cuda.max_memory_allocated()}
    return out


def main():
    cfg, ht, nu, gl, pm = contexto(); del gl, pm
    plan = torch.load(SALIDA/'plan.pt', weights_only=True)
    c = contrato(cfg, plan)
    if leer(SALIDA/'protocolo.json') != c: raise ValueError('Integridad cambiada')
    ramas = {}
    for rama in ['A', 'B']:
        ck, _ = Diario(rama, c).leer()
        if ck is None or ck['paso'] != cfg['actualizaciones']: raise ValueError('Rama incompleta')
        ramas[rama] = ck
    if not all(a['entradas_sha256'] == b['entradas_sha256'] for a, b in zip(ramas['A']['historial'], ramas['B']['historial'])):
        raise ValueError('Entradas A/B distintas')
    val_base = DatosMapas('val', ContornosTACO(RAIZ, 1/64))
    va = SimpleNamespace(registros=val_base.base.origen.registros, objetivos=val_base.base.objetivos,
                         imagenes=val_base.base.imagenes, mascaras=val_base.mascaras)
    ret = cargar_coleccion(RAIZ, RAIZ/cfg['coleccion'], 'val')
    datasets = {'historico_train': ht, 'nuevo_train': nu, 'desarrollo': va, 'externo_retenido': ret}
    regiones = {'historico_train': preparar_regiones(ht), 'desarrollo': preparar_regiones(va)}
    for name, ds in [('nuevo_train', nu), ('externo_retenido', ret)]:
        regiones[name] = [[(cl, mask.numpy(), ident) for cl, mask, ident in rs] for rs in ds.regiones]
    ejemplos = []
    for nombre, ds in datasets.items():
        for cl in [2, 3, 5, 4, 1, 0]:
            ix = [(r['id'], i) for i, r in enumerate(ds.registros) if bool((ds.objetivos[i]['labels'] == cl).any())]
            if ix: ejemplos.append({'dominio': nombre, 'clase': cl, 'indice': min(ix)[1], 'id': min(ix)[0]})
    p = SALIDA/'ejemplos_prefijados.json'
    if not p.exists(): publicar(p, ejemplos)
    elif leer(p) != ejemplos: raise ValueError('Selección de ejemplos cambió')
    res = {}; origen = torch.load(RAIZ/cfg['checkpoint_referencia'], weights_only=True)
    v29 = leer(RAIZ/'resultados/v29_packwise_click/metricas.json')
    with aplicar_backend(politica('completo')):
        for nombre, ck in [('Original', origen), *ramas.items()]:
            m = DetectorMapas(**ck['configuracion']) if nombre == 'Original' else DetectorHibrido(**ck['configuracion_hibrido'])
            m.cuda().eval(); m.load_state_dict(ck['pesos'], strict=True); res[nombre] = {}
            for dom, ds in datasets.items():
                folder = SALIDA/'evaluacion'/nombre/dom
                # Reutilización de la referencia comprobando cada archivo, pesos e inputs.
                equivalencia = {'historico_train': 'historico_train', 'nuevo_train': 'nuevo_train', 'externo_retenido': 'nuevo_retenido'}
                if nombre == 'Original' and dom in equivalencia:
                    prev = v29['resultados']['Original'][equivalencia[dom]]
                    q = RAIZ/prev['analisis']; folder = q.parent
                    if sha(q) != prev['sha256_analisis']: raise ValueError('Referencia analizada alterada')
                    cert = leer(folder/'captura.json')
                    if cert['clave']['pesos'] != huella(m.state_dict()) or cert['clave']['entradas'] != huella(ds.imagenes): raise ValueError('Referencia ajena')
                    for p, h in cert['archivos'].items():
                        if sha(folder/p) != h: raise ValueError('Referencia capturada alterada')
                    an = leer(q)
                else:
                    cert = capturar(m, ds, folder)
                    q = folder/'analisis.json'
                    if q.exists(): an = leer(q)
                    else:
                        an = analizar(folder, ds, regiones[dom]); publicar(q, an)
                res[nombre][dom] = {'metricas': {v: an[v]['metricas'] for v in ['1', '5']},
                    'analisis': q.relative_to(RAIZ).as_posix(), 'sha256_analisis': sha(q), 'costes': cert['costes'],
                    'captura': folder.relative_to(RAIZ).as_posix(), 'referencia_reutilizada': nombre == 'Original' and dom in equivalencia}
                print('Métricas', nombre, dom, {v: res[nombre][dom]['metricas'][v]['map50'] for v in ['1', '5']}, flush=True)
            res[nombre]['benchmark'] = benchmark(m, va)
            del m; torch.cuda.empty_cache()
        b, a, o = [res[n]['desarrollo']['metricas']['5'] for n in ['B', 'A', 'Original']]
        gate = (b['map50'] >= max(a['map50'], o['map50'])+.005 and b['recall50'] >= o['recall50'] and b['fp50'] <= 1.10*o['fp50'] and
                res['B']['externo_retenido']['metricas']['5']['map50'] >= res['A']['externo_retenido']['metricas']['5']['map50'])
        publicar(SALIDA/'metricas.json', {'resultados': res, 'codigo': c['codigo'], 'backend': registro_efectivo(),
            'entradas_A_B_identicas': True, 'puerta_integracion_superada': gate, 'test': False, 'generalizacion_desbloqueada': False,
            'adopcion': False, 'advertencia': 'Val histórica es desarrollo. Retenido externo carece de cartón, vidrio y orgánicos; independencia física no demostrada.'})


def evaluar_piloto():
    """Puerta prefijada a 800, antes de autorizar los pasos restantes."""
    cfg, ht, nu, gl, pm = contexto()
    del ht, nu, gl, pm
    plan = torch.load(SALIDA/'plan.pt', weights_only=True)
    c = contrato(cfg, plan)
    if c != leer(SALIDA/'protocolo.json'): raise ValueError('Protocolo distinto')
    base = DatosMapas('val', ContornosTACO(RAIZ, 1/64)).base
    resultados = {}
    with aplicar_backend(politica('completo')):
        for rama in ['A', 'B']:
            ck, _ = Diario(rama, c).leer()
            if ck is None or ck['paso'] != 800: raise ValueError('Evaluar únicamente piloto A/B 800 terminado')
            m = DetectorHibrido(**ck['configuracion_hibrido']).cuda().eval()
            m.load_state_dict(ck['pesos'], strict=True)
            ps = []; archivos = {}
            for k in range(0, len(base), 8):
                gs = [o['geometria'] for o in base.objetivos[k:k+8]]
                raw = inferir(m, base.imagenes[k:k+8].cuda().float()/255, gs)
                ps.extend(cajas_desde_mapas(raw, .05, 2, gs))
                p = SALIDA/f'piloto_800/{rama}/logits/{k:04d}.pt'
                publicar(p, {'logits': raw.cpu(), 'geometrias': gs, 'ids': [r['id'] for r in base.origen.registros[k:k+8]],
                    'pesos_sha256': huella(m.state_dict()), 'gt_utilizado': False}, tensor=True)
                archivos[p.relative_to(SALIDA).as_posix()] = sha(p)
            publicar(SALIDA/f'piloto_800/{rama}/predicciones.json', [{k: v.tolist() for k, v in p.items()} for p in ps])
            resultados[rama] = resumir_deteccion(evaluar_detecciones(ps, base.objetivos))
            publicar(SALIDA/f'piloto_800/{rama}/resultado.json', {'metricas': resultados[rama], 'archivos': archivos, 'backend': registro_efectivo()})
            print('Piloto 800', rama, resultados[rama], flush=True)
            del m; torch.cuda.empty_cache()
    crit = cfg['puerta_800']; b = resultados['B']
    ok = b['map50'] >= crit['map50_minimo_B'] and b['recall50'] >= crit['recall_minimo_B']
    publicar(SALIDA/'piloto_800/puerta.json', {'aprobada': ok, 'criterio': crit, 'resultados': resultados,
        'contrato': c, 'unica_evaluacion_prefijada': True, 'ampliar_si_falla': False,
        'limitacion': 'Desarrollo histórico sin gradientes V30; utilizado antes por el proyecto, no independencia nueva.'})
    if not ok: print('Puerta no superada: no iniciar las actualizaciones restantes', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--piloto-800', action='store_true')
    a = p.parse_args()
    if a.piloto_800: evaluar_piloto()
    else: main()
