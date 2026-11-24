"""V30: controles, memorización descartable y presupuesto progresivo fijo."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import uuid
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.experimento_v30 import (SALIDA, CONFIG, codigo, contexto, planificar, contrato,
    inicializar, paso, snapshot, comprobar, Diario)
from ecowaste.detector_mapas import perdida_mapas
from ecowaste.detector_hibrido import crear_pareja
from ecowaste.backend_numerico import aplicar_backend, politica, registro_efectivo
from ecowaste.reproducibilidad_v20 import comparar_arbol, cpu
from ecowaste.reanudacion_regional import publicar, bloqueo
from ecowaste.incorporacion import leer, sha


def congelar():
    cfg, ht, nu, gl, pm = contexto()
    plan = planificar(cfg, ht, nu, gl)
    publicar(SALIDA/'plan.pt', plan, tensor=True)
    publicar(SALIDA/'protocolo.json', contrato(cfg, plan))
    modelos = crear_pareja(cfg['semilla'])
    arquitectura = {}
    for nombre, modelo in zip(['A', 'B'], modelos):
        arquitectura[nombre] = {'parametros': sum(p.numel() for p in modelo.parameters()),
            'parametros_cnn': sum(p.numel() for n, p in modelo.named_parameters() if n.startswith('backbone.')),
            'parametros_transformer': sum(p.numel() for n, p in modelo.named_parameters() if n.startswith(('detalle.', 'contexto.'))),
            'inicializacion': 'aleatoria propia', 'pesos_externos': False, 'checkpoint_ep19_utilizado': False}
    publicar(SALIDA/'arquitectura_medida.json', arquitectura)
    # Copia de requisitos anterior a la actualización expresamente autorizada.
    p = RAIZ/'documentacion/01_decisiones_y_requisitos.md'
    from ecowaste.incorporacion import publicar_bytes
    respaldo = SALIDA/'requisitos_previos.md'
    if not respaldo.exists():
        publicar_bytes(respaldo, p.read_bytes())
    if sha(respaldo) != '30df8614ea377ac54088d6bc3cbe1436d51279ed2e62b24ca2a93bb793ee38dc':
        raise ValueError('Respaldo original de requisitos distinto')
    publicar(SALIDA/'actualizacion_requisitos.json', {'respaldo': respaldo.name, 'sha256_previo': sha(respaldo),
        'sha256_actual': sha(p), 'autorizacion': 'CNN propia, pesos aleatorios; prohibidos preentrenados y servicios detectores',
        'fuente': 'corrección V30 comunicada por el usuario'})
    datos = {str(RAIZ/'configuracion/arquitectura.json'): sha(RAIZ/'configuracion/arquitectura.json'), str(p): sha(p)}
    for folder in ['checkpoints', 'datos/nuevos/v27_2/proyecto/seleccion_72']:
        for q in (RAIZ/folder).rglob('*'):
            if q.is_file():
                datos[str(q)] = sha(q)
    # Documentos, fuentes, manifiestos y resultados existentes; excluye V30.
    for folder in ['ecowaste', 'scripts', 'pruebas', 'documentacion', 'configuracion', 'resultados', 'datos/procesados', 'datos/nuevos/v29/coleccion']:
        for q in (RAIZ/folder).rglob('*'):
            if q.is_file() and q.suffix in ['.py', '.json', '.jsonl', '.md', '.csv', '.yaml', '.png', '.jpg'] and '__pycache__' not in q.parts and 'v30' not in q.name and 'v30_hibrido' not in q.parts and q != p:
                datos[str(q)] = sha(q)
    publicar(SALIDA/'protegidos_antes.json', datos)
    print('Protocolo congelado y protegidos', len(datos), flush=True)


def controles():
    cfg, ht, nu, gl, pm = contexto()
    plan = torch.load(SALIDA/'plan.pt', weights_only=True)
    c = contrato(cfg, plan)
    if leer(SALIDA/'protocolo.json') != c:
        raise ValueError('Código/datos cambiados tras congelación')
    resultados = {}
    with aplicar_backend(politica('completo')):
        for rama in ['A', 'B']:
            ref, comps, trazas = [], [], {}
            with tempfile.TemporaryDirectory(prefix='descartar_v30_', dir=SALIDA) as td:
                for camino in ['C1', 'C2', 'R']:
                    m, opt, sc, rg, ra = inicializar(rama, cfg)
                    hist = []
                    for k in range(4):
                        tr, detalle = paso(m, opt, sc, ht, nu, gl, pm, plan, k, rg, ra, True)
                        hist.append(tr)
                        if camino == 'C1':
                            ref.append(detalle)
                        else:
                            comp = comparar_arbol(ref[k], detalle)
                            publicar(SALIDA/f'controles/{rama}/{camino}_paso{k+1}.json', comp)
                            comps.append({'camino': camino, 'paso': k+1, 'exacto': comp['exacto'], 'tolerancia': comp['dentro_tolerancia']})
                        if camino == 'R' and k == 1:
                            ck = snapshot(m, opt, sc, rg, ra, rama, 2, hist, c, None)
                            p = Path(td)/'paso2.pt'; publicar(p, ck, tensor=True)
                            ck = torch.load(p, weights_only=True)
                            comprobar(ck, rama, c)
                            del m, opt, sc
                            m, opt, sc, rg, ra = inicializar(rama, cfg, ck)
                    trazas[camino] = hist
                    del m, opt, sc
                    torch.cuda.empty_cache()
            ok = all(x['tolerancia'] for x in comps)
            resultados[rama] = {'aprobada': ok, 'comparaciones': comps, 'trazas': trazas,
                                'pesos_descartados': True, 'rtol': 1e-4, 'atol': 1e-5}
            if not ok:
                publicar(SALIDA/'controles/fallo.json', resultados)
                raise RuntimeError('Divergencia; no entrenar')
        iguales = all(a['entradas_sha256'] == b['entradas_sha256'] for a, b in zip(resultados['A']['trazas']['C1'], resultados['B']['trazas']['C1']))
        if not iguales:
            raise RuntimeError('Entradas A/B distintas')
        publicar(SALIDA/'controles/reanudacion.json', {'aprobada': True, 'ramas': resultados, 'entradas_A_B_iguales': iguales, 'codigo': codigo(), 'backend': registro_efectivo()})
    print('C1/C2/R aprobados', flush=True)


def memorizar():
    cfg, ht, nu, gl, pm = contexto()
    if leer(SALIDA/'protocolo.json')['codigo'] != codigo():
        raise ValueError('Código cambió')
    if not leer(SALIDA/'controles/reanudacion.json')['aprobada']:
        raise ValueError('Controles pendientes')
    # Selección por GT train, anterior a resultados: una escena por clase y
    # dos escenas con al menos dos objetos del mismo material, sin crear GT.
    ids = []
    for cl in [5, 3, 2, 4, 1, 0]:
        candidatos = [(r['id'], i) for i, r in enumerate(ht.registros) if bool((ht.objetivos[i]['labels'] == cl).any()) and bool((ht.mascaras[i] == cl+1).any())]
        i = min(candidatos)[1]
        if i not in ids: ids.append(i)
    cercanos = []
    for i, o in enumerate(ht.objetivos):
        centros = (o['boxes'][:, :2]+o['boxes'][:, 2:])/2
        cerca = any(int(o['labels'][a]) == int(o['labels'][b]) and float((centros[a]-centros[b]).norm()) <= .20
                    for a in range(len(centros)) for b in range(a+1, len(centros)))
        if cerca:
            cercanos.append((ht.registros[i]['id'], i))
    for _, i in sorted(cercanos):
        if i not in ids:
            ids.append(i)
        if len(ids) >= 8: break
    publicar(SALIDA/'controles/seleccion_memorizacion.json', {'indices': ids, 'ids': [ht.registros[i]['id'] for i in ids],
        'regla': 'mínimo ID por material + escenas con dos objetos de igual material, distancia de centros <=0.20 del lienzo', 'no_evaluacion_independiente': True})
    x = ht.imagenes[ids].cuda().float()/255; y = ht.mascaras[ids].cuda(); v = ht.validos[ids].cuda()
    resultados = {}
    with aplicar_backend(politica('completo')):
        for rama in ['A', 'B']:
            m, opt, sc, rg, ra = inicializar(rama, cfg)
            historia = []; inicio = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            for k in range(cfg['memorizacion']['pasos']):
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    raw = m(x, v); loss = perdida_mapas(raw, y, pm, coeficiente_fondo=3.)
                sc.scale(loss).backward(); sc.unscale_(opt)
                norma = torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True, foreach=False)
                sc.step(opt); sc.update()
                fg = (y > 0) & (y != 255)
                acc = float((raw.argmax(1)[fg] == y[fg]).float().mean())
                historia.append({'paso': k+1, 'loss': float(loss.detach()), 'accuracy_fg': acc, 'norma': float(norma)})
                if (k+1) % 40 == 0: print('Memorización', rama, historia[-1], flush=True)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                final = m(x, v)
            por = {}
            for cl in range(1, 7):
                mask = y == cl
                por[cl] = {'tokens': int(mask.sum()), 'accuracy': float((final.argmax(1)[mask] == cl).float().mean()) if mask.any() else None}
            reduccion = 1-sum(h['loss'] for h in historia[-10:])/sum(h['loss'] for h in historia[:10])
            ok = (reduccion >= cfg['memorizacion']['reduccion_loss_minima'] and historia[-1]['accuracy_fg'] >= cfg['memorizacion']['accuracy_fg_minima']
                  and all(p['accuracy'] is not None and p['accuracy'] >= cfg['memorizacion']['accuracy_por_material_minima'] for p in por.values()))
            resultados[rama] = {'aprobada': ok, 'reduccion_loss': reduccion, 'por_material': por, 'historia': historia,
                'segundos': time.perf_counter()-inicio, 'memoria_max_bytes': torch.cuda.max_memory_allocated(), 'pesos_descartados': True}
            del m, opt, sc
            torch.cuda.empty_cache()
    publicar(SALIDA/'controles/memorizacion.json', {'aprobada': all(r['aprobada'] for r in resultados.values()), 'ramas': resultados, 'codigo': codigo()})
    if not all(r['aprobada'] for r in resultados.values()):
        raise RuntimeError('No aprende en pocas escenas; entrenamiento bloqueado')


def entrenar(rama, hasta):
    cfg, ht, nu, gl, pm = contexto()
    plan = torch.load(SALIDA/'plan.pt', weights_only=True)
    c = contrato(cfg, plan)
    if c != leer(SALIDA/'protocolo.json') or hasta not in cfg['checkpoints'][1:]:
        raise ValueError('Protocolo o presupuesto cambiados')
    for n in ['reanudacion', 'memorizacion']:
        cert = leer(SALIDA/f'controles/{n}.json')
        if not cert['aprobada'] or cert['codigo'] != codigo():
            raise ValueError('Controles pendientes o ajenos')
    tests = leer(SALIDA/'pruebas.json')
    if not tests['aprobada'] or tests['codigo'] != codigo():
        raise ValueError('Pruebas pendientes/código modificado')
    d = Diario(rama, c)
    with bloqueo(d.base/'entrenar.lock'), aplicar_backend(politica('completo')):
        ck, previo = d.leer()
        m, opt, sc, rg, ra = inicializar(rama, cfg, ck)
        hist = copy.deepcopy(ck['historial']) if ck else []
        k0 = ck['paso'] if ck else 0
        if not ck:
            previo = d.guardar(snapshot(m, opt, sc, rg, ra, rama, 0, hist, c, previo))
        if k0 >= hasta:
            raise ValueError('Etapa ya completada')
        if k0 >= 200:
            gate = leer(d.base/'puerta_200.json')
            if not gate['aprobada']:
                raise ValueError('Puerta de aprendizaje fallida')
        elif hasta > 200:
            raise ValueError('Primero completar etapa 200 y comprobar puerta')
        if hasta > 800:
            gate = leer(SALIDA/'piloto_800/puerta.json')
            if not gate['aprobada'] or gate['contrato'] != c:
                raise ValueError('Evaluación de imágenes no usadas en actualizaciones pendiente/fallida')
            if k0 < 800:
                raise ValueError('Primero terminar piloto 800 A/B y evaluar')
        intento = uuid.uuid4().hex
        inicio = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        publicar(d.base/f'intentos/{intento}_inicio.json', {'paso': k0, 'backend': registro_efectivo(opt), 'huellas_restauradas': ck['huellas'] if ck else None})
        for k in range(k0, hasta):
            tr, _ = paso(m, opt, sc, ht, nu, gl, pm, plan, k, rg, ra)
            p = d.base/f'trazas/{intento}/paso_{k+1:04d}.json'
            publicar(p, tr)
            hist.append({**tr, 'traza': p.relative_to(RAIZ).as_posix(), 'sha256_traza': sha(p)})
            if k+1 in cfg['checkpoints']:
                previo = d.guardar(snapshot(m, opt, sc, rg, ra, rama, k+1, hist, c, previo))
            if (k+1) % 50 == 0:
                print(rama, k+1, '/', cfg['actualizaciones'], 'loss', round(tr['loss_total'], 5), flush=True)
        torch.cuda.synchronize()
        publicar(d.base/f'intentos/{intento}_fin.json', {'paso': hasta, 'actualizaciones': hasta-k0,
            'segundos': time.perf_counter()-inicio, 'memoria_gpu_max_bytes': torch.cuda.max_memory_allocated(), 'commit': previo})
        if hasta == 200:
            r = 1-(sum(h['loss_total'] for h in hist[-25:])/sum(h['loss_total'] for h in hist[:25]))
            publicar(d.base/'puerta_200.json', {'aprobada': r >= cfg['puerta_200']['reduccion_loss_minima'], 'reduccion_loss': r, 'train_only': True})
            if r < cfg['puerta_200']['reduccion_loss_minima']:
                raise RuntimeError('Puerta 200 fallida; no ampliar')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fase', choices=['congelar', 'controles', 'memorizar', 'entrenar'], required=True)
    p.add_argument('--rama', choices=['A', 'B']); p.add_argument('--hasta', type=int, choices=[200, 800, 1600, 2400], default=200)
    a = p.parse_args()
    try:
        if a.fase == 'congelar': congelar()
        elif a.fase == 'controles': controles()
        elif a.fase == 'memorizar': memorizar()
        else: entrenar(a.rama, a.hasta)
    except Exception:
        import traceback
        publicar(SALIDA/f'incidencias/{uuid.uuid4().hex}.json', {'error': traceback.format_exc(), 'entrenamiento_detenido': True})
        raise


if __name__ == '__main__': main()
