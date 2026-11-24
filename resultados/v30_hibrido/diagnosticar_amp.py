"""Aísla el primer overflow de memorización; no modifica parámetros tras el fallo."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
RAIZ = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.experimento_v30 import contexto, inicializar, SALIDA
from ecowaste.detector_mapas import perdida_mapas
from ecowaste.backend_numerico import aplicar_backend, politica
from ecowaste.incorporacion import leer
from ecowaste.reanudacion_regional import publicar
from ecowaste.ablacion_fondo import huella


def estado_grads(m):
    filas = []
    for n, p in m.named_parameters():
        if p.grad is None: continue
        g = p.grad.detach().float(); ok = torch.isfinite(g)
        filas.append({'nombre': n, 'elementos_no_finitos': int((~ok).sum()),
            'max_abs_finito': float(g[ok].abs().max()) if ok.any() else None})
    return {'todos_finitos': all(r['elementos_no_finitos'] == 0 for r in filas), 'parametros': filas}


def main():
    cfg, ht, nu, gl, pm = contexto(); del nu, gl
    ix = leer(SALIDA/'controles/seleccion_memorizacion.json')['indices']
    x, y, v = ht.imagenes[ix].cuda().float()/255, ht.mascaras[ix].cuda(), ht.validos[ix].cuda()
    historial = []
    with aplicar_backend(politica('completo')):
        m, opt, sc, rg, ra = inicializar('A', cfg)
        for k in range(240):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                raw = m(x, v); loss = perdida_mapas(raw, y, pm, coeficiente_fondo=3.)
            sc.scale(loss).backward(); sc.unscale_(opt)
            gs = estado_grads(m)
            historial.append({'paso': k+1, 'loss': float(loss.detach()), 'logits_finitos': bool(torch.isfinite(raw).all()),
                'loss_finita': bool(torch.isfinite(loss)), 'scaler': sc.state_dict(), 'gradientes_finitos': gs['todos_finitos']})
            if not gs['todos_finitos']:
                original = raw.detach().clone(); modos = {}
                for nombre, amp, escala in [('AMP_1024', True, 1024.), ('FP32_1', False, 1.)]:
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                        r = m(x, v); l = perdida_mapas(r, y, pm, coeficiente_fondo=3.)
                    (l*escala).backward()
                    for p in m.parameters():
                        if p.grad is not None: p.grad.div_(escala)
                    modos[nombre] = {'loss': float(l.detach()), 'logits_finitos': bool(torch.isfinite(r).all()),
                        'mismos_logits_AMP': bool(torch.equal(original, r)) if amp else None, 'gradientes': estado_grads(m)}
                publicar(SALIDA/'diagnostico_overflow_amp.json', {'paso_fallo': k+1, 'historial': historial,
                    'gradientes_fallo': gs, 'contrafactuales_mismos_pesos_y_entradas': modos,
                    'huella_entradas': huella([x, y, v]), 'sin_actualizaciones_adicionales_tras_fallo': True, 'pesos_descartados': True})
                print('Fallo', k+1, 'loss', float(loss.detach()), {n: r['gradientes']['todos_finitos'] for n, r in modos.items()}, flush=True)
                return
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True, foreach=False)
            sc.step(opt); sc.update()
        raise RuntimeError('No se reprodujo el fallo; no atribuir causa sin evidencia')


if __name__ == '__main__': main()
