"""Smoke CUDA con AdamW real; pesos descartados, sin evaluación val/test."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.experimento_v30 import contexto, planificar, inicializar, paso, SALIDA
from ecowaste.backend_numerico import aplicar_backend, politica
from ecowaste.reanudacion_regional import publicar


def main():
    cfg, ht, nu, gl, pm = contexto(); plan = planificar(cfg, ht, nu, gl)
    out = {}
    with aplicar_backend(politica('completo')):
        for rama in ['A', 'B']:
            m, opt, sc, rg, ra = inicializar(rama, cfg)
            tr, _ = paso(m, opt, sc, ht, nu, gl, pm, plan, 0, rg, ra)
            if any(tr['normas_modulos'][n] <= 0 for n in ['detalle', 'contexto', 'cabezal']+(['backbone'] if rama == 'B' else [])):
                raise RuntimeError('Gradiente desconectado')
            out[rama] = tr
            print(rama, tr, flush=True)
            del m, opt, sc; torch.cuda.empty_cache()
    publicar(SALIDA/'smoke_cuda.json', {'aprobada': True, 'ramas': out, 'pesos_descartados': True})


if __name__ == '__main__': main()
