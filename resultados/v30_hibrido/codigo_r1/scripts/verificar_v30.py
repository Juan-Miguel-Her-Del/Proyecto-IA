"""Cierre V30: presupuestos, reanudación, archivos históricos y manifiesto."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
RAIZ = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(RAIZ))
import torch
from ecowaste.experimento_v30 import SALIDA, CONFIG, codigo, Diario
from ecowaste.incorporacion import leer, sha
from ecowaste.reanudacion_regional import publicar


def main():
    c = leer(SALIDA/'protocolo.json')
    if c['codigo'] != codigo() or c['config'] != leer(CONFIG): raise ValueError('Código/configuración distintos')
    protegidos = leer(SALIDA/'protegidos_antes.json')
    for i, (p, h) in enumerate(protegidos.items()):
        if sha(p) != h: raise ValueError('Histórico modificado: '+p)
        if (i+1) % 4000 == 0: print('Protegidos', i+1, '/', len(protegidos), flush=True)
    for p, h in c['datos'].items():
        if sha(RAIZ/p) != h: raise ValueError('Datos modificados')
    pruebas = leer(SALIDA/'pruebas.json')
    if not pruebas['aprobada'] or pruebas['codigo'] != codigo(): raise ValueError('Pruebas ajenas')
    controles = leer(SALIDA/'controles/reanudacion.json')
    if not controles['aprobada'] or controles['codigo'] != codigo(): raise ValueError('Controles ajenos')
    memo = leer(SALIDA/'controles/memorizacion.json')
    ramas = {}
    met = None
    if memo['aprobada'] and (SALIDA/'metricas.json').exists():
        met = leer(SALIDA/'metricas.json')
        for rama in ['A', 'B']:
            ck, _ = Diario(rama, c).leer()
            if ck['paso'] != c['config']['actualizaciones']: raise ValueError('Entrenamiento incompleto')
            ramas[rama] = ck
        if any(a['entradas_sha256'] != b['entradas_sha256'] for a, b in zip(ramas['A']['historial'], ramas['B']['historial'])):
            raise ValueError('Entradas A/B distintas')
        for n in ['Original', 'A', 'B']:
            for dom in ['historico_train', 'nuevo_train', 'desarrollo', 'externo_retenido']:
                r = met['resultados'][n][dom]
                if sha(RAIZ/r['analisis']) != r['sha256_analisis']: raise ValueError('Análisis alterado')
                folder = RAIZ/r['captura']; cert = leer(folder/'captura.json')
                for p, h in cert['archivos'].items():
                    if sha(folder/p) != h: raise ValueError('Captura/predicción alteradas')
    elif memo['aprobada']:
        gate = leer(SALIDA/'piloto_800/puerta.json')
        if gate['aprobada']:
            raise ValueError('Falta evaluación final; no declarar cierre completo')
        for rama in ['A', 'B']:
            ck, _ = Diario(rama, c).leer()
            if ck['paso'] != 800: raise ValueError('Parada distinta del protocolo')
            ramas[rama] = ck
    archivos = {**codigo(), CONFIG.relative_to(RAIZ).as_posix(): sha(CONFIG)}
    for p in SALIDA.rglob('*'):
        if p.is_file() and p.name not in ['integridad.json', 'recibo_integridad.json'] and p.suffix != '.lock':
            archivos[p.relative_to(RAIZ).as_posix()] = sha(p)
    for p in (RAIZ/'checkpoints/v30_hibrido').rglob('*.pt'):
        archivos[p.relative_to(RAIZ).as_posix()] = sha(p)
    doc = RAIZ/'documentacion/30_arquitectura_hibrida_cnn_transformer.md'
    archivos[doc.relative_to(RAIZ).as_posix()] = sha(doc)
    archivos['documentacion/01_decisiones_y_requisitos.md'] = sha(RAIZ/'documentacion/01_decisiones_y_requisitos.md')
    out = {'version': 'integridad_v30_1', 'aprobada': True, 'protegidos_verificados': len(protegidos),
        'checkpoints_previos_verificados': sum('checkpoints' in Path(p).parts for p in protegidos),
        'pruebas': pruebas['resultados'], 'diagnostico_aprende': memo['aprobada'],
        'pasos': {r: ck['paso'] for r, ck in ramas.items()}, 'test': False, 'generalizacion_desbloqueada': False,
        'predeterminado_modificado': False, 'pesos_externos': False, 'adopcion_automatica': False, 'archivos': archivos}
    publicar(SALIDA/'integridad.json', out)
    publicar(SALIDA/'recibo_integridad.json', {'sha256': sha(SALIDA/'integridad.json')})
    print({k: v for k, v in out.items() if k != 'archivos'}, flush=True)


if __name__ == '__main__': main()
