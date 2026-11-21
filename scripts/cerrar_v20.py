"""Manifiesto final de preparación: no entrena y no evalúa datos retenidos."""
import json
from pathlib import Path
import re
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.reanudacion_regional import publicar,sha
from scripts.preparar_v20 import CONFIG,leer_cfg,huellas_codigo,protocolo


def leer(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    cfg=leer_cfg();base=RAIZ/'resultados/v20_preparacion';destino=base/'entrega.json'
    if destino.exists():raise FileExistsError('Entrega existente')
    d=leer(base/'diagnostico.json');v=leer(base/'verificacion.json');inf=leer(base/'informe.json')
    assert d['aprobada'] and v['aprobada'] and inf['aprobada']
    assert d['hashes_codigo']==v['hashes_codigo']==huellas_codigo()
    assert d['protocolo']==protocolo(cfg) and not cfg['entrenamiento_completo_habilitado']
    assert v['sha256_diagnostico']==sha(base/'diagnostico.json')
    assert not d['validacion_evaluada_en_v20'] and not d['test_evaluado'] and d['actualizaciones_ab_ejecutadas']==0
    previos=leer(base/'protegidos.json');assert all(sha(RAIZ/p)==h for p,h in previos.items())
    assert not list((RAIZ/'checkpoints').glob(cfg['nombre']+'*'))
    doc=RAIZ/'documentacion/20_aprendizaje_discriminativo_mapas.md'
    assert sha(doc)==inf['sha256_documento']
    assert sha(RAIZ/'scripts/informar_v20.py')==inf['sha256_generador']
    assert all(sha(RAIZ/p)==h for p,h in inf['figuras'].items())
    for link in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):assert (doc.parent/link).resolve().exists(),link
    assert d['memorizacion']['aprobada'] and not d['memorizacion']['pesos_guardados']
    assert d['regresion_inferencia_train']['aprobada'] and d['regresion_inferencia_train']['parametros']==439309
    assert all(v['parametros_restaurados_exactos'] for v in d['comparacion_gradientes'])
    assert all(any(p['delta']<0 for p in v['descenso_local_parametros']) for v in d['comparacion_gradientes'])
    archivos=[CONFIG,doc,RAIZ/'README.md',*[RAIZ/p for p in huellas_codigo()],
        RAIZ/'scripts/verificar_v20.py',RAIZ/'scripts/informar_v20.py',Path(__file__),
        *list(base.glob('*.json')),*list((base/'graficas').glob('*.png'))]
    publicar(destino,{'aprobada':True,'pruebas':d['pruebas'],'protocolo':protocolo(cfg),
        'actualizaciones_ab':0,'entrenamiento_completo_habilitado':False,'validacion_evaluada':False,'test_evaluado':False,
        'diagnostico_train_only':True,'historia_preservada_archivos':len(previos),
        'nuevos_parametros':0,'pesos_memorizacion_descartados':True,
        'residual_gradiente_vs_ce6_matched':d['senal_adicional']['fracciones_ortogonales_vs_ce6_matched'],
        'sha256_entregables':{p.relative_to(RAIZ).as_posix():sha(p) for p in archivos}})
    print('Entrega V20 verificada: gradientes adicionales en cabeza principal, históricos intactos y A/B no iniciado.')


if __name__=='__main__':main()
