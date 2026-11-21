"""Integridad final del A/B: métricas, diarios, aliases e históricos reservados."""
import json
import math
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.reanudacion_regional import DiarioRama,publicar,sha
from scripts.supervision_regional_materiales import CONFIG,hashes_codigo,exigir_habilitacion


def leer(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    cfg=leer(CONFIG);exigir_habilitacion(cfg);nombre=cfg['nombre']
    d=leer(RAIZ/cfg['diagnostico']);d['sha256_archivo']=sha(RAIZ/cfg['diagnostico'])
    r=leer(RAIZ/f'resultados/{nombre}.json');e=leer(RAIZ/f'resultados/{nombre}_evaluacion.json')
    comparacion=leer(RAIZ/'resultados/supervision_regional_reanudacion/comparacion.json')
    deltas_reevaluacion={}
    for rama in ['A','B']:
        diario=DiarioRama(RAIZ,cfg,rama,d,hashes_codigo());ck,regs,_=diario.leer()
        assert ck['actualizaciones_desde_origen']==r['ramas'][rama]['actualizaciones']==1452
        assert len(regs)==12 and ck['historial']==r['ramas'][rama]['historial']
        for sufijo,registro in [('ultimo',regs[-1]),('mejor',max(regs,key=lambda x:x['historial_epoca']['map50_val']))]:
            assert sha(RAIZ/f'checkpoints/{diario.nombre}_{sufijo}.pt')==registro['sha256_checkpoint']
            assert e['resultados'][f'{rama}_{sufijo}']['sha256_checkpoint']==registro['sha256_checkpoint']
            esperado=registro['historial_epoca']['map50_val']
            observado=e['resultados'][f'{rama}_{sufijo}']['vistas']['1']['resumen']['map50']
            deltas_reevaluacion[f'{rama}_{sufijo}']=observado-esperado
            assert abs(observado-esperado)<=1e-8
    for modelo in e['resultados'].values():
        for resultado in modelo['vistas'].values():
            x=resultado['resumen'];tp=x['tp50'];fp=x['fp50'];fn=x['fn50']
            assert tp+fn==361
            assert math.isclose(x['recall50'],tp/(tp+fn),rel_tol=1e-10)
            assert math.isclose(x['precision50'],tp/(tp+fp) if tp+fp else 0.,rel_tol=1e-10)
            assert math.isclose(x['map50'],sum(x['ap50_por_material'].values())/6,rel_tol=1e-10)
            assert 0<=x['map50_95']<=x['map50']<=1
    anteriores=leer(RAIZ/'resultados/supervision_regional_reanudacion/estado_previo.json')['sha256_historicos']
    assert all(sha(RAIZ/p)==h for p,h in anteriores.items())
    assert not e['test_evaluado'] and not e['cambio_predeterminado']
    assert comparacion['rng_por_epoca_iguales'] and comparacion['trazas_iguales']
    archivos=[CONFIG,RAIZ/'README.md',RAIZ/'documentacion/18_reanudacion_y_comparacion_regional.md',
        RAIZ/f'resultados/{nombre}.json',RAIZ/f'resultados/{nombre}_evaluacion.json',
        RAIZ/cfg['diagnostico'],RAIZ/cfg['verificacion'],
        *[RAIZ/p for p in hashes_codigo()],RAIZ/'scripts/informar_supervision_regional.py',
        RAIZ/'scripts/graficar_supervision_regional.py',Path(__file__),
        *list((RAIZ/f'resultados/{nombre}_graficas').glob('*.png')),
        *list((RAIZ/f'resultados/{nombre}').rglob('*.json')),
        *list((RAIZ/'resultados/supervision_regional_reanudacion').glob('*.json')),
        *list((RAIZ/'resultados').glob(nombre+'_*v_predicciones_val.json'))]
    publicar(RAIZ/'resultados/supervision_regional_reanudacion/entrega_final.json',{
        'aprobada':True,'actualizaciones_A':1452,'actualizaciones_B':1452,'paso_adamw_final':3751,
        'historicos_intactos':len(anteriores),'metricas_consistentes':True,'pruebas':d['pruebas'],
        'delta_map50_reevaluacion_1vista_vs_historial':deltas_reevaluacion,
        'test_evaluado':False,'checkpoint_predeterminado_modificado':False,
        'sha256_entregables':{p.relative_to(RAIZ).as_posix():sha(p) for p in archivos}})
    print('Entrega verificada: A/B completos, métricas consistentes, históricos intactos y test reservado.')


if __name__=='__main__':main()
