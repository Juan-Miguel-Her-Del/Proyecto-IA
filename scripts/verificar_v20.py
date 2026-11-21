"""Verifica V20 y mide actividad de la pérdida solo en train, sin optimizer.step."""
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.detector_mapas import DetectorMapas
from ecowaste.discriminacion_mapas import perdida_discriminativa
from ecowaste.ablacion_fondo import huella
from ecowaste.reanudacion_regional import publicar,sha
from scripts.preparar_v20 import CONFIG,leer_cfg,huellas_codigo,protocolo,contexto_train,exigir_autorizacion
from scripts.evaluar_v20 import exigir_entrenamiento_completo


def main():
    cfg=leer_cfg();base=RAIZ/'resultados/v20_preparacion';salida=base/'verificacion.json'
    if salida.exists():raise FileExistsError('Verificación existente')
    d=json.loads((base/'diagnostico.json').read_text(encoding='utf-8'))
    assert d['aprobada'] and d['hashes_codigo']==huellas_codigo() and d['protocolo']==protocolo(cfg)
    assert not cfg['entrenamiento_completo_habilitado'] and d['actualizaciones_ab_ejecutadas']==0
    for fn in [exigir_autorizacion,exigir_entrenamiento_completo]:
        try:fn(cfg)
        except ValueError:pass
        else:raise AssertionError('No se bloqueó una acción completa')
    assert not list((RAIZ/'checkpoints').glob(cfg['nombre']+'*'))
    previos=json.loads((base/'protegidos.json').read_text(encoding='utf-8'))
    assert all(sha(RAIZ/p)==h for p,h in previos.items())
    anterior=json.loads((RAIZ/'resultados/supervision_regional_preparacion/diagnostico_revision04.json').read_text(encoding='utf-8'))
    assert d['huella_regiones']==anterior['huella_regiones']
    assert d['huella_mascaras_originales']==anterior['huella_mascaras_semanticas_originales']
    assert d['trazas_preparadas']['A']==d['trazas_preparadas']['B']
    assert sum(x['pasos'] for x in d['trazas_preparadas']['A'])==1452
    assert d['amp']['A']['inicial']==d['amp']['B']['inicial']==d['estado_inicial']
    assert d['amp']['A']['huella_entradas']==d['amp']['B']['huella_entradas']
    assert d['memorizacion']['aprobada'] and not d['memorizacion']['pesos_guardados']
    for prueba in d['comparacion_gradientes']:
        assert prueba['grad_directo_fondo_nuevo_l1']==0
        assert prueba['comparacion_logits']['ce6_mismos_tokens_pesos']['fraccion_ortogonal']>.01
        assert prueba['margenes']['fraccion_no_empeorados']>=1-1e-6
        assert all(v is None for n,v in prueba['normas_parametros_nueva'].items() if n.startswith('clasificador_auxiliar'))
    ck,train,aux,regiones,pm,est=contexto_train(cfg)
    principal=json.loads((base/'principal_train_origen.json').read_text(encoding='utf-8'))
    filas_principal={(f['imagen'],f['id_region']):f for f in principal['filas']}
    m=DetectorMapas(**ck['configuracion']).cuda();m.load_state_dict(ck['pesos'],strict=True);m.eval();m.requires_grad_(False)
    antes=huella(m.state_dict());actividad=[]
    with torch.inference_mode():
        for inicio in range(0,len(train),8):
            ids=list(range(inicio,min(inicio+8,len(train))));x=train.base.imagenes[ids].cuda().float()/255
            with torch.autocast('cuda',dtype=torch.float16):raw=m(x)
            _,detalle=perdida_discriminativa(raw,[regiones[i] for i in ids])
            for f in detalle['filas']:
                f['imagen']=train.base.origen.registros[ids[f['imagen_lote']]]['id']
                anterior_f=filas_principal[(f['imagen'],f['id_region'])]
                f['pred_principal']=anterior_f['pred_principal'];f['pred_condicional']=anterior_f['pred_condicional']
                f['cohorte_fondo_condicional_correcto']=f['pred_principal']==0 and f['pred_condicional']==f['clase']
                actividad.append(f)
    assert huella(m.state_dict())==antes and all(p.grad is None for p in m.parameters())
    resumen={}
    from ecowaste.datasets import CLASES
    for c in CLASES:
        fs=[f for f in actividad if f['material']==c]
        resumen[c]={'instancias':len(fs),'activas':sum(f['tokens_seleccionados']>0 for f in fs),
            'tokens_validos':sum(f['tokens_validos'] for f in fs),'tokens_dificiles':sum(f['tokens_seleccionados'] for f in fs),
            'peso':fs[0]['peso_clase'] if fs else None,
            'solo_fondo_con_ranking_condicional_correcto':sum(f['cohorte_fondo_condicional_correcto'] for f in fs),
            'inactivas_entre_fondo_condicional_correcto':sum(f['cohorte_fondo_condicional_correcto'] and not f['tokens_seleccionados'] for f in fs)}
    organicas=[f for f in actividad if f['material']=='organico'];assert len(organicas)==1 and organicas[0]['peso_clase']==1.
    assert all(f['fraccion_peso_instancia']<=1 for f in actividad)
    publicar(base/'actividad_train.json',{'resumen':resumen,'organico':organicas,'filas':actividad,
        'precision':'AMP FP16, misma agrupación de ocho escenas que principal_train_origen',
        'inferencias_nuevas_train':121,'actualizaciones_optimizador':0,'validacion_evaluada':False,'test_evaluado':False})
    publicar(salida,{'aprobada':True,'pruebas':d['pruebas'],'sha256_diagnostico':sha(base/'diagnostico.json'),
        'hashes_codigo':huellas_codigo(),'sha256_verificador':sha(Path(__file__)),'historicos_intactos':len(previos),
        'geometria_y_mascaras_identicas_v17':True,'trazas_pareadas_1452':True,'estados_iniciales_identicos':True,
        'actividad_train':resumen,'organico':organicas,'arquitectura_inalterada':True,'checkpoint_inalterado':True,
        'actualizaciones_ab':0,'entrenamiento_completo_habilitado':False,'validacion_evaluada':False,'test_evaluado':False})
    print(json.dumps({'aprobada':True,'actividad':resumen,'organico':organicas},ensure_ascii=False))


if __name__=='__main__':main()
