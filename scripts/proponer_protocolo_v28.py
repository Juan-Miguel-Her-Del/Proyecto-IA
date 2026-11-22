"""Protocolo A/B propuesto; sin entrenador ni pasos AdamW sobre datos reales."""
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
import torch
from ecowaste.incorporacion import leer,publicar,sha
from ecowaste.backend_numerico import politica
from ecowaste.reanudacion_regional import huellas_estado
from scripts.respaldar_anotaciones_v28 import SALIDA


def main():
    path=RAIZ/'checkpoints/detector_mapas_v6_auxiliar_ext48_mejor.pt'
    if sha(path)!='eb8fd00ee536ead21e3a2a0ecd5b06baad04acb2e437b22e0eac3686a7bf36b9':raise ValueError('Origen alterado')
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['epoca']!=19:raise ValueError('Origen de otra época')
    grupos=ck['optimizador']['param_groups']
    pasos=sorted({float(x['step']) for x in ck['optimizador']['state'].values() if 'step' in x})
    previo=leer(RAIZ/'resultados/detector_mapas_v6_auxiliar_ext48.json')
    suficiencia=leer(SALIDA/'suficiencia_exploratoria.json')
    objetos=leer(SALIDA/'objetos.json');ids=set(suficiencia['imagenes_candidatas_sin_incidencias_identificadas'])
    n_food=sum(o['imagen'] in ids and o['fuente']=='hpi' and o['material']=='organico' and not o['descartada'] for o in objetos)
    cfg={'version':'v28_auxiliar_usuario_AB_propuesta_1','estado':'bloqueado_por_QA_y_autorizacion_pendientes',
        'entrenamiento_autorizado':False,'evaluacion_autorizada':False,'hash_datos':None,
        'checkpoint_origen':path.relative_to(RAIZ).as_posix(),'sha256_checkpoint_origen':sha(path),
        'arquitectura':ck['configuracion'],'origen_epoca':19,'estado_inicial_huellas':huellas_estado(ck),
        'restaurar':['pesos','optimizador_AdamW','scaler','rng_torch','rng_cuda','rng_loader','rng_auxiliar'],
        'paso_adam_origen':pasos,'backend_numerico':politica('completo'),
        'RAM_GPU':'medir max_memory_allocated/reserved y tiempo sincronizado, sin afirmar estimaciones',
        'semilla_stream_auxiliar':2801,'presupuesto_actualizaciones_por_rama':300,
        'justificacion_presupuesto':'600 exposiciones auxiliares con 2 imágenes/paso: 10–15 pasadas medias sobre 40–60 fotos; piloto de capacidad, no entrenamiento final',
        'lote_detector':8,'ramas':{'A':{'historicas_comunes':6,'historicas_control':2,'nuevas':0},
                                 'B':{'historicas_comunes':6,'historicas_control':0,'nuevas':2}},
        'estimando':'sustitucion_del_25_por_ciento_del_lote_por_auxiliares_revisadas_a_igual_compute',
        'limite_causal':'B ve 1800 muestras históricas y 600 nuevas; A ve 2400 históricas. No es adición con exposición histórica idéntica.',
        'muestreo':'6 anclas históricas idénticas por actualización A/B; dos ranuras control/tratamiento; permutaciones sin reemplazo por ciclo; flujos RNG separados y guardados',
        'aumentos':'flip horizontal p=.5, brillo/contraste uniformes [.85,1.15]; sorteos por posición iguales A/B; auxiliar global con stream propio idéntico',
        'learning_rate':grupos[0]['lr'],'weight_decay':grupos[0]['weight_decay'],
        'betas':list(grupos[0]['betas']),'eps':grupos[0]['eps'],'clipping_norma':1.,
        'justificacion_LR':'continuación breve desde ep19 con AdamW y fase estable restaurados; mantiene LR del origen, no ajuste guiado por val ni nuevo warmup',
        'warmup_nuevo':False,'scheduler_nuevo':False,'AMP':True,'scaler_origen':ck['scaler'],
        'perdida':'perdida_mapas histórica softmax7, foreground+3*fondo+Dice, sin pérdidas V17/V20',
        'pesos_materiales_fijos_historicos':previo['pesos_clase_train'],
        'fuente_pesos':{'archivo':'resultados/detector_mapas_v6_auxiliar_ext48.json','sha256':sha(RAIZ/'resultados/detector_mapas_v6_auxiliar_ext48.json')},
        'recalcular_pesos_con_datos_nuevos':False,'auxiliar_global':{'batch_size':16,'coeficiente':.5,'datos_y_orden_iguales_A_B':True},
        'datos_necesarios':{'imagenes_minimo':40,'imagenes_maximo':60,'poligonos_por_material_minimo':8,
            'orgánicos_alimentarios_prioritarios_minimo':8,'no_sustituir_comida_por_vegetacion_ambigua':True,
            'candidatas_actuales':suficiencia['cantidad_imagenes'],'poligonos_HPI_actuales_candidatos':n_food,
            'faltan_HPI_hasta_8':max(0,8-n_food),'aceptacion_auxiliar':'usuario_unico+QA_resuelta; NO aceptación V26'},
        'seleccion_futura':'tras resolver QA: menor prefijo de permutación determinista semilla2801 que cubra >=40 imágenes y >=8 instancias/clase, hasta60; guardar manifiesto antes de inferencia',
        'targets':'384×384 letterbox, máscara48×48, cobertura1/64; huecos preservados, padding/compuestos/mezcla material ignorados; equivalencia con histórico a verificar',
        'agrupacion':'fuentes enteras, nombres fuente/bonid y duplicados unidos; todo auxiliar candidato train; no fabricar sesiones ni identidades',
        'evaluacion':{'dominios':['train histórico completo','auxiliar train seleccionado'],
            'modelos':['original ep19 bajo mismo backend','A paso300','B paso300'],'vistas':[1,5],
            'metricas':['mAP50','mAP50_95','AP50_seis_materiales','recall','precision','TP','FP','FN',
                        'FP_por_100_imagenes','confusion_material_fondo','IoU_semantico','recuperacion_por_instancia',
                        'geometria_y_fusion','tiempo','memoria_GPU'],
            'predicciones_y_trazas_por_imagen_guardadas':True,'GT_solo_analisis_no_inferencia':True,
            'val_historica':False,'test':False,'generalizacion_bloqueada':True,
            'seleccion_mejor_checkpoint':False,'checkpoint_comparado':'final mismo presupuesto300',
            'conclusion_permitida':'capacidad de ajuste train; NO generalización'},
        'postprocesamiento':{'argmax7':True,'fondo':3,'umbral':.05,'umbral_visual':.35,'area_minima':2,'conectividad':8,'NMS':.5,'top':100,'vistas_predeterminadas':5},
        'checkpoints':{'pasos':[0,100,200,300],'ruta_futura_exclusiva':'checkpoints/v28_auxiliar_<hash_datos>_<id_ejecucion>/A_o_B/paso_N.pt',
                       'sobrescribir':False,'reanudar':'mismo protocolo/dataset/backend, estados AdamW/scaler/RNG/cursores/historial restaurados'},
        'parada_temprana_por_metricas':False,
        'parada_por_error':['integridad','no_finitos','paso_AdamW_omitido','presupuesto_excedido','backend_distinto','reanudacion_incompatible'],
        'criterio_descriptivo_exito':['mAP50 auxiliar B>A','AP50 cartón y vidrio auxiliar B>A',
            'mAP50 histórico B>=A','recall y precisión auxiliares B>=A','FP/100 auxiliares B<=A'],
        'interpretacion':'si mejora material condicional sin AP/oportunidades, declarar que no solucionó detección; informar todos los tradeoffs',
        'adopcion_automatica':False,'cambiar_checkpoint_predeterminado':False,
        'controles_antes_de_ejecutar':['revisión del informe y autorización expresa','QA y dataset final congelados con hash',
            'adaptador de máscaras/huecos y lote mixto verificado','trazas A/B pareadas de entradas/aumentos',
            'prueba de gradientes y reanudación diagnóstica después de autorización; no realizada en V28 preparación'],
        'identidad_bit_a_bit_afirmada':False}
    publicar(RAIZ/'configuracion/entrenamiento_auxiliar_v28_propuesto.json',cfg)
    publicar(SALIDA/'protocolo_propuesto.json',cfg)
    print('Protocolo 300 pasos propuesto y bloqueado; no se ejecutó AdamW.',flush=True)


if __name__=='__main__':main()
