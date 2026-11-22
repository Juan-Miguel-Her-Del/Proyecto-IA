"""Concentracion, cola de revision y propuesta de coleccion; nunca reetiqueta datos."""
from collections import Counter
import json
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.datasets import CLASES
from ecowaste.reanudacion_regional import publicar
from scripts.capturar_v25 import SALIDA,leer
from scripts.informar_v25 import tabla


def main():
    resultados={};cola=[]
    for split in ['train','val']:
        ff=leer(SALIDA/f'censo/instancias_{split}.json');oo=leer(SALIDA/f'censo/oraculos_{split}.json');resultados[split]={}
        or48={(r['imagen'],f['id']):f for r in oo for g in r['rejillas'] if g['lado']==48 for f in g['filas']}
        for c in CLASES:
            filas=[f for f in ff if f['clase']==c];counts=Counter(f['imagen'] for f in filas)
            n=len(filas);sq=sum(v*v for v in counts.values())
            resultados[split][c]={'instancias':n,'escenas':len(counts),'escenas_efectivas_por_concentracion':n*n/sq if sq else 0,
                'escenas_con_mas_instancias':counts.most_common(5),'fraccion_en_cinco_escenas':sum(v for _,v in counts.most_common(5))/n,
                'nota':'Kish de conteos por escena, descriptor de concentracion; no tamano muestral inferencial ni independencia demostrada'}
            for f in filas:
                indicadores=[]
                for campo in ['toca_borde_imagen','poligono_fuera_imagen','bbox_fuera_imagen','cruces_estrictos','posible_multimaterial_por_categoria']:
                    if f[campo]:indicadores.append(campo)
                if not f['region_elegible']['tokens']:indicadores.append('region_sin_supervision_no_ambigua')
                if f['lado_menor_384']<8:indicadores.append('menor_un_token')
                x=or48[(f['imagen'],f['id'])]
                if x['fusion_mayoritaria']:indicadores.append('fusion_semantica_gt')
                cola.append({'split':split,'imagen':f['imagen'],'id':f['id'],'material':c,'categoria':f['categoria'],
                    'ruta':f['ruta'],'prioridad':1 if c in ['organico','vidrio','carton'] or indicadores else 2,
                    'indicadores':';'.join(indicadores),'decision_etiqueta':'pendiente_adjudicacion_sin_cambios'})
    publicar(SALIDA/'concentracion.json',resultados);tabla('cola_revision_instancias.csv',cola)
    tabla('tamanos_y_errores.csv',[{'split':s,'clase':c,'tamano':t,'gt':g['gt'],**g['causas']}
        for s in ['train','val'] for c,ts in leer(SALIDA/f'analisis/resumen_{s}.json')['propuestas']['por_material_y_tamano'].items() for t,g in ts.items()])
    propuesta={'estado':'propuesta_no_recolectada_no_entrenada','intervencion_prioritaria':'coleccion_real_por_instancia_con_cobertura_y_ambiguedad_controladas',
        'instancias_nuevas_train':dict(zip(CLASES,[150,300,300,250,150,400])),
        'total_instancias_nuevas_train':1550,'escenas_nuevas_train_intervalo':[500,650],
        'minimo_escenas_independientes_clases_prioritarias':100,
        'minimo_ejemplares_fisicos_distintos_por_clase_prioritaria':100,
        'tope_instancias_material_por_escena_para_cumplir_cuota':5,
        'minimo_regiones_organicas_sin_ambiguedad_y_validas_48':300,
        'justificacion':'Llevar minoritarias a aproximadamente >=400 instancias totales y ampliar escenas; organico 5->405, papel109->409, carton149->449, vidrio203->453. No es tamano optimo ni garantiza AP.',
        'coleccion_independiente':{'escenas_intervalo':[120,180],'minimo_instancias_por_material':100,
            'minimo_escenas_con_cada_material':50,'separacion_por':['objeto_fisico','secuencia','sesion','dia_sitio'],
            'dominios': ['camaras_web_y_MP4_reales','exterior','interior','fondos_y_dispositivos_no_usados_en_train'],
            'bloquear_antes_de_comparar_modelos':True,'no_se_usa_para_seleccionar_umbrales':True,
            'incertidumbre':'100 instancias independientes darian error aprox maximo de recall +/-9.8 puntos al95%; aqui agrupadas, usar bootstrap por escena y no prometer esa precision.',
            'evaluacion_ahora':False},
        'prioridad_materiales':['organico','vidrio','carton','papel'],
        'politica_anotacion':['solo_imagenes_reales_con_procedencia_y_permiso','una_identidad_por_objeto',
            'poligono_visible_y_bbox_derivada','no_rellenar_a_traves_de_oclusores_ni_envoltorios_opacos',
            'separar_material_compuesto_o_desconocido_sin_forzar_seis_clases','banderas_occlusion_truncamiento_transparencia',
            'conservar_huecos_si_formato_lo_permite_documentar_limitacion_TACO_poligonos_sin_huecos',
            'doble_anotacion_prioritarias_y_20porciento_resto_con_adjudicacion',
            'no_cajas_desde_etiquetas_globales_RealWaste'],
        'puertas_antes_de_entrenar':['censo_y_validacion_de_geometria','adjudicar_composites_y_solapes',
            'congelar_particiones_por_adquisicion','verificar_cobertura_por_instancia_y_no_solo_conteos',
            'aprobar_protocolo_y_una_sola_modificacion_futura'],
        'cambios_arquitectonicos_en_esta_fase':False,'test_original_reservado':True}
    publicar(SALIDA/'propuesta_coleccion.json',propuesta)
    publicar(SALIDA/'revision_visual.json',{'tipo':'triage_exploratorio_no_adjudicacion','train':{'carton':149,'vidrio':203,'organico':5},
        'val':{'carton':25,'vidrio':22},'otros_materiales':'censo_estructural_exhaustivo_y_contactos_generados; sin_adjudicacion_visual_exhaustiva',
        'hojas_revisadas':['train_carton_00','train_carton_01','train_carton_02','train_vidrio_00','train_vidrio_01','train_vidrio_02','train_vidrio_03','val_carton_00','val_vidrio_00','organicos_contexto'],
        'hallazgos':[{'imagen':'taco:68','id':163,'observacion':'comida dentro de lata; solape con metal, no etiqueta falsa demostrada'},
            {'imagen':'taco:72','id':171,'observacion':'pan bajo envoltura transparente; limites de material superpuestos'},
            {'imagen':'taco:81','ids':[203,204],'observacion':'restos pequenos dentro de caja; poligono carton cubre contenido'},
            {'imagen':'taco:92','id':244,'observacion':'resto de manzana visible, fondo exterior; unico organico elegible'},
            {'imagen':'taco:1179','id':3649,'observacion':'carton muy fino/oscuro, material incierto a la vista; revisar original'},
            {'imagen':'taco:1335','ids':[4255,4256,4257,4258,4259,4260],'observacion':'oclusiones y partes de botella: revisar criterio parte/instancia sin fusionar automaticamente'},
            {'imagen':'taco:1321','observacion':'muchos fragmentos vidrio del mismo escenario; instancias no son ejemplos visuales independientes'}],
        'etiquetas_modificadas':0,'certeza':'no se diagnostica etiqueta erronea solo por apariencia, cruce o borde'})
    publicar(SALIDA/'incidencias_presentacion.json',{'afecta_inferencia_o_censo':False,'historicos_alterados':False,
        'incidencias':['matplotlib ausente en venv ML; figuras separadas al interprete grafico ya instalado',
            'resumen organicos no cubria escena sin otro material; corregido maximo de conjunto vacio con mascara cero',
            'refactor grafico por comando con comillas fallo antes de escribir; rehecho mediante here-string literal'],
        'recuperacion':'Se conservaron los resúmenes publicados y se completaron solo artefactos pendientes; no se repitio inferencia ni se sobrescribio evidencia.'})


if __name__=='__main__':main()
