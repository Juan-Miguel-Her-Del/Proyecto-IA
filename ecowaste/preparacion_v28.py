"""Selección conservadora y barreras de ejecución V28; no contiene entrenamiento."""
from collections import Counter
from ecowaste.incorporacion import CLASES


def suficiencia(imagenes,observaciones):
    bloqueadas={o['imagen'] for o in observaciones if o.get('requiere_decision',True)}
    elegibles=[f for f in imagenes if f['estado']=='completa_usuario_unico' and not f['errores']
               and not f['avisos_visuales'] and f['imagen'] not in bloqueadas]
    conteos=Counter(a['material'] for f in elegibles for a in f['qa_objetos']
                    if not a['descartada'] and a['material'] in CLASES and a['valido_geometria'])
    faltan={c:max(0,8-conteos[c]) for c in CLASES}
    return {'imagenes_candidatas_sin_incidencias_identificadas':[f['imagen'] for f in elegibles],
            'cantidad_imagenes':len(elegibles),'materiales':{c:conteos[c] for c in CLASES},
            'faltan_hasta_minimo_40_imagenes':max(0,40-len(elegibles)),
            'faltan_hasta_8_por_material':faltan,
            'cumple_referencia_numerica':len(elegibles)>=40 and not any(faltan.values()),
            'no_es_certificacion_exhaustividad_visual':True,'entrenamiento_autorizado':False}


def exigir_ejecucion(protocolo,revision):
    if not protocolo.get('entrenamiento_autorizado'):
        raise PermissionError('V28 requiere autorización expresa posterior al informe; no entrenar')
    if not revision.get('aprobada_qa_para_entrenar') or not revision.get('coleccion_congelada_para_entrenar'):
        raise PermissionError('Faltan QA visual y colección final congelada')
    if not revision.get('hash_datos') or protocolo.get('hash_datos')!=revision['hash_datos']:
        raise ValueError('La autorización no corresponde a estos datos')
    return True
