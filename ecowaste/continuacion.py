"""Valida ampliaciones de presupuesto de un mismo entrenamiento propio."""
from ecowaste.datasets import CLASES


def validar_continuacion(previo,config,hashes):
    if (previo.get('tipo')!='ecowaste_mapas_desde_cero_v1' or previo.get('pesos_externos') is not False
            or previo.get('clases')!=CLASES):raise ValueError('Origen no es un entrenamiento propio compatible')
    if previo['hashes_manifiestos']!=hashes:raise ValueError('Los datos de continuación deben coincidir')
    permitidas={'nombre','epocas','min_epocas','paciencia'}
    antes={k:v for k,v in previo['config_entrenamiento'].items() if k not in permitidas}
    ahora={k:v for k,v in config.items() if k not in permitidas}
    if antes!=ahora:raise ValueError('Solo se puede ampliar presupuesto y parada, sin cambiar el aprendizaje')
    if config['nombre']==previo['config_entrenamiento']['nombre'] or config['epocas']<=previo['epoca']:
        raise ValueError('La continuación requiere otro nombre y épocas adicionales')
