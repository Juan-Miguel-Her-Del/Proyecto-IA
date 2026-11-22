"""Selección de calidad sin adjudicación humana: conservar inicial y cuarentena."""
from pathlib import Path
import sys
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from ecowaste.incorporacion import leer, publicar, digest, sha
from scripts.preparar_anotacion_v27_2 import PROYECTO
from scripts.fuentes_v27_2 import SALIDA


def main():
    inicial = leer(PROYECTO / 'PROYECTO.json')
    activo = {k: v for k, v in inicial.items() if k != 'huella'}
    activo['coleccion_id'] = 'v27_2_piloto_72_seleccion_2'
    activo['imagenes'] = [i for i in inicial['imagenes'] if i['fuente'] != 'dwsd']
    activo['revision_preparacion_anterior_sha256'] = sha(PROYECTO / 'PROYECTO.json')
    activo['huella'] = digest(activo)
    publicar(PROYECTO / 'seleccion_72/PROYECTO.json', activo)
    publicar(SALIDA / 'seleccion_calidad_piloto.json', {
        'tipo': 'seleccion_asistente_no_revision_humana', 'proyecto_inicial_imagenes': 80,
        'piloto_activo_imagenes': 72, 'anotaciones_publicas_pendientes': sum(len(i['anotaciones']) for i in activo['imagenes']),
        'imagenes_excluidas': [i['id'] for i in inicial['imagenes'] if i['fuente'] == 'dwsd'],
        'motivos': ['DWSD: muestras giradas con bordes y variación artificial de color; originales sin transformar no identificados',
                    'DWSD: sin leyenda de máscaras ni identidad verificable de instancia; no generar pseudo-GT desde componentes',
                    'WSC: conservar solo 12 fotos publicadas de valid; cinco variantes por nombre fuente en train excluidas',
                    'WSC Organic: indicios visuales de material vegetal, no evidencia suficiente para mapear a restos alimentarios'],
        'revisiones_humanas': 0, 'aceptadas_V26': 0, 'entrenamiento': False,
        'proyecto_activo_sha256': sha(PROYECTO / 'seleccion_72/PROYECTO.json')})
    print('Piloto activo: 72 imágenes; 256 anotaciones públicas pendientes; 0 revisiones humanas.', flush=True)


if __name__ == '__main__':
    main()
