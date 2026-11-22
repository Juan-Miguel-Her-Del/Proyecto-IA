"""Matriz de fuentes y cantidades V27: hechos declarados frente a archivos verificados."""
import csv
from pathlib import Path
import sys
RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.incorporacion import leer,publicar,publicar_bytes,CLASES
from scripts.investigar_v27 import DATOS,SALIDA


def main():
    z=leer(SALIDA/'piloto_zerowaste.json');w=leer(SALIDA/'piloto_warp.json');h=leer(SALIDA/'piloto_hpi.json')
    filas=[]
    def agregar(nombre,url,imagenes,instancias,categorias,formato,licencia,academico,comercial,disponibilidad,calidad,diversidad,duplicados,potencial,*,verificadas=None,inst_verificadas=None,descargadas=0):
        filas.append({'nombre':nombre,'url_oficial':url,'imagenes_declaradas':imagenes,'instancias_declaradas':instancias,
            'categorias_originales':categorias,'formato':formato,'licencia_y_alcance':licencia,'academico':academico,'comercial':comercial,
            'disponibilidad_comprobada':disponibilidad,'calidad_y_limitaciones':calidad,'diversidad':diversidad,
            'duplicados_y_linaje':duplicados,'potencial_materiales_no_aceptados':potencial,
            'imagenes_train_en_metadatos_verificados':verificadas,'instancias_train_verificadas_en_archivos':inst_verificadas,
            'imagenes_piloto_descargadas':descargadas,'instancias_aceptadas_V26':0})
    agregar('ZeroWaste-f','https://ai.bu.edu/zerowaste/',4503,None,['cardboard','metal','rigid_plastic','soft_plastic'],'COCO instancias/poligonos; fotos deblurred del proveedor',
        'Autores CC BY-NC 4.0; Zenodo CC BY 4.0: contradicción preservada. Piloto académico bajo condiciones NC comunes; no permiso comercial presumido.',
        'sí, no comercial y atribución según autores','no confirmado; no asumir BY del registro como eliminación de NC',
        'ZIP oficial leído por rangos; 24 fotos y JSON train realmente obtenidos','128 polígonos válidos; 128 bbox difieren de caja derivada. Clases incompletas para nuestras seis, papel visible sin GT.',
        'multiobjeto, desorden, oclusión; una cinta industrial, fotogramas correlacionados',
        '0 exactos/perceptuales contra históricos SOLO en piloto; no usar ZeroWasteAug, que añade derivados/TACO',
        {'plastico':z['train_categorias_instancias']['soft_plastic']+z['train_categorias_instancias']['rigid_plastic'],'papel':0,'carton':11941,'vidrio':0,'metal':259,'organico':0},
        verificadas=3002,inst_verificadas=18002,descargadas=24)
    agregar('WaRP-D, distribuidor enlazado por AIRI','https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset',2974,None,w['categorias_originales'],'YOLO bbox; sin polígonos en WaRP-D',
        'CC0: Public Domain en metadata Kaggle original; autor enlazado desde AIRI','sí según CC0 declarado','sí según CC0 declarado',
        'ZIP oficial público por rangos; 2452 TXT train y 24 fotos obtenidos','97 cajas de piloto válidas; todas necesitan polígono manual para V26; 27 sin mapeo de material confirmado.',
        'plantas industriales, botellas deformadas, reflejos, oclusiones; no da tracking físico',
        '0 en piloto contra históricos; WaRP-C son recortes derivados, no objetos nuevos',
        {'plastico':6071,'papel':0,'carton':0,'vidrio':448,'metal':None,'organico':0,'cans_para_revisar':562,'cartones_multicapa_para_revisar':650},
        verificadas=2452,inst_verificadas=8823,descargadas=24)
    agregar('SpectralWaste original / HF semántico','https://huggingface.co/datasets/ferpb/spectralwaste-segmentation',
        {'repo_original':7655,'HF':7653,'etiquetadas':852},None,['film','basket','cardboard','video_tape','filament','bag','background'],
        'COCO instancias en release original; HF 256x256 con máscaras SEMÁNTICAS',
        'CC BY 4.0 en licencia original HF/Zenodo','sí con atribución','sí con atribución',
        'metadatos y LICENSE accesibles; 16 shards train ~1GB c/u; no se descargaron fotos',
        'HF fusiona instancias al convertir a semántica; no reconstruir IDs físicos/cajas desde componentes',
        'planta industrial y secuencias; 256x256 pierde detalle de pequeños',
        'mismo dataset en original/HF/Voxel51: no fuentes independientes; hashes de fotos no comprobados',
        {c:None for c in CLASES})
    agregar('Food Waste HPI, fuente original','https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset',375,0,
        ['ingredientes y pesos por recibo; sin instancias espaciales'],'imágenes + tabla nutricional, no bbox/máscaras',
        'CC BY 4.0 explícita para fotografías y registros; misma en revisión de assets','sí con atribución','sí con atribución',
        '215 filas train consultadas; 12 fotos, una por cada recibo train; mismas huellas LFS en revisión actual/cache',
        'necesita anotación completa por instancia; etiqueta de ingrediente no define bbox',
        'sobras reales después de comer; solo 12 comidas train, fondo/plato similar, 12-23 vistas por recibo',
        '0 hashes duplicados en piloto; espejo Voxel51 deriva de esta fuente, no sumarlo como nuevo',
        {'organico_anotado':0,'fotos_para_anotacion_manual_descargadas':12,'comidas_train_documentadas':12},
        verificadas=215,inst_verificadas=0,descargadas=12)
    agregar('FoodInsSeg','https://github.com/jamesjg/FoodInsSeg',7118,119048,['103 ingredientes alimentarios'],'COCO polígonos por ingrediente',
        'CC BY-NC-SA 4.0 + autores especifican uso académico, sin fines comerciales','sí bajo condiciones académicas/NC/SA','no según autores',
        'enlace Google Drive publicado; no se materializó archivo ni se verificó contenido local',
        'declaran train 4983 imágenes / 82716 máscaras; no verificadas aquí; ingredientes no equivalen a residuos físicos',
        'platos servidos/comida, no necesariamente descartada; cambio de dominio',
        'deriva de FoodSeg103; no probado contra históricos',{'organico':None})
    agregar('UC Food Waste','https://github.com/CollaborativeRoboticsLab/UC-Food-Waste',542,None,
        ['Soft Plastics','Hard plastics','Glass','Metal','Background (organic)'],'polígonos CSV + máscaras/numpy multimodal',
        'no se confirmó permiso de dataset en la fuente; no descargar fotos','pendiente permiso explícito','sin confirmar',
        'README/Drive enlazado; fotos no descargadas',
        'orgánico es FONDO en cinco canales; no son instancias orgánicas identificadas',
        'comida y contaminantes en laboratorio controlado',
        'sin comprobación de duplicados de imágenes',{'vidrio':None,'plastico':None,'metal':None,'organico_instancias_confirmadas':0})
    agregar('TrashCan Material','https://irvlab.cs.umn.edu/resources/trashcan',7212,None,
        ['trash por material, ROV, flora y fauna; dos variantes'],'máscaras por instancia',
        'laboratorio limita datasets a investigación académica; permisos JAMSTEC para otros usos no corroborados en archivo local',
        'declaración académica en sitio del laboratorio','no autorizado por fuente consultada',
        'sitio primario consultado en web; repositorio de datos devolvió 403; no fotos locales',
        'no confundir fauna viva con orgánicos; dominio submarino distinto',
        'baja luz, agua, vídeos ROV correlacionados',
        'espejos con CC BY/MIT contradicen restricciones de proveedor; no usarlos para saltarlas',
        {c:None for c in CLASES})
    agregar('FLIC','https://www.mdpi.com/2076-3417/16/11/5465',802,None,
        ['first course','second course','side dish','bread','food','background'],'segmentación semántica refinada tras SAM',
        'licencia del artículo no demuestra por sí misma licencia de fotografías del Drive; pendiente confirmar','pendiente condiciones del dataset','sin confirmar',
        '401 pares full/leftover declarados, enlace Drive; no archivos locales',
        'máscaras semánticas, no identidades físicas por instancia; no generar cajas desde componentes',
        'comedor real 22 días; versiones antes/después de la misma bandeja',
        'pares de una bandeja deben mantenerse juntos; no fotos comparadas aquí',{'organico':None})
    agregar('Roboflow GARBAGE CLASSIFICATION 3 / CEP','https://universe.roboflow.com/cep-owycw/garbage-classification-3-i5yna',10464,None,
        ['PAPER','PLASTIC','GLASS','METAL','CARDBOARD','CLOTH','BIODEGRADABLE'],'bbox declaradas en Universe',
        'ficha anuncia CC BY 4.0; procedencia y permisos de fotos no corroborados','pendiente corroborar fuente y archivo','sin confirmar para fotografías',
        'preview/6 versiones declaradas; descarga de archivo no verificada, solicitud directa 403',
        'BIODEGRADABLE no equivale automáticamente a comida; no validar calidad solo por mAP del modelo alojado',
        'sin verificar contra archivo',
        'posible mezcla de repositorios/clasificación; no demostrar unicidad con número de filas',
        {c:None for c in CLASES})
    agregar('Roboflow wast / myfast','https://universe.roboflow.com/myfast/wast-6srsc',425,None,
        ['Paper','Cardboard','Plastic Bottle','Plastic Bag','Paper Bag','Food Waste'],'instancia declarada',
        'CC BY 4.0 anunciada; origen de fotografías sin corroborar','pendiente','sin confirmar',
        'preview 425 imágenes y 0 versiones de dataset; no se obtuvo exportación',
        'anotaciones no verificadas en archivo',
        'no comprobada',
        'no comparado por hashes',{c:None for c in CLASES})
    agregar('WaRP-D espejo Roboflow','https://universe.roboflow.com/trash-recycling-data/warp-d-jsdws',2965,None,
        w['categorias_originales'],'bbox',
        'espejo anuncia CC BY 4.0; fuente original vinculada por AIRI se declara CC0','preferir original','no inferir de etiqueta del espejo',
        'preview consultado; archivo no descargado',
        'difiere el número de fotos declarado; no equivale a nuevas instancias',
        'derivado de WaRP',
        'linaje común conocido; solapamiento por hashes de espejo no medido',{'aporte_independiente_acreditado':0})
    agregar('Kaggle Garbage Dataset clasificación','https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/versions/8',19762,None,
        ['metal','glass','biological','paper','battery','trash','cardboard','shoes','clothes','plastic'],'etiqueta global de clasificación',
        'MIT anunciada; cadena de procedencia de fotos sin verificar','pendiente corroboración','sin confirmar',
        'ficha accesible; no descargar fotos sin corroboración',
        'sin evidencia de bbox/polígonos; no fabricar caja de imagen completa',
        'declarada, no inspeccionada',
        'posible reutilización de TrashNet/otras fuentes; no hashes medidos',{'propuestas_espaciales_confirmadas':0})
    agregar('Voxel51 Food Waste (derivado HPI)','https://huggingface.co/datasets/Voxel51/food-waste-dataset',
        {'card':375,'preview_train':430},None,['ingredientes alimentarios traducidos'],'máscaras propuestas por YOLO-E',
        'MIT anunciada en conversión; las fotografías originales HPI tienen CC BY 4.0, que debe conservarse','solo con derechos de fuente preservados','no basarse solo en MIT del espejo',
        'ficha descargada; sin fotos del espejo',
        'segmentación automática NO sustituye revisión humana; discrepancia en cifras de ficha/preview',
        'mismo linaje HPI, no aporta diversidad acreditada',
        'fuente declarada AI-ServicesBB; no sumar fotos como colección independiente',{'aporte_espacial_humano_acreditado':0})
    publicar(SALIDA/'matriz_datasets.json',filas)
    with (SALIDA/'matriz_datasets.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(filas[0]));writer.writeheader()
        for row in filas:writer.writerow({k:json_serial(v) for k,v in row.items()})
    cuotas={'plastico':150,'papel':300,'carton':300,'vidrio':250,'metal':150,'organico':400}
    piloto=leer(SALIDA/'piloto_preparado.json')
    totales={c:sum(s['materiales_propuestos'].get(c,0) for s in piloto['fuentes'].values()) for c in CLASES}
    plan={'cuotas_train':{c:{'objetivo':cuotas[c],'aceptadas':0,'pendientes':cuotas[c],'propuestas_piloto_material':totales[c]} for c in CLASES},
        'train_total_pendiente':1550,'generalizacion_pendiente':600,'organicos_utilizables_pendientes':300,
        'suficiente_revisada_congelada':False,'V28_habilitada':False,
        'limites':'Propuestas no son objetos físicos distintos ni GT aceptado. Fuentes públicas no sustituyen generalización capturada de forma separada.',
        'captura_manual_prioritaria':['papel:300, sin fuente espacial suficientemente verificada en el piloto',
            'organico:400 con >=300 regiones útiles; HPI solo 12 comidas conocidas, no multiplicar vistas',
            'generalizacion:600 de sitios, objetos y sesiones nuevos bajo protocolo bloqueado',
            'carton/vidrio/metal: completar escenas y objetos físicos diversos; no confiar solo en repeticiones de cinta industrial']}
    publicar(SALIDA/'plan_cuotas_pendientes.json',plan)
    publicar(SALIDA/'revision_visual_asistente.json',{'tipo':'inspeccion_asistente_no_humana_no_adjudicacion','imagenes_inspeccionadas':60,
        'observaciones':['ZeroWaste muestra papel visible no anotado como papel; revisar cobertura completa de seis clases, no tratarlo como fondo negativo',
            'ZeroWaste 2706,2709,2710 comparten apariencia de escena y objetos pese a pHash>4; no afirmar independencia física',
            'WaRP presenta reflejos, oclusión, objetos pequeños y envases multicapa; completar polígonos',
            'HPI ofrece restos reales visibles, pero fondo homogéneo de plato y pocas comidas; requiere polígonos humanos y diversidad exterior/interior'],
        'revisores_humanos':0,'instancias_aceptadas':0})
    publicar_bytes(DATOS/'ATRIBUCION.txt',('''Uso local de investigación; no se publican fotografías de terceros en Git.
ZeroWaste: Dina Bashkirova, Mohamed Abdelfattah, Ziliang Zhu, James Akl, Fadi Alladkani,
Ping Hu, Vitaly Ablavsky, Berk Calli, Sarah Adel Bargal y Kate Saenko. CVPR 2022.
https://ai.bu.edu/zerowaste/ https://github.com/dbash/zerowaste
Autores: CC BY-NC 4.0; el registro Zenodo anuncia CC BY 4.0. Se mantiene alcance
no comercial del autor; discrepancia registrada, sin permiso comercial presumido.
Subconjunto original zerowaste-f-final/train deblurred por el proveedor. No ZeroWasteAug.
Fotografías sin cambios de píxeles; montajes de inspección locales redimensionados.
Cajas derivadas desde polígonos solo como propuestas diagnósticas, originales conservados.

WaRP: Dmitry Yudin y colaboradores. Hierarchical waste detection with weakly supervised
segmentation in images from recycling plants. Engineering Applications of AI 128, 107542.
https://github.com/AIRI-Institute/WaRP
https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset
Licencia declarada en distribuidor original enlazado: CC0: Public Domain.
Solo Warp-D/train. Fotos y TXT originales conservados; coordenadas YOLO convertidas
a cajas diagnósticas. No se crean polígonos a partir de cajas.

Food Waste HPI: AI Service Centre Berlin-Brandenburg, Hasso-Plattner-Institut,
L. Stroetmann y a la QUARTO. Food Waste Dataset, 2025.
https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset
Fotografías y registros bajo CC BY 4.0. Una foto de train por bonid; 12 comidas.
Se conserva JPEG servido por la fuente; no se fabrican cajas/máscaras/instancias
desde los ingredientes o pesos globales. Ver recibos de descarga y revisión cacheada.

Ningún archivo de esta adquisición es GT aceptado. No se ha entrenado ni abierto test.
''').encode('utf-8'))
    print(f'Matriz: {len(filas)} fuentes/espejos. Piloto: 60 fotografías y 225 propuestas, 0 aceptadas.',flush=True)


def json_serial(v):
    import json
    return json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v


if __name__=='__main__':main()
