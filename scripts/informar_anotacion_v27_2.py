"""Comparación auditable de fuentes, límites del piloto y herramientas V27.2."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import zipfile

RAIZ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAIZ))
from ecowaste.auxiliares_publicos import Taller
from ecowaste.incorporacion import leer,publicar,publicar_bytes,sha
from scripts.fuentes_v27_2 import SALIDA,DATOS
from scripts.anotar_v27_2 import PROYECTO


def main():
    w=leer(SALIDA/'auditoria_wsc.json');d=leer(SALIDA/'auditoria_dwsd.json')
    rwzip=RAIZ/'datos/originales/realwaste.zip'
    with zipfile.ZipFile(rwzip) as z:
        fotos=[i for i in z.infolist() if i.filename.lower().endswith(('.jpg','.jpeg','.png'))]
        rwconteos=dict(Counter(Path(i.filename).parent.name for i in fotos))
    publicar(SALIDA/'realwaste_historico_verificado.json',{'archivo_historico':rwzip.relative_to(RAIZ).as_posix(),
        'sha256_zip_actual':sha(rwzip),'referencias_fotograficas_en_indice':len(fotos),'categorias_en_indice':rwconteos,
        'contenidos_de_fotos_abiertos':0,'anotaciones_instancia_originales':0,
        'no_se_descargo_otra_copia':True,'no_es_adquisicion_nueva':True,'test_original_evaluado':False})
    filas=[
      {'dataset':'WasteSegCluster v4','oficial':'https://data.mendeley.com/datasets/3wnf3n242s/4',
       'licencia':'CC BY 4.0 ficha y README del archivo; atribuir autores y procedencia Roboflow',
       'uso_academico':'sí con atribución','uso_comercial':'sí según CC BY; conservar derechos de terceros y atribución',
       'declaradas_imagenes':1947,'declaradas_instancias':'aproximadamente 3900',
       'verificado':'1947 JPG en índice; COCO train 1795/3559, valid 103/249; train son 359 nombres fuente con 5 variantes cada uno',
       'categorias':['Organic','Paper','Plastic','Composite_Rubber_Textile'],
       'formato':'COCO polígonos por instancia; categoría 0 es supercategoría sin instancias train',
       'archivo_real':'wasteSegCluster.zip, 100414243 bytes, SHA256 proveedor comprobado',
       'materiales':'Paper→papel y Plastic→plástico como propuestas; Organic sin asignar; compuesto conservado como compuesto',
       'calidad':'fotos fit-within 640; train con flips/giros/ruido documentados; no 1947 escenas independientes',
       'diversidad':'agua urbana; objetos pequeños, vegetación y basura mezclada; nueva apariencia pero dominio limitado',
       'duplicados':'nombres fuente agrupan variantes; 0 candidatos hash/pHash≤4 de las 12 seleccionadas frente a train/val y V27; independencia física no acreditada',
       'piloto_activo_imagenes':12,'anotaciones_publicas_piloto':w['anotaciones_piloto'],'aceptadas_V26':0,
       'decision':'solo valid externo publicado, una variante por nombre; candidatas auxiliares train, no evaluación EcoWasteFormer'},
      {'dataset':'DWSD v1','oficial':'https://data.mendeley.com/datasets/gr99ny6b8p/1',
       'licencia':'CC BY 4.0 en ficha oficial','uso_academico':'sí con atribución','uso_comercial':'sí según CC BY con atribución',
       'declaradas_imagenes':784,'declaradas_instancias':2350,
       'verificado':'784 pares imagen/máscara en índice, 640 Train y 144 Test; 640 máscaras Train L leídas; no LabelMe JSON ni leyenda',
       'categorias':['plastic containers','plastic bottles','thermocol','metal bottles','plastic cardboard','glass','thermocol plates','plastic','paper','plastic cups','paper cups','aluminum foil','cloth','nylon'],
       'formato':'PNG L valores discretos; sin IDs de instancia o clave valor→material verificables',
       'archivo_real':'DSWD.zip, 252719363 bytes, SHA256 proveedor comprobado',
       'materiales':'nombres declarados compatibles en parte con papel/vidrio/plástico/metal; no asignar materiales a códigos sin leyenda',
       'calidad':'muestras giradas y con cambios fuertes de color; faltan originales identificados y trazabilidad de aumentos',
       'diversidad':'muestras exteriores, césped y calles; sin garantías de objetos físicos nuevos',
       'duplicados':'0 candidatos de las 8 muestras frente a train/val y V27; no demuestra independencia de variantes',
       'piloto_activo_imagenes':0,'anotaciones_publicas_piloto':0,'aceptadas_V26':0,
       'decision':'8 muestras privadas en cuarentena; no convertir componentes semánticos en instancias GT'},
      {'dataset':'RealWaste UCI','oficial':'https://archive.ics.uci.edu/dataset/908/realwaste',
       'licencia':'CC BY 4.0 en UCI','uso_academico':'sí con atribución','uso_comercial':'sí según CC BY con atribución',
       'declaradas_imagenes':4752,'declaradas_instancias':'0 anotaciones espaciales; 4752 imágenes de clasificación',
       'verificado':f'{len(fotos)} fotografías en índice del ZIP histórico; nueve categorías; no se abrieron imágenes test',
       'categorias':list(rwconteos),'formato':'JPEG 524×524 publicado, etiquetas globales sin máscara/bbox',
       'archivo_real':'ZIP ya existente, no descargado nuevamente; SHA actual registrado',
       'materiales':'Food Organics y Paper son candidatos para anotación humana, no polígonos existentes; Vegetation no se convierte automáticamente en comida',
       'calidad':'no evaluación geométrica posible sin anotaciones; históricos ya empleados por clasificador',
       'diversidad':'vertedero real; independencia respecto del proyecto no existe como fuente nueva',
       'duplicados':'fuente ya histórica; no acreditar fotos como nuevas ni abrir test para buscar candidatos',
       'piloto_activo_imagenes':0,'anotaciones_publicas_piloto':0,'aceptadas_V26':0,
       'decision':'no repetir descarga; cualquier anotación futura solo de train histórico y con reuso explícito'},
    ]
    for fuente,nombre,url,licencia,imgs,anns,nota in [
        ('zerowaste','ZeroWaste-f','https://github.com/dbash/zerowaste','CC BY-NC 4.0 autores frente a CC BY Zenodo; alcance académico NC',24,128,'Cartón/plástico/metal; completar objetos omitidos de las seis clases'),
        ('warp','WaRP-D','https://github.com/AIRI-Institute/WaRP','CC0 declarada en Kaggle original enlazado',24,97,'86 propuestas SAM geométricas; 11 cajas por refinar; materiales compuestos no se fuerzan'),
        ('hpi','HPI Food Waste','https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset','CC BY 4.0 fotos y registros',12,0,'12 comidas, no 215 objetos nuevos; indicar restos con cajas y SAM, sin etiquetas globales como máscaras')]:
        filas.append({'dataset':nombre,'oficial':url,'licencia':licencia,'uso_academico':'según informe V27',
          'uso_comercial':'ver alcance y conflicto en V27','declaradas_imagenes':None,'declaradas_instancias':None,
          'verificado':'recibos y hashes V27 conservados','categorias':'ver catálogo original V27','formato':'ver V27',
          'archivo_real':'fotografías originales del piloto V27, sin descargar de nuevo','materiales':nota,
          'calidad':nota,'diversidad':'ver informe V27','duplicados':'mismas fotografías V27 reutilizadas explícitamente',
          'piloto_activo_imagenes':imgs,'anotaciones_publicas_piloto':anns,'aceptadas_V26':0,'decision':'reutilizar tareas en editor local'})
    publicar(SALIDA/'comparacion_fuentes.json',filas)
    with (SALIDA/'comparacion_fuentes.csv').open('x',encoding='utf-8-sig',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(filas[0]));out.writeheader()
        for row in filas:out.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    t=Taller(PROYECTO,RAIZ);anns=[a for i in t.imagenes for a in t.tarea(i)['anotaciones']]
    publicar(SALIDA/'estado_entrega_piloto.json',{'imagenes_activas':72,'fotografias_V27_reutilizadas':60,'fotografias_WSC_publicadas_activas':12,
        'DWSD_cuarentena':8,'propuestas_publicas':len(anns),'con_poligonos':sum(bool(a['poligonos']) for a in anns),
        'solo_caja_pendiente':sum(not a['poligonos'] for a in anns),'propuestas_SAM_activas':sum(bool(a.get('propuesta_ia_id')) for a in anns),
        'con_huecos_requieren_compatibilidad_V26':sum(bool(a.get('huecos')) for a in anns),
        'materiales_propuestos_NO_GT':dict(Counter(a['material'] for a in anns)),
        'imagenes_HPI_sin_anotaciones_espaciales':12,'regiones_organicas_48_aceptadas':0,
        'revisiones_humanas':sum(len(t.diario(i)) for i in t.imagenes),'aceptadas_V26':0,'cuotas_train_pendientes':1550,
        'cuotas_generalizacion_pendientes':600,'organicos_utilizables_pendientes':300,'entrenamiento':False,'test_evaluado':False})
    publicar_bytes(DATOS/'ATRIBUCION_V27_2.txt',('''Archivos privados de investigación. No redistribuidos en Git.
WasteSegCluster v4: Rezaur Rahman Ratul, Kona Moni, Aranya Kishor Das,
Aditya Sowdagor, Suman Ahmmed, Mohammad Nurul Huda y Ohidujjaman.
https://doi.org/10.17632/3wnf3n242s.4 CC BY 4.0.
El ZIP contiene exportación Roboflow Extened_Dataset of Waste Classification v3,
14 marzo 2026. Fotografías publicadas con auto-orientación y resize fit-within 640.
No se usan las cinco variantes aumentadas de train como fotos independientes.
Se conservan 12 fotografías de valid externo y sus COCO originales; no son val/test EcoWasteFormer.
COCO originales conservados. Mapeos, normalización de contornos y propuestas son derivados pendientes.

DWSD: Asfak Ali, Suvojit Acharjee, MD Manarul Sk, Salman Alharthi,
Sheli Sinha Chaudhuri y Adnan Akhunzada. https://doi.org/10.17632/gr99ny6b8p.1
CC BY 4.0 según Mendeley. Ocho pares Train conservados solo en cuarentena.
No se asignan IDs físicos ni GT por instancia desde máscaras L.

RealWaste: Sam Single, Saeid Iranmanesh, Raad Raad. https://doi.org/10.24432/C5SS4G.
CC BY 4.0, fuente histórica; sin nueva descarga ni conversión de etiquetas globales a máscaras.

ZeroWaste, WaRP y HPI: conservar ATRIBUCION.txt y licencias originales del piloto V27.
SAM 2.1 Tiny: Meta Platforms y colaboradores, Apache-2.0 para código/pesos.
https://github.com/facebookresearch/sam2 Copia completa de LICENSE en modelos/fuente/LICENSE.
SAM propone regiones, no materiales. Sus máscaras no constituyen aprobación humana.
Los montajes y capturas de editor son derivados locales; no se publican fotografías de terceros.
''').encode('utf-8'))
    print('Comparación de seis fuentes y estado real del piloto guardados.',flush=True)


if __name__=='__main__':main()
