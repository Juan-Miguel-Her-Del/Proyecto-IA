"""Detección local por imagen con cajas originales y JSON utilizable por el asistente."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from PIL import Image, ImageDraw, ImageOps
import torch
from ecowaste.cajas import area, deshacer_letterbox
from ecowaste.datasets import CLASES, letterbox
from ecowaste.detector import cargar_detector
from ecowaste.detector_mapas import cargar_detector_mapas, cajas_desde_mapas
from ecowaste.detector_consultas import cargar_detector_consultas,postprocesar_consultas
from ecowaste.detector_centros import cargar_detector_centros,postprocesar_centros
from ecowaste.mosaico import predecir_mosaico,VERSION as VERSION_MOSAICO
from ecowaste.inferencia_detector import postprocesar
from ecowaste.refinamiento import refinar_imagen,VERSION as VERSION_REFINAMIENTO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imagen", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--umbral", type=float)
    parser.add_argument('--refinar',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--consultas-candidatas',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--mosaico',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument("--imagen-salida", type=Path)
    parser.add_argument("--json-salida", type=Path)
    args = parser.parse_args()
    seleccion=None
    if args.checkpoint is None:
        arquitectura = json.loads((RAIZ / "configuracion/arquitectura.json").read_text(encoding="utf-8"))
        args.checkpoint = RAIZ / arquitectura["detector"]["checkpoint"]
        seleccion=arquitectura['detector']
        if args.mosaico is None:args.mosaico=seleccion.get('modo_inferencia')=='mosaico_cinco_vistas'
        if args.refinar is None:args.refinar=arquitectura['detector'].get('modo_inferencia')=='regiones_y_clasificacion_auxiliar'
    if args.refinar is None:args.refinar=False
    if args.mosaico is None:args.mosaico=False
    if args.mosaico and args.refinar:parser.error('Seleccionar un solo modo de inferencia')
    if not args.imagen.is_file() or not args.checkpoint.is_file() or (args.umbral is not None and not 0 <= args.umbral <= 1):
        parser.error("Se requieren una imagen, un checkpoint y un umbral entre cero y uno")
    destinos=[p.resolve() for p in [args.imagen_salida,args.json_salida] if p]
    if (any(p in [args.imagen.resolve(),args.checkpoint.resolve()] for p in destinos)
            or len(set(destinos))!=len(destinos)):
        parser.error("Las salidas deben ser distintas y conservar la imagen y el checkpoint")
    torch.set_num_threads(4)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    cabecera = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    es_mapas = cabecera.get("tipo") == "ecowaste_mapas_desde_cero_v1"
    es_consultas = cabecera.get('tipo')=='ecowaste_consultas_desde_cero_v1'
    es_centros = cabecera.get('tipo')=='ecowaste_centros_desde_cero_v1'
    del cabecera
    cargador=cargar_detector_centros if es_centros else (cargar_detector_consultas if es_consultas else (cargar_detector_mapas if es_mapas else cargar_detector))
    modelo, metadatos = cargador(args.checkpoint, dispositivo)
    if args.mosaico and not (es_mapas or es_centros):parser.error('El mosaico requiere mapas o centros')
    descartar_origen=metadatos.get('config_entrenamiento',{}).get('descartar_nula',True)
    descartar_nula=seleccion.get('descartar_clase_nula',descartar_origen) if seleccion else descartar_origen
    if args.consultas_candidatas is not None:descartar_nula=not args.consultas_candidatas
    if args.consultas_candidatas and not es_consultas:parser.error('La opción de candidatas requiere un detector de consultas')
    if args.refinar and (not es_mapas or modelo.clasificador_auxiliar is None):
        parser.error('El refinamiento requiere un checkpoint con clasificación auxiliar')
    if args.umbral is None:
        args.umbral = metadatos.get("config_entrenamiento", {}).get("umbral_visualizacion", .35 if es_mapas else .25)
    with Image.open(args.imagen) as original:
        imagen = ImageOps.exif_transpose(original).convert("RGB")
    entrada, g = letterbox(imagen, modelo.imagen)
    with torch.inference_mode():
        raw = modelo(entrada.unsqueeze(0).to(dispositivo))
        if args.mosaico:
            pred=predecir_mosaico(modelo,imagen,args.umbral,amp=dispositivo=='cuda',
                umbral_propuestas=metadatos.get('config_entrenamiento',{}).get('umbral_evaluacion',.05))
        elif es_centros:
            pred=postprocesar_centros(raw,[g],args.umbral,metadatos.get('config_entrenamiento',{}).get('nms_iou',.5))[0]
        elif es_consultas:
            pred=postprocesar_consultas(raw,[g],args.umbral,descartar_nula=descartar_nula)[0]
        elif args.refinar:
            pred=refinar_imagen(modelo,raw[0],imagen,g,umbral=args.umbral)
        else:
            pred = cajas_desde_mapas(raw, umbral=args.umbral, area_minima=metadatos.get('config_entrenamiento',{}).get('area_minima_celdas',2), geometrias=[g],formulacion=modelo.configuracion_mapas.get('formulacion','softmax7'))[0] if es_mapas else postprocesar(raw, umbral=args.umbral, geometrias=[g], radio_centros=modelo.configuracion.get("radio_centros", 0))[0]
    boxes = deshacer_letterbox(pred["boxes"], g)
    detecciones = []
    for caja, score, etiqueta in zip(boxes, pred["scores"], pred["labels"]):
        if float(area(caja.unsqueeze(0))[0]) <= 0:
            continue
        detecciones.append({"material": CLASES[int(etiqueta)], "puntuacion": float(score),
            "bbox_xyxy": [round(v, 2) for v in caja.tolist()]})
    resultado = {"origen": "detector_transformer_propio_piloto_imagen_real", "imagen": str(args.imagen),
        "fecha_utc": datetime.now(timezone.utc).isoformat(), "modelo": 'EcoWasteFormer-Centros experimental' if es_centros else ('EcoWasteFormer-Consultas piloto' if es_consultas else ("EcoWasteFormer-DetMapas piloto" if es_mapas else "EcoWasteFormer-Det piloto")),
        "epoca_checkpoint": metadatos["epoca"], "umbral_puntuacion": args.umbral,
        "checkpoint": str(args.checkpoint), "estado_detector": "experimental_validacion_insuficiente",
        "map50_validacion_checkpoint": metadatos.get('mejor_map') if metadatos.get('epoca')==metadatos.get('mejor_epoca') and not args.refinar and not args.mosaico and (not es_consultas or descartar_origen==descartar_nula) else None,
        "mejor_map50_historial_entrenamiento": metadatos.get('mejor_map'),
        "map50_validacion_inferencia_seleccionada": seleccion.get('map50_validacion') if seleccion and args.mosaico==(seleccion.get('modo_inferencia')=='mosaico_cinco_vistas') else None,
        "map50_validacion_checkpoint_modo": 'centros_focales' if es_centros else ('consultas_directas' if es_consultas else "mapas_o_cajas_directos"),
        "fuente_supervision": metadatos.get('config_entrenamiento',{}).get('fuente_supervision','cajas'),
        "postprocesamiento": 'picos_locales_y_nms_por_clase' if es_centros else ('consultas_sin_nms' if es_consultas else ("componentes_solo_contenido_real_v2" if es_mapas else "cajas_y_nms_por_clase")),
        "formulacion": 'centros_gaussianos_focales' if es_centros else ('consultas_univocas' if es_consultas else (modelo.configuracion_mapas.get('formulacion','softmax7') if es_mapas else 'cajas_densas')),
        "modo_inferencia": VERSION_MOSAICO if args.mosaico else ('centros_focales' if es_centros else ('consultas_directas' if es_consultas else (VERSION_REFINAMIENTO if args.refinar else 'mapas_o_cajas_directos'))),
        "vistas_procesadas": 5 if args.mosaico else 1,
        "umbral_formacion_propuestas": metadatos.get('config_entrenamiento',{}).get('umbral_evaluacion',.05) if args.mosaico else args.umbral,
        "descartar_clase_nula": descartar_nula if es_consultas else None,
        "coordenadas": "xyxy_pixeles_imagen_con_orientacion_corregida", "ancho": imagen.width, "alto": imagen.height,
        "detecciones": detecciones, "limite": "Prototipo con resultados limitados. La puntuación no está calibrada y no demuestra limpieza o aprovechabilidad."}
    texto = json.dumps(resultado, ensure_ascii=False, indent=2)
    if args.json_salida:
        args.json_salida.parent.mkdir(parents=True, exist_ok=True)
        args.json_salida.write_text(texto + "\n", encoding="utf-8")
    if args.imagen_salida:
        dibujar = ImageDraw.Draw(imagen)
        colores = ["#0ca678", "#1c7ed6", "#e67700", "#7048e8", "#d6336c", "#5c940d"]
        for d in detecciones:
            color = colores[CLASES.index(d["material"])]
            dibujar.rectangle(d["bbox_xyxy"], outline=color, width=max(2, imagen.width // 250))
            dibujar.text((d["bbox_xyxy"][0], d["bbox_xyxy"][1]), f"{d['material']} {d['puntuacion']:.2f}", fill=color, stroke_width=1, stroke_fill="white")
        args.imagen_salida.parent.mkdir(parents=True, exist_ok=True)
        imagen.save(args.imagen_salida)
    print(texto)


if __name__ == "__main__":
    main()
