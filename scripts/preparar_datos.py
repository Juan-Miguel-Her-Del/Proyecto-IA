"""Audita imágenes/cajas, agrupa duplicados y exporta manifiestos y COCO sin copiar imágenes."""
from collections import Counter, defaultdict
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ecowaste.datos import ArbolHash, UnionFind, caja_valida, recortar_desborde_minimo

CONFIG = json.loads((RAIZ / "configuracion/datasets.json").read_text(encoding="utf-8"))
CLASES = CONFIG["clases"]
SALIDA = RAIZ / "datos/procesados"
REPORTE = RAIZ / "datos/reportes"


def escribir(path, datos):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(datos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonl(path, registros):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in registros), encoding="utf-8")


def inspeccionar(registro):
    r = dict(registro)
    try:
        archivo = RAIZ / r["ruta"]
        with Image.open(archivo) as imagen:
            imagen.verify()
        with Image.open(archivo) as imagen:
            r["orientacion_exif_original"] = imagen.getexif().get(274, 1)
            if r["fuente"] == "taco" and r["orientacion_exif_original"] in range(2, 9):
                # La interfaz de anotación usa la orientación visual. No rotar las cajas otra vez.
                rgb = ImageOps.exif_transpose(imagen).convert("RGB")
                if rgb.size != tuple(r["dimensiones_declaradas"]):
                    raise ValueError("EXIF no reconcilia las dimensiones de COCO")
                normalizada = RAIZ / "datos/normalizados/taco" / (str(r["id_original"]) + ".png")
                normalizada.parent.mkdir(parents=True, exist_ok=True)
                rgb.save(normalizada, compress_level=3)
                r["ruta_original"] = r["ruta"]
                r["sha256_original"] = hashlib.sha256(archivo.read_bytes()).hexdigest()
                r["ruta"] = normalizada.relative_to(RAIZ).as_posix()
                r["normalizacion"] = "exif_transpose_a_png_sin_recompresion_con_perdida"
                archivo = normalizada
            else:
                rgb = imagen.convert("RGB")
            r["ancho"], r["alto"] = rgb.size
            r["sha256_pixeles"] = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            gris = np.array(rgb.resize((32, 32), Image.Resampling.LANCZOS).convert("L"), dtype=np.float32)
            dct = cv2.dct(gris)[:8, :8].flatten()
            bits = dct > np.median(dct[1:])
            bits[0] = False
            r["phash"] = f"{sum(int(bit) << i for i, bit in enumerate(bits)):016x}"
            pequeno = np.array(rgb.resize((9, 8), Image.Resampling.LANCZOS).convert("L"))
            r["dhash"] = f"{sum(int(b) << i for i, b in enumerate((pequeno[:, 1:] > pequeno[:, :-1]).flatten())):016x}"
            r["orientacion_exif"] = imagen.getexif().get(274, 1)
        r["sha256_archivo"] = hashlib.sha256(archivo.read_bytes()).hexdigest()
        if r["fuente"] == "taco" and (r["ancho"], r["alto"]) != tuple(r["dimensiones_declaradas"]):
            raise ValueError(f"Dimensiones reales {r['ancho']}x{r['alto']} distintas a COCO {r['dimensiones_declaradas']}")
        r["integridad"] = "correcta"
    except Exception as error:
        r["integridad"] = "fallida"
        r["error"] = f"{type(error).__name__}: {error}"
    return r


def registros_originales():
    registros = []
    real = RAIZ / "datos/originales/realwaste"
    for p in sorted(real.rglob("*")):
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        categoria = p.parent.name
        numero = re.search(r"(\d+)$", p.stem)
        registros.append({"id": f"realwaste:{p.relative_to(real).as_posix()}", "fuente": "realwaste",
            "ruta": p.relative_to(RAIZ).as_posix(), "categoria_original": categoria,
            "clase": CONFIG["realwaste"].get(categoria), "licencia": "CC BY 4.0",
            "licencia_url": "https://creativecommons.org/licenses/by/4.0/",
            "url_fuente": "https://archive.ics.uci.edu/dataset/908/realwaste",
            "grupo_adquisicion": f"realwaste:{categoria}:bloque_{int(numero.group(1)) // 10}" if numero else f"realwaste:{categoria}:{p.stem}",
            "grupo_adquisicion_metodo": "heuristica_conservadora_de_10_nombres_consecutivos_no_sesion_demostrada"})
    datos = json.loads((RAIZ / "datos/originales/taco/annotations.json").read_text(encoding="utf-8"))
    categorias = {c["id"]: c["name"] for c in datos["categories"]}
    if set(categorias.values()) != set(CONFIG["taco"]) | set(CONFIG["categorias_taco_excluidas"]):
        raise ValueError("El mapeo TACO no cubre exactamente todas las categorías de origen")
    anotaciones = defaultdict(list)
    errores_cajas, correcciones_cajas = [], []
    imagenes = {im["id"]: im for im in datos["images"]}
    # La fuente contiene IDs repetidos; usar un ID interno único sin perder el original.
    ids = Counter(a["id"] for a in datos["annotations"])
    escribir(REPORTE / "ids_origen_repetidos.json", {str(k): n for k, n in ids.items() if n > 1})
    for indice_anotacion, a in enumerate(datos["annotations"], start=1):
        im = imagenes.get(a["image_id"])
        if im is None or a["category_id"] not in categorias:
            errores_cajas.append({"id": a["id"], "motivo": "referencia_invalida"})
            continue
        bbox = a["bbox"]
        if not caja_valida(bbox, im["width"], im["height"]):
            bbox = recortar_desborde_minimo(bbox, im["width"], im["height"])
            if bbox is None:
                errores_cajas.append({"id": a["id"], "motivo": "caja_invalida", "bbox": a["bbox"]})
                # No introducir falsos negativos al omitir silenciosamente una caja defectuosa.
                anotaciones[a["image_id"]].append({"invalida": True})
                continue
            correcciones_cajas.append({"id_original": a["id"], "image_id": im["id"], "bbox_original": a["bbox"], "bbox": bbox, "motivo": "interseccion_con_imagen_desborde_menor_2px_area_conservada_98pct"})
        categoria = categorias[a["category_id"]]
        anotaciones[a["image_id"]].append({"id_original": a["id"], "id_interno": indice_anotacion, "bbox": bbox, "bbox_original": a["bbox"],
            "clase": CONFIG["taco"].get(categoria), "categoria_original": categoria,
            "iscrowd": a.get("iscrowd", 0)})
    for im in datos["images"]:
        p = RAIZ / "datos/originales/taco/imagenes" / im["file_name"]
        registros.append({"id": f"taco:{im['id']}", "fuente": "taco", "id_original": im["id"],
            "ruta": p.relative_to(RAIZ).as_posix(), "dimensiones_declaradas": [im["width"], im["height"]],
            "licencia_id_original": im.get("license"),
            "grupo_adquisicion": "taco:" + im["file_name"].split("/")[0],
            "grupo_adquisicion_metodo": "lote_original_completo_no_se_divide_entre_particiones",
            "licencia": "no_especificada_en_metadatos" if im.get("license") is None else im["license"],
            "nota_licencia": "Autores declaran imágenes bajo licencias públicas: https://arxiv.org/html/2003.06975v1#S2; conservar procedencia, sin atribuir licencia MIT a las fotos.",
            "url_fuente": im.get("flickr_url"), "cajas": anotaciones[im["id"]]})
    escribir(REPORTE / "correcciones_cajas.json", correcciones_cajas)
    return registros, errores_cajas


def agrupar(registros):
    uf = UnionFind(len(registros))
    adquisicion = {}
    for i, r in enumerate(registros):
        clave = r["grupo_adquisicion"]
        if clave in adquisicion:
            uf.unir(i, adquisicion[clave])
        else:
            adquisicion[clave] = i
    exactos, similares = {}, []
    arbol = ArbolHash()
    for i, r in enumerate(registros):
        clave = r["sha256_pixeles"]
        if clave in exactos:
            j = exactos[clave]
            uf.unir(i, j)
            similares.append({"a": registros[j]["id"], "b": r["id"], "tipo": "pixeles_identicos"})
        else:
            exactos[clave] = i
        phash, dhash = int(r["phash"], 16), int(r["dhash"], 16)
        for j in arbol.buscar(phash, 4):
            otro = registros[j]
            ratio_a, ratio_b = r["ancho"] / r["alto"], otro["ancho"] / otro["alto"]
            if abs(ratio_a / ratio_b - 1) <= 0.05 and (dhash ^ int(otro["dhash"], 16)).bit_count() <= 8:
                uf.unir(i, j)
                if clave != otro["sha256_pixeles"]:
                    similares.append({"a": otro["id"], "b": r["id"], "tipo": "similitud_perceptual", "distancia_phash": (phash ^ int(otro["phash"], 16)).bit_count()})
        arbol.agregar(phash, i)
    for i, r in enumerate(registros):
        r["grupo"] = f"escena_{uf.find(i):06d}"
    return similares


def asignar_particiones(registros):
    grupos = defaultdict(list)
    for r in registros:
        grupos[r["grupo"]].append(r)
    vectores = {}
    for grupo, filas in grupos.items():
        vector = Counter()
        for r in filas:
            etiquetas = [r["clase"]] if r["fuente"] == "realwaste" else {a["clase"] for a in r["cajas"] if a.get("clase")}
            vector.update(f"{r['fuente']}:{c}" for c in etiquetas)
            vector.update([r["fuente"] + ":total"])
        vectores[grupo] = vector
    total = sum(vectores.values(), Counter())
    actuales = {s: Counter() for s in CONFIG["particiones"]}
    rng = random.Random(CONFIG["semilla"])
    orden = list(grupos)
    rng.shuffle(orden)
    orden.sort(key=lambda g: max(1 / total[c] for c in vectores[g]), reverse=True)
    for grupo in orden:
        vector = vectores[grupo]
        def puntuar(split):
            fraccion = CONFIG["particiones"][split]
            # Reducción del error cuadrático relativo después de asignar el grupo completo.
            # Incluye el tamaño del lote para no sobrellenar val/test por su déficit previo.
            score = sum((2 * (total[c] * fraccion - actuales[split][c]) * n - n * n) / max(total[c] * fraccion, 1) ** 2 for c, n in vector.items())
            return score, fraccion
        split = max(CONFIG["particiones"], key=puntuar)
        actuales[split].update(vector)
        for r in grupos[grupo]:
            r["split"] = split


def exportar(registros):
    for s in CONFIG["particiones"]:
        clasificacion = [r for r in registros if r["fuente"] == "realwaste" and r["split"] == s]
        deteccion = [r for r in registros if r["fuente"] == "taco" and r["split"] == s]
        jsonl(SALIDA / "clasificacion" / f"{s}.jsonl", clasificacion)
        jsonl(SALIDA / "deteccion" / f"{s}.jsonl", deteccion)
        coco = {"images": [], "annotations": [], "ignore_regions": [],
            "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASES)]}
        for r in deteccion:
            coco["images"].append({"id": r["id_original"], "file_name": r["ruta"], "width": r["ancho"], "height": r["alto"], "grupo": r["grupo"]})
            for a in r["cajas"]:
                registro = {"id": a["id_interno"], "id_original": a["id_original"], "image_id": r["id_original"], "bbox": a["bbox"],
                    "area": a["bbox"][2] * a["bbox"][3], "iscrowd": a["iscrowd"]}
                if a["clase"]:
                    registro["category_id"] = CLASES.index(a["clase"]) + 1
                    coco["annotations"].append(registro)
                else:
                    registro["motivo"] = CONFIG["categorias_taco_excluidas"][a["categoria_original"]]
                    coco["ignore_regions"].append(registro)
        escribir(SALIDA / "deteccion" / f"{s}_coco.json", coco)


def muestras(registros):
    # Panel de muestras de entrenamiento; no se eligen modelos inspeccionando el test.
    for fuente in ["realwaste", "taco"]:
        panel = Image.new("RGB", (4 * 240, 6 * 200), "white")
        dibujo = ImageDraw.Draw(panel)
        for ci, clase in enumerate(CLASES):
            candidatos = [r for r in registros if r["fuente"] == fuente and r["split"] == "train" and
                (r.get("clase") == clase or any(a.get("clase") == clase for a in r.get("cajas", [])))]
            for col, r in enumerate(candidatos[:4]):
                with Image.open(RAIZ / r["ruta"]) as im:
                    vista = im.convert("RGB")
                    if fuente == "taco":
                        a = next(a for a in r["cajas"] if a.get("clase") == clase)
                        x, y, w, h = a["bbox"]
                        vista = vista.crop((int(x), int(y), int(np.ceil(x + w)), int(np.ceil(y + h))))
                    vista = ImageOps.contain(vista, (230, 165))
                    panel.paste(vista, (col * 240, ci * 200 + 25))
                dibujo.text((col * 240 + 4, ci * 200 + 4), f"{clase} {r['id'][:18]}", fill="black")
        REPORTE.mkdir(parents=True, exist_ok=True)
        panel.save(REPORTE / f"muestras_{fuente}.jpg", quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reusar-inventario", action="store_true", help="Reutiliza inspecciones solo si los hashes de archivo original y normalizado siguen coincidiendo")
    args = parser.parse_args()
    originales, errores_cajas = registros_originales()
    print(f"Auditando {len(originales)} imágenes...", flush=True)
    auditados = []
    cache = {}
    inventario = REPORTE / "inventario_imagenes.jsonl"
    if args.reusar_inventario and inventario.is_file():
        cache = {r["id"]: r for r in (json.loads(l) for l in inventario.read_text(encoding="utf-8").splitlines())}
    def tarea(r):
        previa = cache.get(r["id"], {})
        if previa.get("integridad") == "correcta":
            original = RAIZ / r["ruta"]
            vigente = RAIZ / previa["ruta"]
            if original.is_file() and vigente.is_file():
                sha_original = hashlib.sha256(original.read_bytes()).hexdigest()
                sha_vigente = hashlib.sha256(vigente.read_bytes()).hexdigest() if original != vigente else sha_original
                if sha_original == previa.get("sha256_original", previa["sha256_archivo"]) and sha_vigente == previa["sha256_archivo"]:
                    campos = ["ancho", "alto", "phash", "dhash", "sha256_pixeles", "sha256_archivo", "orientacion_exif", "orientacion_exif_original", "normalizacion", "ruta_original", "sha256_original", "ruta", "integridad"]
                    return dict(r, **{k: previa[k] for k in campos if k in previa})
        return inspeccionar(r)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(tarea, originales):
            auditados.append(r)
            if len(auditados) % 500 == 0:
                print(f"Verificadas {len(auditados)} imágenes", flush=True)
    validos = [r for r in auditados if r["integridad"] == "correcta"]
    pares = agrupar(validos)
    excluidos, elegibles = [], []
    vistos = {}
    conflictos = set()
    clases_por_hash = defaultdict(set)
    for r in validos:
        if r.get("clase"):
            clases_por_hash[r["sha256_pixeles"]].add(r["clase"])
    conflictos = {h for h, clases in clases_por_hash.items() if len(clases) > 1}
    for r in validos:
        motivo = None
        if r["sha256_pixeles"] in conflictos:
            motivo = "etiquetas_contradictorias_mismos_pixeles"
        elif r["fuente"] == "realwaste" and not r["clase"]:
            motivo = "categoria_fuera_de_alcance"
        elif r["fuente"] == "taco" and any(a.get("invalida") for a in r["cajas"]):
            motivo = "anotacion_invalida"
        elif r["fuente"] == "taco" and not any(a.get("clase") for a in r["cajas"]):
            motivo = "sin_cajas_de_las_seis_clases_no_usar_como_negativo"
        clave = (r["fuente"], r["sha256_pixeles"])
        if not motivo and clave in vistos:
            motivo = "duplicado_exacto"
            if r["fuente"] == "taco":
                canonico = vistos[clave]
                ya = {(a.get("clase"), tuple(a["bbox"])) for a in canonico["cajas"]}
                canonico["cajas"].extend(a for a in r["cajas"] if (a.get("clase"), tuple(a["bbox"])) not in ya)
        if motivo:
            excluidos.append({"id": r["id"], "motivo": motivo})
        else:
            vistos[clave] = r
            elegibles.append(r)
    asignar_particiones(elegibles)
    exportar(elegibles)
    jsonl(REPORTE / "inventario_imagenes.jsonl", auditados)
    escribir(REPORTE / "exclusiones.json", excluidos)
    escribir(REPORTE / "cajas_invalidas.json", errores_cajas)
    escribir(REPORTE / "pares_similares.json", pares)
    conteos = {}
    for fuente in ["realwaste", "taco"]:
        conteos[fuente] = {}
        for s in CONFIG["particiones"]:
            filas = [r for r in elegibles if r["fuente"] == fuente and r["split"] == s]
            cuenta = Counter(r["clase"] for r in filas) if fuente == "realwaste" else Counter(a["clase"] for r in filas for a in r["cajas"] if a.get("clase"))
            conteos[fuente][s] = {"imagenes": len(filas), "grupos": len({r["grupo"] for r in filas}), "clases": {c: cuenta[c] for c in CLASES}}
    pequenos = Counter()
    for r in elegibles:
        if r["fuente"] == "taco":
            escala = 384 / max(r["ancho"], r["alto"])
            pequenos.update(a["clase"] for a in r["cajas"] if a.get("clase") and min(a["bbox"][2:]) * escala < 8)
    faltantes = [r for r in auditados if r["integridad"] != "correcta"]
    informe = {"fecha_utc": datetime.now(timezone.utc).isoformat(), "semilla": CONFIG["semilla"],
        "imagenes_inventariadas": len(originales), "imagenes_integras": len(validos),
        "imagenes_orientacion_normalizada": sum("normalizacion" in r for r in validos),
        "fallos_integridad": [{"id": r["id"], "error": r["error"]} for r in faltantes],
        "cajas_invalidas": len(errores_cajas), "exclusiones": dict(Counter(r["motivo"] for r in excluidos)),
        "pares_similitud": len(pares), "conflictos_etiqueta": len(conflictos), "conteos": conteos,
        "objetos_con_lado_menor_8px_al_redimensionar_384": dict(pequenos),
        "clasificacion": "preparada_para_experimentos_con_limitacion_de_escenas_inferidas",
        "deteccion": "preparada_estructuralmente_cobertura_organicos_insuficiente",
        "licencias_taco": "La fuente declara licencias públicas; metadatos individuales no especifican licencia. Conservar URLs y no atribuir MIT a imágenes.",
        "limitaciones": ["Solo ocho Food waste en TACO original; no demuestra generalización para orgánicos.",
            "Las regiones ignoradas deben respetarse en el entrenamiento y evaluación del detector; los COCO incluyen ignore_regions propio.",
            "La agrupación usa píxeles y similitud perceptual; no hay identificadores completos de sesión para garantizar independencia de capturas.",
            "Se mantienen juntos los lotes TACO completos y bloques de diez nombres consecutivos de RealWaste; las proporciones 70/15/15 son aproximadas por esa restricción.",
            "No hay videos propios independientes ni escenas vacías verificadas en estas fuentes.",
            "Los paneles son muestreo visual de entrenamiento; las etiquetas completas provienen del dataset y no se revisaron una por una."]}
    escribir(REPORTE / "preparacion.json", informe)
    muestras(elegibles)
    print(json.dumps({k: informe[k] for k in ["imagenes_inventariadas", "imagenes_integras", "imagenes_orientacion_normalizada", "cajas_invalidas", "exclusiones", "conteos"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
