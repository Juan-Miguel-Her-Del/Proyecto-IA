"""Descarga reanudable por archivo de fuentes de datos, sin descargar modelos."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
import zipfile

RAIZ = Path(__file__).resolve().parents[1]
ORIGINALES = RAIZ / "datos" / "originales"


def guardar_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def descargar(url, destino, intentos=3):
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.is_file() and destino.stat().st_size:
        return
    parcial = destino.with_suffix(destino.suffix + ".part")
    for intento in range(intentos):
        try:
            req = Request(url, headers={"User-Agent": "EcoWasteFormer-academic-dataset-preparation/1.0"})
            with urlopen(req, timeout=45) as entrada, parcial.open("wb") as salida:
                while bloque := entrada.read(1024 * 1024):
                    salida.write(bloque)
            if not parcial.stat().st_size:
                raise ValueError("Archivo descargado vacío")
            parcial.replace(destino)
            return
        except Exception:
            if intento + 1 == intentos:
                raise
            time.sleep(1 + intento)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as archivo:
        while bloque := archivo.read(1024 * 1024):
            digest.update(bloque)
    return digest.hexdigest()


def metadatos():
    destino = ORIGINALES / "taco" / "annotations.json"
    url = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
    descargar(url, destino)
    contenido = json.loads(destino.read_text(encoding="utf-8"))
    guardar_json(ORIGINALES / "taco" / "procedencia.json", {
        "fuente": "TACO", "url_anotaciones": url,
        "sha256_anotaciones": sha256(destino),
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "imagenes_declaradas": len(contenido["images"]),
        "anotaciones": len(contenido["annotations"]),
        "licencias_declaradas_en_coco": contenido.get("licenses", []),
        "atribucion": "Pedro F. Proença y Pedro Simões, TACO: Trash Annotations in Context for Litter Detection (2020)",
    })
    print(f"TACO: {len(contenido['images'])} imágenes, {len(contenido['annotations'])} anotaciones", flush=True)
    return contenido


def realwaste():
    url = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
    archivo = ORIGINALES / "realwaste.zip"
    print("Descargando RealWaste desde UCI...", flush=True)
    descargar(url, archivo)
    destino = (ORIGINALES / "realwaste").resolve()
    destino.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archivo) as comprimido:
        for miembro in comprimido.infolist():
            objetivo = (destino / miembro.filename).resolve()
            if not objetivo.is_relative_to(destino):
                raise ValueError(f"Ruta insegura en ZIP: {miembro.filename}")
            if miembro.is_dir():
                objetivo.mkdir(parents=True, exist_ok=True)
                continue
            if objetivo.is_file() and objetivo.stat().st_size == miembro.file_size:
                continue
            objetivo.parent.mkdir(parents=True, exist_ok=True)
            with comprimido.open(miembro) as entrada, objetivo.open("wb") as salida:
                while bloque := entrada.read(1024 * 1024):
                    salida.write(bloque)
    guardar_json(destino / "procedencia.json", {
        "fuente": "RealWaste", "url": url, "sha256_zip": sha256(archivo),
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "licencia": "CC BY 4.0", "licencia_url": "https://creativecommons.org/licenses/by/4.0/",
        "atribucion": "Sam Single, Saeid Iranmanesh y Raad Raad (2023), RealWaste, UCI, DOI:10.24432/C5SS4G",
        "ficha": "https://archive.ics.uci.edu/dataset/908/realwaste",
    })
    print("RealWaste descargado y extraído.", flush=True)


def taco(datos):
    raiz = (ORIGINALES / "taco" / "imagenes").resolve()
    fallos, completos = [], 0

    def tarea(imagen):
        objetivo = (raiz / imagen["file_name"]).resolve()
        if not objetivo.is_relative_to(raiz):
            raise ValueError("Ruta de imagen fuera del dataset")
        # La URL normal (no flickr_640_url) corresponde a las dimensiones COCO.
        url = imagen.get("flickr_url") or imagen.get("url")
        if not url or not url.startswith("https://"):
            raise ValueError("Falta URL HTTPS de imagen original")
        descargar(url, objetivo)
        return imagen["id"]

    with ThreadPoolExecutor(max_workers=6) as pool:
        tareas = {pool.submit(tarea, im): im for im in datos["images"]}
        for futuro in as_completed(tareas):
            im = tareas[futuro]
            try:
                futuro.result()
                completos += 1
            except Exception as error:
                fallos.append({"image_id": im["id"], "archivo": im["file_name"], "error": str(error)})
            if (completos + len(fallos)) % 100 == 0:
                print(f"TACO: {completos} disponibles, {len(fallos)} fallidas", flush=True)
    guardar_json(ORIGINALES / "taco" / "descarga.json", {
        "disponibles": completos, "fallos": fallos,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"TACO terminado: {completos} disponibles, {len(fallos)} fallidas", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fuente", choices=["metadatos", "realwaste", "taco", "todo"], default="todo")
    args = parser.parse_args()
    datos = metadatos() if args.fuente in ["metadatos", "taco", "todo"] else None
    if args.fuente in ["realwaste", "todo"]:
        realwaste()
    if args.fuente in ["taco", "todo"]:
        taco(datos)


if __name__ == "__main__":
    main()
