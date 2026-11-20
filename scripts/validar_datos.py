"""Comprobaciones independientes de manifiestos, geometría y fugas entre particiones."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from PIL import Image
from ecowaste.datos import caja_valida


def leer(path):
    return [json.loads(linea) for linea in path.read_text(encoding="utf-8").splitlines() if linea.strip()]


def main():
    errores, advertencias = [], []
    clases = json.loads((RAIZ / "configuracion/datasets.json").read_text(encoding="utf-8"))["clases"]
    global_grupos, global_pixeles, global_archivos = defaultdict(set), defaultdict(set), defaultdict(set)
    global_adquisicion = defaultdict(set)
    conteos, hashes = {}, {}
    for tarea in ["clasificacion", "deteccion"]:
        conteos[tarea] = {}
        todos_ids = set()
        for split in ["train", "val", "test"]:
            path = RAIZ / "datos/procesados" / tarea / f"{split}.jsonl"
            hashes[path.relative_to(RAIZ).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            registros = leer(path)
            cuenta = Counter()
            for r in registros:
                if r["id"] in todos_ids:
                    errores.append(f"ID repetido entre particiones: {r['id']}")
                todos_ids.add(r["id"])
                if r["split"] != split:
                    errores.append(f"Split inconsistente en {r['id']}")
                global_grupos[r["grupo"]].add(split)
                global_adquisicion[r["grupo_adquisicion"]].add(split)
                global_pixeles[r["sha256_pixeles"]].add(split)
                global_archivos[r["sha256_archivo"]].add(split)
                archivo = RAIZ / r["ruta"]
                if not archivo.is_file():
                    errores.append(f"Archivo ausente: {r['id']}")
                    continue
                if hashlib.sha256(archivo.read_bytes()).hexdigest() != r["sha256_archivo"]:
                    errores.append(f"Archivo modificado tras preparación: {r['id']}")
                with Image.open(archivo) as imagen:
                    if imagen.size != (r["ancho"], r["alto"]):
                        errores.append(f"Dimensiones inconsistentes: {r['id']}")
                if tarea == "clasificacion":
                    if r["clase"] not in clases:
                        errores.append(f"Clase desconocida: {r['id']}")
                    cuenta.update([r["clase"]])
                    if "cajas" in r:
                        errores.append(f"Se fabricaron cajas para clasificación: {r['id']}")
                else:
                    for a in r["cajas"]:
                        if not caja_valida(a["bbox"], r["ancho"], r["alto"]):
                            errores.append(f"Caja inválida: {r['id']}")
                        if a["clase"]:
                            if a["clase"] not in clases:
                                errores.append(f"Clase desconocida: {a['clase']}")
                            cuenta.update([a["clase"]])
            if not registros:
                errores.append(f"Partición vacía: {tarea}/{split}")
            conteos[tarea][split] = {"imagenes": len(registros), "clases": {c: cuenta[c] for c in clases}}
            for c in clases:
                if cuenta[c] == 0:
                    advertencias.append(f"Sin ejemplos de {c} en {tarea}/{split}")
            if tarea == "deteccion":
                coco_path = path.with_name(f"{split}_coco.json")
                coco = json.loads(coco_path.read_text(encoding="utf-8"))
                ids_imagen = {im["id"] for im in coco["images"]}
                ids_anno = [a["id"] for a in coco["annotations"] + coco["ignore_regions"]]
                if len(ids_imagen) != len(registros) or len(ids_anno) != len(set(ids_anno)):
                    errores.append(f"IDs de COCO inválidos: {split}")
                if len(coco["annotations"]) != sum(cuenta.values()):
                    errores.append(f"Conteo COCO y manifiesto difieren: {split}")
                if any(a["image_id"] not in ids_imagen for a in coco["annotations"] + coco["ignore_regions"]):
                    errores.append(f"Referencia de imagen inexistente en COCO: {split}")
    fugas = {}
    for nombre, grupos in [("grupos", global_grupos), ("adquisicion", global_adquisicion), ("pixeles", global_pixeles), ("archivos", global_archivos)]:
        fugas[nombre] = sum(len(splits) > 1 for splits in grupos.values())
        if fugas[nombre]:
            errores.append(f"Fuga entre particiones por {nombre}: {fugas[nombre]}")
    advertencias.extend([
        "Cobertura real de orgánicos en detección insuficiente: ocho cajas en la fuente TACO antes de filtrado.",
        "No hay garantía de independencia por sesión de captura; los grupos son inferidos por similitud.",
        "Los ignore_regions requieren soporte explícito en el futuro loader del detector.",
        "Licencias individuales TACO no especificadas en JSON; se conserva fuente y declaración pública de los autores.",
    ])
    # Corpus de lenguaje: disyunción de grupos y preguntas completas normalizadas.
    from ecowaste.asistente import palabras
    grupos_texto, preguntas_texto = defaultdict(set), defaultdict(set)
    for split in ["train", "val", "test"]:
        for r in leer(RAIZ / "datos/asistente" / f"{split}.jsonl"):
            grupos_texto[r["grupo"]].add(split)
            preguntas_texto[tuple(palabras(r["pregunta"]))].add(split)
    fugas["grupos_texto"] = sum(len(s) > 1 for s in grupos_texto.values())
    fugas["preguntas_texto"] = sum(len(s) > 1 for s in preguntas_texto.values())
    if fugas["grupos_texto"] or fugas["preguntas_texto"]:
        errores.append("Fuga en corpus del asistente")
    reporte = {"validacion_estructural_correcta": not errores, "errores": errores,
        "advertencias": advertencias, "fugas_detectadas": fugas, "conteos": conteos,
        "sha256_manifiestos": hashes,
        "listo_clasificacion_experimental": not errores,
        "listo_detector_seis_clases_con_evaluacion_representativa": False,
        "motivo_detector": "Faltan ejemplos reales diversos de orgánicos y videos independientes, aunque la geometría y particiones se comprueben correctamente."}
    salida = RAIZ / "datos/reportes/validacion.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reporte, ensure_ascii=False, indent=2))
    return 0 if not errores else 1


if __name__ == "__main__":
    raise SystemExit(main())
