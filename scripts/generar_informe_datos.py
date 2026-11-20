"""Informe legible y comprobación de carga de los datos preparados."""
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import torch
from PIL import Image, ImageDraw, ImageOps
from ecowaste.datasets import CLASES, ResiduosDataset, collate_deteccion


def leer(path):
    return json.loads((RAIZ / path).read_text(encoding="utf-8"))


def main():
    preparacion = leer("datos/reportes/preparacion.json")
    validacion = leer("datos/reportes/validacion.json")
    asistente = leer("resultados/asistente_v1.json")
    baseline = leer("resultados/baseline_asistente.json")
    correcciones = leer("datos/reportes/correcciones_cajas.json")
    ids = leer("datos/reportes/ids_origen_repetidos.json")
    carga = {}
    for tarea in ["clasificacion", "deteccion"]:
        dataset = ResiduosDataset(RAIZ, tarea, "train")
        muestras = []
        for clase in CLASES:
            indice = next(i for i, r in enumerate(dataset.registros) if r.get("clase") == clase or any(a.get("clase") == clase for a in r.get("cajas", [])))
            imagen, objetivo = dataset[indice]
            assert torch.isfinite(imagen).all() and 0 <= imagen.min() <= imagen.max() <= 1
            if tarea == "deteccion":
                for key in ["boxes", "ignore_boxes"]:
                    assert torch.isfinite(objetivo[key]).all()
                    assert bool(((objetivo[key] >= 0) & (objetivo[key] <= 1)).all())
            muestras.append((imagen, objetivo))
        if tarea == "deteccion":
            lote, objetivos = collate_deteccion(muestras)
            carga[tarea] = {"forma_lote": list(lote.shape), "formato_cajas": "xyxy normalizado; labels 0..5",
                "regiones_ignoradas_en_lote": sum(len(o["ignore_boxes"]) for o in objetivos)}
        else:
            carga[tarea] = {"forma_lote": list(torch.stack([x for x, _ in muestras]).shape), "clases_cargadas": CLASES}
    (RAIZ / "datos/reportes/carga_tensores.json").write_text(json.dumps(carga, indent=2), encoding="utf-8")

    # Comprobar visualmente la geometría de muestras de entrenamiento normalizadas.
    filas = ResiduosDataset(RAIZ, "deteccion", "train").registros
    elegidas = []
    for orientacion in [6, 8, 3]:
        candidatos = [r for r in filas if r.get("orientacion_exif_original") == orientacion]
        if candidatos:
            elegidas.append(candidatos[0])
    panel = Image.new("RGB", (500 * max(len(elegidas), 1), 560), "white")
    dibujo = ImageDraw.Draw(panel)
    for col, r in enumerate(elegidas):
        with Image.open(RAIZ / r["ruta"]) as im:
            vista = ImageOps.contain(im.convert("RGB"), (490, 490))
        sx, sy = vista.width / r["ancho"], vista.height / r["alto"]
        pincel = ImageDraw.Draw(vista)
        for a in r["cajas"]:
            x, y, w, h = a["bbox"]
            color = "red" if a["clase"] else "orange"
            pincel.rectangle([x * sx, y * sy, (x + w) * sx, (y + h) * sy], outline=color, width=3)
            pincel.text((x * sx, y * sy), a["clase"] or "ignorar", fill=color)
        panel.paste(vista, (col * 500, 45))
        dibujo.text((col * 500 + 5, 5), f"{r['id']} EXIF original {r['orientacion_exif_original']}", fill="black")
    panel.save(RAIZ / "datos/reportes/orientacion_y_cajas.jpg", quality=90)

    lineas = ["# Preparación y validación de datos — 19/11/2026", "",
        "El diseño del asistente quedó decidido por el usuario y no requiere aprobación adicional del profesor. Se implementó y entrenó un prototipo Transformer propio. Los modelos visuales aún no se han implementado ni entrenado.", "",
        "## Resultado de la preparación visual", "",
        f"Se descargaron e inspeccionaron {preparacion['imagenes_inventariadas']:,} imágenes: 4.752 de RealWaste y 1.500 de TACO. Tras normalizar orientación, {preparacion['imagenes_integras']:,} se decodifican y coinciden con la geometría esperada.", "",
        "| Dataset preparado | Entrenamiento | Validación | Prueba | Total |", "|---|---:|---:|---:|---:|"]
    for nombre, tarea in [("RealWaste, clasificación", "clasificacion"), ("TACO, detección", "deteccion")]:
        cantidades = [validacion["conteos"][tarea][s]["imagenes"] for s in ["train", "val", "test"]]
        lineas.append(f"| {nombre} | {cantidades[0]} | {cantidades[1]} | {cantidades[2]} | {sum(cantidades)} |")
    lineas += ["", "Las rutas de los manifiestos son relativas a la raíz del proyecto. No se duplican las fotografías en carpetas train/val/test. Los COCO usan categorías 1..6; el lector PyTorch devuelve etiquetas 0..5 y cajas xyxy normalizadas.", "",
        "## Correcciones y controles", "",
        f"- {preparacion['imagenes_orientacion_normalizada']} imágenes normalizadas con EXIF a PNG, preservando los archivos originales. Las 152 discrepancias de ancho/alto se explicaban por orientación; se incluyó también una rotación de 180 grados.",
        f"- {len(correcciones)} cajas con desbordes menores de dos píxeles recortadas por intersección geométrica; se conserva al menos el 98 % del área de cada caja y se registran las coordenadas originales.",
        f"- {len(ids)} identificadores de anotación repetidos en la fuente ({', '.join(ids)}), sustituidos por identificadores internos únicos con trazabilidad.",
        f"- Exclusiones de los manifiestos: {json.dumps(preparacion['exclusiones'], ensure_ascii=False)}. Los originales se conservan.",
        f"- Validación estructural: {'correcta' if validacion['validacion_estructural_correcta'] else 'fallida'}. Fugas detectadas por grupos, adquisición, píxeles, archivos y corpus: {json.dumps(validacion['fugas_detectadas'])}.",
        "- Lotes TACO completos y bloques conservadores de diez nombres consecutivos por categoría RealWaste se mantienen juntos. Se añaden coincidencias exactas y perceptuales. Esta agrupación reduce correlaciones, pero no sustituye una prueba externa con nuevas sesiones.",
        "- Las categorías ambiguas/compuestas o ajenas al alcance de TACO se conservan como regiones ignoradas. El lector ecowaste/datasets.py entrega ignore_boxes; la futura pérdida del detector debe enmascarar esas regiones para no contarlas como negativos.",
        "- Se cargó una imagen real por clase para cada tarea y se comprobaron tensores, rangos, clases y geometría; evidencia en datos/reportes/carga_tensores.json.",
        "- Revisión visual por el asistente de paneles de muestras de entrenamiento y de ejemplos con orientación corregida; no es una revisión humana exhaustiva de las etiquetas.", "",
        "## Cobertura por clase", "", "| Clase | Clasificación: imágenes | Detección: cajas |", "|---|---:|---:|"]
    for c in CLASES:
        clas = sum(validacion["conteos"]["clasificacion"][s]["clases"][c] for s in ["train", "val", "test"])
        det = sum(validacion["conteos"]["deteccion"][s]["clases"][c] for s in ["train", "val", "test"])
        lineas.append(f"| {c} | {clas} | {det} |")
    lineas += ["", "La clasificación permite iniciar experimentos con las seis clases. **La detección dispone de solo ocho cajas orgánicas en total**: queda preparada técnicamente para pruebas, pero no permite una evaluación representativa de esa clase. Harán falta imágenes orgánicas reales con cajas revisadas y videos independientes. No se fabricaron cajas a partir de imágenes de clasificación ni se sustituyó diversidad por duplicación de muestras.", "",
        "## Asistente implementado", "",
        f"Transformer propio: {asistente['parametros']:,} parámetros, vocabulario de {asistente['vocabulario']} tokens construido solo con entrenamiento, dos bloques y cuatro cabezas. Se entrenó desde cero en CUDA con semilla 42. Checkpoint: checkpoints/asistente_v1.pt.", "",
        "Corpus inicial: 144 preguntas base de entrenamiento con tres variantes por grupo (432 registros), 36 de validación y 36 de prueba. Los contextos son sintéticos y están marcados como tales; las variantes no se reparten entre particiones. Su autoría asistida por IA y la revisión humana pendiente constan en datos/asistente/procedencia.json.", "",
        f"Resultados en la prueba inicial: accuracy {asistente['test']['accuracy']:.1%}, F1 macro {asistente['test']['f1_macro']:.3f}. Con abstención: cobertura {asistente['test']['cobertura_con_abstencion']:.1%} y accuracy entre respuestas aceptadas {asistente['test']['accuracy_aceptadas']:.1%}. El umbral {asistente['umbral_por_val']} se seleccionó con validación.", "",
        f"La referencia simple Naive Bayes obtuvo accuracy {baseline['test']['accuracy']:.1%} y F1 macro {baseline['test']['f1_macro']:.3f} en la misma prueba. El Transformer no supera todavía esa referencia. Es un prototipo funcional con comprensión limitada; requiere ampliar y revisar el corpus antes de utilizarlo como asistente fiable. No son métricas de detección visual ni de usuarios independientes.", "",
        "Las respuestas se construyen a partir de detecciones suministradas y reglas documentadas. Se probó la consulta de separación con datos sintéticos explícitamente identificados. El motor no analiza por sí solo una cámara y no declara conocer limpieza o composición interna no suministradas.", "",
        "## Reproducción", "", "Desde la carpeta EcoWasteFormer en PowerShell:", "", "```powershell",
        "$pythonEco = Join-Path $env:USERPROFILE '.venvs\\EcoWasteFormer\\Scripts\\python.exe'",
        "& $pythonEco scripts/descargar_datos.py --fuente todo",
        "& $pythonEco scripts/preparar_datos.py --reusar-inventario",
        "& $pythonEco scripts/validar_datos.py",
        "& $pythonEco scripts/generar_informe_datos.py",
        "& $pythonEco -m unittest discover -s pruebas -v",
        "& $pythonEco scripts/consultar_asistente.py '¿Dónde pongo el plástico?' --detecciones datos/ejemplos/detecciones_sinteticas.json",
        "```", "", "Para reconstruir el asistente: preparar_corpus_asistente.py, entrenar_asistente.py y comparar_baseline_asistente.py, en ese orden. Mantener la prueba fija; los experimentos futuros deben seleccionar cambios con validación y reservar una nueva prueba externa para la evaluación final.", "",
        "## Archivos de evidencia", "",
        "- datos/reportes/validacion.json: controles, conteos y hashes de manifiestos.",
        "- datos/reportes/preparacion.json: integridad, exclusiones, tamaño de objetos y límites.",
        "- datos/reportes/inventario_imagenes.jsonl: procedencia y hashes por imagen.",
        "- datos/reportes/correcciones_cajas.json e ids_origen_repetidos.json: cambios trazables.",
        "- datos/reportes/muestras_realwaste.jpg, muestras_taco.jpg y orientacion_y_cajas.jpg: paneles de inspección.",
        "- resultados/asistente_v1.json y baseline_asistente.json: métricas completas y configuración.",
        "- pruebas/test_componentes.py: siete pruebas superadas de geometría, atención, checkpoint y fundamento de respuestas.", "",
        "## Fuentes y atribución", "",
        "[RealWaste, UCI](https://archive.ics.uci.edu/dataset/908/realwaste), Single, Iranmanesh y Raad (2023), DOI 10.24432/C5SS4G, CC BY 4.0. ZIP de origen y hash conservados.", "",
        "[TACO, repositorio original](https://github.com/pedropro/TACO), Proença y Simões (2020), [artículo](https://arxiv.org/abs/2003.06975). Las anotaciones descargadas no especifican licencias por imagen y los autores declaran imágenes bajo licencias públicas. Se conservan URLs e identificadores; no se atribuye a las fotos la licencia MIT del código. No se han republicado las imágenes.", "",
        "[MinAmbiente](https://www.minambiente.gov.co/preguntas-frecuentes/): fuente de las reglas de separación. La recomendación depende también del estado del objeto y del servicio local.", ""]
    destino = RAIZ / "documentacion/02_datasets_y_asistente.md"
    destino.write_text("\n".join(lineas), encoding="utf-8")
    print(json.dumps({"informe": str(destino), "carga": carga}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
