"""Clasifica un residuo predominante en una imagen usando el checkpoint local."""
import argparse
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from PIL import Image, ImageOps
import torch
from ecowaste.clasificador import cargar_clasificador
from ecowaste.datasets import CLASES, letterbox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imagen", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--salida", type=Path)
    args = parser.parse_args()
    if args.checkpoint is None:
        arquitectura = json.loads((RAIZ / 'configuracion/arquitectura.json').read_text(encoding='utf-8'))
        args.checkpoint = RAIZ / arquitectura['clasificador']['checkpoint']
    if args.salida and args.salida.resolve() in [args.imagen.resolve(), args.checkpoint.resolve()]:
        parser.error('La salida debe conservar la imagen y el checkpoint')
    if not args.imagen.is_file() or not args.checkpoint.is_file():
        parser.error("La imagen y el checkpoint deben existir")
    torch.set_num_threads(4)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, metadatos = cargar_clasificador(args.checkpoint, dispositivo)
    with Image.open(args.imagen) as imagen:
        tensor, _ = letterbox(ImageOps.exif_transpose(imagen).convert("RGB"), modelo.imagen)
    with torch.inference_mode():
        probabilidades = modelo(tensor.unsqueeze(0).to(dispositivo)).softmax(-1)[0].cpu().tolist()
    indice = max(range(len(CLASES)), key=probabilidades.__getitem__)
    resultado = {"imagen": str(args.imagen), "material_predicho": CLASES[indice],
        "categoria_operativa": "organico" if CLASES[indice] == "organico" else "inorganico",
        "puntuacion_no_calibrada": probabilidades[indice],
        "probabilidades_modelo": dict(zip(CLASES, probabilidades)),
        "modelo": "EcoWasteFormer-Cls experimental", "epoca_checkpoint": metadatos["epoca"],
        "checkpoint": str(args.checkpoint),
        "f1_macro_validacion_checkpoint": metadatos.get('mejor_f1') if metadatos.get('epoca') == metadatos.get('mejor_epoca') else None,
        "alcance": "Clasificación de la imagen completa con un residuo predominante; no localiza ni cuenta objetos.",
        "limitacion": "Prototipo de seis clases cerradas; no identifica automáticamente objetos fuera de alcance, limpieza o aprovechabilidad."}
    texto = json.dumps(resultado, ensure_ascii=False, indent=2)
    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(texto + "\n", encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
