"""Rutas portables: no dependen del directorio de trabajo ni de Windows."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("plastico", "papel", "carton", "vidrio", "metal", "organicos")
TRASHNET_CLASSES = {
    "plastic": "plastico", "paper": "papel", "cardboard": "carton",
    "glass": "vidrio", "metal": "metal",
}
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def data_root(value=None):
    # --data-dir tiene prioridad sobre la variable de entorno.
    configured = value or os.environ.get("VISION_IA_DATA_DIR")
    return Path(configured).expanduser().resolve() if configured else PROJECT_ROOT / "data"


def add_data_argument(parser):
    parser.add_argument("--data-dir", help="Raiz de datos; por defecto VISION_IA_DATA_DIR o <proyecto>/data.")
