"""Lectura de manifiestos existentes; no divide, modifica ni descarga imagenes."""
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

from utils.config import CLASSES, IMAGE_EXTENSIONS, PROJECT_ROOT, SPLITS
from utils.dataset import sha256


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_manifest(dataset_dir):
    root = Path(dataset_dir).resolve()
    path = root / "manifest.csv"
    summary = json.loads((root / "split_summary.json").read_text(encoding="utf-8"))
    if sha256(path) != summary["manifest_sha256"]:
        raise ValueError("El manifiesto cambio respecto de split_summary.json.")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen_paths, seen_hashes, seen_pixels, groups = set(), set(), set(), {}
    for row in rows:
        relative = Path(row["path"])
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or row["split"] not in SPLITS or row["class"] not in CLASSES:
            raise ValueError("Ruta, clase o particion invalida en el manifiesto.")
        if relative.parts[:2] != (row["split"], row["class"]) or relative.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("La ruta no corresponde a la clase y particion declaradas.")
        if row["path"] in seen_paths or row["sha256"] in seen_hashes or row["pixel_sha256"] in seen_pixels:
            raise ValueError("Duplicados en el manifiesto.")
        previous = groups.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("Hay un grupo visual en varias particiones.")
        seen_paths.add(row["path"])
        seen_hashes.add(row["sha256"])
        seen_pixels.add(row["pixel_sha256"])
    for split in SPLITS:
        counts = Counter(r["class"] for r in rows if r["split"] == split)
        if any(counts[label] <= 0 or counts[label] != summary["counts"][split][label] for label in CLASSES):
            raise ValueError(f"Cantidades incompletas o incorrectas en {split}.")
    if len(rows) != summary["total"]:
        raise ValueError("El total del manifiesto no coincide con el resumen.")
    return rows, summary


def verify_files(dataset_dir, rows, splits=("train", "val")):
    """El entrenamiento solo verifica archivos train/val; no abre fotos test."""
    root = Path(dataset_dir).resolve()
    for split in splits:
        expected = {r["path"] for r in rows if r["split"] == split}
        actual = {p.relative_to(root).as_posix() for p in (root / split).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
        if expected != actual:
            raise ValueError(f"Archivos extra o ausentes en {split}.")
        for row in rows:
            if row["split"] == split and sha256(root / row["path"]) != row["sha256"]:
                raise ValueError(f"Imagen modificada: {row['path']}")


class ImageTransform:
    def __init__(self, training=False, normalization=None, augmentation=None):
        self.training = training
        self.normalization = normalization or {"mean": [.5] * 3, "std": [.5] * 3}
        self.augmentation = augmentation or {}
        self.mean = torch.tensor(self.normalization["mean"], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(self.normalization["std"], dtype=torch.float32)[:, None, None]

    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        if self.training:
            options = self.augmentation
            if random.random() < options.get("horizontal_flip_probability", .5):
                image = ImageOps.mirror(image)
            angle = options.get("rotation_degrees", 10)
            image = image.rotate(random.uniform(-angle, angle), resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
            for option, enhancer in (("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast)):
                magnitude = options.get(option, .1)
                image = enhancer(image).enhance(random.uniform(1 - magnitude, 1 + magnitude))
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
        return (torch.from_numpy(pixels) - self.mean) / self.std


class WasteDataset(Dataset):
    def __init__(self, dataset_dir, split, config, rows=None):
        if split not in SPLITS:
            raise ValueError("Particion desconocida.")
        self.root = Path(dataset_dir).resolve()
        all_rows = rows if rows is not None else read_manifest(self.root)[0]
        self.rows = [row for row in all_rows if row["split"] == split]
        self.class_to_idx = {label: index for index, label in enumerate(CLASSES)}
        if config.get("architecture") == "VisionResidual-v2":
            from training.compact import build_transform
            self.transform = build_transform(config, split == "train")
        else:
            self.transform = ImageTransform(split == "train", config["normalization"], config["augmentation"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as image:
            tensor = self.transform(image)
        return tensor, self.class_to_idx[row["class"]]


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset_dir, split, config, rows=None, pin_memory=False):
    dataset = WasteDataset(dataset_dir, split, config, rows)
    generator = torch.Generator().manual_seed(config["seed"] + SPLITS.index(split))
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=split == "train",
                      num_workers=config["num_workers"], pin_memory=pin_memory,
                      worker_init_fn=seed_worker, generator=generator, drop_last=False)


def training_loaders(dataset_dir, config, rows=None, pin_memory=False):
    # Deliberadamente no construye un DataLoader test.
    return {split: make_loader(dataset_dir, split, config, rows, pin_memory) for split in ("train", "val")}
