# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Prepare the DAWN adverse-weather dataset as a unified six-class YOLO detection dataset.

DAWN (Kenk & Hassaballah, 2020, https://doi.org/10.17632/766ygrbt8y.3) ships Pascal VOC annotations and 1,027 images
split into four weather sets (Fog 300, Rain 200, Sand 323, Snow 204). This script maps the annotations onto the six
autonomous-driving classes used by CWAD-YOLO, drops the single ``train`` instance (no counterpart in the target class
space), and writes a weather-stratified train/val split plus a conditions manifest for per-weather evaluation.

Example:
    python scripts/cwad/prepare_dawn.py --raw datasets/raw/dawn --out datasets/dawn --val-frac 0.2
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

CLASSES = ("car", "bus", "truck", "motorcycle", "bicycle", "person")  # canonical six-class order
WEATHER = {"Fog": "fog", "Rain": "rain", "Sand": "sand", "Snow": "snow"}


def parse_voc(xml_file: Path, width: int, height: int, skipped: Counter) -> list[str]:
    """Convert one Pascal VOC annotation file to YOLO label lines."""
    rows = []
    for obj in ET.parse(xml_file).getroot().findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASSES:  # e.g. the single 'train' box has no counterpart in the six-class scheme
            skipped[name] += 1
            continue
        box = obj.find("bndbox")
        x1, y1, x2, y2 = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
        y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:  # degenerate box after clipping
            skipped[name] += 1
            continue
        rows.append(
            f"{CLASSES.index(name)} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{w / width:.6f} {h / height:.6f}"
        )
    return rows


def main(opt: argparse.Namespace) -> None:
    """Convert DAWN VOC annotations into a stratified YOLO dataset."""
    raw, out = Path(opt.raw), Path(opt.out)
    rng = random.Random(opt.seed)
    skipped: Counter = Counter()
    manifest = []

    for folder, weather in WEATHER.items():
        root = raw / folder / folder
        images_dir = root
        xml_dir = root / f"{folder}_PASCAL_VOC"
        images = sorted(images_dir.glob("*.jpg"))
        assert images, f"no images found in {images_dir}"

        order = list(images)
        rng.shuffle(order)
        n_val = round(len(order) * opt.val_frac)
        for i, image in enumerate(order):
            split = "val" if i < n_val else "train"
            xml_file = xml_dir / f"{image.stem}.xml"
            size = ET.parse(xml_file).getroot().find("size")
            width, height = int(size.findtext("width")), int(size.findtext("height"))
            rows = parse_voc(xml_file, width, height, skipped)
            if not rows:  # image keeps zero labels so negatives stay represented in both splits
                print(f"warning: {image.name} has no usable box")

            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, out / "images" / split / image.name)
            (out / "labels" / split / f"{image.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            manifest.append(
                {"file": image.name, "weather": weather, "split": split, "boxes": len(rows), "width": width, "height": height}
            )

    with open(out / "conditions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "weather", "split", "boxes", "width", "height"])
        writer.writeheader()
        writer.writerows(sorted(manifest, key=lambda r: r["file"]))

    print(f"images: {len(manifest)} -> {out}")
    for split in ("train", "val"):
        per_weather = Counter(r["weather"] for r in manifest if r["split"] == split)
        print(f"  {split}: {sum(per_weather.values())} {dict(sorted(per_weather.items()))}")
    print(f"  boxes: {sum(r['boxes'] for r in manifest)}, skipped instances: {dict(skipped)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="datasets/raw/dawn", help="raw DAWN root containing the four weather folders")
    parser.add_argument("--out", default="datasets/dawn", help="output YOLO dataset root")
    parser.add_argument("--val-frac", type=float, default=0.2, help="per-weather validation fraction")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed")
    main(parser.parse_args())
