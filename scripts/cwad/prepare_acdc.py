# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert the ACDC adverse-conditions dataset into the six-class YOLO layout used for unseen-domain testing.

ACDC (Sakaridis et al., ICCV 2021, https://acdc.vision.ee.ethz.ch) ships semantic and instance masks rather than
detection boxes, so boxes are derived from the instance mask of every "thing" region and mapped onto the CWAD-YOLO
class space: bicycle, bus, car, truck, motorcycle and person. Riders are folded into ``person``; ``train`` has no
counterpart in the six-class scheme and is dropped.

ACDC is used **only** for final cross-domain evaluation: this script never touches training data, and the default
split is the labelled ``val`` split (1,067 images, ~equal fog/night/rain/snow).

Example:
    python scripts/cwad/prepare_acdc.py --raw datasets/raw/ACDC --out datasets/acdc6 --split val
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("car", "bus", "truck", "motorcycle", "bicycle", "person")
# Cityscapes-style thing label ids used by ACDC instance masks -> unified class index
INSTANCE_LABELS = {1: "bicycle", 2: "bus", 3: "car", 4: "motorcycle", 5: "person", 6: "person", 8: "truck"}
CONDITIONS = ("fog", "night", "rain", "snow")


def find_instance_mask(root: Path, image: Path, split: str) -> Path | None:
    """Locate the ACDC instance mask that corresponds to an RGB frame."""
    frame = image.name.split("_rgb_anon")[0]
    rel = image.parent.relative_to(next(p for p in image.parents if p.name == "rgb_anon")).parent
    for base in (root / "gt", root / "gtFine", root):
        for candidate in (
            base / split / rel / f"{frame}_gt_instanceIds.png",
            base / rel / f"{frame}_gt_instanceIds.png",
            base / split / rel / f"{frame}_gtFine_instanceIds.png",
            base / rel / f"{frame}_gtFine_instanceIds.png",
        ):
            if candidate.is_file():
                return candidate
    matches = list(root.rglob(f"{frame}_*instance*Ids*.png"))  # last resort for unusual layouts
    return matches[0] if matches else None


def mask_to_boxes(mask: np.ndarray, dropped: Counter) -> list[str]:
    """Convert one instance mask into YOLO label lines."""
    height, width = mask.shape[:2]
    rows = []
    for value in np.unique(mask):
        if value == 0:
            continue
        label_id = int(value) // 1000 if value >= 1000 else int(value)  # instanceIds are label*1000+instance
        name = INSTANCE_LABELS.get(label_id)
        if name is None:
            dropped[label_id] += 1
            continue
        ys, xs = np.nonzero(mask == value)
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:
            dropped[label_id] += 1
            continue
        rows.append(
            f"{CLASSES.index(name)} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{w / width:.6f} {h / height:.6f}"
        )
    return rows


def main(opt: argparse.Namespace) -> None:
    """Derive six-class detection labels from ACDC instance masks."""
    root, out = Path(opt.raw), Path(opt.out)
    rgb_roots = [p for p in root.rglob("rgb_anon") if p.is_dir()]
    assert rgb_roots, f"no rgb_anon folder found under {root}"

    images = sorted(p for p in rgb_roots[0].rglob(f"{opt.split}/*/*/*_rgb_anon.png"))
    assert images, f"no {opt.split} images found under {rgb_roots[0]}"

    dropped: Counter = Counter()
    manifest, missing = [], []
    (out / "images" / opt.split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / opt.split).mkdir(parents=True, exist_ok=True)

    for image in images:
        condition = next((c for c in CONDITIONS if c in image.parts), "unknown")
        mask_file = find_instance_mask(root, image, opt.split)
        if mask_file is None:
            missing.append(image.name)
            rows = []
        else:
            rows = mask_to_boxes(cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED), dropped)
        shutil.copy2(image, out / "images" / opt.split / image.name)
        (out / "labels" / opt.split / f"{image.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        manifest.append({"file": image.name, "condition": condition, "split": opt.split, "boxes": len(rows)})

    with open(out / "conditions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "condition", "split", "boxes"])
        writer.writeheader()
        writer.writerows(sorted(manifest, key=lambda r: r["file"]))

    per_condition = Counter(r["condition"] for r in manifest)
    print(f"images: {len(manifest)} -> {out} ({dict(sorted(per_condition.items()))})")
    print(f"boxes: {sum(r['boxes'] for r in manifest)}")
    print(f"missing instance masks: {len(missing)}")
    print(f"dropped instance label ids (train/signs/etc.): {dict(dropped)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="datasets/raw/ACDC", help="raw ACDC root containing rgb_anon and gt")
    parser.add_argument("--out", default="datasets/acdc6", help="output YOLO dataset root")
    parser.add_argument("--split", default="val", help="ACDC split to convert (val holds the labelled frames)")
    main(parser.parse_args())
