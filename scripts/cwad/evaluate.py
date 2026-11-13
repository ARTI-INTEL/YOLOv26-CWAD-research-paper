# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate the trained CWAD-YOLO models in-domain, per adverse condition, and on the unseen ACDC domain.

For every `runs/cwad/<run>/weights/best.pt` this reports Precision, Recall, mAP@0.5, mAP@0.5:0.95 on the DAWN
validation split overall and per weather condition, the same metrics on the strictly unseen ACDC split (evaluated
without retraining or fine-tuning), the resulting generalization drop, and the model cost (parameters, GFLOPs, CPU
latency, FPS). Outputs `runs/cwad/results.md` and `runs/cwad/results.csv`.

`--tta` optionally adapts each model to ACDC at inference time with `ultralytics.utils.tta.TestTimeAdapter ('bn' for
AdaBN, 'entropy' for AdaBN + entropy minimization) before the unseen-domain metrics are computed; the default `none`
keeps them strictly zero-shot.

Example:
    python scripts/cwad/evaluate.py --runs cwad-yolo11s cwad-yolo26s --imgsz 512
    python scripts/cwad/evaluate.py --runs cwad-yolo26s-plus --tta bn --imgsz 512
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import yaml

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[2]
CONDITION_KEY = {"dawn": "weather", "acdc6": "condition"}


def load_conditions(dataset_root: Path, project: Path) -> dict[str, list[Path]]:
    """Group validation images by adverse condition using the manifest written by the prep script."""
    key = CONDITION_KEY[dataset_root.name]
    grouped: dict[str, list[Path]] = {}
    with open(project / dataset_root / "conditions.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "val":
                grouped.setdefault(row[key], []).append((project / dataset_root / "images" / "val" / row["file"]).resolve())
    return dict(sorted(grouped.items()))


def condition_yaml(project: Path, dataset_root: Path, condition: str, files: list[Path], imgsz: int) -> Path:
    """Write a one-condition dataset YAML pointing at an explicit image list."""
    list_file = project / dataset_root / "lists" / f"val_{condition}.txt"
    list_file.parent.mkdir(exist_ok=True)
    list_file.write_text("\n".join(str(f) for f in files) + "\n")
    base = yaml.safe_load((project / "ultralytics/cfg/datasets" / f"{dataset_root.name}.yaml").read_text())
    cfg = {"path": str((project / dataset_root).resolve()), "train": str(list_file), "val": str(list_file),
           "names": base["names"]}
    out = list_file.with_suffix(".yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def val_metrics(net: YOLO | Path, data: Path | str, imgsz: int) -> dict[str, float]:
    """Run Ultralytics validation and return box metrics."""
    net = net if isinstance(net, YOLO) else YOLO(str(net))
    r = net.val(data=str(data), split="val", imgsz=imgsz, batch=8, device="cpu", plots=False)
    return {"precision": float(r.box.mp), "recall": float(r.box.mr), "map50": float(r.box.map50), "map": float(r.box.map)}


def cost(weights: Path, imgsz: int) -> dict[str, float]:
    """Measure parameters, GFLOPs and CPU latency for one checkpoint."""
    net = YOLO(str(weights)).model.float().eval()
    params = sum(p.numel() for p in net.parameters())
    try:
        flops = get_flops(net, imgsz=imgsz)
    except Exception:  # thop occasionally cannot trace an end-to-end head
        flops = float("nan")
    x = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        for _ in range(2):  # warmup
            net(x)
        t0 = time.time()
        for _ in range(5):
            net(x)
        ms = (time.time() - t0) / 5 * 1000
    return {"params_m": params / 1e6, "gflops": flops, "ms_img": ms, "fps": 1000 / ms}


def adapt_to_acdc(weights: Path, project: Path, opt: argparse.Namespace) -> YOLO | None:
    """Adapt a checkpoint to the unseen ACDC domain at test time and return the adapted model."""
    if not (project / "datasets/acdc6" / "conditions.csv").is_file():
        print("skip TTA: ACDC dataset not prepared")
        return None
    files = [p for group in load_conditions(Path("datasets/acdc6"), project).values() for p in group]
    from ultralytics.utils.tta import TestTimeAdapter  # lazy so the default path never imports the adapter

    net = YOLO(str(weights))
    TestTimeAdapter(net.model, mode=opt.tta, imgsz=opt.imgsz, device="cpu").adapt(files)
    print(f"TTA ({opt.tta}): adapted on {len(files)} ACDC images")
    return net


def main(opt: argparse.Namespace) -> None:
    """Evaluate every requested run and write the comparison tables."""
    project = Path(opt.project_dir).resolve()
    runs = [r if isinstance(r, str) else r for r in opt.runs]
    rows, detail = [], []

    for run in runs:
        weights = project / "runs/cwad" / run / "weights/best.pt"
        if not weights.is_file():
            print(f"skip {run}: {weights} not found")
            continue
        print(f"\n=== {run} ===")
        c = cost(weights, opt.imgsz)
        print(f"cost: {c['params_m']:.2f}M params, {c['gflops']:.1f} GFLOPs, {c['ms_img']:.0f} ms/img")

        source = val_metrics(weights, "ultralytics/cfg/datasets/dawn6.yaml", opt.imgsz)
        tta_net = adapt_to_acdc(weights, project, opt) if opt.tta != "none" else None
        print("DAWN val:", {k: round(v, 4) for k, v in source.items()})
        row = {"model": run, **c, **{f"dawn_{k}": v for k, v in source.items()}}

        for split_name, dataset_root in (("dawn", Path("datasets/dawn")), ("acdc", Path("datasets/acdc6"))):
            if not (project / dataset_root / "conditions.csv").is_file():
                print(f"skip {split_name} per-condition: dataset not prepared")
                continue
            for condition, files in load_conditions(dataset_root, project).items():
                cfg = condition_yaml(project, dataset_root, condition, files, opt.imgsz)
                m = val_metrics(tta_net if split_name == "acdc" and tta_net is not None else weights, cfg, opt.imgsz)
                print(f"  {split_name}/{condition}: {len(files)} imgs, mAP50 {m['map50']:.4f}")
                detail.append({"model": run, "dataset": split_name, "condition": condition, "images": len(files), **m})
                if split_name == "acdc":
                    row.setdefault("acdc_map50", []).append(m["map50"])
                    row.setdefault("acdc_map", []).append(m["map"])

        if row.get("acdc_map50"):  # unseen-domain aggregate over all evaluated conditions
            row["acdc_map50"] = sum(row["acdc_map50"]) / len(row["acdc_map50"])
            row["acdc_map"] = sum(row["acdc_map"]) / len(row["acdc_map"])
            row["drop_map50"] = (source["map50"] - row["acdc_map50"]) / source["map50"] * 100
        rows.append(row)

    out_dir = project / "runs/cwad"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "results_detail.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "dataset", "condition", "images", "precision", "recall", "map50", "map"])
        writer.writeheader()
        writer.writerows(detail)

    md = ["# CWAD-YOLO results", "", "## In-domain (DAWN val) vs unseen (ACDC)", ""]
    md += ["| Model | Params (M) | GFLOPs | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | ms/img (CPU) | FPS (CPU) | ACDC mAP@0.5 | Generalization drop |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        acdc = f"{r['acdc_map50']:.4f}" if "acdc_map50" in r else "n/a"
        drop = f"{r['drop_map50']:.1f}%" if "drop_map50" in r else "n/a"
        md.append(
            f"| {r['model']} | {r['params_m']:.2f} | {r['gflops']:.1f} | {r['dawn_precision']:.4f} | {r['dawn_recall']:.4f} | "
            f"{r['dawn_map50']:.4f} | {r['dawn_map']:.4f} | {r['ms_img']:.0f} | {r['fps']:.2f} | {acdc} | {drop} |"
        )
    for dataset in ("dawn", "acdc"):
        subset = [d for d in detail if d["dataset"] == dataset]
        if not subset:
            continue
        md += ["", f"## Per-condition mAP@0.5 ({'DAWN val' if dataset == 'dawn' else 'ACDC unseen'})", "",
               "| Model | Condition | Images | mAP@0.5 | mAP@0.5:0.95 |", "| --- | --- | --- | --- | --- |"]
        md += [f"| {d['model']} | {d['condition']} | {d['images']} | {d['map50']:.4f} | {d['map']:.4f} |" for d in subset]
    (out_dir / "results.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {out_dir / 'results.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=str(ROOT))
    parser.add_argument("--runs", nargs="+", default=["cwad-yolo11s", "cwad-yolo26s"])
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--tta", choices=["none", "bn", "entropy"], default="none", help="adapt to ACDC at test time")
    main(parser.parse_args())
