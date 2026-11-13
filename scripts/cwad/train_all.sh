#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Train the two CWAD-YOLO models on the DAWN six-class dataset with identical settings.
# CPU-only reference machine (i5-8365U, 4 cores): ~1.4 s/img at 512 px, i.e. ~21 min/epoch, so the schedule is
# 36 epochs at 512 px rather than the 640 px / 100-epoch GPU protocol. Every setting is shared between the two models
# so the comparison is controlled; only the architecture differs.
set -euo pipefail

cd "$(dirname "$0")/../.."  # run from the repository root

YOLO=.venv/Scripts/yolo.exe
# An absolute project path keeps outputs inside the repo; a relative one is resolved against Ultralytics' runs dir.
COMMON=(
  data=ultralytics/cfg/datasets/dawn6.yaml
  imgsz=512 epochs=36 batch=8 device=cpu workers=0 cache=True
  seed=0 patience=0 plots=False pretrained=False exist_ok=True
  "project=$PWD/runs/cwad"
)

$YOLO detect train model=ultralytics/cfg/models/11/cwad-yolo11s.yaml name=cwad-yolo11s "${COMMON[@]}"
$YOLO detect train model=ultralytics/cfg/models/26/yolo26s-cwad.yaml name=cwad-yolo26s "${COMMON[@]}"
