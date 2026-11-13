#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Train CWAD-YOLO26-plus, the DANN variant of CWAD-YOLO26, on the DAWN six-class dataset.
# Same schedule as scripts/cwad/train_all.sh so the plus model is directly comparable with cwad-yolo11s and
# cwad-yolo26s; the only addition is the weather-domain adversarial branch (dann, dann_domains, dann_lambda in the
# model YAML). The `yolo` CLI cannot be used here because it builds a plain DetectionModel with no weather labels.
set -euo pipefail

cd "$(dirname "$0")/../.."  # run from the repository root

.venv/Scripts/python.exe -c "
from ultralytics.models.yolo.detect.cwad_plus import CWADPlusTrainer

CWADPlusTrainer(overrides=dict(
    model='ultralytics/cfg/models/26/yolo26s-cwad-plus.yaml',
    data='ultralytics/cfg/datasets/dawn6.yaml',
    imgsz=512, epochs=36, batch=8, device='cpu', workers=0, cache=True,
    seed=0, patience=0, plots=False, pretrained=False, exist_ok=True,
    project='$PWD/runs/cwad', name='cwad-yolo26s-plus',
)).train()
"
