# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Test-time adaptation of a trained detector to an unlabeled target domain.

The DANN branch in `CWAD-YOLO26-plus` makes the backbone weather-invariant during training; this module is the
inference-time counterpart, adapting an already trained detector to an unseen domain (ACDC) without labels, retraining
or access to source data.

Two modes are supported:
  - `bn`: AdaBN. Reset the BatchNorm running statistics and re-estimate them cumulatively over the target images, which
    shifts every normalization layer from the source (DAWN) to the target (ACDC) statistics.
  - `entropy`: AdaBN followed by TENT-style adaptation, minimizing the entropy of the detection scores with respect to
    the BatchNorm affine parameters.

Examples:
    >>> from ultralytics import YOLO
    >>> from ultralytics.utils.tta import TestTimeAdapter
    >>> net = YOLO("runs/cwad/cwad-yolo26s-plus/weights/best.pt").model
    >>> TestTimeAdapter(net, mode="bn", imgsz=512, device="cpu").adapt(["datasets/acdc6/images/val/fog_0001.png"])
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER

__all__ = ("TestTimeAdapter",)


class TestTimeAdapter:
    """Adapt a trained detector to a target domain at inference time, without labels.

    Attributes:
        model (torch.nn.Module): Detector being adapted, modified in place and returned to eval mode afterwards.
        mode (str): 'bn' for AdaBN only, 'entropy' for AdaBN plus entropy minimization.
        imgsz (int): Inference image size the target images are letterboxed to.
        batch (int): Number of target images per forward pass.
        steps (int): Number of entropy-minimization passes; ignored in 'bn' mode.
        lr (float): Learning rate of the entropy-minimization optimizer.
        device (str): Device the model and images live on.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = "bn",
        imgsz: int = 640,
        batch: int = 8,
        steps: int = 1,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        """Initialize the adapter.

        Args:
            model (torch.nn.Module): Detector to adapt in place, e.g. `YOLO(weights).model`.
            mode (str): 'bn' or 'entropy'.
            imgsz (int): Inference image size.
            batch (int): Number of target images per forward pass.
            steps (int): Number of entropy-minimization passes.
            lr (float): Learning rate for entropy minimization.
            device (str): Device to adapt on.
        """
        assert mode in {"bn", "entropy"}, f"Unsupported TTA mode '{mode}', expected 'bn' or 'entropy'."
        self.model = model.float().to(device)
        self.mode, self.imgsz, self.batch, self.steps, self.lr, self.device = mode, imgsz, batch, steps, lr, device
        self.stride = int(self.model.stride.max()) if hasattr(self.model, "stride") else 32

    def adapt(self, images: list[str | Path]) -> nn.Module:
        """Adapt the model on the given unlabeled target images.

        Args:
            images (list[str | Path]): Target-domain images to adapt on.

        Returns:
            (torch.nn.Module): The adapted model, left in eval mode.
        """
        images = [Path(p) for p in images]
        if not images:
            LOGGER.warning("TTA: no target images supplied, model left unchanged.")
            return self.model
        norms = [m for m in self.model.modules() if isinstance(m, nn.BatchNorm2d)]
        self.model.train()
        for m in norms:
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average over the target set
        with torch.no_grad():
            for x in self._batches(images):
                self.model(x)
        LOGGER.info(f"TTA (AdaBN): re-estimated {len(norms)} normalization layers on {len(images)} target images.")
        if self.mode == "entropy":
            self._minimize_entropy(images, norms)
        self.model.eval()
        return self.model

    def _minimize_entropy(self, images: list[Path], norms: list[nn.BatchNorm2d]) -> None:
        """Update the normalization affine parameters to sharpen predictions on the target domain."""
        self.model.eval()
        params = [p for m in norms for p in (m.weight, m.bias) if p is not None]
        for p in params:
            p.requires_grad_(True)
        optimizer = torch.optim.Adam(params, lr=self.lr)
        for _ in range(self.steps):
            for x in self._batches(images):
                optimizer.zero_grad()
                scores = self._scores(self.model(x)).sigmoid()
                q = scores.clamp(1e-6, 1 - 1e-6)
                loss = -(scores * q.log() + (1 - scores) * (1 - q).log()).mean()
                loss.backward()
                optimizer.step()
        for p in params:
            p.requires_grad_(False)
        LOGGER.info(f"TTA (entropy): {self.steps} step(s) on {len(images)} target images.")

    @staticmethod
    def _scores(preds) -> torch.Tensor:
        """Return the raw class scores from a training dict or an eval-mode (predictions, raw) tuple."""
        if isinstance(preds, tuple):
            preds = preds[1]
        if "one2one" in preds:
            preds = preds["one2one"]
        elif "one2many" in preds:
            preds = preds["one2many"]
        return preds["scores"]

    def _batches(self, images: list[Path]):
        """Yield letterboxed target images as normalized BGR tensors of shape (batch, 3, imgsz, imgsz)."""
        box = LetterBox(self.imgsz, auto=False, scaleup=True, stride=self.stride)
        for i in range(0, len(images), self.batch):
            ims = [box(image=cv2.imread(str(p))) for p in images[i : i + self.batch]]
            batch = np.stack([im[:, :, ::-1].transpose(2, 0, 1) for im in ims])
            yield torch.from_numpy(np.ascontiguousarray(batch)).to(self.device).float() / 255
