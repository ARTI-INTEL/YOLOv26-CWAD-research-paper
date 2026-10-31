# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Test-time adaptation (TTA) for cross-domain inference of the pro-YOLO26-CWAD framework.

The adapter re-fits the model to the unlabeled target domain as it predicts, so a detector trained on a source domain
(e.g. SODA10M) keeps working on a shifted target domain (e.g. DAWN fog/night). Two levels are available:

- ``bn``: recompute every BatchNorm running statistic from the incoming test batch (fully test-time adaptation's
  normalization step), no gradients and no labels.
- ``entropy``: the ``bn`` step followed by a few gradient steps on the BatchNorm affine parameters that minimize the
  entropy of the confident detections (TENT-style self-training).

Usage:
    >>> from ultralytics import YOLO
    >>> from ultralytics.utils.tta import TestTimeAdapter
    >>> model = YOLO("pro-yolo26s-cwad.pt")
    >>> results = model.predict("dawn_test/", predictor=TestTimeAdapter(mode="bn").predictor())
"""

from __future__ import annotations

import functools

import torch
from torch import nn

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER

__all__ = ("TestTimeAdapter", "TestTimeAdaptivePredictor")


class TestTimeAdapter:
    """Adapt a detector to an unlabeled target batch while it predicts.

    Args:
        mode (str): ``bn`` (BatchNorm statistics only), ``entropy`` (BatchNorm statistics plus entropy minimization) or
            ``none`` to disable adaptation.
        steps (int): Gradient steps per batch in ``entropy`` mode.
        lr (float): SGD learning rate of the entropy minimization steps.
        momentum (float | None): BatchNorm momentum during adaptation; ``None`` computes the statistics of the test
            batch alone instead of blending them with the running estimate.
        threshold (float): Detection confidence above which a prediction contributes to the entropy term.
        keep (bool): Keep the adapted parameters for the next batch (online adaptation) instead of restoring the
            pre-batch weights after every batch (episodic adaptation).
        eps (float): Numerical floor for the confidence clamp.
    """

    def __init__(
        self,
        mode: str = "bn",
        steps: int = 1,
        lr: float = 1e-3,
        momentum: float | None = None,
        threshold: float = 0.5,
        keep: bool = False,
        eps: float = 1e-6,
    ):
        """Initialize the adapter."""
        if mode not in {"bn", "entropy", "none"}:
            raise ValueError(f"mode must be 'bn', 'entropy' or 'none', not '{mode}'")
        self.mode = mode
        self.steps = steps
        self.lr = lr
        self.momentum = momentum
        self.threshold = threshold
        self.keep = keep
        self.eps = eps
        self.batches = 0
        self.warned = False

    @staticmethod
    def _bn_layers(model: nn.Module) -> list[nn.BatchNorm2d]:
        """Return the BatchNorm layers of a model."""
        return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]

    def predictor(self):
        """Return a `TestTimeAdaptivePredictor` bound to this adapter, for `YOLO.predict(predictor=...)`."""
        return functools.partial(TestTimeAdaptivePredictor, adapter=self)

    def adapt(self, model: nn.Module, im: torch.Tensor) -> None:
        """Adapt the model in place to the preprocessed test images `im` of shape (batch, 3, h, w)."""
        layers = self._bn_layers(model)
        if self.mode == "none" or not layers:
            if self.mode != "none" and not self.warned:
                LOGGER.warning("No BatchNorm layers to adapt (fused model?); skipping test-time adaptation.")
                self.warned = True
            return
        with torch.inference_mode(False):  # the predictor runs inside inference_mode, which forbids autograd
            im = im.clone() if im.is_inference() else im  # inference tensors cannot enter an autograd graph
            was_training = model.training
            self._adapt_bn(model, im.detach(), layers)
            if self.mode == "entropy":
                self._adapt_entropy(model, im.detach(), layers)
            model.train(was_training)
        self.batches += 1

    def _adapt_bn(self, model: nn.Module, im: torch.Tensor, layers: list[nn.BatchNorm2d]) -> None:
        """Recompute the BatchNorm running statistics from the test batch."""
        momentums = [m.momentum for m in layers]
        model.train()
        for m in layers:
            m.momentum = self.momentum
        try:
            with torch.no_grad():
                model(im)
        finally:
            for m, momentum in zip(layers, momentums):
                m.momentum = momentum

    def _adapt_entropy(self, model: nn.Module, im: torch.Tensor, layers: list[nn.BatchNorm2d]) -> None:
        """Minimize the entropy of the confident detections over the BatchNorm affine parameters."""
        params = [p for m in layers for p in (m.weight, m.bias) if p is not None and p.requires_grad]
        if not params:
            return
        saved = [p.detach().clone() for p in params]
        optimizer = torch.optim.SGD(params, lr=self.lr)
        model.eval()  # gradients reach the affine parameters without disturbing the refreshed statistics
        for _ in range(self.steps):
            scores = self._scores(model(im))
            if scores is None:
                break
            loss = self._entropy(scores)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if not self.keep:
            with torch.no_grad():
                for p, s in zip(params, saved):
                    p.copy_(s)

    @staticmethod
    def _scores(out) -> torch.Tensor | None:
        """Return the raw class logits of a forward pass, or None when the model exposes no raw predictions."""
        if isinstance(out, tuple):  # (decoded, raw predictions) from an end-to-end head
            out = out[1]
        elif isinstance(out, list):
            out = out[-1]
        if isinstance(out, dict):
            out = out.get("one2one", out)  # prefer the NMS-free branch of a dual-head model
            return out.get("scores")
        return None

    def _entropy(self, scores: torch.Tensor) -> torch.Tensor:
        """Return the mean binary entropy of the predictions above the confidence threshold."""
        probs = scores.sigmoid()
        mask = probs.amax(1) > self.threshold  # (batch, anchors) confident detections
        p = probs.permute(0, 2, 1)[mask] if mask.any() else probs.permute(0, 2, 1).flatten(0, 1)
        p = p.clamp(self.eps, 1 - self.eps)
        return -(p * p.log() + (1 - p) * (1 - p).log()).mean()


class TestTimeAdaptivePredictor(DetectionPredictor):
    """Detection predictor that adapts the model to every incoming batch before predicting it.

    Args:
        cfg (Any): Predictor configuration.
        overrides (dict | None): Configuration overrides.
        _callbacks (dict | None): Callbacks registered on the predictor.
        adapter (TestTimeAdapter | None): Adapter to use; defaults to BatchNorm-statistics adaptation.
    """

    fuse_model = False  # keep BatchNorm layers so the adapter can update their statistics and affine weights

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, adapter: TestTimeAdapter | None = None):
        """Initialize the predictor with a test-time adapter."""
        super().__init__(cfg, overrides, _callbacks)
        self.adapter = adapter or TestTimeAdapter()

    def inference(self, im, *args, **kwargs):
        """Adapt the model to `im`, then run the regular detection inference on it."""
        model = getattr(self.model, "model", None)  # AutoBackend wraps the PyTorch module in .model
        if getattr(self.model, "format", None) == "pt" and isinstance(model, nn.Module):
            self.adapter.adapt(model, im)
        elif not self.adapter.warned:
            LOGGER.warning("Test-time adaptation requires a PyTorch model; predicting without adaptation.")
            self.adapter.warned = True
        return super().inference(im, *args, **kwargs)
