# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CWAD-YOLO26-plus: DANN weather adaptation around the CWAD-YOLO26 detector.

The network is the CWAD-YOLO26 architecture from `cfg/models/26/yolo26-cwad-plus.yaml` plus a domain-adversarial
branch: the detection head's multi-scale features are pooled, passed through a gradient-reversal layer and a small
classifier that predicts the DAWN weather condition (fog, rain, sand, snow). The reversed gradient pushes the backbone
toward weather-invariant features. This module is the training half of the DANN framework; `ultralytics/utils/tta.py`
is the inference half that adapts the trained detector to an unseen target domain.

The plain `yolo detect train` entry point cannot be used for this model because it builds a `DetectionModel` and never
creates weather labels, so use `CWADPlusTrainer` (see `scripts/cwad/train_plus.sh`):

Examples:
    >>> from ultralytics.models.yolo.detect.cwad_plus import CWADPlusTrainer
    >>> args = dict(model="yolo26s-cwad-plus.yaml", data="dawn6.yaml", epochs=36, device="cpu")
    >>> CWADPlusTrainer(overrides=args).train()
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules.cwad import DomainClassifier
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model


class CWADPlusModel(DetectionModel):
    """CWAD-YOLO26 detector with a DANN weather-domain discriminator.

    Attributes:
        domain_classifier (DomainClassifier | None): Domain branch, or None when `dann: False` in the model YAML.
        dann_lambda_max (float): Peak weight of the domain loss, reached at the last epoch.
        dann_alpha (float): Current gradient-reversal strength, ramped by the trainer each epoch.

    Methods:
        loss: Add the reversed-gradient domain cross-entropy to the detection loss.
        set_dann_alpha: Update the gradient-reversal strength.
    """

    def __init__(self, cfg="yolo26-cwad-plus.yaml", ch=3, nc=None, verbose=True):
        """Initialize the model and its domain branch.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg, ch, nc, verbose)
        self.dann_lambda_max = float(self.yaml.get("dann_lambda", 0.1))
        self.dann_alpha = 0.0
        num_domains = int(self.yaml.get("dann_domains", 4))
        self.domain_classifier = self._build_domain_classifier(num_domains) if self.yaml.get("dann", True) else None
        if self.domain_classifier is not None:
            LOGGER.info(f"CWAD-YOLO26-plus DANN: {num_domains} domains, lambda_max={self.dann_lambda_max}")

    def _build_domain_classifier(self, num_domains: int) -> DomainClassifier:
        """Build the discriminator on top of the detection head's input channels."""
        head = self.model[-1]
        if not hasattr(head, "cv2"):
            raise NotImplementedError("The DANN branch needs a Detect head to source its feature channels.")
        return DomainClassifier([m[0].conv.in_channels for m in head.cv2], num_domains)

    def set_dann_alpha(self, alpha: float) -> None:
        """Set the current gradient-reversal strength.

        Args:
            alpha (float): Reversal strength, typically ramped from 0 to `dann_lambda_max`.
        """
        self.dann_alpha = float(alpha)
        if self.domain_classifier is not None:
            self.domain_classifier.grl.alpha = float(alpha)

    def loss(self, batch: dict[str, Any], preds=None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the detection loss plus the reversed-gradient domain loss.

        Args:
            batch (dict[str, Any]): Batch with images, targets and the optional `domain` weather labels.
            preds (dict, optional): Precomputed predictions.

        Returns:
            loss (torch.Tensor): Total loss, batch-scaled like the detection criterion.
            loss_items (dict[str, torch.Tensor]): Detached loss components.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])
        loss, loss_items = self.criterion(preds, batch)
        domain = batch.get("domain")
        if domain is None or self.domain_classifier is None:
            return loss, loss_items
        feats = preds["one2many"]["feats"] if "one2many" in preds else preds["feats"]
        domain_loss = F.cross_entropy(self.domain_classifier(feats), domain)
        loss_items["dann_loss"] = domain_loss.detach()
        # The criterion returns the batch-scaled box/cls/dfl vector, so sum it and add the domain term once.
        return loss.sum() + domain_loss * batch["img"].shape[0] * self.dann_alpha, loss_items


class CWADPlusTrainer(DetectionTrainer):
    """Detection trainer that labels every DAWN batch with its weather domain for the DANN branch.

    Weather labels are read from `datasets/dawn/conditions.csv`, written by `scripts/cwad/prepare_dawn.py`; they never
    reach the dataloader, so the dataset and its augmentation pipeline stay unchanged.

    Attributes:
        domain_file (Path | None): Manifest the weather labels are read from.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize the trainer.

        Args:
            cfg (dict, optional): Default configuration dictionary.
            overrides (dict, optional): Parameter overrides for the default configuration.
            _callbacks (dict, optional): Callbacks to run during training.
        """
        super().__init__(cfg, overrides, _callbacks)
        self._domains: dict[str, int] | None = None
        self.domain_file: Path | None = None

    def get_model(self, cfg: str | dict | None = None, weights: torch.nn.Module | None = None, verbose: bool = True):
        """Return the plus model, which carries the DANN branch.

        Args:
            cfg (str | dict, optional): Model configuration file path or dictionary.
            weights (torch.nn.Module, optional): Pretrained model whose weights are loaded into the new model.
            verbose (bool): Whether to display model information.

        Returns:
            (CWADPlusModel): Detection model with a domain classifier attached.
        """
        model = self.set_model_names_for_load(
            CWADPlusModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model

    def _load_domains(self) -> dict[str, int]:
        """Map every training image filename to its weather-domain index."""
        self.domain_file = Path(self.data["path"]) / "conditions.csv"
        if not self.domain_file.is_file():
            raise FileNotFoundError(f"DANN needs per-image weather labels, but {self.domain_file} is missing.")
        with self.domain_file.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
        names = sorted({r["weather"] for r in rows})
        LOGGER.info(f"DANN weather domains ({len(names)}): {names}")
        return {r["file"]: names.index(r["weather"]) for r in rows}

    def _dann_alpha(self) -> float:
        """Return the ramped gradient-reversal strength for the current epoch."""
        progress = self.epoch / max(self.epochs - 1, 1)
        return unwrap_model(self.model).dann_lambda_max * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Attach the weather-domain labels and the current reversal strength to a batch.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.

        Returns:
            (dict[str, Any]): Batch with an added `domain` tensor.
        """
        batch = super().preprocess_batch(batch)
        model = unwrap_model(self.model)
        if getattr(model, "domain_classifier", None) is None:  # DANN disabled in the model YAML
            return batch
        if self._domains is None:
            self._domains = self._load_domains()
        names = [Path(p).name for p in batch["im_file"]]
        missing = [n for n in names if n not in self._domains]
        if missing:
            raise KeyError(f"{len(missing)} training images have no weather label in {self.domain_file}: {missing[:2]}")
        model.set_dann_alpha(self._dann_alpha())
        batch["domain"] = torch.tensor([self._domains[n] for n in names], device=self.device, dtype=torch.long)
        return batch
