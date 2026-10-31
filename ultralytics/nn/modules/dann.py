# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Domain-Adversarial Neural Network (DANN) head for the pro YOLO26-CWAD detector.

The head extends the native end-to-end Detect head with an auxiliary domain classifier attached to the deepest
detection feature through a gradient reversal layer, so the backbone learns features that stay discriminative for
detection while becoming indistinguishable between weather/domain sources (e.g. SODA10M source vs DAWN target).
The adversarial loss is added by `ultralytics.utils.loss.DANNE2ELoss`.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function

from .head import Detect

__all__ = ("DANNDetect", "DomainDiscriminator", "GradientReversal")


class GradientReversal(Function):
    """Identity in the forward pass and a negated, alpha-scaled gradient in the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Return the input unchanged and store the reversal strength for backward."""
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Return the negated gradient so the upstream features maximize the domain loss."""
        return -ctx.alpha * grad, None


class DomainDiscriminator(nn.Module):
    """Classify the domain of a feature map through a gradient reversal layer.

    Args:
        c1 (int): Input channel count of the feature map being classified.
        n_domains (int): Number of domains (classes) to discriminate.
        hidden (int | None): Hidden unit count; defaults to a quarter of the input channels, at least 128.
        dropout (float): Dropout probability before the output layer.
    """

    def __init__(self, c1: int, n_domains: int = 2, hidden: int | None = None, dropout: float = 0.5):
        """Initialize the domain classifier."""
        super().__init__()
        hidden = hidden or max(c1 // 4, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, hidden),
            nn.LayerNorm(hidden),  # layer norm, unlike batch norm, works for a single-image batch
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_domains),
        )
        self.alpha = 1.0  # gradient reversal strength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return domain logits of shape (batch, n_domains) for a (batch, c1, h, w) feature map."""
        return self.fc(GradientReversal.apply(self.pool(x).flatten(1), self.alpha))


class DANNDetect(Detect):
    """Detect head with an auxiliary domain-adversarial branch used during training only.

    Args:
        nc (int): Number of classes.
        n_domains (int): Number of domains for the adversarial classifier.
        reg_max (int): Maximum number of DFL channels.
        end2end (bool): Whether to use end-to-end NMS-free detection.
        ch (list[int] | tuple[int, ...]): Channel sizes from backbone feature maps.
        domain_weight (float): Weight of the domain adversarial loss relative to the detection loss.
    """

    def __init__(
        self,
        nc: int = 80,
        n_domains: int = 2,
        reg_max: int = 16,
        end2end: bool = False,
        ch: list[int] | tuple[int, ...] = (),
        domain_weight: float = 0.1,
    ):
        """Initialize the head and the domain classifier on the deepest detection feature."""
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.n_domains = n_domains
        self.domain_weight = domain_weight
        self.domain = DomainDiscriminator(ch[-1], n_domains)

    def forward(self, x: list[torch.Tensor]) -> dict[str, torch.Tensor] | torch.Tensor | tuple:
        """Run the detection head and, in training, attach the domain logits to the returned predictions dict."""
        preds = super().forward(x)
        if self.training and isinstance(preds, dict):
            preds["domain"] = self.domain(x[-1])
        return preds
