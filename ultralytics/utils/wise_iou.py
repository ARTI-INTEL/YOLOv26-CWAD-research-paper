# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Wise-IoU v2 bounding box regression loss with dynamic focusing mechanism.

Adapted from the official Wise-IoU v2 implementation (Instinct323/Wise-IoU, branch v2, ``iou.py``) from Tong et al.,
"Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism", https://arxiv.org/abs/2301.10051.

Instead of a hand-tuned penalty term the loss multiplies the per-box IoU loss by an enclosing-box distance term and
rescales it by a dynamic focusing mechanism driven by a running mean of the IoU loss.
"""

from __future__ import annotations

import torch
from torch import nn


class IouLoss(nn.Module):
    """Wise-IoU v2 IoU regression loss with dynamic focusing mechanism.

    The per-box IoU loss (``1 - IoU``) is scaled by ``exp(l2_center / l2_box.detach())``, an exponential distance term
    between the boxes' centers relative to their smallest enclosing box, then rescaled by the focusing factor derived
    from ``iou_mean``, an exponential moving average of the per-box IoU loss updated while training.

    Args:
        monotonous (bool | None): Focusing mechanism variant: False for the non-monotonic Wise-IoU v2 mechanism
            (default), True for the monotonic v1-style mechanism, None to disable dynamic focusing.
    """

    momentum = 1e-2  # EMA momentum of the running IoU mean
    alpha = 1.7  # non-monotonic focusing hyperparameter
    delta = 2.7  # non-monotonic focusing hyperparameter

    def __init__(self, monotonous: bool | None = False):
        """Initialize Wise-IoU v2 loss with the running IoU mean buffer."""
        super().__init__()
        self.monotonous = monotonous
        self.register_buffer("iou_mean", torch.tensor(1.0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the Wise-IoU v2 loss for predicted and target boxes.

        Args:
            pred (torch.Tensor): Predicted boxes in xyxy coordinates, shape (N, 4).
            target (torch.Tensor): Target boxes in xyxy coordinates, shape (N, 4).

        Returns:
            (torch.Tensor): Per-box regression loss, shape (N,).
        """
        # Overlap
        min_coord = torch.minimum(pred, target)
        max_coord = torch.maximum(pred, target)
        wh_inter = torch.relu(min_coord[..., 2:] - max_coord[..., :2])
        s_inter = torch.prod(wh_inter, dim=-1)
        pred_wh = pred[..., 2:] - pred[..., :2]
        target_wh = target[..., 2:] - target[..., :2]
        s_union = torch.prod(pred_wh, dim=-1) + torch.prod(target_wh, dim=-1) - s_inter

        # Smallest enclosing box and central point distance
        wh_box = max_coord[..., 2:] - min_coord[..., :2]
        l2_box = torch.square(wh_box).sum(dim=-1)
        d_center = (pred[..., :2] + pred[..., 2:]) / 2 - (target[..., :2] + target[..., 2:]) / 2
        l2_center = torch.square(d_center).sum(dim=-1)

        # Per-box IoU loss term (1 - IoU)
        iou_term = 1 - s_inter / s_union

        if self.training and iou_term.numel():  # skip empty batches, their mean would poison the EMA with NaN
            self.iou_mean.mul_(1 - self.momentum)
            self.iou_mean.add_(self.momentum * iou_term.detach().mean())

        loss = torch.exp(l2_center / l2_box.detach()) * iou_term
        if self.monotonous is not None:  # dynamic focusing mechanism
            beta = iou_term.detach() / self.iou_mean
            if self.monotonous:
                loss = loss * beta.sqrt()
            else:
                divisor = self.delta * torch.pow(self.alpha, beta - self.delta)
                loss = loss * beta / divisor
        return loss
