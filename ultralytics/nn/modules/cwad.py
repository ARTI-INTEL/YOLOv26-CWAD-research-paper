# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Custom modules for the CWAD-YOLO research model.

Phase 2 adds LightSDI (Light Semantics and Detail Infusion), the multi-scale feature fusion module from the
CWAD-YOLO paper that replaces the neck's Concat layers with depthwise/pointwise refinement and Hadamard product
fusion. Phase 3 adds DKSCDown, the dual-kernel (3x3/5x5) depthwise stride-2 downsampler with GAP+Softmax dynamic
branch weighting that replaces the backbone/neck's stride-2 Conv layers. Phase 4 adds the DANN building blocks,
GradientReversal and DomainClassifier, which let CWAD-YOLO26-plus learn weather-invariant features adversarially.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .conv import Conv

__all__ = ("DKSCDown", "DomainClassifier", "GradientReversal", "LightSDI")


class LightSDI(nn.Module):
    """Light Semantics and Detail Infusion (LightSDI) multi-scale feature fusion module.

    Every incoming feature is resized to the anchor spatial size (bilinear interpolation for smaller features,
    adaptive average pooling for larger ones), refined by a depthwise 3x3 convolution followed by a pointwise 1x1
    convolution that aligns it to the output width, and all refined features are fused by element-wise (Hadamard)
    multiplication instead of concatenation.
    """

    def __init__(self, c1, c2):
        """Initialize LightSDI.

        Args:
            c1 (list[int]): Input channel counts of each incoming feature.
            c2 (int): Output channel count every input is aligned to before fusion.
        """
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1, groups=c),  # depthwise 3x3 spatial refinement
                nn.Conv2d(c, c2, 1),  # pointwise 1x1 channel alignment to c2
            )
            for c in c1
        )

    def forward(self, xs):
        """Resize, refine, and element-wise multiply all input features.

        Args:
            xs (list[torch.Tensor]): Input features; xs[0] anchors the target spatial size.

        Returns:
            (torch.Tensor): Fused feature map of shape (batch, c2, H, W).
        """
        size = xs[0].shape[-2:]
        out = None
        for conv, x in zip(self.convs, xs):
            if x.shape[-2:] != size:  # smaller features are upsampled, larger ones average-pooled
                x = (
                    F.adaptive_avg_pool2d(x, size)
                    if x.shape[-2] > size[0]
                    else F.interpolate(x, size=size, mode="bilinear", align_corners=True)
                )
            x = conv(x)
            out = x if out is None else out * x
        return out


class DKSCDown(nn.Module):
    """Dual-Kernel Scale-Consistent Downsample (DK_SCDown) from the CWAD-YOLO paper.

    A 1x1 convolution reduces the input to c2 channels, which are then processed by two parallel stride-2
    depthwise convolutions (3x3 for local detail, 5x5 for wider context). Global average pooling over the reduced
    features feeds a linear layer whose 2-way Softmax produces per-sample dynamic branch weights, and the output
    is their weighted sum.
    """

    def __init__(self, c1, c2):
        """Initialize DK_SCDown: 1x1 reduce, dual depthwise stride-2 branches, dynamic weighting head.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.reduce = Conv(c1, c2, 1, 1)  # 1x1 channel reduction (BN + SiLU)
        self.branch3 = Conv(c2, c2, 3, 2, g=c2)  # depthwise 3x3 detail branch (BN + SiLU)
        self.branch5 = Conv(c2, c2, 5, 2, g=c2)  # depthwise 5x5 context branch (BN + SiLU)
        self.pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.gate = nn.Linear(c2, 2)  # dynamic branch weights

    def forward(self, x):
        """Downsample x by the GAP+Softmax-weighted sum of the dual depthwise branches.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, c1, H, W).

        Returns:
            (torch.Tensor): Downsampled tensor of shape (batch, c2, H/2, W/2).
        """
        x = self.reduce(x)
        w = torch.softmax(self.gate(self.pool(x).flatten(1)), dim=1)  # (B, 2)
        w3, w5 = w[:, 0, None, None, None], w[:, 1, None, None, None]
        return w3 * self.branch3(x) + w5 * self.branch5(x)


class _GradientReversal(torch.autograd.Function):
    """Identity in the forward pass, gradient reversed and scaled by alpha in the backward pass."""

    @staticmethod
    def forward(ctx, x, alpha):
        """Return x unchanged and remember the reversal strength."""
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        """Return the negated, alpha-scaled gradient."""
        return -ctx.alpha * grad, None


class GradientReversal(nn.Module):
    """Domain-adversarial gradient reversal layer (Ganin et al., DANN).

    Features pass through unchanged, but the gradient flowing back from the domain classifier is negated and scaled by
    `alpha`. Minimizing the domain loss therefore makes the domain classifier accurate while pushing the features it
    reads toward domain invariance, which is the adversarial objective of the DANN framework.
    """

    def __init__(self, alpha=1.0):
        """Initialize the layer.

        Args:
            alpha (float): Gradient reversal strength, ramped from 0 to its peak over a training run.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        """Return x unchanged with reversed gradients.

        Args:
            x (torch.Tensor): Input features of shape (batch, channels).

        Returns:
            (torch.Tensor): The same tensor.
        """
        return _GradientReversal.apply(x, self.alpha)


class DomainClassifier(nn.Module):
    """DANN domain discriminator over the detection head's multi-scale features.

    Each detection level is global-average-pooled, the pooled vectors are concatenated, and a two-layer perceptron
    predicts which weather domain the image came from. The pooled vector passes through `GradientReversal` first, so
    the backbone receives a domain-confusing gradient.
    """

    def __init__(self, ch, num_domains=4, hidden=256, p=0.1, alpha=1.0):
        """Initialize the discriminator.

        Args:
            ch (list[int]): Channel count of each detection level, in feature-pyramid order.
            num_domains (int): Number of domains to discriminate between.
            hidden (int): Width of the hidden layer.
            p (float): Dropout probability before the output layer.
            alpha (float): Initial gradient reversal strength.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.grl = GradientReversal(alpha)
        self.head = nn.Sequential(
            nn.Linear(sum(ch), hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, feats):
        """Predict the domain logits of a batch from its multi-scale features.

        Args:
            feats (list[torch.Tensor]): Detection-level feature maps.

        Returns:
            (torch.Tensor): Domain logits of shape (batch, num_domains).
        """
        return self.head(self.grl(torch.cat([self.pool(f).flatten(1) for f in feats], 1)))
