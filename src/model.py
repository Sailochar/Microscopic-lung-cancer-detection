"""
model.py  -  Step 2: EfficientNet-B0 with fully fine-tuned 3-class head
========================================================================
All backbone layers are trainable by default (freeze_backbone=False).
Frozen backbone was a root cause of lung_aca/lung_scc confusion in Review-2.

Usage:
    from model import get_model
    model = get_model()          # 3-class, fully trainable
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class LungCancerModel(nn.Module):

    def __init__(self, num_classes: int = 3, freeze_backbone: bool = False,
                 pretrained: bool = True):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ── Flower parameter helpers ───────────────────────────────────────────
    def get_parameters(self):
        """Return model weights as list of numpy arrays (for Flower)."""
        import numpy as np
        return [v.cpu().numpy() for v in self.state_dict().values()]

    def set_parameters(self, parameters):
        """Load list of numpy arrays back into the model (from Flower)."""
        import numpy as np
        state_dict = OrderedDict(
            {k: torch.tensor(v)
             for k, v in zip(self.state_dict().keys(), parameters)}
        )
        self.load_state_dict(state_dict, strict=True)


def get_model(num_classes: int = 3,
              freeze_backbone: bool = False,
              pretrained: bool = True) -> LungCancerModel:
    return LungCancerModel(num_classes=num_classes,
                            freeze_backbone=freeze_backbone,
                            pretrained=pretrained)


if __name__ == "__main__":
    model = get_model()
    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    print(f"Output shape   : {out.shape}")    # (2, 3)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params   : {total:,}")
    print(f"Trainable      : {trainable:,}")
    print("model.py self-test passed.")
