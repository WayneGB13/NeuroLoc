"""
models/resnet50_model.py
------------------------
Module: CNN Feature Extractor + Classifier

Responsibilities:
  - Load a pre-trained ResNet50 backbone (ImageNet weights).
  - Replace the final FC layer with a 4-class head for tumor classification.
  - Expose `extract_features()` — strips the FC head and returns a flat
    2048-d feature vector.  
  - Support Grad-CAM by exposing the last convolutional layer.

Architecture diagram:
    Input (3×224×224)
        │
    ResNet50 backbone  ← frozen in Stage-1, unfrozen in Stage-2
        │
    AdaptiveAvgPool  →  (2048,)
        │
    Dropout(0.5)
        │
    Linear(2048 → 4)   ← replaced FC head
        │
    Softmax  (during inference)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


NUM_CLASSES = 4          # glioma, meningioma, notumor, pituitary
FEATURE_DIM = 2048       # ResNet50 penultimate layer width


class BrainTumorCNN(nn.Module):
    """
    ResNet50-based classifier for brain tumour MRI images.

    Parameters
    ----------
    num_classes  : int   Number of output classes (default 4).
    freeze_backbone : bool
        If True, backbone weights are frozen and only the new FC head
        is trained.  Set False for fine-tuning all layers.
    pretrained   : bool  Use ImageNet pre-trained weights.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        # ── Load pre-trained ResNet50 ────────────────────────────────────────
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # ── Freeze backbone layers (Stage-1 training) ───────────────────────
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # ── Replace the 1000-class FC with a 4-class head ───────────────────
        in_features = backbone.fc.in_features   # 2048
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes),
        )

        self.backbone = backbone

        # ── Reference to last conv layer (needed for Grad-CAM) ───────────────
        self.last_conv_layer = backbone.layer4

    # ────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass → class logits.

        Parameters
        ----------
        x : (B, 3, 224, 224)

        Returns
        -------
        logits : (B, num_classes)
        """
        return self.backbone(x)

    # ────────────────────────────────────────────────────────────────────────
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the 2048-d feature vector BEFORE the classification head.
        Used by the DQN agent to build the RL state.

        Parameters
        ----------
        x : (B, 3, 224, 224)

        Returns
        -------
        features : (B, 2048)
        """
        b = self.backbone

        # Mirror ResNet50 forward() up to avgpool
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)

        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)

        x = b.avgpool(x)              # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)      # (B, 2048)
        return x

    # ────────────────────────────────────────────────────────────────────────
    def unfreeze_backbone(self):
        """Unfreeze all backbone layers for Stage-2 fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[BrainTumorCNN] Backbone unfrozen — entering fine-tuning stage.")

    # ────────────────────────────────────────────────────────────────────────
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method: returns (class_indices, probabilities).

        Parameters
        ----------
        x : (B, 3, 224, 224)

        Returns
        -------
        pred_class : (B,) int64
        probs      : (B, num_classes) float
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)
            pred_class = probs.argmax(dim=1)
        return pred_class, probs


# ── Convenience factory ──────────────────────────────────────────────────────
def build_cnn(
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    device: str = "cpu",
) -> BrainTumorCNN:
    """Instantiate BrainTumorCNN and move to device."""
    model = BrainTumorCNN(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        pretrained=pretrained,
    ).to(device)
    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[BrainTumorCNN] Total params: {total_params:,} | "
          f"Trainable: {trainable_params:,}")
    return model
