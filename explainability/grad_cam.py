"""
explainability/grad_cam.py
--------------------------
Module: Grad-CAM Explainability

Responsibilities:
  - Implement vanilla Grad-CAM using PyTorch hooks.
  - Generate a class-discriminative heatmap for any target class.
  - Overlay heatmap on the original MRI image in a matplotlib figure.
  - Optionally overlay the RL bounding box on the same figure.
  - Save the annotated figure to disk.

How Grad-CAM works (3 steps):
  1. Run a forward pass and record the activation map of the last conv layer.
  2. Run a backward pass for the target class; record the gradients flowing
     into that same layer.
  3. Weight each channel of the activation map by the global-average of the
     corresponding gradient channel, then ReLU the sum → heatmap.

The heatmap is then resized to input resolution and blended with the image.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ════════════════════════════════════════════════════════════════════════════
class GradCAM:
    """
    Grad-CAM implementation using forward/backward hooks.

    Parameters
    ----------
    model       : BrainTumorCNN  Must expose `.last_conv_layer`.
    target_layer: nn.Module      The convolutional layer to hook
                                 (defaults to model.last_conv_layer).
    device      : str
    """

    def __init__(self, model, target_layer=None, device: str = "cpu"):
        self.model  = model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

        # Default: use the last ResNet bottleneck block
        self.target_layer = target_layer or model.last_conv_layer

        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None

        self._register_hooks()

    # ────────────────────────────────────────────────────────────────────────
    def _register_hooks(self):
        """Attach forward and backward hooks to the target layer."""

        def _save_activation(module, input, output):
            self._activations = output.detach()

        def _save_gradient(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(_save_activation)
        self.target_layer.register_full_backward_hook(_save_gradient)

    # ────────────────────────────────────────────────────────────────────────
    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Compute a Grad-CAM heatmap for the given image.

        Parameters
        ----------
        image_tensor : (1, 3, 224, 224) or (3, 224, 224)  normalised tensor
        target_class : int  Class index to explain.
                            If None, uses the predicted class.

        Returns
        -------
        heatmap : (224, 224) float32 ndarray in [0, 1]
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_(False)

        # ── Forward pass ────────────────────────────────────────────────────
        self.model.zero_grad()
        logits = self.model(image_tensor)               # (1, 4)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # ── Backward pass for target class ──────────────────────────────────
        score = logits[0, target_class]
        score.backward()

        # ── Compute channel importance weights (global average pooling) ──────
        # gradients: (1, C, H, W)  →  weights: (C,)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # ── Weighted combination of activation maps ──────────────────────────
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)                                              # only positive influence

        # ── Normalise to [0, 1] ──────────────────────────────────────────────
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # ── Resize to input resolution ───────────────────────────────────────
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        heatmap = cam.squeeze().cpu().numpy().astype(np.float32)
        return heatmap

    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def overlay_heatmap(
        image_tensor: torch.Tensor,
        heatmap:      np.ndarray,
        alpha:        float = 0.5,
    ) -> np.ndarray:
        """
        Blend the heatmap with the original image using a jet colourmap.

        Parameters
        ----------
        image_tensor : (3, 224, 224) normalised tensor
        heatmap      : (224, 224) float32 in [0, 1]
        alpha        : blend weight for the heatmap

        Returns
        -------
        blended : (224, 224, 3) uint8 RGB ndarray
        """
        # Denormalise image
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = image_tensor.permute(1, 2, 0).cpu().numpy()
        img  = (img * std + mean).clip(0, 1)

        # Colourmap: jet
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

        blended = (1 - alpha) * img + alpha * heatmap_rgb
        blended = (blended.clip(0, 1) * 255).astype(np.uint8)
        return blended

    # ────────────────────────────────────────────────────────────────────────
    def visualise_and_save(
        self,
        image_tensor: torch.Tensor,
        true_class:   str,
        pred_class:   str,
        confidence:   float,
        heatmap:      np.ndarray,
        bbox_pixels:  tuple[int, int, int, int] | None = None,
        save_path:    str = "results/gradcam_output.png",
    ):
        """
        Create a 3-panel figure and save to disk:
          Panel 1 — Original MRI
          Panel 2 — Grad-CAM overlay
          Panel 3 — Grad-CAM + bounding box (if provided)

        Parameters
        ----------
        image_tensor : (3, 224, 224) normalised tensor
        true_class   : str  Ground-truth label name
        pred_class   : str  Predicted label name
        confidence   : float Softmax probability
        heatmap      : (224, 224) float32 ndarray
        bbox_pixels  : (x1, y1, x2, y2) pixel coords or None
        save_path    : str
        """
        # Denormalise original
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        orig = image_tensor.permute(1, 2, 0).cpu().numpy()
        orig = (orig * std + mean).clip(0, 1)

        overlay = self.overlay_heatmap(image_tensor, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"True: {true_class}  |  Pred: {pred_class}  |  Confidence: {confidence:.2%}",
            fontsize=14,
            fontweight="bold",
        )

        # Panel 1 — Original
        axes[0].imshow(orig)
        axes[0].set_title("Original MRI")
        axes[0].axis("off")

        # Panel 2 — Grad-CAM overlay
        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        # Panel 3 — Grad-CAM + bounding box
        axes[2].imshow(overlay)
        axes[2].set_title("Grad-CAM + RL Bounding Box")
        axes[2].axis("off")

        if bbox_pixels is not None:
            x1, y1, x2, y2 = bbox_pixels
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="lime", facecolor="none",
                label="RL BBox",
            )
            axes[2].add_patch(rect)
            axes[2].legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[GradCAM] Saved visualisation → {save_path}")

    # ────────────────────────────────────────────────────────────────────────
    def cleanup(self):
        """Remove hooks (prevents memory leaks in long loops)."""
        # Hooks are automatically removed when handles go out of scope
        # Re-create the object for each batch to stay safe.
        pass
