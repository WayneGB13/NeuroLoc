"""
environment/tumor_env.py
------------------------
Module: Reinforcement Learning Environment

Responsibilities:
  - Provide a Gymnasium-compatible environment where an RL agent searches
    for a tumor region in an MRI image.
  - Maintain a bounding box that the agent adjusts step-by-step.
  - Compute rewards based on IoU improvement and classification correctness.
  - Build the state vector = CNN features + normalised bbox coordinates.

The environment does NOT require ground-truth pixel masks.
Instead it uses the CNN's confidence on the cropped sub-region as a
proxy reward signal (self-supervised localisation).

State  : np.ndarray (2052,) = [CNN_features(2048) | bbox_norm(4)]
Action : int in {0…6}
Reward : float
Done   : bool  (agent says STOP or max_steps exceeded)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF


# ── Action indices ───────────────────────────────────────────────────────────
ACTION_UP     = 0
ACTION_DOWN   = 1
ACTION_LEFT   = 2
ACTION_RIGHT  = 3
ACTION_ZOOM_IN  = 4
ACTION_ZOOM_OUT = 5
ACTION_STOP   = 6

ACTION_NAMES = {
    0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT",
    4: "ZOOM_IN", 5: "ZOOM_OUT", 6: "STOP",
}

# ── Hyper-parameters ─────────────────────────────────────────────────────────
STEP_SIZE   = 0.10   # bbox moves by 10 % of image dimension per step
ZOOM_FACTOR = 0.10   # each zoom changes box size by 10 %
MIN_BOX_SIZE = 0.20  # bounding box never smaller than 20 % of image
MAX_BOX_SIZE = 1.00  # never exceeds full image
MAX_STEPS    = 40    # episode terminates after this many steps

IMG_SIZE = 224       # pixels (matches ResNet50 input)

# Rewards
REWARD_CORRECT_STOP  = +5.0
REWARD_WRONG_STOP    = -3.0
REWARD_STEP_IMPROVE  = +1.0
REWARD_STEP_WORSEN   = -1.0
REWARD_TIMEOUT       = -2.0


class TumorEnv:
    """
    MRI bounding-box localisation environment.

    Parameters
    ----------
    cnn_model  : BrainTumorCNN  Pre-trained CNN (in eval mode).
    image      : torch.Tensor   Single MRI image tensor (3, 224, 224), normalised.
    true_label : int            Ground-truth class index.
    device     : str
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        cnn_model,
        image:      torch.Tensor,
        true_label: int,
        device:     str = "cpu",
    ):
        self.cnn      = cnn_model.eval()
        self.image    = image              # (3, H, W) on CPU
        self.true_label = true_label
        self.device   = torch.device(device)

        self.action_space_n = 7
        self.state_dim      = 2052        # 2048 + 4

        self.bbox:  list[float] = []
        self.steps: int = 0
        self.prev_confidence: float = 0.0

    # ────────────────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        """
        Reset episode: bbox covers the full image, step counter = 0.

        Returns
        -------
        state : (2052,) numpy array
        """
        # normalised [x_min, y_min, x_max, y_max] in [0, 1]
        self.bbox  = [0.0, 0.0, 1.0, 1.0]
        self.steps = 0
        self.prev_confidence = self._get_confidence()
        return self._get_state()

    # ────────────────────────────────────────────────────────────────────────
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one action and return (next_state, reward, done, info).

        Parameters
        ----------
        action : int

        Returns
        -------
        next_state : np.ndarray (2052,)
        reward     : float
        done       : bool
        info       : dict
        """
        self.steps += 1

        # ── STOP action ──────────────────────────────────────────────────────
        if action == ACTION_STOP:
            _, pred_label = self._classify_current_crop()
            if pred_label == self.true_label:
                reward = REWARD_CORRECT_STOP
            else:
                reward = REWARD_WRONG_STOP
            return self._get_state(), reward, True, {"action": "STOP", "pred": pred_label}

        # ── Movement / zoom actions ──────────────────────────────────────────
        self._apply_action(action)
        new_conf = self._get_confidence()

        if new_conf > self.prev_confidence:
            reward = REWARD_STEP_IMPROVE
        else:
            reward = REWARD_STEP_WORSEN

        self.prev_confidence = new_conf

        done = self.steps >= MAX_STEPS
        if done:
            reward += REWARD_TIMEOUT

        return self._get_state(), reward, done, {"confidence": new_conf}

    # ────────────────────────────────────────────────────────────────────────
    def _apply_action(self, action: int):
        """Mutate self.bbox according to the chosen action."""
        x1, y1, x2, y2 = self.bbox
        w = x2 - x1
        h = y2 - y1

        if action == ACTION_UP:
            shift = STEP_SIZE * h
            y1 = max(0.0, y1 - shift)
            y2 = y1 + h

        elif action == ACTION_DOWN:
            shift = STEP_SIZE * h
            y2 = min(1.0, y2 + shift)
            y1 = y2 - h

        elif action == ACTION_LEFT:
            shift = STEP_SIZE * w
            x1 = max(0.0, x1 - shift)
            x2 = x1 + w

        elif action == ACTION_RIGHT:
            shift = STEP_SIZE * w
            x2 = min(1.0, x2 + shift)
            x1 = x2 - w

        elif action == ACTION_ZOOM_IN:
            dw = ZOOM_FACTOR * w / 2
            dh = ZOOM_FACTOR * h / 2
            x1, x2 = x1 + dw, x2 - dw
            y1, y2 = y1 + dh, y2 - dh
            # Prevent box becoming too small
            if (x2 - x1) < MIN_BOX_SIZE:
                x1, x2 = self.bbox[0], self.bbox[2]
            if (y2 - y1) < MIN_BOX_SIZE:
                y1, y2 = self.bbox[1], self.bbox[3]

        elif action == ACTION_ZOOM_OUT:
            dw = ZOOM_FACTOR * w / 2
            dh = ZOOM_FACTOR * h / 2
            x1, x2 = x1 - dw, x2 + dw
            y1, y2 = y1 - dh, y2 + dh
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(1.0, x2), min(1.0, y2)

        self.bbox = [x1, y1, x2, y2]

    # ────────────────────────────────────────────────────────────────────────
    def _crop_to_bbox(self) -> torch.Tensor:
        """
        Crop the stored image tensor to the current bounding box and
        resize back to IMG_SIZE × IMG_SIZE.

        Returns
        -------
        cropped : (1, 3, 224, 224)
        """
        _, H, W = self.image.shape
        x1, y1, x2, y2 = self.bbox

        px1 = max(0, int(x1 * W))
        py1 = max(0, int(y1 * H))
        px2 = min(W, int(x2 * W))
        py2 = min(H, int(y2 * H))

        # Safety: ensure non-zero crop
        if px2 <= px1:
            px2 = min(W, px1 + 1)
        if py2 <= py1:
            py2 = min(H, py1 + 1)

        crop = self.image[:, py1:py2, px1:px2]   # (3, h, w)
        crop = F.interpolate(
            crop.unsqueeze(0),
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        )                                          # (1, 3, 224, 224)
        return crop

    # ────────────────────────────────────────────────────────────────────────
    def _classify_current_crop(self) -> tuple[float, int]:
        """
        Run CNN on the current bbox crop.

        Returns
        -------
        confidence : float  P(true_label | crop)
        pred_label : int    argmax of softmax
        """
        crop = self._crop_to_bbox().to(self.device)
        with torch.no_grad():
            logits = self.cnn(crop)               # (1, 4)
            probs  = torch.softmax(logits, dim=1) # (1, 4)
        pred_label = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, self.true_label].item())
        return confidence, pred_label

    # ────────────────────────────────────────────────────────────────────────
    def _get_confidence(self) -> float:
        conf, _ = self._classify_current_crop()
        return conf

    # ────────────────────────────────────────────────────────────────────────
    def _get_state(self) -> np.ndarray:
        """
        Build the 2052-d state vector.

        state = [CNN_features(2048) | bbox_normalised(4)]
        """
        crop = self._crop_to_bbox().to(self.device)
        with torch.no_grad():
            features = self.cnn.extract_features(crop)   # (1, 2048)
        feat_np = features.squeeze(0).cpu().numpy()      # (2048,)
        bbox_np = np.array(self.bbox, dtype=np.float32)  # (4,)
        return np.concatenate([feat_np, bbox_np])        # (2052,)

    # ────────────────────────────────────────────────────────────────────────
    def render(self) -> np.ndarray:
        """
        Return an RGB numpy array of the image with the current bbox drawn.
        Useful for visualisation / debugging.
        """
        import cv2
        img_np = self.image.permute(1, 2, 0).cpu().numpy()
        # Denormalise
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_np = (img_np * std + mean).clip(0, 1)
        img_bgr = (img_np * 255).astype(np.uint8)[:, :, ::-1].copy()

        H, W = img_bgr.shape[:2]
        x1, y1, x2, y2 = self.bbox
        cv2.rectangle(
            img_bgr,
            (int(x1 * W), int(y1 * H)),
            (int(x2 * W), int(y2 * H)),
            color=(0, 255, 0),
            thickness=2,
        )
        return img_bgr[:, :, ::-1]  # back to RGB

    # ────────────────────────────────────────────────────────────────────────
    def get_bbox_pixels(self) -> tuple[int, int, int, int]:
        """Return bounding box in pixel coordinates (x1, y1, x2, y2)."""
        _, H, W = self.image.shape
        x1, y1, x2, y2 = self.bbox
        return int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
