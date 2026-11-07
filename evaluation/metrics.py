"""
evaluation/metrics.py
---------------------
Module: Evaluation Metrics

Responsibilities:
  - Classification metrics: Accuracy, Precision, Recall, F1 (per-class + macro).
  - Localisation metrics:   IoU (Jaccard Index), Dice Score.
  - Run full evaluation on the test DataLoader.
  - Print a classification report and confusion matrix.
  - Save all metrics to results/evaluation_report.json.
  - Save a confusion matrix figure to results/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

import sys
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# Classification Helpers
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a DataLoader and collect all predictions + true labels.

    Returns
    -------
    y_true : (N,) int
    y_pred : (N,) int
    """
    model.eval()
    all_true, all_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_pred.extend(preds)
        all_true.extend(labels.numpy())

    return np.array(all_true), np.array(all_pred)


def compute_classification_metrics(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    class_names:  list[str],
    save_path:    str = str(RESULTS_DIR / "classification_report.json"),
) -> dict:
    """
    Compute and return a dict with accuracy + per-class precision/recall/F1.
    Also prints a formatted report and saves JSON.

    Parameters
    ----------
    y_true, y_pred : integer label arrays
    class_names    : ordered list matching label indices

    Returns
    -------
    report_dict : dict
    """
    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report["accuracy"] = acc

    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print(f"  Overall Accuracy: {acc:.4f}")
    print("=" * 60)

    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Metrics] Saved classification report → {save_path}")

    return report


def plot_confusion_matrix(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: list[str],
    save_path:   str = str(RESULTS_DIR / "confusion_matrix.png"),
):
    """Save a normalised heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Normalised Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Metrics] Saved confusion matrix → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# Localisation Helpers
# ════════════════════════════════════════════════════════════════════════════

def compute_iou(
    pred_box:  tuple[int, int, int, int],
    gt_box:    tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection-over-Union (IoU / Jaccard Index) between two boxes.

    Boxes are in (x1, y1, x2, y2) pixel format.

    Returns
    -------
    iou : float in [0, 1]
    """
    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    pred_area = max(0, px2 - px1) * max(0, py2 - py1)
    gt_area   = max(0, gx2 - gx1) * max(0, gy2 - gy1)

    union_area = pred_area + gt_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_dice(
    pred_box: tuple[int, int, int, int],
    gt_box:   tuple[int, int, int, int],
) -> float:
    """
    Dice Score = 2 * |A ∩ B| / (|A| + |B|)

    Equivalent to F1 at the voxel level.
    """
    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    pred_area  = max(0, px2 - px1) * max(0, py2 - py1)
    gt_area    = max(0, gx2 - gx1) * max(0, gy2 - gy1)

    denom = pred_area + gt_area
    if denom == 0:
        return 0.0
    return (2.0 * inter_area) / denom


def evaluate_localisation(
    pred_boxes: list[tuple],
    gt_boxes:   list[tuple],
    save_path:  str = str(RESULTS_DIR / "localisation_metrics.json"),
) -> dict:
    """
    Compute mean IoU and mean Dice over a list of prediction/GT box pairs.

    Parameters
    ----------
    pred_boxes : list of (x1, y1, x2, y2)
    gt_boxes   : list of (x1, y1, x2, y2)  (ground-truth)

    Returns
    -------
    results : dict with 'mean_iou' and 'mean_dice'
    """
    assert len(pred_boxes) == len(gt_boxes), "Mismatched box lists."

    ious, dices = [], []
    for pred, gt in zip(pred_boxes, gt_boxes):
        ious.append(compute_iou(pred, gt))
        dices.append(compute_dice(pred, gt))

    results = {
        "mean_iou":  float(np.mean(ious)),
        "std_iou":   float(np.std(ious)),
        "mean_dice": float(np.mean(dices)),
        "std_dice":  float(np.std(dices)),
        "n_samples": len(ious),
    }

    print("\n" + "=" * 40)
    print("  LOCALISATION METRICS")
    print("=" * 40)
    print(f"  Mean IoU  : {results['mean_iou']:.4f} ± {results['std_iou']:.4f}")
    print(f"  Mean Dice : {results['mean_dice']:.4f} ± {results['std_dice']:.4f}")
    print("=" * 40)

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Metrics] Saved localisation metrics → {save_path}")

    return results


# ════════════════════════════════════════════════════════════════════════════
# Full Evaluation Pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    cnn_model,
    test_loader,
    class_names: list[str],
    device: str = "cpu",
):
    """
    Convenience function: collect predictions, compute classification metrics,
    and plot confusion matrix.

    Returns
    -------
    report : dict
    """
    print("[Metrics] Running full evaluation on test set…")
    y_true, y_pred = collect_predictions(cnn_model, test_loader, device)
    report = compute_classification_metrics(y_true, y_pred, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names)
    return report
