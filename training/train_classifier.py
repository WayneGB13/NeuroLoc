import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

# ── Project imports ───────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.resnet50_model import build_cnn
from utils.preprocessing import load_datasets

# ── Hyper-parameters ─────────────────────────────────────────────────────────
STAGE1_EPOCHS = 10      # train only FC head
STAGE2_EPOCHS = 20      # fine-tune full network
STAGE1_LR     = 1e-3
STAGE2_LR     = 1e-4
WEIGHT_DECAY  = 1e-4

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = RESULTS_DIR / "best_cnn.pth"
METRICS_PATH    = RESULTS_DIR / "cnn_training_metrics.json"
PLOT_PATH       = RESULTS_DIR / "cnn_training_curves.png"


# ════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    """Run one full training pass over the DataLoader."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Run one evaluation pass and return (loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ════════════════════════════════════════════════════════════════════════════
def plot_curves(history: dict, save_path: Path):
    """Save a 2-panel loss / accuracy plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"],   label="Val Acc")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Train] Saved training curves → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
def run_stage(
    model,
    train_loader,
    val_loader,
    epochs:   int,
    lr:       float,
    device,
    best_val_acc: float,
    history:  dict,
    stage_name: str,
) -> float:
    """Generic training loop for a single stage."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)

    print(f"\n{'='*60}")
    print(f"  {stage_name} — {epochs} epochs, lr={lr}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>3}/{epochs} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✓ New best val acc {val_acc:.4f} — model saved.")

    return best_val_acc


# ════════════════════════════════════════════════════════════════════════════
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names = load_datasets()

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_cnn(
        num_classes=len(class_names),
        freeze_backbone=True,
        pretrained=True,
        device=device,
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_acc = 0.0

    # ── Stage 1: FC head only ─────────────────────────────────────────────────
    best_val_acc = run_stage(
        model, train_loader, val_loader,
        epochs=STAGE1_EPOCHS, lr=STAGE1_LR,
        device=device, best_val_acc=best_val_acc,
        history=history, stage_name="Stage-1 (frozen backbone)",
    )

    # ── Stage 2: Fine-tune full network ──────────────────────────────────────
    model.unfreeze_backbone()
    best_val_acc = run_stage(
        model, train_loader, val_loader,
        epochs=STAGE2_EPOCHS, lr=STAGE2_LR,
        device=device, best_val_acc=best_val_acc,
        history=history, stage_name="Stage-2 (fine-tune all layers)",
    )

    # ── Save metrics ──────────────────────────────────────────────────────────
    with open(METRICS_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Train] Saved metrics → {METRICS_PATH}")

    # ── Save plots ────────────────────────────────────────────────────────────
    plot_curves(history, PLOT_PATH)

    print(f"\n[Train] Done. Best Val Accuracy: {best_val_acc:.4f}")
    print(f"[Train] Best model saved at: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
