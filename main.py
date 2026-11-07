"""
main.py
-------
Orchestrator — Brain Tumor RL Detection Pipeline

This script ties all modules together.  It supports four modes via CLI:

  python main.py --mode train_cnn        →  Stage-1 + Stage-2 CNN training
  python main.py --mode train_rl         →  DQN agent training
  python main.py --mode evaluate         →  Full evaluation on test set
  python main.py --mode infer --img PATH →  Single-image inference + Grad-CAM

Expected workflow:
  1. Download dataset from Kaggle, place under dataset/Training & dataset/Testing
  2. python main.py --mode train_cnn
  3. python main.py --mode train_rl
  4. python main.py --mode evaluate
  5. python main.py --mode infer --img dataset/Testing/glioma/some_image.jpg
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

# ── Project imports ───────────────────────────────────────────────────────────
from models.resnet50_model import build_cnn, NUM_CLASSES
from models.dqn_agent import DQNAgent
from environment.tumor_env import TumorEnv
from explainability.grad_cam import GradCAM
from evaluation.metrics import run_full_evaluation, evaluate_localisation
from utils.preprocessing import (
    load_datasets,
    get_eval_transforms,
    CLASS_NAMES,
)

BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATASET_DIR = BASE_DIR / "dataset"


# ════════════════════════════════════════════════════════════════════════════
def load_pretrained_cnn(device: str):
    cnn_path = RESULTS_DIR / "best_cnn.pth"
    model = build_cnn(
        num_classes=NUM_CLASSES,
        freeze_backbone=False,
        pretrained=False,
        device=device,
    )
    model.load_state_dict(torch.load(cnn_path, map_location=device))
    model.eval()
    print(f"[Main] CNN loaded from {cnn_path}")
    return model


# ════════════════════════════════════════════════════════════════════════════
def mode_train_cnn():
    """Launch CNN training (stages 1 + 2)."""
    from training.train_classifier import main as train_main
    train_main()


# ════════════════════════════════════════════════════════════════════════════
def mode_train_rl():
    """Launch RL agent training."""
    from training.train_rl_agent import main as rl_main
    rl_main()


# ════════════════════════════════════════════════════════════════════════════
def mode_evaluate():
    """
    Evaluate the trained CNN on the test set.
    Also runs a mini RL localization evaluation using the DQN agent.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── CNN evaluation ────────────────────────────────────────────────────────
    _, _, test_loader, class_names = load_datasets(dataset_root=str(DATASET_DIR))
    cnn = load_pretrained_cnn(device)
    run_full_evaluation(cnn, test_loader, class_names, device)

    # ── RL localization evaluation (sample 50 test images) ───────────────────
    dqn_path = RESULTS_DIR / "best_dqn.pth"
    if not dqn_path.exists():
        print("[Main] DQN checkpoint not found — skipping localization eval.")
        return

    agent = DQNAgent(device=device)
    agent.load(str(dqn_path))
    agent.epsilon = 0.0   # greedy at test time

    from torchvision import datasets
    test_ds = datasets.ImageFolder(
        root=str(DATASET_DIR / "Testing"),
        transform=get_eval_transforms(),
    )

    N_EVAL   = min(50, len(test_ds))
    pred_boxes, gt_boxes = [], []

    for idx in range(N_EVAL):
        image, label = test_ds[idx]
        env   = TumorEnv(cnn_model=cnn, image=image, true_label=label, device=device)
        state = env.reset()
        done  = False

        while not done:
            action = agent.select_action(state)
            state, _, done, _ = env.step(action)

        pred_box = env.get_bbox_pixels()
        # No GT pixel mask in this dataset — use full-image as proxy GT
        _, H, W = image.shape
        gt_box   = (0, 0, W, H)
        pred_boxes.append(pred_box)
        gt_boxes.append(gt_box)

    evaluate_localisation(pred_boxes, gt_boxes)
    print("[Main] Evaluation complete.")


# ════════════════════════════════════════════════════════════════════════════
def mode_infer(img_path: str):
    """
    Single-image inference pipeline:
      1. Classify tumor type with CNN.
      2. Localize tumor region with DQN agent.
      3. Generate Grad-CAM heatmap.
      4. Save annotated figure.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load image ────────────────────────────────────────────────────────────
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    transform = get_eval_transforms()
    pil_image = Image.open(img_path).convert("RGB")
    image_tensor = transform(pil_image)   # (3, 224, 224)

    # ── CNN classification ───────────────────────────────────────────────────
    cnn = load_pretrained_cnn(device)

    input_batch = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = cnn(input_batch)
        probs  = torch.softmax(logits, dim=1)
    pred_idx   = int(probs.argmax(dim=1).item())
    confidence = float(probs[0, pred_idx].item())
    pred_label = CLASS_NAMES[pred_idx]

    print(f"\n[Inference] Predicted Class : {pred_label}")
    print(f"[Inference] Confidence      : {confidence:.2%}")

    # ── DQN localization ─────────────────────────────────────────────────────
    bbox_pixels = None
    dqn_path = RESULTS_DIR / "best_dqn.pth"

    if dqn_path.exists():
        agent = DQNAgent(device=device)
        agent.load(str(dqn_path))
        agent.epsilon = 0.0

        env   = TumorEnv(cnn_model=cnn, image=image_tensor, true_label=pred_idx, device=device)
        state = env.reset()
        done  = False
        step  = 0

        while not done and step < 40:
            action = agent.select_action(state)
            state, _, done, info = env.step(action)
            step += 1

        bbox_pixels = env.get_bbox_pixels()
        print(f"[Inference] Bounding Box (pixels): {bbox_pixels}")
    else:
        print("[Inference] DQN model not found — skipping localization.")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    gradcam  = GradCAM(model=cnn, device=device)
    heatmap  = gradcam.generate_heatmap(image_tensor, target_class=pred_idx)

    # Derive true class from folder name (if path has a parent folder label)
    true_label = img_path.parent.name

    out_path = str(RESULTS_DIR / f"gradcam_{img_path.stem}.png")
    gradcam.visualise_and_save(
        image_tensor  = image_tensor,
        true_class    = true_label,
        pred_class    = pred_label,
        confidence    = confidence,
        heatmap       = heatmap,
        bbox_pixels   = bbox_pixels,
        save_path     = out_path,
    )

    print(f"\n[Inference] ✓ Output saved → {out_path}")
    print(f"[Inference] Summary:")
    print(f"  • Tumor Type    : {pred_label}")
    print(f"  • Confidence    : {confidence:.2%}")
    print(f"  • Bounding Box  : {bbox_pixels}")
    print(f"  • Grad-CAM      : {out_path}")


# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Brain Tumor RL Detection — Main Orchestrator"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train_cnn", "train_rl", "evaluate", "infer"],
        required=True,
        help="Pipeline mode to execute.",
    )
    parser.add_argument(
        "--img",
        type=str,
        default=None,
        help="Path to MRI image (required for --mode infer).",
    )
    args = parser.parse_args()

    if args.mode == "train_cnn":
        mode_train_cnn()

    elif args.mode == "train_rl":
        mode_train_rl()

    elif args.mode == "evaluate":
        mode_evaluate()

    elif args.mode == "infer":
        if args.img is None:
            parser.error("--img is required when --mode infer.")
        mode_infer(args.img)


if __name__ == "__main__":
    main()
