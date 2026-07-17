"""
utils/preprocessing.py
-----------------------
Module: Data Preprocessing & Loading

Responsibilities:
  - Load the Kaggle Brain Tumor MRI dataset from a local folder.
  - Apply image transforms: resize, normalize, and augment.
  - Split dataset into Train / Validation / Test subsets.
  - Return PyTorch DataLoader objects ready for training.

Dataset expected folder layout (after Kaggle download):
    dataset/
        Training/
            glioma/
            meningioma/
            notumor/
            pituitary/
        Testing/
            glioma/
            meningioma/
            notumor/
            pituitary/
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ── Class labels (order must match torchvision folder-sorting) ──────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# ── Image statistics (ImageNet values; fine for transfer learning) ───────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE = 224          # ResNet50 input size
BATCH_SIZE = 32
NUM_WORKERS = 4         # Adjust based on CPU cores


def get_train_transforms() -> transforms.Compose:
    """
    Augmented transforms for the training split.
    Augmentations reduce overfitting on the limited medical dataset.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),   # slightly larger before crop
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """
    Deterministic transforms for validation / test splits.
    No augmentation — we want reproducible evaluation.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_datasets(dataset_root: str = "dataset"):
    """
    Load Training and Testing splits from the Kaggle folder layout.

    The Kaggle dataset ships with a pre-defined Training/ and Testing/ split.
    We further split Training → train (70 %) + val (15 %).
    Testing/ is used as-is for the held-out test set (≈15 %).

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    class_names                           : list[str]
    """
    root = Path(dataset_root)
    train_dir = root / "Training"
    test_dir  = root / "Testing"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training folder not found at '{train_dir}'.\n"
            "Download the Kaggle dataset and place it under dataset/Training & dataset/Testing."
        )

    # ── Full training set (with augmentation) ───────────────────────────────
    full_train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=get_train_transforms(),
    )

    # ── 70 / 15 split inside the Kaggle Training folder ─────────────────────
    total = len(full_train_dataset)
    train_size = int(0.70 * total)
    val_size   = total - train_size          # ~30 % → we keep 15 % as val

    train_dataset, val_dataset_aug = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Validation should NOT use augmentation; rebuild with eval transforms
    val_base = datasets.ImageFolder(
        root=str(train_dir),
        transform=get_eval_transforms(),
    )
    _, val_dataset = random_split(
        val_base,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # same seed → same indices
    )

    # ── Test set ─────────────────────────────────────────────────────────────
    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=get_eval_transforms(),
    )

    class_names = full_train_dataset.classes   # sorted alphabetically by torchvision

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"[Preprocessing] Train: {len(train_dataset)} | "
          f"Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"[Preprocessing] Classes: {class_names}")

    return train_loader, val_loader, test_loader, class_names


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalization so images can be displayed with matplotlib.

    Parameters
    ----------
    tensor : (C, H, W) normalized tensor

    Returns
    -------
    tensor : (C, H, W) in [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
