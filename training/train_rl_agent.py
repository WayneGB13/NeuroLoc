"""
training/train_rl_agent.py
--------------------------
Module: RL Agent Training

Responsibilities:
  - Load the pre-trained CNN (from train_classifier.py output).
  - Build a TumorEnv for each training sample.
  - Run DQN training episodes:
      * Reset environment → get initial state
      * Agent selects action → env returns (next_state, reward, done)
      * Push transition to replay buffer
      * Agent learns from mini-batch
  - Decay epsilon after each episode.
  - Log episode rewards and losses.
  - Save the best DQN checkpoint.
  - Plot RL training curves.

Usage
-----
    python training/train_rl_agent.py
"""

import random
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets

import sys
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from models.resnet50_model import build_cnn, NUM_CLASSES
from models.dqn_agent import DQNAgent
from environment.tumor_env import TumorEnv
from utils.preprocessing import get_eval_transforms

# ── Hyper-parameters ─────────────────────────────────────────────────────────
NUM_EPISODES  = 500         # total training episodes
WARMUP_STEPS  = 200         # random actions before learning starts
LOG_EVERY     = 20          # print every N episodes
SAVE_EVERY    = 100         # save checkpoint every N episodes

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATASET_DIR = BASE_DIR / "dataset"

BEST_DQN_PATH    = RESULTS_DIR / "best_dqn.pth"
RL_METRICS_PATH  = RESULTS_DIR / "rl_training_metrics.json"
RL_PLOT_PATH     = RESULTS_DIR / "rl_training_curves.png"


# ════════════════════════════════════════════════════════════════════════════
def load_cnn(device: str) -> torch.nn.Module:
    """Load the pre-trained CNN from disk."""
    cnn_path = RESULTS_DIR / "best_cnn.pth"
    if not cnn_path.exists():
        raise FileNotFoundError(
            f"CNN checkpoint not found at '{cnn_path}'.\n"
            "Run training/train_classifier.py first."
        )
    cnn = build_cnn(num_classes=NUM_CLASSES, freeze_backbone=False, pretrained=False, device=device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn.eval()
    # Freeze CNN — it acts only as a feature extractor during RL training
    for p in cnn.parameters():
        p.requires_grad = False
    print(f"[RL Train] CNN loaded from {cnn_path} and frozen.")
    return cnn


# ════════════════════════════════════════════════════════════════════════════
def sample_training_images(dataset_root: str = "dataset", n: int = NUM_EPISODES):
    """
    Build a flat list of (image_tensor, label) tuples by sampling from
    the Training folder.  This list drives one episode per sample.
    """
    train_dir = Path(dataset_root) / "Training"
    dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=get_eval_transforms(),
    )
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    # Repeat indices if num_episodes > dataset size
    while len(indices) < n:
        extra = list(range(len(dataset)))
        random.shuffle(extra)
        indices += extra
    indices = indices[:n]

    samples = [(dataset[i][0], dataset[i][1]) for i in indices]
    print(f"[RL Train] Prepared {len(samples)} episode samples.")
    return samples


# ════════════════════════════════════════════════════════════════════════════
def plot_rl_curves(history: dict, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["episode_rewards"], alpha=0.6, label="Episode Reward")
    # Rolling average
    rewards = np.array(history["episode_rewards"])
    if len(rewards) >= 10:
        roll = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        axes[0].plot(range(9, len(rewards)), roll, label="10-ep avg", linewidth=2)
    axes[0].set_title("Episode Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].legend()
    axes[0].grid(True)

    valid_losses = [(i, l) for i, l in enumerate(history["losses"]) if l is not None]
    if valid_losses:
        xs, ys = zip(*valid_losses)
        axes[1].plot(xs, ys, alpha=0.5, label="Loss")
        axes[1].set_title("DQN Loss")
        axes[1].set_xlabel("Step")
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[RL Train] Saved RL curves → {save_path}")


# ════════════════════════════════════════════════════════════════════════════
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RL Train] Using device: {device}")

    cnn     = load_cnn(device)
    agent   = DQNAgent(device=device)
    samples = sample_training_images(dataset_root=str(DATASET_DIR), n=NUM_EPISODES)

    history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "epsilons":        [],
        "losses":          [],
    }

    best_avg_reward = -float("inf")
    global_step     = 0

    for ep, (image, label) in enumerate(samples, start=1):
        env   = TumorEnv(cnn_model=cnn, image=image, true_label=label, device=device)
        state = env.reset()

        ep_reward = 0.0
        ep_steps  = 0
        done      = False

        while not done:
            # ε-greedy action (pure random during warm-up)
            if global_step < WARMUP_STEPS:
                action = random.randrange(agent.num_actions)
            else:
                action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            agent.store(state, action, reward, next_state, done)
            loss = agent.learn()
            history["losses"].append(loss)

            state      = next_state
            ep_reward += reward
            ep_steps  += 1
            global_step += 1

        # End of episode
        agent.decay_epsilon()

        history["episode_rewards"].append(ep_reward)
        history["episode_lengths"].append(ep_steps)
        history["epsilons"].append(agent.epsilon)

        if ep % LOG_EVERY == 0:
            avg_r = np.mean(history["episode_rewards"][-LOG_EVERY:])
            print(
                f"  Ep {ep:>4}/{NUM_EPISODES} | "
                f"Avg Reward {avg_r:+.2f} | "
                f"Steps {ep_steps:>3} | "
                f"ε {agent.epsilon:.3f} | "
                f"Buffer {len(agent.memory):>5}"
            )
            # Save best checkpoint
            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                agent.save(str(BEST_DQN_PATH))

        if ep % SAVE_EVERY == 0:
            ckpt_path = RESULTS_DIR / f"dqn_ep{ep}.pth"
            agent.save(str(ckpt_path))

    # ── Final save ────────────────────────────────────────────────────────────
    agent.save(str(RESULTS_DIR / "dqn_final.pth"))

    with open(RL_METRICS_PATH, "w") as f:
        json.dump(history, f, indent=2)

    plot_rl_curves(history, RL_PLOT_PATH)
    print(f"\n[RL Train] Done. Best avg reward: {best_avg_reward:.2f}")


if __name__ == "__main__":
    main()
