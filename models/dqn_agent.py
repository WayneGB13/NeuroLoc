"""
models/dqn_agent.py
-------------------
Module: Deep Q-Network (DQN) Agent

Responsibilities:
  - Define the Q-network architecture (state → Q-values for 7 actions).
  - Implement an Experience Replay Buffer for stable off-policy learning.
  - Implement the DQNAgent class, which wraps:
      * Online Q-network  (updated every step)
      * Target Q-network  (frozen copy, synced every N steps)
      * ε-greedy action selection
      * Bellman loss computation and gradient update

Actions (7 total):
    0 → Move bounding box UP
    1 → Move bounding box DOWN
    2 → Move bounding box LEFT
    3 → Move bounding box RIGHT
    4 → Zoom IN  (shrink box)
    5 → Zoom OUT (expand box)
    6 → STOP    (declare tumor found)

State vector:
    [CNN features (2048-d)] + [bbox normalised coords (4-d)]  = 2052-d
"""

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ── Constants ────────────────────────────────────────────────────────────────
NUM_ACTIONS   = 7
FEATURE_DIM   = 2048
BBOX_DIM      = 4      # [x_min, y_min, x_max, y_max] normalised to [0,1]
STATE_DIM     = FEATURE_DIM + BBOX_DIM   # 2052

MEMORY_CAPACITY   = 10_000
BATCH_SIZE        = 64
GAMMA             = 0.99      # discount factor
LR                = 1e-4      # Adam learning rate
EPS_START         = 1.0
EPS_END           = 0.05
EPS_DECAY         = 0.995     # multiplicative decay per episode
TARGET_SYNC_EVERY = 200       # steps between target-network sync

# ── Transition named-tuple stored in replay buffer ───────────────────────────
Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done"),
)


# ════════════════════════════════════════════════════════════════════════════
# 1.  Q-Network
# ════════════════════════════════════════════════════════════════════════════
class QNetwork(nn.Module):
    """
    Fully-connected Q-network.

    Architecture
    ------------
    Input  (2052,)
        │
    Linear(2052 → 1024) + LayerNorm + ReLU + Dropout(0.3)
        │
    Linear(1024 → 512)  + LayerNorm + ReLU + Dropout(0.3)
        │
    Linear(512  → 256)  + ReLU
        │
    Linear(256  → 7)    ← one Q-value per action
    """

    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = NUM_ACTIONS):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, num_actions),
        )

        # Xavier initialisation for faster convergence
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, state_dim)

        Returns
        -------
        q_values : (B, num_actions)
        """
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════════
# 2.  Replay Buffer
# ════════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    """
    Fixed-size circular replay buffer.

    Stores Transition tuples (s, a, r, s', done) and returns random mini-batches
    for experience-replay training.
    """

    def __init__(self, capacity: int = MEMORY_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ════════════════════════════════════════════════════════════════════════════
# 3.  DQN Agent
# ════════════════════════════════════════════════════════════════════════════
class DQNAgent:
    """
    Deep Q-Network agent for bounding-box localisation.

    Parameters
    ----------
    state_dim   : int   Dimension of the state vector (default 2052).
    num_actions : int   Number of discrete actions     (default 7).
    device      : str   'cuda' or 'cpu'.
    lr          : float Adam learning rate.
    gamma       : float Discount factor for future rewards.
    eps_start   : float Initial ε for ε-greedy exploration.
    eps_end     : float Minimum ε.
    eps_decay   : float Multiplicative decay applied after each episode.
    memory_cap  : int   Replay buffer capacity.
    batch_size  : int   Mini-batch size for learning.
    target_sync : int   Steps between target-network hard updates.
    """

    def __init__(
        self,
        state_dim:   int   = STATE_DIM,
        num_actions: int   = NUM_ACTIONS,
        device:      str   = "cpu",
        lr:          float = LR,
        gamma:       float = GAMMA,
        eps_start:   float = EPS_START,
        eps_end:     float = EPS_END,
        eps_decay:   float = EPS_DECAY,
        memory_cap:  int   = MEMORY_CAPACITY,
        batch_size:  int   = BATCH_SIZE,
        target_sync: int   = TARGET_SYNC_EVERY,
    ):
        self.device      = torch.device(device)
        self.num_actions = num_actions
        self.gamma       = gamma
        self.epsilon     = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay
        self.batch_size  = batch_size
        self.target_sync = target_sync

        # ── Online + Target networks ──────────────────────────────────────────
        self.online_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # ── Optimiser ────────────────────────────────────────────────────────
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # ── Replay buffer ────────────────────────────────────────────────────
        self.memory = ReplayBuffer(capacity=memory_cap)

        # ── Step counter (for target-network sync) ───────────────────────────
        self.steps_done = 0

    # ────────────────────────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        """
        ε-greedy action selection.

        With probability ε → random action (exploration).
        Otherwise         → argmax Q(s, ·) (exploitation).

        Parameters
        ----------
        state : (state_dim,) numpy array

        Returns
        -------
        action : int in [0, num_actions)
        """
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, D)
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_t)            # (1, num_actions)
        self.online_net.train()
        return int(q_values.argmax(dim=1).item())

    # ────────────────────────────────────────────────────────────────────────
    def store(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        """Push one (s, a, r, s', done) tuple into the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    # ────────────────────────────────────────────────────────────────────────
    def learn(self) -> float | None:
        """
        Sample a mini-batch from the replay buffer and perform one
        gradient descent step using the Bellman MSE loss.

        Returns
        -------
        loss : float  (None if buffer is too small)
        """
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # ── Convert to tensors ──────────────────────────────────────────────
        states      = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions     = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones       = torch.FloatTensor(batch.done).to(self.device)

        # ── Current Q-values ────────────────────────────────────────────────
        # Q(s, a) for the actions that were actually taken
        current_q = self.online_net(states).gather(1, actions).squeeze(1)  # (B,)

        # ── Target Q-values (Double DQN style) ──────────────────────────────
        # Action selected by online net, value estimated by target net
        with torch.no_grad():
            best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            target_q_next = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target_q = rewards + self.gamma * target_q_next * (1.0 - dones)

        # ── Huber loss (smooth L1) is more robust than MSE to outlier rewards ─
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping avoids exploding gradients in early training
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ── Sync target network ─────────────────────────────────────────────
        self.steps_done += 1
        if self.steps_done % self.target_sync == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    # ────────────────────────────────────────────────────────────────────────
    def decay_epsilon(self):
        """Call once per episode to reduce exploration over time."""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    # ────────────────────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save(
            {
                "online_net":  self.online_net.state_dict(),
                "target_net":  self.target_net.state_dict(),
                "optimizer":   self.optimizer.state_dict(),
                "epsilon":     self.epsilon,
                "steps_done":  self.steps_done,
            },
            path,
        )
        print(f"[DQNAgent] Saved checkpoint → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon    = ckpt["epsilon"]
        self.steps_done = ckpt["steps_done"]
        print(f"[DQNAgent] Loaded checkpoint ← {path}")
def load(self, path):
    self.q_network.load_state_dict(torch.load(path))
    self.q_network.eval()