from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import ActorCritic

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
CLIP_EPSILON  = 0.2
ENTROPY_COEF  = 0.01
VALUE_COEF    = 0.5
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
N_EPOCHS      = 4
MINIBATCH     = 256
MAX_GRAD_NORM = 0.5
LR            = 3e-4


# ------------------------------------------------------------------
# Rollout buffer
# ------------------------------------------------------------------

class RolloutBuffer:
    """Stores a flat collection of (obs, act, log_prob, value, reward, done) tuples."""

    def __init__(self):
        self.obs:       list[np.ndarray] = []
        self.actions:   list[np.ndarray] = []
        self.log_probs: list[float]      = []
        self.values:    list[float]      = []
        self.rewards:   list[float]      = []
        self.dones:     list[bool]       = []

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        log_prob: float,
        value:    float,
        reward:   float,
        done:     bool,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_gae(
        self,
        last_values: np.ndarray,   # (N,) bootstrap values for the final obs
        gamma:       float = GAMMA,
        gae_lambda:  float = GAE_LAMBDA,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (advantages, returns) both shape (T,).
        last_values is a flat array of bootstrap values — one per trajectory
        that appears in this buffer. For a single flat trajectory it's a
        scalar wrapped in a 1-element array.
        """
        n = len(self.rewards)
        values    = np.array(self.values,    dtype=np.float32)
        rewards   = np.array(self.rewards,   dtype=np.float32)
        dones     = np.array(self.dones,     dtype=np.float32)

        # Treat last_values as a single scalar (last bootstrap value)
        next_val  = float(last_values)

        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                nv = next_val
            else:
                nv = values[t + 1]
            mask       = 1.0 - dones[t]
            delta      = rewards[t] + gamma * nv * mask - values[t]
            last_gae   = delta + gamma * gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns


# ------------------------------------------------------------------
# PPO agent
# ------------------------------------------------------------------

class PPOAgent:
    def __init__(
        self,
        obs_dim:     int   = 42,
        act_dim:     int   = 2,
        lr:          float = LR,
        total_steps: int   = 5_000_000,
    ):
        self.model = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

        # Linear LR decay — scheduler step is called once per PPO update
        # total_updates is an estimate; clip at 0 to avoid negative LR
        self._total_steps  = total_steps
        self._rollout_size = 1          # updated by train.py when known
        self._update_count = 0

        self._scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda u: max(0.0, 1.0 - u * self._rollout_size / total_steps),
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Single observation → (action, log_prob, value)."""
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, log_prob, _, value = self.model.get_action_and_value(t)
        return (
            action.squeeze(0).numpy(),
            log_prob.item(),
            value.item(),
        )

    @torch.no_grad()
    def act_batch(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched: (N, obs_dim) → actions (N, act_dim), log_probs (N,), values (N,)."""
        t = torch.as_tensor(obs, dtype=torch.float32)
        actions, log_probs, _, values = self.model.get_action_and_value(t)
        return actions.numpy(), log_probs.numpy(), values.numpy()

    # ------------------------------------------------------------------
    def update(
        self,
        obs:       np.ndarray,   # (T, obs_dim)
        actions:   np.ndarray,   # (T, act_dim)
        log_probs: np.ndarray,   # (T,)
        advantages: np.ndarray,  # (T,)
        returns:   np.ndarray,   # (T,)
    ) -> dict[str, float]:
        """Run N_EPOCHS of minibatch PPO updates. Returns loss metrics."""
        self._update_count += 1

        # Normalise advantages
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        t_obs  = torch.as_tensor(obs,       dtype=torch.float32)
        t_acts = torch.as_tensor(actions,   dtype=torch.float32)
        t_lp   = torch.as_tensor(log_probs, dtype=torch.float32)
        t_adv  = torch.as_tensor(adv,       dtype=torch.float32)
        t_ret  = torch.as_tensor(returns,   dtype=torch.float32)

        n = len(obs)
        pg_losses, v_losses, ent_vals = [], [], []

        for _ in range(N_EPOCHS):
            idx = np.random.permutation(n)
            for start in range(0, n, MINIBATCH):
                mb = idx[start : start + MINIBATCH]
                if len(mb) < 2:
                    continue

                _, new_lp, entropy, new_v = self.model.get_action_and_value(
                    t_obs[mb], t_acts[mb]
                )

                ratio = torch.exp(new_lp - t_lp[mb])
                pg1   = -t_adv[mb] * ratio
                pg2   = -t_adv[mb] * ratio.clamp(1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                pg_loss = torch.max(pg1, pg2).mean()

                v_loss  = nn.functional.mse_loss(new_v, t_ret[mb])
                ent_loss = -entropy.mean()

                loss = pg_loss + VALUE_COEF * v_loss + ENTROPY_COEF * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_vals.append(-ent_loss.item())

        self._scheduler.step()

        return {
            'policy_loss': float(np.mean(pg_losses)),
            'value_loss':  float(np.mean(v_losses)),
            'entropy':     float(np.mean(ent_vals)),
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            'model':        self.model.state_dict(),
            'optimizer':    self.optimizer.state_dict(),
            'scheduler':    self._scheduler.state_dict(),
            'update_count': self._update_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self._scheduler.load_state_dict(ckpt['scheduler'])
        self._update_count = ckpt.get('update_count', 0)
