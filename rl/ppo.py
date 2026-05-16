from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import Actor, Critic

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
CLIP_EPSILON      = 0.2
ENTROPY_COEF      = 0.015
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
N_EPOCHS          = 4
MINIBATCH         = 256
MAX_GRAD_NORM     = 0.5
LR                = 3e-4
MAX_CRITIC_EPOCHS = 1000
CRITIC_PATIENCE   = 2
N_QUANTILES       = 51


# ------------------------------------------------------------------
# Quantile Huber loss (QR-DQN style)
# ------------------------------------------------------------------

def quantile_huber_loss(
    quantiles: torch.Tensor,
    targets:   torch.Tensor,
    kappa:     float = 1.0,
) -> torch.Tensor:
    """QR-DQN quantile Huber loss with distributional targets.
    quantiles: (B, N)  predicted quantile values
    targets:   (B, N)  distributional target returns (one per scenario)
    """
    N   = quantiles.shape[-1]
    tau = (torch.arange(N, device=quantiles.device, dtype=torch.float32) + 0.5) / N

    # u[b, i, j] = targets[b, j] - quantiles[b, i]
    u = targets.unsqueeze(1) - quantiles.unsqueeze(2)              # (B, N, N)
    huber = torch.where(u.abs() <= kappa,
                        0.5 * u.pow(2),
                        kappa * (u.abs() - 0.5 * kappa))
    qloss = (tau.view(1, N, 1) - (u < 0).float()).abs() * huber / kappa
    return qloss.sum(dim=2).mean()                                 # sum over targets, mean over pred+batch


def compute_distributional_returns(
    rewards:         np.ndarray,   # (T,)
    dones:           np.ndarray,   # (T,)
    quantile_values: np.ndarray,   # (T, N)
    last_quantiles:  np.ndarray,   # (N,)
    gamma:           float = GAMMA,
    gae_lambda:      float = GAE_LAMBDA,
) -> np.ndarray:                   # (T, N)
    """
    For each of N return scenarios, sample one quantile value per timestep
    and run a GAE computation. Returns distributional returns (T, N).
    """
    T, N = quantile_values.shape

    # For each scenario k and each timestep t, sample one quantile index
    idx      = np.random.randint(0, N, size=(T, N))                    # (T, N)
    sampled  = quantile_values[np.arange(T)[:, None], idx]             # (T, N)

    last_idx     = np.random.randint(0, N, size=(N,))
    last_sampled = last_quantiles[last_idx]                             # (N,)

    adv      = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N,     dtype=np.float32)
    for t in reversed(range(T)):
        nv       = last_sampled if t == T - 1 else sampled[t + 1]      # (N,)
        mask     = 1.0 - dones[t]
        delta    = rewards[t] + gamma * nv * mask - sampled[t]         # (N,)
        last_gae = delta + gamma * gae_lambda * mask * last_gae        # (N,)
        adv[t]   = last_gae

    return adv + sampled                                               # (T, N)


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
        last_values: np.ndarray,
        gamma:       float = GAMMA,
        gae_lambda:  float = GAE_LAMBDA,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.rewards)
        values  = np.array(self.values,  dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones   = np.array(self.dones,   dtype=np.float32)

        next_val   = float(last_values)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0
        for t in reversed(range(n)):
            nv       = next_val if t == n - 1 else values[t + 1]
            mask     = 1.0 - dones[t]
            delta    = rewards[t] + gamma * nv * mask - values[t]
            last_gae = delta + gamma * gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns


# ------------------------------------------------------------------
# PPO agent
# ------------------------------------------------------------------

class PPOAgent:
    def __init__(
        self,
        obs_dim:     int   = 62,
        act_dim:     int   = 3,
        lr:          float = LR,
        total_steps: int   = 50_000_000,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor  = Actor(obs_dim, act_dim).to(self.device)
        self.critic = Critic(obs_dim, n_quantiles=N_QUANTILES).to(self.device)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)
        self._update_count = 0
        self._rollout_size = 1  # updated by train.py when known
        self._actor_scheduler = optim.lr_scheduler.LambdaLR(
            self.actor_optimizer,
            lr_lambda=lambda u: max(0.0, 1.0 - u * self._rollout_size / total_steps),
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action, log_prob, _ = self.actor.get_action_and_log_prob(t)
        value = self.critic.mean_value(t)
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    @torch.no_grad()
    def act_batch(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        actions, log_probs, _ = self.actor.get_action_and_log_prob(t)
        values = self.critic(t)                    # (B, N_QUANTILES)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    # ------------------------------------------------------------------
    def update(
        self,
        obs:        np.ndarray,
        actions:    np.ndarray,
        log_probs:  np.ndarray,
        advantages: np.ndarray,
        returns:    np.ndarray,
    ) -> dict[str, float]:
        self._update_count += 1

        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        t_obs  = torch.as_tensor(obs,       dtype=torch.float32).to(self.device)
        t_acts = torch.as_tensor(actions,   dtype=torch.float32).to(self.device)
        t_lp   = torch.as_tensor(log_probs, dtype=torch.float32).to(self.device)
        t_adv  = torch.as_tensor(adv,       dtype=torch.float32).to(self.device)
        t_ret  = torch.as_tensor(returns,   dtype=torch.float32).to(self.device)

        n = len(obs)
        pg_losses, ent_vals = [], []

        # ---- Actor update (PPO clipped objective) ----
        self.actor.train()
        for _ in range(N_EPOCHS):
            idx = np.random.permutation(n)
            for start in range(0, n, MINIBATCH):
                mb = idx[start : start + MINIBATCH]
                if len(mb) < 2:
                    continue

                _, new_lp, entropy = self.actor.get_action_and_log_prob(t_obs[mb], t_acts[mb])
                ratio   = torch.exp(new_lp - t_lp[mb])
                pg1     = -t_adv[mb] * ratio
                pg2     = -t_adv[mb] * ratio.clamp(1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                pg_loss = torch.max(pg1, pg2).mean()
                ent_loss = -entropy.mean()

                loss = pg_loss + ENTROPY_COEF * ent_loss
                self.actor_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
                self.actor_optimizer.step()

                pg_losses.append(pg_loss.item())
                ent_vals.append(-ent_loss.item())

        self._actor_scheduler.step()

        # ---- Critic update (train/val 4:1 split + early stopping) ----
        self.critic.train()
        perm    = np.random.permutation(n)
        n_train = int(n * 0.8)
        tr_idx  = torch.as_tensor(perm[:n_train], dtype=torch.long)
        va_idx  = torch.as_tensor(perm[n_train:],  dtype=torch.long)

        t_obs_tr, t_ret_tr = t_obs[tr_idx], t_ret[tr_idx]
        t_obs_va, t_ret_va = t_obs[va_idx], t_ret[va_idx]

        best_val = float('inf')
        no_improve = 0
        v_losses = []

        for _ in range(MAX_CRITIC_EPOCHS):
            ep_v = []
            idx = np.random.permutation(n_train)
            for start in range(0, n_train, MINIBATCH):
                mb = idx[start : start + MINIBATCH]
                if len(mb) < 2:
                    continue
                v_loss = quantile_huber_loss(self.critic(t_obs_tr[mb]), t_ret_tr[mb])
                self.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
                self.critic_optimizer.step()
                ep_v.append(v_loss.item())
            v_losses.extend(ep_v)

            with torch.no_grad():
                val_loss = quantile_huber_loss(self.critic(t_obs_va), t_ret_va).item()

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= CRITIC_PATIENCE:
                    break

        return {
            'policy_loss': float(np.mean(pg_losses)),
            'value_loss':  float(np.mean(v_losses)),
            'entropy':     float(np.mean(ent_vals)),
        }

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            'actor':            self.actor.state_dict(),
            'critic':           self.critic.state_dict(),
            'actor_optimizer':  self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_scheduler':  self._actor_scheduler.state_dict(),
            'update_count':     self._update_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        self._actor_scheduler.load_state_dict(ckpt['actor_scheduler'])
        self._update_count = ckpt.get('update_count', 0)
