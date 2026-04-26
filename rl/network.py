import torch
import torch.nn as nn


def _ortho(layer, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 42, act_dim: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.shared_mlp = nn.Sequential(
            _ortho(nn.Linear(obs_dim, 128)),
            nn.LayerNorm(128),
            nn.ReLU(),
            _ortho(nn.Linear(128, 128)),
            nn.ReLU(),
        )
        # Small gain on actor output encourages exploration early
        self.actor_head = nn.Sequential(
            _ortho(nn.Linear(128, act_dim), gain=0.01),
            nn.Tanh(),
        )
        self.critic_head = _ortho(nn.Linear(128, 1), gain=1.0)

        # State-independent learnable log-std (one per action dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    # ------------------------------------------------------------------
    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.shared_mlp(obs)

    def forward(self, obs: torch.Tensor):
        """Deterministic forward — returns (action_mean, value)."""
        f = self._features(obs)
        return self.actor_head(f), self.critic_head(f).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """
        Sample (or re-evaluate) an action from the stochastic policy.

        Returns:
            action      : (*, act_dim)  clamped to [-1, 1]
            log_prob    : (*,)          sum over action dims
            entropy     : (*,)          sum over action dims
            value       : (*,)
        """
        f = self._features(obs)
        mean = self.actor_head(f)
        value = self.critic_head(f).squeeze(-1)

        std = torch.exp(self.log_std.clamp(-4, 2)).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample().clamp(-1, 1)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value
