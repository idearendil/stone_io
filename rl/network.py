import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli


def _ortho(layer, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 62, act_dim: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim  # act_dim-1 continuous (movement) + 1 discrete (boost)

        self.shared_mlp = nn.Sequential(
            _ortho(nn.Linear(obs_dim, 128)),
            nn.LayerNorm(128),
            nn.ReLU(),
            _ortho(nn.Linear(128, 128)),
            nn.ReLU(),
        )
        # Raw linear output: first (act_dim-1) → tanh+Normal, last → sigmoid+Bernoulli
        self.actor_head = nn.Sequential(
            _ortho(nn.Linear(128, act_dim), gain=0.01),
        )
        self.critic_head = _ortho(nn.Linear(128, 1), gain=1.0)

        # Learnable log-std for continuous movement dims only
        self.log_std = nn.Parameter(torch.zeros(act_dim - 1))

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.shared_mlp(obs)

    def forward(self, obs: torch.Tensor):
        """Deterministic forward — returns (raw_actor_out, value)."""
        f = self._features(obs)
        return self.actor_head(f), self.critic_head(f).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """
        Sample (or re-evaluate) an action from the stochastic policy.

        action layout: [dx, dy, boost]
          - dx, dy : continuous, Normal(tanh(mean), std), clamped to [-1,1]
          - boost  : discrete 0/1, Bernoulli(sigmoid(logit))

        Returns:
            action   : (*, act_dim)
            log_prob : (*,)   sum over all dims
            entropy  : (*,)   sum over all dims
            value    : (*,)
        """
        f = self._features(obs)
        raw = self.actor_head(f)
        value = self.critic_head(f).squeeze(-1)

        # Movement (continuous)
        move_mean = torch.tanh(raw[:, :-1])
        std = torch.exp(self.log_std.clamp(-4, 2)).expand_as(move_mean)
        move_dist = Normal(move_mean, std)

        # Boost (discrete)
        boost_dist = Bernoulli(torch.sigmoid(raw[:, -1]))

        if action is None:
            action_move = move_dist.sample().clamp(-1, 1)
            action_boost = boost_dist.sample()
            action = torch.cat([action_move, action_boost.unsqueeze(1)], dim=1)
        else:
            action_move = action[:, :-1]
            action_boost = action[:, -1]

        log_prob = move_dist.log_prob(action_move).sum(-1) + boost_dist.log_prob(action_boost)
        entropy = move_dist.entropy().sum(-1) + boost_dist.entropy()

        return action, log_prob, entropy, value
