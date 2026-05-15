import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli


def _ortho(layer, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Actor(nn.Module):
    def __init__(self, obs_dim: int = 62, act_dim: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.shared_mlp = nn.Sequential(
            _ortho(nn.Linear(obs_dim, 256)),
            nn.LayerNorm(256),
            nn.ReLU(),
            _ortho(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            _ortho(nn.Linear(256, act_dim), gain=0.01),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim - 1))

    def forward(self, obs: torch.Tensor):
        return self.actor_head(self.shared_mlp(obs))

    def get_action_and_log_prob(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """
        Returns: action (*, act_dim), log_prob (*,), entropy (*,)
        action layout: [dx, dy, boost]
          - dx, dy: Normal(tanh(mean), std), clamped to [-1, 1]
          - boost:  Bernoulli(sigmoid(logit))
        """
        raw = self.actor_head(self.shared_mlp(obs))

        move_mean = torch.tanh(raw[:, :-1])
        std = torch.exp(self.log_std.clamp(-4, 2)).expand_as(move_mean)
        move_dist = Normal(move_mean, std)
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
        return action, log_prob, entropy


class Critic(nn.Module):
    def __init__(self, obs_dim: int = 62, n_quantiles: int = 51):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_quantiles = n_quantiles

        self.shared_mlp = nn.Sequential(
            _ortho(nn.Linear(obs_dim, 256)),
            nn.LayerNorm(256),
            nn.ReLU(),
            _ortho(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.critic_head = _ortho(nn.Linear(256, n_quantiles), gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.shared_mlp(obs))  # (*, n_quantiles)

    def mean_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs).mean(dim=-1)
