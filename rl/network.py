import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli


def _ortho(layer, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 74, act_dim: int = 4):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim  # (act_dim-1) continuous [dx, dy, power] + 1 discrete [boost]

        self.shared_mlp = nn.Sequential(
            _ortho(nn.Linear(obs_dim, 256)),
            nn.LayerNorm(256),
            nn.ReLU(),
            _ortho(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        # Output layout: [mean_x, mean_y, mean_power, log_std_x, log_std_y, log_std_power, boost_logit]
        self.actor_head = nn.Sequential(
            _ortho(nn.Linear(256, 2 * act_dim - 1), gain=0.01),
        )
        self.critic_head = _ortho(nn.Linear(256, 1), gain=1.0)

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

        action layout: [x, y, power, boost]
          - x, y   : continuous, Normal(tanh(mean), std), clamped to [-1,1];
                     applied as unit-normalized direction outside the network
          - power  : continuous, Normal(sigmoid(mean), std), clamped to [0,1]
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

        n_cont = self.act_dim - 1  # 3 continuous dims: x, y, power
        # Means: raw[:, :n_cont]; log_stds: raw[:, n_cont:2*n_cont]; boost: raw[:, -1]
        dir_mean = torch.tanh(raw[:, :2])
        pow_mean = torch.sigmoid(raw[:, 2:3])
        move_mean = torch.cat([dir_mean, pow_mean], dim=-1)
        std = torch.exp(raw[:, n_cont:2 * n_cont].clamp(-4, 2))
        move_dist = Normal(move_mean, std)

        # Boost (discrete)
        boost_dist = Bernoulli(torch.sigmoid(raw[:, -1]))

        if action is None:
            action_move = move_dist.sample()
            action_move = torch.cat([
                action_move[:, :2].clamp(-1, 1),   # x, y
                action_move[:, 2:].clamp(0, 1),    # power
            ], dim=1)
            action_boost = boost_dist.sample()
            action = torch.cat([action_move, action_boost.unsqueeze(1)], dim=1)
        else:
            action_move = action[:, :-1]
            action_boost = action[:, -1]

        log_prob = move_dist.log_prob(action_move).sum(-1) + boost_dist.log_prob(action_boost)
        entropy = move_dist.entropy().sum(-1) + boost_dist.entropy()

        return action, log_prob, entropy, value
