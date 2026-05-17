import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli

VEC_DIM      = 5
IMG_CHANNELS = 8
IMG_SIZE     = 32


def _ortho(layer, gain: float = 1.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Actor(nn.Module):
    def __init__(self, vec_dim: int = VEC_DIM, img_channels: int = IMG_CHANNELS, act_dim: int = 3):
        super().__init__()
        self.vec_dim      = vec_dim
        self.img_channels = img_channels
        self.act_dim      = act_dim

        self.cnn = nn.Sequential(
            _ortho(nn.Conv2d(img_channels, 16, 4, stride=2, padding=1)), nn.ReLU(),  # 32→16
            _ortho(nn.Conv2d(16, 16, 4, stride=2, padding=1)),           nn.ReLU(),  # 16→8
            _ortho(nn.Conv2d(16, 16, 4, stride=2, padding=1)),           nn.ReLU(),  # 8→4
            nn.Flatten(),                                                              # → 256
        )
        self.vec_mlp = nn.Sequential(
            _ortho(nn.Linear(vec_dim, 16)),
            nn.ReLU(),
        )
        self.shared_mlp = nn.Sequential(
            _ortho(nn.Linear(272, 256)),
            nn.LayerNorm(256),
            nn.ReLU(),
            _ortho(nn.Linear(256, 128)),
            nn.ReLU(),
        )
        self.actor_head = _ortho(nn.Linear(128, act_dim), gain=0.01)
        self.log_std    = nn.Parameter(torch.zeros(act_dim - 1))

    def forward(self, vec: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(img)                                   # (B, 256)
        vec_feat = self.vec_mlp(vec)                               # (B, 16)
        merged   = torch.cat([cnn_feat, vec_feat], dim=-1)         # (B, 272)
        return self.actor_head(self.shared_mlp(merged))

    def get_action_and_log_prob(
        self,
        vec:    torch.Tensor,
        img:    torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """
        Returns: action (*, act_dim), log_prob (*,), entropy (*,)
        action layout: [dx, dy, boost]
          - dx, dy: Normal(tanh(mean), std), clamped to [-1, 1]
          - boost:  Bernoulli(sigmoid(logit))
        """
        raw = self.forward(vec, img)

        move_mean  = torch.tanh(raw[:, :-1])
        std        = torch.exp(self.log_std.clamp(-4, 2)).expand_as(move_mean)
        move_dist  = Normal(move_mean, std)
        boost_dist = Bernoulli(torch.sigmoid(raw[:, -1]))

        if action is None:
            action_move  = move_dist.sample().clamp(-1, 1)
            action_boost = boost_dist.sample()
            action = torch.cat([action_move, action_boost.unsqueeze(1)], dim=1)
        else:
            action_move  = action[:, :-1]
            action_boost = action[:, -1]

        log_prob = move_dist.log_prob(action_move).sum(-1) + boost_dist.log_prob(action_boost)
        entropy  = move_dist.entropy().sum(-1) + boost_dist.entropy()
        return action, log_prob, entropy


class Critic(nn.Module):
    def __init__(self, vec_dim: int = VEC_DIM, img_channels: int = IMG_CHANNELS, n_quantiles: int = 51):
        super().__init__()
        self.vec_dim      = vec_dim
        self.img_channels = img_channels
        self.n_quantiles  = n_quantiles

        self.cnn = nn.Sequential(
            _ortho(nn.Conv2d(img_channels, 32, 4, stride=2, padding=1)), nn.ReLU(),
            _ortho(nn.Conv2d(32, 64, 4, stride=2, padding=1)),           nn.ReLU(),
            _ortho(nn.Conv2d(64, 64, 4, stride=2, padding=1)),           nn.ReLU(),
            _ortho(nn.Conv2d(64, 64, 4, stride=2, padding=1)),           nn.ReLU(),
            nn.Flatten(),
        )
        self.vec_mlp = nn.Sequential(
            _ortho(nn.Linear(vec_dim, 64)),
            nn.ReLU(),
        )
        self.shared_mlp = nn.Sequential(
            _ortho(nn.Linear(320, 256)),
            nn.LayerNorm(256),
            nn.ReLU(),
            _ortho(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.critic_head = _ortho(nn.Linear(256, n_quantiles), gain=1.0)

    def forward(self, vec: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(img)
        vec_feat = self.vec_mlp(vec)
        merged   = torch.cat([cnn_feat, vec_feat], dim=-1)
        return self.critic_head(self.shared_mlp(merged))  # (B, n_quantiles)

    def mean_value(self, vec: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        return self.forward(vec, img).mean(dim=-1)
