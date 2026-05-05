from __future__ import annotations

import numpy as np
import gymnasium as gym

from .bridge import HeadlessBridge

OBS_SIZE = 86

# Auto-increment port so multiple envs in the same process don't collide
_next_port: list[int] = [7777]


class StoneEnv(gym.Env):
    """
    Gymnasium wrapper around the Stone.io GameEngine.

    Single-agent mode (n_agents=1):
        obs, info            = env.reset()
        obs, rew, term, trunc, info = env.step(action)   # action: np.ndarray shape (2,)

    Multi-agent mode (n_agents > 1):
        obs_dict, info       = env.reset()               # dict[str, np.ndarray]
        obs_dict, rew_dict, term_dict, trunc_dict, info = env.step(action_dict)
    """

    metadata = {'render_modes': []}

    def __init__(self, n_agents: int = 1, num_bots: int = 0, port: int | None = None):
        super().__init__()
        self.n_agents = n_agents
        self._num_bots = num_bots
        if port is None:
            self._port = _next_port[0]
            _next_port[0] += 1
        else:
            self._port = port
        self._bridge: HeadlessBridge | None = None

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bridge_(self) -> HeadlessBridge:
        if self._bridge is None:
            self._bridge = HeadlessBridge(self._port, self.n_agents, self._num_bots)
        return self._bridge

    @staticmethod
    def _to_list(action) -> list[float]:
        return action.tolist() if hasattr(action, 'tolist') else list(action)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        result = self._bridge_().post('/reset')
        obs_dict: dict = result['observations']
        if self.n_agents == 1:
            return np.array(obs_dict['agent_0'], dtype=np.float32), {}
        return {k: np.array(v, dtype=np.float32) for k, v in obs_dict.items()}, {}

    def step(self, action):
        if self.n_agents == 1:
            actions = {'agent_0': self._to_list(action)}
        else:
            actions = {k: self._to_list(v) for k, v in action.items()}

        result = self._bridge_().post('/step', {'actions': actions})
        obs_d  = result['observations']
        rew_d  = result['rewards']
        term_d = result['terminated']
        trunc_d = result['truncated']

        if self.n_agents == 1:
            return (
                np.array(obs_d['agent_0'], dtype=np.float32),
                float(rew_d['agent_0']),
                bool(term_d['agent_0']),
                bool(trunc_d['agent_0']),
                {},
            )

        return (
            {k: np.array(v, dtype=np.float32) for k, v in obs_d.items()},
            {k: float(v)                       for k, v in rew_d.items()},
            {k: bool(v)                        for k, v in term_d.items()},
            {k: bool(v)                        for k, v in trunc_d.items()},
            {},
        )

    def get_radii(self) -> list[float]:
        result = self._bridge_().get('/radii')
        return result.get('radii', [])

    def close(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
