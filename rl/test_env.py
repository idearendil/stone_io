"""Smoke test: 3 episodes × 200 steps, validates shapes and types."""
import sys
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from stone_env import StoneEnv

OBS_SIZE = 36
N_EPISODES = 3
N_STEPS = 200


def test_single_agent():
    env = StoneEnv(n_agents=1, num_bots=5)
    try:
        for ep in range(N_EPISODES):
            obs, info = env.reset()
            assert obs.shape == (OBS_SIZE,), f'ep {ep}: obs shape {obs.shape}'
            assert obs.dtype == np.float32,   f'ep {ep}: obs dtype {obs.dtype}'

            total_reward = 0.0
            deaths = 0
            for step in range(N_STEPS):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                assert obs.shape == (OBS_SIZE,), f'ep {ep} step {step}: bad obs shape'
                assert isinstance(reward,     float), f'ep {ep} step {step}: reward not float'
                assert isinstance(terminated, bool),  f'ep {ep} step {step}: terminated not bool'
                assert isinstance(truncated,  bool),  f'ep {ep} step {step}: truncated not bool'

                total_reward += reward
                if terminated:
                    deaths += 1

            print(f'  episode {ep}: total_reward={total_reward:.2f}  deaths={deaths}')
    finally:
        env.close()


def test_multi_agent():
    n = 3
    env = StoneEnv(n_agents=n, num_bots=2)
    try:
        obs_dict, info = env.reset()
        assert len(obs_dict) == n
        for k, v in obs_dict.items():
            assert v.shape == (OBS_SIZE,), f'multi-agent reset obs shape {v.shape}'

        actions = {f'agent_{i}': env.action_space.sample() for i in range(n)}
        obs_dict, rew_dict, term_dict, trunc_dict, info = env.step(actions)
        assert len(obs_dict) == n
        assert len(rew_dict) == n
        for k in rew_dict:
            assert isinstance(rew_dict[k], float)
            assert isinstance(term_dict[k], bool)
        print(f'  multi-agent step OK: rewards={list(rew_dict.values())}')
    finally:
        env.close()


if __name__ == '__main__':
    print('--- single-agent ---')
    test_single_agent()
    print('--- multi-agent ---')
    test_multi_agent()
    print('All tests passed.')
