"""
Evaluate a trained checkpoint against rule-based bots.

Usage:
    python eval.py checkpoints/final.pt [--n-episodes 20] [--ep-steps 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from stone_env import StoneEnv
from ppo import PPOAgent

OBS_DIM   = 42
N_BOTS    = 99     # rule-based opponents
BASE_PORT = 8100


def evaluate(checkpoint: str, n_episodes: int, ep_steps: int) -> dict:
    agent = PPOAgent(obs_dim=OBS_DIM)
    agent.load(checkpoint)
    agent.model.eval()
    print(f'Loaded checkpoint: {checkpoint}')

    env = StoneEnv(n_agents=1, num_bots=N_BOTS, port=BASE_PORT)

    survival_times: list[int]   = []
    final_radii:    list[float] = []
    total_rewards:  list[float] = []
    wins:           list[bool]  = []

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            survival      = 0
            last_radius   = 0.0
            alive_steps   = 0

            for step in range(ep_steps):
                action, _, _ = agent.act(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                # Get raw state to inspect radius and rank
                raw = env._bridge_().get('/state')
                stones = raw.get('stones', [])
                my_stone = next((s for s in stones if s['id'] == 1), None)

                if my_stone and my_stone['alive']:
                    alive_steps  += 1
                    last_radius   = my_stone['radius']
                    survival      = step + 1

                if terminated or truncated:
                    break

            # Win = largest stone at episode end
            raw   = env._bridge_().get('/state')
            stones = [s for s in raw.get('stones', []) if s['alive']]
            if stones:
                best_r  = max(s['radius'] for s in stones)
                my_stone = next((s for s in stones if s['id'] == 1), None)
                won = my_stone is not None and my_stone['radius'] >= best_r * 0.99
            else:
                won = False

            survival_times.append(survival)
            final_radii.append(last_radius)
            total_rewards.append(total_reward)
            wins.append(won)

            print(
                f'  ep {ep+1:3d}/{n_episodes}  '
                f'survival={survival:5d}  '
                f'radius={last_radius:6.1f}  '
                f'reward={total_reward:8.2f}  '
                f'won={won}'
            )
    finally:
        env.close()

    results = {
        'mean_survival':  float(np.mean(survival_times)),
        'mean_radius':    float(np.mean(final_radii)),
        'mean_reward':    float(np.mean(total_rewards)),
        'win_rate':       float(np.mean(wins)),
    }
    print()
    print('--- Results ---')
    for k, v in results.items():
        print(f'  {k:<20s}: {v:.4f}')
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint',   type=str)
    p.add_argument('--n-episodes', type=int, default=20)
    p.add_argument('--ep-steps',   type=int, default=2000)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args.checkpoint, args.n_episodes, args.ep_steps)
