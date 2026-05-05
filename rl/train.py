"""
Self-play PPO training for Stone.io.

Usage:
    python train.py [options]

Each run spawns n_envs worker processes; each worker runs a Node.js
HeadlessServer on its own port and steps the game via HTTP.
"""
from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

# Allow importing sibling modules
sys.path.insert(0, str(Path(__file__).parent))
from stone_env import StoneEnv
from ppo import PPOAgent, RolloutBuffer, GAMMA, GAE_LAMBDA
from network import ActorCritic

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
OBS_DIM    = 86
ACT_DIM    = 3
BASE_PORT  = 8000
POOL_SIZE  = 5
POOL_EVERY = 5   # updates
CKPT_EVERY = 10  # updates


# ------------------------------------------------------------------
# Parallel environment workers
# ------------------------------------------------------------------

def _env_worker(worker_id: int, env_kwargs: dict, pipe: mp.connection.Connection) -> None:
    """Runs in a child process. Owns a StoneEnv + Node server."""
    env = StoneEnv(**env_kwargs)
    try:
        while True:
            cmd, payload = pipe.recv()
            if cmd == 'reset':
                obs, info = env.reset()
                pipe.send(obs)
            elif cmd == 'step':
                result = env.step(payload)
                pipe.send(result)
            elif cmd == 'radii':
                pipe.send(env.get_radii())
            elif cmd == 'close':
                env.close()
                break
    except Exception:
        pass
    finally:
        env.close()


class ParallelEnv:
    """Thin multiprocessing wrapper: n_envs workers, each with its own StoneEnv."""

    def __init__(self, n_envs: int, env_kwargs_list: list[dict]):
        self._n = n_envs
        self._parent_pipes: list[mp.connection.Connection] = []
        self._procs: list[mp.Process] = []

        for i, kwargs in enumerate(env_kwargs_list):
            parent, child = mp.Pipe(duplex=True)
            p = mp.Process(
                target=_env_worker,
                args=(i, kwargs, child),
                daemon=True,
            )
            p.start()
            self._parent_pipes.append(parent)
            self._procs.append(p)

    def reset(self) -> list[dict]:
        for pipe in self._parent_pipes:
            pipe.send(('reset', None))
        return [pipe.recv() for pipe in self._parent_pipes]

    def step(self, actions_list: list[dict]) -> list[tuple]:
        for pipe, acts in zip(self._parent_pipes, actions_list):
            pipe.send(('step', acts))
        return [pipe.recv() for pipe in self._parent_pipes]

    def get_radii(self) -> list[list[float]]:
        for pipe in self._parent_pipes:
            pipe.send(('radii', None))
        return [pipe.recv() for pipe in self._parent_pipes]

    def close(self) -> None:
        for pipe in self._parent_pipes:
            try:
                pipe.send(('close', None))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()


# ------------------------------------------------------------------
# Self-play pool helpers
# ------------------------------------------------------------------

def _snapshot(model: ActorCritic) -> ActorCritic:
    device = next(model.parameters()).device
    snap = ActorCritic(model.obs_dim, model.act_dim).to(device)
    snap.load_state_dict(copy.deepcopy(model.state_dict()))
    snap.eval()
    return snap


@torch.no_grad()
def _pool_act_batch(obs_batch: np.ndarray, pool_model: ActorCritic) -> np.ndarray:
    """Batch inference for all opponents sharing the same pool model.
    obs_batch: (N, obs_dim) → actions: (N, act_dim)
    """
    device = next(pool_model.parameters()).device
    t = torch.as_tensor(obs_batch, dtype=torch.float32).to(device)
    actions, *_ = pool_model.get_action_and_value(t)
    return actions.cpu().numpy()


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    n_self  = args.n_agents
    n_opp   = args.n_opponents
    n_total = n_self + n_opp
    n_envs  = args.n_envs

    env_kwargs_list = [
        dict(n_agents=n_total, num_bots=0, port=BASE_PORT + i)
        for i in range(n_envs)
    ]

    agent = PPOAgent(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        total_steps=args.total_steps,
    )
    # Inform scheduler about rollout size for LR decay
    agent._rollout_size = args.rollout_steps * n_envs * n_self

    pool: list[ActorCritic] = []

    # ---- rollout storage (steps × envs × agents, flattened per update) ----
    T, E, A = args.rollout_steps, n_envs, n_self
    buf_obs      = np.zeros((T, E, A, OBS_DIM), dtype=np.float32)
    buf_actions  = np.zeros((T, E, A, ACT_DIM), dtype=np.float32)
    buf_log_probs = np.zeros((T, E, A),          dtype=np.float32)
    buf_values   = np.zeros((T, E, A),           dtype=np.float32)
    buf_rewards  = np.zeros((T, E, A),           dtype=np.float32)
    buf_dones    = np.zeros((T, E, A),           dtype=np.float32)

    # per-agent episode stats for logging
    ep_rewards   = np.zeros((E, A), dtype=np.float64)
    ep_lengths   = np.zeros((E, A), dtype=np.int32)
    finished_rewards: list[float] = []
    finished_lengths: list[int]   = []

    wandb.init(
        entity='leetm2021-postech',
        project='stone_io',
        config=vars(args),
    )

    print('Launching environments...')
    vec_env = ParallelEnv(n_envs, env_kwargs_list)
    obs_dicts = vec_env.reset()
    print(f'Envs ready. n_self={n_self}  n_opp={n_opp}  n_envs={n_envs}')

    total_steps_done = 0
    update_count     = 0
    t_start          = time.time()

    while total_steps_done < args.total_steps:
        # ---- collect rollout ----
        agent.model.eval()

        # Assign each opponent a fixed pool model for this entire rollout
        opp_pool_indices: dict[tuple[int, int], int | None] = {
            (e, i): random.randrange(len(pool)) if pool else None
            for e in range(E) for i in range(A, n_total)
        }

        # Group opponents by pool model index for batched inference
        # {idx_or_None: [(e, i), ...]}
        opp_groups: dict[int | None, list[tuple[int, int]]] = {}
        for (e, i), idx in opp_pool_indices.items():
            opp_groups.setdefault(idx, []).append((e, i))

        for step in range(T):
            # Batch all self-agent obs across envs: (E*A, OBS_DIM)
            self_obs_batch = np.stack([
                obs_dicts[e][f'agent_{i}']
                for e in range(E) for i in range(A)
            ])
            all_actions, all_log_probs, all_values = agent.act_batch(self_obs_batch)
            # Reshape to (E, A, ...)
            all_actions   = all_actions.reshape(E, A, ACT_DIM)
            all_log_probs = all_log_probs.reshape(E, A)
            all_values    = all_values.reshape(E, A)

            # Store self-agent data
            for e in range(E):
                for i in range(A):
                    buf_obs[step, e, i]       = obs_dicts[e][f'agent_{i}']
                    buf_actions[step, e, i]   = all_actions[e, i]
                    buf_log_probs[step, e, i] = all_log_probs[e, i]
                    buf_values[step, e, i]    = all_values[e, i]

            # Compute opponent actions — one batch forward pass per unique pool model
            opp_actions: dict[tuple[int, int], np.ndarray] = {}
            for idx, group in opp_groups.items():
                obs_batch = np.stack([obs_dicts[e][f'agent_{i}'] for (e, i) in group])
                model = agent.model if idx is None else pool[idx]
                batch_acts = _pool_act_batch(obs_batch, model)
                for j, (e, i) in enumerate(group):
                    opp_actions[(e, i)] = batch_acts[j]

            # Build full action dicts (self + opponents)
            env_action_dicts = []
            for e in range(E):
                acts: dict = {}
                for i in range(A):
                    acts[f'agent_{i}'] = all_actions[e, i]
                for i in range(A, n_total):
                    acts[f'agent_{i}'] = opp_actions[(e, i)]
                env_action_dicts.append(acts)

            # Step all envs
            step_results = vec_env.step(env_action_dicts)
            new_obs_dicts = []
            for e, (obs_d, rew_d, term_d, trunc_d, _) in enumerate(step_results):
                new_obs_dicts.append(obs_d)
                for i in range(A):
                    key = f'agent_{i}'
                    r   = float(rew_d[key])
                    d   = bool(term_d[key])
                    buf_rewards[step, e, i] = r
                    buf_dones[step, e, i]   = float(d)
                    ep_rewards[e, i] += r
                    ep_lengths[e, i] += 1
                    if d:
                        finished_rewards.append(float(ep_rewards[e, i]))
                        finished_lengths.append(int(ep_lengths[e, i]))
                        ep_rewards[e, i] = 0.0
                        ep_lengths[e, i] = 0

            obs_dicts = new_obs_dicts
            total_steps_done += A * E

        # ---- bootstrap last values ----
        last_obs_batch = np.stack([
            obs_dicts[e][f'agent_{i}']
            for e in range(E) for i in range(A)
        ])
        _, _, last_vals = agent.act_batch(last_obs_batch)
        last_vals = last_vals.reshape(E, A)

        # ---- GAE per (env, agent) trajectory ----
        advantages = np.zeros((T, E, A), dtype=np.float32)
        returns    = np.zeros((T, E, A), dtype=np.float32)
        for e in range(E):
            for i in range(A):
                buf = RolloutBuffer()
                buf.obs       = list(buf_obs[:, e, i])
                buf.actions   = list(buf_actions[:, e, i])
                buf.log_probs = list(buf_log_probs[:, e, i])
                buf.values    = list(buf_values[:, e, i])
                buf.rewards   = list(buf_rewards[:, e, i])
                buf.dones     = list(buf_dones[:, e, i])
                adv, ret = buf.compute_gae(np.array([last_vals[e, i]]), GAMMA, GAE_LAMBDA)
                advantages[:, e, i] = adv
                returns[:, e, i]    = ret

        # ---- PPO update ----
        agent.model.train()
        loss_dict = agent.update(
            obs=buf_obs.reshape(-1, OBS_DIM),
            actions=buf_actions.reshape(-1, ACT_DIM),
            log_probs=buf_log_probs.reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
        )
        update_count += 1

        # ---- radius histogram (all alive stones across all envs) ----
        all_radii = []
        for radii_list in vec_env.get_radii():
            all_radii.extend(radii_list)
        clamped = [min(r, 150.0) for r in all_radii]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(clamped, bins=40, range=(0, 150), color='steelblue', edgecolor='black')
        ax.set_xlabel('Radius')
        ax.set_ylabel('Count')
        ax.set_xlim(0, 150)
        ax.set_title(f'Stone Radius Distribution (update {update_count})')
        hist_path = os.path.join(args.checkpoint_dir, '_radius_hist.png')
        fig.savefig(hist_path, dpi=80, bbox_inches='tight')
        plt.close(fig)

        # ---- wandb logging (every update) ----
        mean_rew_log = np.mean(finished_rewards[-200:]) if finished_rewards else 0.0
        mean_len_log = np.mean(finished_lengths[-200:]) if finished_lengths else 0.0
        wandb.log({
            'mean_reward':  mean_rew_log,
            'ep_len':       mean_len_log,
            'policy_loss':  loss_dict['policy_loss'],
            'value_loss':   loss_dict['value_loss'],
            'entropy':      loss_dict['entropy'],
            'lr':           agent.optimizer.param_groups[0]['lr'],
            'pool_size':    len(pool),
            'radius_hist':  wandb.Image(hist_path),
        }, step=total_steps_done)

        # ---- pool / checkpoint management ----
        if update_count % POOL_EVERY == 0:
            pool.append(_snapshot(agent.model))
            if len(pool) > POOL_SIZE:
                pool.pop(0)
            print(f'  [pool] updated, size={len(pool)}')

        if update_count % CKPT_EVERY == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'ckpt_{update_count:06d}.pt')
            agent.save(ckpt_path)
            print(f'  [checkpoint] saved → {ckpt_path}')

        # ---- logging ----
        if update_count % args.log_interval == 0:
            elapsed = time.time() - t_start
            mean_rew = np.mean(finished_rewards[-200:]) if finished_rewards else 0.0
            mean_len = np.mean(finished_lengths[-200:]) if finished_lengths else 0.0
            lr_now   = agent.optimizer.param_groups[0]['lr']
            print(
                f'update={update_count:5d}  '
                f'steps={total_steps_done:9d}  '
                f'mean_reward={mean_rew:7.3f}  '
                f'ep_len={mean_len:6.1f}  '
                f'policy_loss={loss_dict["policy_loss"]:7.4f}  '
                f'value_loss={loss_dict["value_loss"]:7.4f}  '
                f'entropy={loss_dict["entropy"]:6.4f}  '
                f'lr={lr_now:.2e}  '
                f'elapsed={elapsed:.0f}s'
            )

    # ---- final save ----
    final_path = os.path.join(args.checkpoint_dir, 'final.pt')
    agent.save(final_path)
    print(f'Training complete. Final checkpoint → {final_path}')
    vec_env.close()
    wandb.finish()


# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='PPO self-play training for Stone.io')
    p.add_argument('--n-agents',        type=int,   default=40)
    p.add_argument('--n-opponents',     type=int,   default=60)
    p.add_argument('--total-steps',     type=int,   default=80_000_000)
    p.add_argument('--rollout-steps',   type=int,   default=2500)
    p.add_argument('--n-envs',          type=int,   default=2)
    p.add_argument('--checkpoint-dir',  type=str,   default='checkpoints')
    p.add_argument('--log-interval',    type=int,   default=10)
    return p.parse_args()


if __name__ == '__main__':
    # Required on Windows to prevent recursive subprocess spawning
    mp.set_start_method('spawn', force=True)
    train(parse_args())
