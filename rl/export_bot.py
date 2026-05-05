"""
Export a trained ActorCritic checkpoint to:
  1. ONNX  → bot.onnx        (for Python/serving inference)
  2. JSON  → bot.json / src/bots/bot.json  (for in-browser TrainedBot.js)

Usage:
    python export_bot.py checkpoints/final.pt [--out-dir .]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from network import ActorCritic

OBS_DIM = 86
ACT_DIM = 3


# ------------------------------------------------------------------
# ONNX export
# ------------------------------------------------------------------

def export_onnx(model: ActorCritic, path: str) -> None:
    model.eval()
    dummy = torch.zeros(1, OBS_DIM)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=['obs'],
        output_names=['action_mean', 'value'],
        dynamic_axes={'obs': {0: 'batch_size'}},
        opset_version=17,
    )
    print(f'ONNX exported → {path}')


# ------------------------------------------------------------------
# JSON export  (sequential layer description for JS MLP)
# ------------------------------------------------------------------

def _t(tensor: torch.Tensor) -> list:
    return tensor.detach().tolist()


def export_json(model: ActorCritic, path: str) -> None:
    model.eval()
    sm = model.shared_mlp
    ah = model.actor_head

    payload = {
        'obs_dim': OBS_DIM,
        'act_dim': ACT_DIM,
        # Sequential layer list — maps 1-to-1 to TrainedBot._forward()
        'layers': [
            {'type': 'linear',     'weight': _t(sm[0].weight), 'bias': _t(sm[0].bias)},
            {'type': 'layer_norm', 'weight': _t(sm[1].weight), 'bias': _t(sm[1].bias)},
            {'type': 'relu'},
            {'type': 'linear',     'weight': _t(sm[3].weight), 'bias': _t(sm[3].bias)},
            {'type': 'relu'},
            {'type': 'linear',     'weight': _t(ah[0].weight), 'bias': _t(ah[0].bias)},
            # No activation — JS applies tanh to [:2] and sigmoid threshold to [2]
        ],
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    print(f'JSON exported → {path}  ({Path(path).stat().st_size // 1024} KB)')


# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint',  type=str)
    p.add_argument('--out-dir',   type=str, default='.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out  = Path(args.out_dir)

    model = ActorCritic(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    ckpt  = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])

    export_onnx(model, str(out / 'bot.onnx'))
    export_json(model, str(out / 'bot.json'))

    # Also copy to src/bots/ for the game to load
    src_bots = Path(__file__).parent.parent / 'src' / 'bots' / 'bot.json'
    export_json(model, str(src_bots))
