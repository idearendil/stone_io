"""
Export a trained Actor checkpoint to:
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
from network import Actor, VEC_DIM, IMG_CHANNELS, IMG_SIZE

ACT_DIM = 3


# ------------------------------------------------------------------
# ONNX export
# ------------------------------------------------------------------

def export_onnx(model: Actor, path: str) -> None:
    model.eval()
    dummy_vec = torch.zeros(1, VEC_DIM)
    dummy_img = torch.zeros(1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        (dummy_vec, dummy_img),
        path,
        input_names=['obs_vec', 'obs_img'],
        output_names=['action_raw'],
        dynamic_axes={
            'obs_vec': {0: 'batch_size'},
            'obs_img': {0: 'batch_size'},
        },
        opset_version=17,
    )
    print(f'ONNX exported → {path}')


# ------------------------------------------------------------------
# JSON export  (sequential layer description for JS TrainedBot)
# ------------------------------------------------------------------

def _t(tensor: torch.Tensor) -> list:
    return tensor.detach().tolist()


def export_json(model: Actor, path: str) -> None:
    model.eval()
    cnn = model.cnn
    vm  = model.vec_mlp
    sm  = model.shared_mlp
    ah  = model.actor_head

    payload = {
        'vec_dim':      VEC_DIM,
        'img_channels': IMG_CHANNELS,
        'img_size':     IMG_SIZE,
        'cnn_layers': [
            {'type': 'conv2d', 'weight': _t(cnn[0].weight), 'bias': _t(cnn[0].bias), 'stride': 2, 'padding': 1},
            {'type': 'relu'},
            {'type': 'conv2d', 'weight': _t(cnn[2].weight), 'bias': _t(cnn[2].bias), 'stride': 2, 'padding': 1},
            {'type': 'relu'},
            {'type': 'conv2d', 'weight': _t(cnn[4].weight), 'bias': _t(cnn[4].bias), 'stride': 2, 'padding': 1},
            {'type': 'relu'},
            {'type': 'flatten'},
        ],
        'vec_mlp_layers': [
            {'type': 'linear', 'weight': _t(vm[0].weight), 'bias': _t(vm[0].bias)},
            {'type': 'relu'},
        ],
        'shared_mlp_layers': [
            {'type': 'linear',     'weight': _t(sm[0].weight), 'bias': _t(sm[0].bias)},
            {'type': 'layer_norm', 'weight': _t(sm[1].weight), 'bias': _t(sm[1].bias)},
            {'type': 'relu'},
            {'type': 'linear',     'weight': _t(sm[3].weight), 'bias': _t(sm[3].bias)},
            {'type': 'relu'},
        ],
        'actor_head_layers': [
            {'type': 'linear', 'weight': _t(ah.weight), 'bias': _t(ah.bias)},
            # JS applies tanh to [:2] and sigmoid threshold to [2]
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

    model = Actor(vec_dim=VEC_DIM, img_channels=IMG_CHANNELS, act_dim=ACT_DIM)
    ckpt  = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['actor'])

    export_onnx(model, str(out / 'bot.onnx'))
    export_json(model, str(out / 'bot.json'))

    # Also copy to src/bots/ for the game to load
    src_bots = Path(__file__).parent.parent / 'src' / 'bots' / 'bot.json'
    export_json(model, str(src_bots))
