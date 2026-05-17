/**
 * TrainedBot — runs a trained Actor policy entirely in plain JS.
 * Weights are loaded from a JSON file exported by export_bot.py.
 *
 * Interface matches RuleBasedBot:  update(deltaMs, state, engine)
 *
 * NOTE: The CNN forward pass is ~89M float ops → expect ~180ms/inference in JS.
 * This bot is intended for demo purposes; real-time play should use ONNX Runtime Web.
 */

const BASE_RADIUS  = 16;
const ZOOM_MIN     = 0.20;
const ZOOM_MAX     = 1.00;
const IMG_CHANNELS = 8;
const IMG_SIZE     = 32;
const VEC_SIZE     = 5;
const VP           = 200;

export class TrainedBot {
  constructor(stoneId, weightsJson) {
    this.stoneId = stoneId;
    this._cnnLayers       = weightsJson.cnn_layers;
    this._vecMlpLayers    = weightsJson.vec_mlp_layers;
    this._sharedLayers    = weightsJson.shared_mlp_layers;
    this._actorHeadLayers = weightsJson.actor_head_layers;
    this._lastAction = null;
    this._ACTION_REPEAT = 3;
    this._AR_counter = 0;
  }

  update(deltaMs, state, engine) {
    const stone = engine.stones.get(this.stoneId);
    if (!stone || !stone.alive) {
      this._lastAction = null;
      this._AR_counter = 0;
      return;
    }

    if (this._AR_counter > 0 && this._lastAction) {
      const [dx, dy, boost] = this._lastAction;
      engine.setInput(this.stoneId, VP / 2 + dx * 120, VP / 2 + dy * 120, VP, VP);
      if (boost) engine.boost(this.stoneId);
    } else {
      const { vec, img } = this._buildObs(stone, engine);
      const raw   = this._forward(vec, img);
      const dx    = Math.tanh(raw[0]);
      const dy    = Math.tanh(raw[1]);
      const boost = 1 / (1 + Math.exp(-raw[2])) > 0.5;
      this._lastAction = [dx, dy, boost];

      engine.setInput(this.stoneId, VP / 2 + dx * 120, VP / 2 + dy * 120, VP, VP);
      if (boost) engine.boost(this.stoneId);
    }

    this._AR_counter += 1;
    if (this._AR_counter >= this._ACTION_REPEAT) this._AR_counter = 0;
  }

  // ---------------------------------------------------------------------------
  // Observation builder — must match HeadlessServer.js buildObs exactly
  // ---------------------------------------------------------------------------

  _buildObs(stone, engine) {
    const { MAP_WIDTH, MAP_HEIGHT } = engine.config;
    const { x, y, vx, vy, radius } = stone;

    // vec[0-4]: vx, vy, log(radius), zone_index, invincibility_flag
    const vec = new Float32Array(VEC_SIZE);
    vec[0] = Math.log(Math.abs(vx) + 1) * Math.sign(vx);
    vec[1] = Math.log(Math.abs(vy) + 1) * Math.sign(vy);
    vec[2] = Math.log(radius + 1);
    vec[3] = Math.floor((MAP_HEIGHT - y) / MAP_HEIGHT * 5);
    vec[4] = engine._totalTime < stone.invincibleUntil ? 1.0 : 0.0;

    // Image
    const img = new Float32Array(IMG_CHANNELS * IMG_SIZE * IMG_SIZE);

    const zoom       = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, BASE_RADIUS * 4 / (radius + BASE_RADIUS * 3)));
    const pixelScale = zoom * IMG_SIZE / VP;

    const half = IMG_SIZE / 2;
    const inBounds = (px, py) => px >= 0 && px < IMG_SIZE && py >= 0 && py < IMG_SIZE;
    const setPixel = (ch, px, py, val) => { img[ch * IMG_SIZE * IMG_SIZE + py * IMG_SIZE + px] = val; };
    const toPx = (wx, wy) => ({
      px: Math.floor(half + (wx - x) * pixelScale),
      py: Math.floor(half + (wy - y) * pixelScale),
    });

    // Channels 0-1: fragments (3 small + 2 large)
    const allFrags = engine.getFragmentsNear(x, y)
      .sort((a, b) => (a.x - x) ** 2 + (a.y - y) ** 2 - ((b.x - x) ** 2 + (b.y - y) ** 2));
    const fragSlots = [
      ...allFrags.filter(f => f.radius <= 8).slice(0, 3),
      ...allFrags.filter(f => f.radius >= 9).slice(0, 2),
    ];
    for (const f of fragSlots) {
      const { px, py } = toPx(f.x, f.y);
      if (!inBounds(px, py)) continue;
      setPixel(0, px, py, Math.log(f.area + 1));
      setPixel(1, px, py, Math.log(Math.max(0, Math.hypot(f.x - x, f.y - y) - radius) + 1));
    }

    // Channels 2-5: all visible alive stones
    for (const s of engine.stones.values()) {
      if (s.id === this.stoneId || !s.alive) continue;
      const { px, py } = toPx(s.x, s.y);
      if (!inBounds(px, py)) continue;
      setPixel(2, px, py, Math.log(s.radius / radius));
      setPixel(3, px, py, Math.log(Math.abs(s.vx - vx) + 1) * Math.sign(s.vx - vx));
      setPixel(4, px, py, Math.log(Math.abs(s.vy - vy) + 1) * Math.sign(s.vy - vy));
      setPixel(5, px, py, Math.log(Math.max(0, Math.hypot(s.x - x, s.y - y) - s.radius - radius) + 1));
    }

    // Channel 6: gear distance (center pixel)
    for (const g of engine.gears) {
      const { px, py } = toPx(g.x, g.y);
      if (!inBounds(px, py)) continue;
      const dist = Math.max(0.01, Math.hypot(x - g.x, y - g.y) - radius - g.collisionRadius);
      setPixel(6, px, py, Math.log(dist + 1));
    }

    // Channel 7: danger zone (dense)
    const leftEnd    = Math.ceil(half  - x * pixelScale);
    const rightStart = Math.floor(half + (MAP_WIDTH  - x) * pixelScale);
    const topEnd     = Math.ceil(half  - y * pixelScale);
    const botStart   = Math.floor(half + (MAP_HEIGHT - y) * pixelScale);
    const ch7base    = 7 * IMG_SIZE * IMG_SIZE;
    for (let py = 0; py < IMG_SIZE; py++) {
      for (let px = 0; px < IMG_SIZE; px++) {
        if (px < leftEnd || px >= rightStart || py < topEnd || py >= botStart) {
          img[ch7base + py * IMG_SIZE + px] = 1.0;
        }
      }
    }
    for (const g of engine.gears) {
      const dangerPxR    = (radius + g.collisionRadius) * pixelScale;
      const { px: cx, py: cy } = toPx(g.x, g.y);
      const minPx = Math.max(0,           Math.floor(cx - dangerPxR));
      const maxPx = Math.min(IMG_SIZE - 1, Math.ceil(cx  + dangerPxR));
      const minPy = Math.max(0,           Math.floor(cy - dangerPxR));
      const maxPy = Math.min(IMG_SIZE - 1, Math.ceil(cy  + dangerPxR));
      for (let py = minPy; py <= maxPy; py++) {
        for (let px = minPx; px <= maxPx; px++) {
          if ((px - cx) ** 2 + (py - cy) ** 2 <= dangerPxR ** 2) {
            img[ch7base + py * IMG_SIZE + px] = 1.0;
          }
        }
      }
    }

    return { vec, img };
  }

  // ---------------------------------------------------------------------------
  // Two-branch forward pass: CNN(img) + MLP(vec) → shared MLP → actor head
  // ---------------------------------------------------------------------------

  _forward(vec, img) {
    // CNN branch
    let feat = { data: img, c: IMG_CHANNELS, h: IMG_SIZE, w: IMG_SIZE };
    for (const layer of this._cnnLayers) {
      switch (layer.type) {
        case 'conv2d':
          feat = this._conv2d(feat.data, feat.c, feat.h, feat.w,
                              layer.weight, layer.bias, layer.stride, layer.padding);
          break;
        case 'relu':
          feat = { data: this._relu(feat.data), c: feat.c, h: feat.h, w: feat.w };
          break;
        case 'flatten':
          feat = { data: feat.data, c: feat.c * feat.h * feat.w, h: 1, w: 1 };
          break;
      }
    }
    const cnnFeat = feat.data;  // Float32Array(1024)

    // Vec branch
    let vecFeat = vec;
    for (const layer of this._vecMlpLayers) {
      switch (layer.type) {
        case 'linear': vecFeat = this._linear(vecFeat, layer.weight, layer.bias); break;
        case 'relu':   vecFeat = this._relu(vecFeat);                             break;
      }
    }

    // Concat → (1088,)
    const merged = new Float32Array(cnnFeat.length + vecFeat.length);
    merged.set(cnnFeat, 0);
    merged.set(vecFeat, cnnFeat.length);

    // Shared MLP
    let shared = merged;
    for (const layer of this._sharedLayers) {
      switch (layer.type) {
        case 'linear':     shared = this._linear(shared, layer.weight, layer.bias);    break;
        case 'layer_norm': shared = this._layerNorm(shared, layer.weight, layer.bias); break;
        case 'relu':       shared = this._relu(shared);                                break;
      }
    }

    // Actor head
    let out = shared;
    for (const layer of this._actorHeadLayers) {
      switch (layer.type) {
        case 'linear': out = this._linear(out, layer.weight, layer.bias); break;
        case 'relu':   out = this._relu(out);                             break;
      }
    }

    return out;  // Float32Array[3]: raw_dx, raw_dy, raw_boost
  }

  // ---------------------------------------------------------------------------
  // Primitive ops
  // ---------------------------------------------------------------------------

  _conv2d(input, inC, inH, inW, weight, bias, stride, padding) {
    const outC = weight.length;
    const kH   = weight[0][0].length;
    const kW   = weight[0][0][0].length;
    const outH = Math.floor((inH + 2 * padding - kH) / stride + 1);
    const outW = Math.floor((inW + 2 * padding - kW) / stride + 1);
    const output = new Float32Array(outC * outH * outW);

    for (let oc = 0; oc < outC; oc++) {
      const wOc = weight[oc];
      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          let sum = bias[oc];
          for (let ic = 0; ic < inC; ic++) {
            const wIc   = wOc[ic];
            const icOff = ic * inH * inW;
            for (let kh = 0; kh < kH; kh++) {
              const ih = oh * stride - padding + kh;
              if (ih < 0 || ih >= inH) continue;
              const wKh  = wIc[kh];
              const ihOff = ih * inW;
              for (let kw = 0; kw < kW; kw++) {
                const iw = ow * stride - padding + kw;
                if (iw < 0 || iw >= inW) continue;
                sum += wKh[kw] * input[icOff + ihOff + iw];
              }
            }
          }
          output[oc * outH * outW + oh * outW + ow] = sum;
        }
      }
    }
    return { data: output, c: outC, h: outH, w: outW };
  }

  _linear(x, weight, bias) {
    const out = new Float32Array(bias.length);
    for (let i = 0; i < bias.length; i++) {
      let s = bias[i];
      const row = weight[i];
      for (let j = 0; j < x.length; j++) s += row[j] * x[j];
      out[i] = s;
    }
    return out;
  }

  _layerNorm(x, weight, bias, eps = 1e-5) {
    let mean = 0;
    for (let i = 0; i < x.length; i++) mean += x[i];
    mean /= x.length;
    let variance = 0;
    for (let i = 0; i < x.length; i++) variance += (x[i] - mean) ** 2;
    variance /= x.length;
    const std = Math.sqrt(variance + eps);
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = weight[i] * (x[i] - mean) / std + bias[i];
    return out;
  }

  _relu(x) {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
    return out;
  }
}
