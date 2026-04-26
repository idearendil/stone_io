/**
 * TrainedBot — runs a trained ActorCritic policy entirely in plain JS.
 * Weights are loaded from a JSON file exported by export_bot.py.
 *
 * Interface matches RuleBasedBot:  update(deltaMs, state, engine)
 */
export class TrainedBot {
  constructor(stoneId, weightsJson) {
    this.stoneId = stoneId;
    this._layers = weightsJson.layers;
  }

  update(deltaMs, state, engine) {
    const stone = engine.stones.get(this.stoneId);
    if (!stone || !stone.alive) return;

    const obs    = this._buildObs(stone, engine);
    const action = this._forward(obs);          // [dx, dy] ∈ [-1, 1]²
    const VP     = 200;
    engine.setInput(
      this.stoneId,
      VP / 2 + action[0] * 120,
      VP / 2 + action[1] * 120,
      VP, VP,
    );
  }

  // ---------------------------------------------------------------------------
  // Observation builder — must match HeadlessServer.js buildObs exactly
  // Layout: [0-5] self  [6-20] 5 frags×3  [21-32] 4 stones×3  [33-41] 3 gears×3
  // ---------------------------------------------------------------------------

  _buildObs(stone, engine) {
    const obs = new Float32Array(42);
    const { MAP_WIDTH, MAP_HEIGHT, MAX_SPEED } = engine.config;
    const { x, y, vx, vy, radius } = stone;

    // [0-5] self
    obs[0] = x / MAP_WIDTH;
    obs[1] = y / MAP_HEIGHT;
    obs[2] = vx / MAX_SPEED;
    obs[3] = vy / MAX_SPEED;
    obs[4] = radius / 1000;
    obs[5] = 1.0;

    // [6-20] 5 nearest fragments (uses spatial grid — same O(1) as RuleBasedBot)
    const nearFrags = engine.getFragmentsNear(x, y)
      .sort((a, b) => (a.x - x) ** 2 + (a.y - y) ** 2 - ((b.x - x) ** 2 + (b.y - y) ** 2));
    for (let i = 0; i < 5 && i < nearFrags.length; i++) {
      const f    = nearFrags[i];
      const base = 6 + i * 3;
      obs[base]     = (f.x - x) / 1200;
      obs[base + 1] = (f.y - y) / 1200;
      obs[base + 2] = f.area   / 200;
    }

    // [21-32] 4 nearest other alive stones
    const others = [...engine.stones.values()]
      .filter(s => s.id !== this.stoneId && s.alive)
      .sort((a, b) => Math.hypot(a.x - x, a.y - y) - Math.hypot(b.x - x, b.y - y));
    for (let i = 0; i < 4 && i < others.length; i++) {
      const s    = others[i];
      const base = 21 + i * 3;
      obs[base]     = (s.x - x)      / 1200;
      obs[base + 1] = (s.y - y)      / 1200;
      obs[base + 2] = s.radius / radius;
    }

    // [33-41] 3 nearest gears
    const gears = [...engine.gears]
      .sort((a, b) => Math.hypot(a.x - x, a.y - y) - Math.hypot(b.x - x, b.y - y));
    for (let i = 0; i < 3 && i < gears.length; i++) {
      const g    = gears[i];
      const base = 33 + i * 3;
      obs[base]     = (g.x - x)           / 1200;
      obs[base + 1] = (g.y - y)           / 1200;
      obs[base + 2] = g.collisionRadius   / 1200;
    }

    return obs;
  }

  // ---------------------------------------------------------------------------
  // Minimal MLP forward pass — sequential layer list from export_bot.py
  // ---------------------------------------------------------------------------

  _forward(obs) {
    let x = obs;
    for (const layer of this._layers) {
      switch (layer.type) {
        case 'linear':     x = this._linear(x, layer.weight, layer.bias); break;
        case 'layer_norm': x = this._layerNorm(x, layer.weight, layer.bias); break;
        case 'relu':       x = this._relu(x);  break;
        case 'tanh':       x = this._tanh(x);  break;
      }
    }
    return x;  // Float32Array [dx, dy]
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

  _tanh(x) {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = Math.tanh(x[i]);
    return out;
  }
}
