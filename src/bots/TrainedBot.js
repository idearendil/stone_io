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

    const obs = this._buildObs(stone, engine);
    const raw = this._forward(obs);   // raw linear output: [dx_raw, dy_raw, boost_raw]
    const dx    = Math.tanh(raw[0]);
    const dy    = Math.tanh(raw[1]);
    const boost = 1 / (1 + Math.exp(-raw[2])) > 0.5;

    const VP = 200;
    engine.setInput(
      this.stoneId,
      VP / 2 + dx * 120,
      VP / 2 + dy * 120,
      VP, VP,
    );
    if (boost) engine.boost(this.stoneId);
  }

  // ---------------------------------------------------------------------------
  // Observation builder — must match HeadlessServer.js buildObs exactly
  // [0-3] wall log-dists  [4-7] self  [8-27] 5 frags×4
  // [28-51] 4 stones×6   [52-60] 3 gears×3
  // ---------------------------------------------------------------------------

  _buildObs(stone, engine) {
    const obs = new Float32Array(62);
    const { MAP_WIDTH, MAP_HEIGHT } = engine.config;
    const { x, y, vx, vy, radius } = stone;

    // [0-3] log wall distances (capped at 300)
    obs[0] = Math.log(Math.max(0, Math.min(x - radius * 0.5, 300)) + 1);
    obs[1] = Math.log(Math.max(0, Math.min(MAP_WIDTH - x - radius * 0.5, 300)) + 1);
    obs[2] = Math.log(Math.max(0, Math.min(y - radius * 0.5, 300)) + 1);
    obs[3] = Math.log(Math.max(0, Math.min(MAP_HEIGHT - y - radius * 0.5, 300)) + 1);

    // [4-7] self
    obs[4] = Math.log(Math.abs(vx) + 1) * Math.sign(vx);
    obs[5] = Math.log(Math.abs(vy) + 1) * Math.sign(vy);
    obs[6] = Math.log(Math.abs(radius) + 1);
    obs[7] = Math.floor((MAP_HEIGHT - y) / MAP_HEIGHT * 5);

    // [8-27] 5 nearest fragments (dx, dy, area, dist)
    const nearFrags = engine.getFragmentsNear(x, y)
      .sort((a, b) => (a.x - x) ** 2 + (a.y - y) ** 2 - ((b.x - x) ** 2 + (b.y - y) ** 2));
    for (let i = 0; i < 5 && i < nearFrags.length; i++) {
      const f    = nearFrags[i];
      const base = 8 + i * 4;
      obs[base]     = Math.log(Math.abs(f.x - x) + 1) * Math.sign(f.x - x);
      obs[base + 1] = Math.log(Math.abs(f.y - y) + 1) * Math.sign(f.y - y);
      obs[base + 2] = Math.log(f.area + 1);
      obs[base + 3] = Math.log(Math.max(0, Math.hypot(f.x - x, f.y - y) - radius) + 1);
    }

    // [28-51] 4 nearest other alive stones (dx, dy, radius_ratio, dvx, dvy, dist)
    const others = [...engine.stones.values()]
      .filter(s => s.id !== this.stoneId && s.alive)
      .sort((a, b) => Math.hypot(a.x - x, a.y - y) - Math.hypot(b.x - x, b.y - y));
    for (let i = 0; i < 4 && i < others.length; i++) {
      const s    = others[i];
      const base = 28 + i * 6;
      obs[base]     = Math.log(Math.abs(s.x - x) + 1) * Math.sign(s.x - x);
      obs[base + 1] = Math.log(Math.abs(s.y - y) + 1) * Math.sign(s.y - y);
      obs[base + 2] = Math.log(s.radius / radius);
      obs[base + 3] = Math.log(Math.abs(s.vx - vx) + 1) * Math.sign(s.vx - vx);
      obs[base + 4] = Math.log(Math.abs(s.vy - vy) + 1) * Math.sign(s.vy - vy);
      obs[base + 5] = Math.log(Math.max(0, Math.hypot(s.x - x, s.y - y) - s.radius - radius) + 1);
    }

    // [52-60] 3 nearest gears
    const gears = [...engine.gears]
      .sort((a, b) => Math.hypot(a.x - x, a.y - y) - Math.hypot(b.x - x, b.y - y));
    for (let i = 0; i < 3 && i < gears.length; i++) {
      const g    = gears[i];
      const base = 52 + i * 3;
      obs[base]     = Math.log(Math.abs(g.x - x) + 1) * Math.sign(g.x - x);
      obs[base + 1] = Math.log(Math.abs(g.y - y) + 1) * Math.sign(g.y - y);
      obs[base + 2] = Math.log(Math.max(0, Math.hypot(x - g.x, y - g.y) - radius - g.collisionRadius) + 1);
    }

    // [61] spawn invincibility flag
    obs[61] = engine._totalTime < stone.invincibleUntil ? 1.0 : 0.0;

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
