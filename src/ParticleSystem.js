/**
 * Ephemeral visual effects (sparks, bursts, flashes) — purely cosmetic,
 * completely separate from the engine's gameplay Fragment entities.
 *
 * All particle positions live in WORLD coordinates; draw() projects them through
 * the camera so effects track the world as it scrolls/zooms. Counts respect
 * prefers-reduced-motion (halved) and a hard global cap for the perf budget.
 */

const REDUCED =
  typeof window !== 'undefined' && window.matchMedia
    ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
    : false;
const DENSITY = REDUCED ? 0.5 : 1;
const MAX_PARTICLES = 200;

const TAU = Math.PI * 2;

export class ParticleSystem {
  constructor() {
    /** @type {Array<object>} unified pool of ring / dot / flash particles */
    this._items = [];
  }

  get count() {
    return this._items.length;
  }

  _push(p) {
    // Hard cap — drop the oldest particles to make room (perf budget)
    if (this._items.length >= MAX_PARTICLES) {
      this._items.splice(0, this._items.length - MAX_PARTICLES + 1);
    }
    this._items.push(p);
  }

  _scaled(n) {
    return Math.max(1, Math.round(n * DENSITY));
  }

  // ---------------------------------------------------------------------------
  // Effect spawners
  // ---------------------------------------------------------------------------

  /** Expanding ring + a handful of sparkle dots when a fragment is absorbed. */
  absorb(x, y, color = '#ffe9a8') {
    this._push({ kind: 'ring', x, y, color, t: 0, dur: 250, r0: 4, r1: 26 });
    const dots = this._scaled(6);
    for (let i = 0; i < dots; i++) {
      const a = (i / dots) * TAU + Math.random() * 0.5;
      const spd = 0.04 + Math.random() * 0.05; // world units / ms
      this._push({
        kind: 'dot', x, y,
        vx: Math.cos(a) * spd, vy: Math.sin(a) * spd,
        g: 0, drag: 0.9,
        color, t: 0, dur: 250,
        size: 1.5 + Math.random() * 1.5,
      });
    }
  }

  /** Outward burst of colored debris when a stone is destroyed. */
  death(x, y, color = '#cc5555') {
    const dots = this._scaled(20);
    for (let i = 0; i < dots; i++) {
      const a = Math.random() * TAU;
      const spd = 0.06 + Math.random() * 0.14;
      this._push({
        kind: 'dot', x, y,
        vx: Math.cos(a) * spd, vy: Math.sin(a) * spd,
        g: 0, drag: 0.94,
        color, t: 0, dur: 600,
        size: 2 + Math.random() * 2.5,
      });
    }
  }

  /** Brief white flash at a collision point. */
  collision(x, y) {
    this._push({ kind: 'flash', x, y, t: 0, dur: 150, r: 10 });
  }

  /**
   * A white comet tail behind a stone when it boosts. The tail points opposite
   * the stone's movement direction and fades out. It stays attached to the
   * stone (by id) and is re-anchored every frame in update() so it follows the
   * stone for its whole lifetime.
   */
  boostTail(stoneId, x, y, vx, vy, radius) {
    const sp = Math.hypot(vx, vy);
    let dx = -1, dy = 0;
    if (sp > 1e-3) { dx = -vx / sp; dy = -vy / sp; } // backward
    this._push({
      kind: 'streak',
      stoneId,
      x: x + dx * radius * 0.6,   // anchor head just behind the stone
      y: y + dy * radius * 0.6,
      dx, dy,
      len: radius * 4.0 + Math.min(sp, 80) * 1.3,
      width: Math.max(3, radius * 0.95),
      t: 0, dur: 360,
    });
  }

  /** Upward golden shimmer when crossing into a new zone. */
  zoneEntry(x, y) {
    const dots = this._scaled(8);
    for (let i = 0; i < dots; i++) {
      const spread = (Math.random() - 0.5) * 0.04;
      this._push({
        kind: 'dot',
        x: x + (Math.random() - 0.5) * 30,
        y, vx: spread, vy: -(0.05 + Math.random() * 0.06),
        g: 0.00012, drag: 0.99,
        color: '#E0C060', t: 0, dur: 800,
        size: 1.5 + Math.random() * 2,
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  update(deltaMs, stones = null) {
    const dt = deltaMs;
    const live = [];
    for (const p of this._items) {
      p.t += dt;
      if (p.t >= p.dur) continue;
      if (p.kind === 'dot') {
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        if (p.g) p.vy += p.g * dt;
        if (p.drag) {
          const f = Math.pow(p.drag, dt / 16);
          p.vx *= f;
          p.vy *= f;
        }
      } else if (p.kind === 'streak' && p.stoneId != null && stones) {
        // Follow the boosting stone: re-anchor behind it using its live
        // position + velocity (so length/width track the decaying boost too).
        const s = stones.find(st => st.id === p.stoneId);
        if (s && s.alive) {
          const sp = Math.hypot(s.vx, s.vy);
          if (sp > 1e-3) { p.dx = -s.vx / sp; p.dy = -s.vy / sp; }
          p.x = s.x + p.dx * s.radius * 0.6;
          p.y = s.y + p.dy * s.radius * 0.6;
          p.len = s.radius * 4.0 + Math.min(sp, 80) * 1.3;
          p.width = Math.max(3, s.radius * 0.95);
        }
      }
      live.push(p);
    }
    this._items = live;
  }

  draw(ctx, camera) {
    for (const p of this._items) {
      const life = p.t / p.dur;        // 0 → 1
      const fade = 1 - life;
      const z = camera.zoom;

      if (p.kind === 'ring') {
        if (!camera.isVisible(p.x, p.y, p.r1)) continue;
        const { sx, sy } = camera.worldToScreen(p.x, p.y);
        const r = (p.r0 + (p.r1 - p.r0) * life) * z;
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, TAU);
        ctx.strokeStyle = `rgba(255,233,168,${(fade * 0.9).toFixed(3)})`;
        ctx.lineWidth = 2;
        ctx.stroke();

      } else if (p.kind === 'flash') {
        if (!camera.isVisible(p.x, p.y, p.r)) continue;
        const { sx, sy } = camera.worldToScreen(p.x, p.y);
        ctx.beginPath();
        ctx.arc(sx, sy, p.r * z, 0, TAU);
        ctx.fillStyle = `rgba(255,255,255,${(0.6 * fade).toFixed(3)})`;
        ctx.fill();

      } else if (p.kind === 'streak') {
        if (!camera.isVisible(p.x, p.y, p.len + p.width)) continue;
        const { sx, sy } = camera.worldToScreen(p.x, p.y);
        const ex = sx + p.dx * p.len * z;
        const ey = sy + p.dy * p.len * z;
        const perpX = -p.dy, perpY = p.dx;      // perpendicular to the tail
        const hw = p.width * 0.5 * z * (1 - life * 0.5); // head widens → tapers to a point
        ctx.beginPath();
        ctx.moveTo(sx + perpX * hw, sy + perpY * hw);
        ctx.lineTo(sx - perpX * hw, sy - perpY * hw);
        ctx.lineTo(ex, ey);                     // tail tip
        ctx.closePath();
        const grad = ctx.createLinearGradient(sx, sy, ex, ey);
        grad.addColorStop(0, `rgba(255,255,255,${(0.85 * fade).toFixed(3)})`);
        grad.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = grad;
        ctx.fill();

      } else { // dot
        if (!camera.isVisible(p.x, p.y, p.size + 2)) continue;
        const { sx, sy } = camera.worldToScreen(p.x, p.y);
        ctx.globalAlpha = fade;
        ctx.beginPath();
        ctx.arc(sx, sy, p.size * z, 0, TAU);
        ctx.fillStyle = p.color;
        ctx.fill();
        ctx.globalAlpha = 1;
      }
    }
  }
}
