import { Camera } from './Camera.js';
import { Minimap } from './Minimap.js';
import { DebugOverlay } from './DebugOverlay.js';

const ZONE_COLORS = ['#1A0F05', '#3D250F', '#7A4F20', '#B07A3A', '#D4A860'];
const GRAIN_SPACING = 18; // world-px between wood grain lines

const MINIMAP_W = 200;
const MINIMAP_H = 160;
const MINIMAP_MARGIN = 10;

export class Renderer {
  constructor(canvas, config) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.config = config;
    this.camera = new Camera(canvas.width, canvas.height);
    this.minimap = new Minimap(config);
    this._debugOverlay = new DebugOverlay(config);
    this._debugEnabled = false;
    this._time = 0;
    this._effects = []; // absorption pulse effects
  }

  toggleDebug() {
    this._debugEnabled = !this._debugEnabled;
  }

  resize(w, h) {
    this.canvas.width = w;
    this.canvas.height = h;
    this.camera.viewportW = w;
    this.camera.viewportH = h;
  }

  /** Main draw call — invoke once per rAF frame. timestamp is the rAF DOMHighResTimeStamp. */
  render(gameState, myStoneId, timestamp = performance.now()) {
    this._time = timestamp;
    const myStone = gameState.stones.find(s => s.id === myStoneId) || null;

    if (myStone) this.camera.update(myStone.x, myStone.y, myStone.radius);

    // Collect absorption effects from events
    if (gameState.events) {
      for (const ev of gameState.events) {
        if (ev.type === 'absorb') {
          this._effects.push({ x: ev.x, y: ev.y, startTime: timestamp, duration: 200 });
        }
      }
    }

    const { ctx, camera } = this;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    this._drawBackground(camera);
    this._drawGears(gameState.gears, camera);
    this._drawFragments(gameState.fragments, camera);
    this._drawStones(gameState.stones, camera, myStoneId);
    this._drawEffects(camera, timestamp);
    this._drawHUD(gameState, myStoneId);
    this.minimap.draw(ctx, gameState, myStoneId, camera, this.canvas.width, this.canvas.height);
    this._drawScoreboard(gameState, myStoneId);
    if (this._debugEnabled) {
      this._debugOverlay.draw(ctx, gameState, camera, timestamp);
    }
  }

  // ---------------------------------------------------------------------------
  // Background
  // ---------------------------------------------------------------------------

  _drawBackground(camera) {
    const ctx = this.ctx;
    const { MAP_WIDTH, MAP_HEIGHT, ZONES } = this.config;
    const zoneH = MAP_HEIGHT / ZONES.length;
    const worldTop    = camera.y - camera.viewportH * 0.5 / camera.zoom;
    const worldBottom = camera.y + camera.viewportH * 0.5 / camera.zoom;

    // Out-of-bounds base fill
    ctx.fillStyle = '#0a0808';
    ctx.fillRect(0, 0, camera.viewportW, camera.viewportH);

    for (let z = 0; z < ZONE_COLORS.length; z++) {
      const zTop    = z * zoneH;
      const zBottom = (z + 1) * zoneH;
      if (zBottom < worldTop || zTop > worldBottom) continue;

      const { sy: sTop    } = camera.worldToScreen(0, Math.max(zTop, worldTop));
      const { sy: sBottom } = camera.worldToScreen(0, Math.min(zBottom, worldBottom));

      // Zone fill
      ctx.fillStyle = ZONE_COLORS[z];
      ctx.fillRect(0, sTop, camera.viewportW, sBottom - sTop);

      // Wood grain — horizontal quadratic beziers spaced every GRAIN_SPACING world-px
      const firstY = Math.floor(Math.max(zTop, worldTop) / GRAIN_SPACING) * GRAIN_SPACING;
      for (let wy = firstY; wy < Math.min(zBottom, worldBottom); wy += GRAIN_SPACING) {
        const { sy } = camera.worldToScreen(0, wy);
        const curve  = Math.sin(wy * 0.031) * 5;
        const opacity = 0.025 + (Math.sin(wy * 0.11) * 0.5 + 0.5) * 0.035;
        ctx.strokeStyle = `rgba(255,200,120,${opacity.toFixed(3)})`;
        ctx.lineWidth = 0.6;
        ctx.beginPath();
        ctx.moveTo(0, sy);
        ctx.quadraticCurveTo(camera.viewportW * 0.5, sy + curve, camera.viewportW, sy - curve * 0.5);
        ctx.stroke();
      }

      // Zone boundary — thin golden line
      if (z < ZONE_COLORS.length - 1) {
        const { sy: bY } = camera.worldToScreen(0, zBottom);
        if (bY >= -1 && bY <= camera.viewportH + 1) {
          ctx.strokeStyle = '#C8A84B';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, bY);
          ctx.lineTo(camera.viewportW, bY);
          ctx.stroke();
        }
      }
    }

    // Left / right out-of-bounds bars (only visible when near map edge)
    const { sx: leftEdge  } = camera.worldToScreen(0, 0);
    const { sx: rightEdge } = camera.worldToScreen(MAP_WIDTH, 0);
    ctx.fillStyle = '#0a0808';
    if (leftEdge  > 0) ctx.fillRect(0, 0, leftEdge, camera.viewportH);
    if (rightEdge < camera.viewportW) ctx.fillRect(rightEdge, 0, camera.viewportW - rightEdge, camera.viewportH);
  }

  // ---------------------------------------------------------------------------
  // Gears
  // ---------------------------------------------------------------------------

  _drawGears(gears, camera) {
    const ctx = this.ctx;
    for (const gear of gears) {
      if (!camera.isVisible(gear.x, gear.y, gear.radius * 1.4)) continue;
      const { sx, sy } = camera.worldToScreen(gear.x, gear.y);
      const r = gear.radius * camera.zoom;
      const N = Math.max(8, Math.floor(gear.radius / 8));

      ctx.save();
      ctx.translate(sx, sy);
      ctx.rotate(gear.angle);

      // Danger halo at collision radius (r * 1.05)
      ctx.beginPath();
      ctx.arc(0, 0, r * 1.05, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,30,30,0.12)';
      ctx.fill();

      // Gear body + teeth
      this._buildGearPath(ctx, r, N);
      ctx.fillStyle = '#888780';
      ctx.fill();
      ctx.strokeStyle = '#555450';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Center bolt
      ctx.beginPath();
      ctx.arc(0, 0, r * 0.18, 0, Math.PI * 2);
      ctx.fillStyle = '#2a2825';
      ctx.fill();
      ctx.strokeStyle = '#555450';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.restore();
    }
  }

  /**
   * Builds the gear path in the current rotated/translated context.
   * N teeth as isoceles triangles pointing outward from a polygon body.
   */
  _buildGearPath(ctx, radius, N) {
    const period    = (Math.PI * 2) / N;
    const innerR    = radius * 0.80;
    const outerR    = radius * 1.25;  // tooth tip
    const toothHalf = period * 0.18;  // half-angle of tooth base

    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const valleyA = i * period;
      const toothA  = valleyA + period * 0.5;

      if (i === 0) ctx.moveTo(Math.cos(valleyA) * innerR, Math.sin(valleyA) * innerR);
      else          ctx.lineTo(Math.cos(valleyA) * innerR, Math.sin(valleyA) * innerR);

      // Left tooth base → tip → right tooth base
      ctx.lineTo(Math.cos(toothA - toothHalf) * radius, Math.sin(toothA - toothHalf) * radius);
      ctx.lineTo(Math.cos(toothA)              * outerR, Math.sin(toothA)              * outerR);
      ctx.lineTo(Math.cos(toothA + toothHalf) * radius, Math.sin(toothA + toothHalf) * radius);
    }
    ctx.closePath();
  }

  // ---------------------------------------------------------------------------
  // Fragments
  // ---------------------------------------------------------------------------

  _drawFragments(fragments, camera) {
    const ctx = this.ctx;
    for (const frag of fragments) {
      if (!camera.isVisible(frag.x, frag.y, frag.radius + 1)) continue;
      const { sx, sy } = camera.worldToScreen(frag.x, frag.y);
      const lifeFrac = frag.maxTtl > 0 ? frag.ttl / frag.maxTtl : 1;
      ctx.globalAlpha = lifeFrac * 0.7;
      ctx.beginPath();
      ctx.arc(sx, sy, frag.radius * camera.zoom, 0, Math.PI * 2);
      ctx.fillStyle = frag.color || '#c8a460';
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  // ---------------------------------------------------------------------------
  // Stones
  // ---------------------------------------------------------------------------

  _drawStones(stones, camera, myStoneId) {
    const ctx = this.ctx;
    for (const stone of stones) {
      if (!stone.alive) continue;
      if (!camera.isVisible(stone.x, stone.y, stone.radius)) continue;

      // Invincibility flash — skip draw every other 100ms while immune
      if (stone.invincible && Math.floor(this._time / 100) % 2 === 1) continue;

      const { sx, sy } = camera.worldToScreen(stone.x, stone.y);
      const r = stone.radius * camera.zoom;

      // Pulsing ownership ring (opacity 0.4 → 0, radius +18px, period 1.2 s)
      if (stone.id === myStoneId) {
        const t = (this._time % 1200) / 1200;
        ctx.beginPath();
        ctx.arc(sx, sy, r + t * 18, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(255,255,255,${(0.4 * (1 - t)).toFixed(3)})`;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Body
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = stone.color;
      ctx.fill();

      // Glossy highlight — small white circle, top-left, 35% opacity
      ctx.beginPath();
      ctx.arc(sx - r * 0.30, sy - r * 0.30, r * 0.45, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.35)';
      ctx.fill();

      // Nickname (skip very small stones)
      if (r > 12) {
        const fontSize = Math.max(10, Math.min(14, r * 0.65));
        ctx.font = `${fontSize}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.shadowColor = 'rgba(0,0,0,0.75)';
        ctx.shadowBlur = 3;
        ctx.fillStyle = '#ffffff';
        ctx.fillText(stone.nickname, sx, sy);
      }
    }

    // Reset shadow state so it doesn't bleed into later draw calls
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
  }

  // ---------------------------------------------------------------------------
  // Absorption effects
  // ---------------------------------------------------------------------------

  _drawEffects(camera, timestamp) {
    const ctx = this.ctx;
    this._effects = this._effects.filter(e => timestamp - e.startTime < e.duration);
    for (const e of this._effects) {
      if (!camera.isVisible(e.x, e.y, 30)) continue;
      const t = (timestamp - e.startTime) / e.duration;
      const { sx, sy } = camera.worldToScreen(e.x, e.y);
      ctx.beginPath();
      ctx.arc(sx, sy, (4 + t * 22) * camera.zoom, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,255,180,${(1 - t).toFixed(3)})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  // ---------------------------------------------------------------------------
  // HUD
  // ---------------------------------------------------------------------------

  _drawHUD(gameState, myStoneId) {
    const ctx = this.ctx;
    const myStone = gameState.stones.find(s => s.id === myStoneId);
    if (!myStone) return;

    const aliveStones = gameState.stones.filter(s => s.alive);
    let rank = '--';
    if (myStone.alive) {
      rank = [...aliveStones]
        .sort((a, b) => b.radius - a.radius)
        .findIndex(s => s.id === myStoneId) + 1;
    }

    ctx.save();
    ctx.font = 'bold 13px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.shadowColor = 'rgba(0,0,0,0.85)';
    ctx.shadowBlur = 4;
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    const speed = Math.hypot(myStone.vx, myStone.vy).toFixed(2);
    const frg_cnt = gameState.fragments.length;
    ctx.fillText(`r: ${myStone.radius.toFixed(1)}   spd: ${speed}   rank: ${rank} / ${aliveStones.length}   fragments: ${frg_cnt}`, 12, 12);
    ctx.restore();
  }

  // ---------------------------------------------------------------------------
  // Scoreboard
  // ---------------------------------------------------------------------------

  _drawScoreboard(gameState, myStoneId) {
    const ctx = this.ctx;
    const sorted = [...gameState.stones]
      .filter(s => s.alive)
      .sort((a, b) => b.radius - a.radius)
      .slice(0, 5);

    if (sorted.length === 0) return;

    const MARGIN = MINIMAP_MARGIN;
    const W = MINIMAP_W;
    const ITEM_H = 18;
    const PAD = 8;
    const x = this.canvas.width - W - MARGIN;
    const y = MARGIN + MINIMAP_H + 6;
    const panelH = PAD + sorted.length * ITEM_H + PAD;

    ctx.save();
    ctx.fillStyle = 'rgba(12,8,6,0.72)';
    ctx.fillRect(x, y, W, panelH);
    ctx.strokeStyle = 'rgba(200,168,75,0.3)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, W, panelH);

    ctx.textBaseline = 'middle';
    ctx.shadowColor = 'rgba(0,0,0,0.6)';
    ctx.shadowBlur = 2;

    for (let i = 0; i < sorted.length; i++) {
      const stone = sorted[i];
      const sy = y + PAD + i * ITEM_H + ITEM_H / 2;
      const isMe = stone.id === myStoneId;

      // Rank number
      ctx.fillStyle = isMe ? '#C8A84B' : '#555';
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`${i + 1}`, x + PAD, sy);

      // Color dot
      ctx.beginPath();
      ctx.arc(x + PAD + 14, sy, 4, 0, Math.PI * 2);
      ctx.fillStyle = stone.color;
      ctx.fill();

      // Nickname
      ctx.fillStyle = isMe ? '#e8dcc8' : '#999';
      ctx.font = isMe ? 'bold 11px monospace' : '11px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(stone.nickname.slice(0, 11), x + PAD + 24, sy);

      // Radius
      ctx.fillStyle = isMe ? '#C8A84B' : '#555';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(stone.radius.toFixed(1), x + W - PAD, sy);
    }

    ctx.restore();
  }
}
