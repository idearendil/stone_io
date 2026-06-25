import { Camera } from './Camera.js';
import { Minimap } from './Minimap.js';
import { DebugOverlay } from './DebugOverlay.js';
import { ParticleSystem } from './ParticleSystem.js';

const TAU = Math.PI * 2;

// Per-zone wood aesthetic: base lacquer colour + grain line styling.
// Zone 0 (top) = darkest ebony, zone 4 (bottom) = light maple.
const ZONE_STYLE = [
  { base: '#1A0F05', grainSpacing: 12, grainColor: '#241507', grainOpacity: 0.6  },
  { base: '#3D250F', grainSpacing: 15, grainColor: '#4E3218', grainOpacity: 0.5  },
  { base: '#7A4F20', grainSpacing: 18, grainColor: '#8F6030', grainOpacity: 0.45 },
  { base: '#B07A3A', grainSpacing: 20, grainColor: '#C08A4A', grainOpacity: 0.4  },
  { base: '#D4A860', grainSpacing: 22, grainColor: '#E0B870', grainOpacity: 0.35 },
];
// Width (px) of the pre-rendered wood texture per zone. The full map is huge
// (8000×16000), so each zone is rendered once at this fixed resolution and
// stretched to fit on screen via drawImage — premium look, one-time cost.
const WOOD_TEX_W = 1600;

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

    this.particles = new ParticleSystem();
    this.soundEngine = null;
    this._playerZone = null;        // last zone the player occupied (zone-cross detection)
    this._zoneCanvases = null;      // pre-rendered wood textures, one per zone
    this._zoneTexH = null;          // zone height the textures were built for
  }

  setSoundEngine(soundEngine) {
    this.soundEngine = soundEngine;
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
    const dt = gameState._deltaMS ?? 16;
    const myStone = gameState.stones.find(s => s.id === myStoneId) || null;

    if (myStone) this.camera.update(myStone.x, myStone.y, myStone.radius);

    const { ctx, camera } = this;

    // Engine events → cosmetic particles + procedural sound (on-screen only)
    this._processEvents(gameState.events, camera, gameState.stones, myStoneId);
    this._checkZoneCross(myStone);
    this.particles.update(dt, gameState.stones);

    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    this._drawBackground(camera);
    this._drawGears(gameState.gears, camera);
    this._drawFragments(gameState.fragments, camera);
    this._drawStones(gameState.stones, camera, myStoneId);
    this.particles.draw(ctx, camera);
    this._drawHUD(gameState, myStoneId);
    this.minimap.draw(ctx, gameState, myStoneId, camera, this.canvas.width, this.canvas.height);
    this._drawScoreboard(gameState, myStoneId);
    if (this._debugEnabled) {
      this._debugOverlay.draw(ctx, gameState, camera, timestamp);
    }
  }

  // ---------------------------------------------------------------------------
  // Events → particles + sound
  // ---------------------------------------------------------------------------

  _processEvents(events, camera, stones, myStoneId) {
    if (!events) return;
    const se = this.soundEngine;
    for (const ev of events) {
      // Boost comet tail — look up the stone for its position + velocity
      if (ev.type === 'boost') {
        const s = stones && stones.find(st => st.id === ev.stoneId);
        if (s && s.alive && camera.isVisible(s.x, s.y, s.radius * 4)) {
          this.particles.boostTail(s.id, s.x, s.y, s.vx, s.vy, s.radius);
        }
        continue;
      }

      // Only react to effects on (or near) the visible screen — keeps the
      // particle budget and sound spam confined to what the player can see.
      const near = ev.x == null || camera.isVisible(ev.x, ev.y, 40);
      if (!near) continue;

      if (ev.type === 'absorb') {
        this.particles.absorb(ev.x, ev.y, ev.color || '#ffe9a8');
        if (se) se.absorb();
      } else if (ev.type === 'death') {
        this.particles.death(ev.x, ev.y, ev.color || '#cc5555');
        if (se) (ev.stoneId === myStoneId ? se.playerDeath() : se.death());
      } else if (ev.type === 'collision') {
        this.particles.collision(ev.x, ev.y);
        if (se) se.collision();
      }
    }
  }

  _checkZoneCross(myStone) {
    if (!myStone || !myStone.alive) { this._playerZone = null; return; }
    const { MAP_HEIGHT, ZONES } = this.config;
    const zoneH = MAP_HEIGHT / ZONES.length;
    const z = Math.min(ZONES.length - 1, Math.max(0, Math.floor(myStone.y / zoneH)));
    if (this._playerZone !== null && z !== this._playerZone) {
      this.particles.zoneEntry(myStone.x, myStone.y);
      if (this.soundEngine) this.soundEngine.zoneCross();
    }
    this._playerZone = z;
  }

  // ---------------------------------------------------------------------------
  // Colour helpers
  // ---------------------------------------------------------------------------

  _parseHex(hex) {
    let h = hex.replace('#', '');
    if (h.length === 3) h = h.split('').map(c => c + c).join('');
    const n = parseInt(h, 16);
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
  }

  _darken(hex, f) {
    const { r, g, b } = this._parseHex(hex);
    return `rgb(${Math.round(r * f)},${Math.round(g * f)},${Math.round(b * f)})`;
  }

  _lighten(hex, amt) {
    const { r, g, b } = this._parseHex(hex);
    const L = (c) => Math.round(c + (255 - c) * amt);
    return `rgb(${L(r)},${L(g)},${L(b)})`;
  }

  _hexToRgba(hex, a) {
    const { r, g, b } = this._parseHex(hex);
    return `rgba(${r},${g},${b},${a})`;
  }

  // ---------------------------------------------------------------------------
  // Background
  // ---------------------------------------------------------------------------

  /**
   * Pre-render one wood texture per zone onto an offscreen canvas. Each is a
   * full-map-width strip of lacquered wood with bezier grain lines, built once
   * and reused — only rebuilt if the zone height changes (e.g. config edit).
   */
  _buildZoneTextures(zoneH) {
    const { MAP_WIDTH } = this.config;
    const scale = WOOD_TEX_W / MAP_WIDTH;
    const texW = WOOD_TEX_W;
    const texH = Math.max(1, Math.round(zoneH * scale));

    this._zoneCanvases = ZONE_STYLE.map((style) => {
      const c = document.createElement('canvas');
      c.width = texW;
      c.height = texH;
      const g = c.getContext('2d');

      // Base lacquer fill
      g.fillStyle = style.base;
      g.fillRect(0, 0, texW, texH);

      // Broad "cathedral" figure — a few wide, soft, darker sweeps that give the
      // wood depth and a sense of solid timber beneath the surface.
      g.lineWidth = 2.5;
      const figures = 6;
      for (let k = 0; k < figures; k++) {
        const baseY = (k + 0.5) / figures * texH + (Math.random() * 2 - 1) * texH * 0.04;
        const amp   = texH * (0.02 + Math.random() * 0.03);
        g.strokeStyle = this._hexToRgba(style.grainColor, (style.grainOpacity * 0.45).toFixed(3));
        g.beginPath();
        g.moveTo(0, baseY);
        g.bezierCurveTo(texW * 0.3, baseY - amp, texW * 0.6, baseY + amp, texW, baseY - amp * 0.4);
        g.stroke();
      }

      // Wood grain — horizontal bezier curves with two slightly-offset control
      // points for a natural, hand-finished look.
      const spacing = Math.max(2, style.grainSpacing * scale);
      g.lineWidth = 0.8;
      for (let y = -spacing; y < texH + spacing; y += spacing) {
        const yy   = y + (Math.random() * 2 - 1) * spacing * 0.3;
        const off1 = (Math.random() * 2 - 1) * 3;
        const off2 = (Math.random() * 2 - 1) * 3;
        const op   = style.grainOpacity * (0.5 + Math.random() * 0.5);
        g.strokeStyle = this._hexToRgba(style.grainColor, op.toFixed(3));
        g.beginPath();
        g.moveTo(0, yy);
        g.bezierCurveTo(
          texW * 0.33, yy + off1,
          texW * 0.66, yy + off2,
          texW,        yy + (off1 + off2) * 0.3,
        );
        g.stroke();
      }

      // Fine pores — faint speckle that breaks up the flat fill at close zoom.
      const pores = Math.floor((texW * texH) / 1400);
      for (let i = 0; i < pores; i++) {
        g.fillStyle = this._hexToRgba(style.grainColor, (0.06 + Math.random() * 0.1).toFixed(3));
        g.fillRect(Math.random() * texW, Math.random() * texH, 1, 1);
      }

      // Lacquer sheen — a soft diagonal light-to-shadow wash for a glossy finish.
      const sheen = g.createLinearGradient(0, 0, texW, texH);
      sheen.addColorStop(0,   'rgba(255,240,210,0.06)');
      sheen.addColorStop(0.5, 'rgba(255,240,210,0)');
      sheen.addColorStop(1,   'rgba(0,0,0,0.06)');
      g.fillStyle = sheen;
      g.fillRect(0, 0, texW, texH);

      return c;
    });
    this._zoneTexH = zoneH;
  }

  _drawBackground(camera) {
    const ctx = this.ctx;
    const { MAP_WIDTH, MAP_HEIGHT, ZONES } = this.config;
    const zoneH = MAP_HEIGHT / ZONES.length;

    if (!this._zoneCanvases || this._zoneTexH !== zoneH) this._buildZoneTextures(zoneH);

    const worldTop    = camera.y - camera.viewportH * 0.5 / camera.zoom;
    const worldBottom = camera.y + camera.viewportH * 0.5 / camera.zoom;

    // Out-of-bounds base fill
    ctx.fillStyle = '#0a0808';
    ctx.fillRect(0, 0, camera.viewportW, camera.viewportH);

    // Horizontal screen span of the playfield, clamped to the viewport
    const sxLeft  = camera.worldToScreen(0, 0).sx;
    const sxRight = camera.worldToScreen(MAP_WIDTH, 0).sx;
    const fullW   = sxRight - sxLeft;
    const drawX   = Math.max(0, sxLeft);
    const drawW   = Math.min(camera.viewportW, sxRight) - drawX;
    if (fullW <= 0 || drawW <= 0) return;

    const uLeft  = (drawX - sxLeft) / fullW;
    const uRight = (drawX + drawW - sxLeft) / fullW;

    for (let z = 0; z < ZONE_STYLE.length; z++) {
      const zTop    = z * zoneH;
      const zBottom = (z + 1) * zoneH;
      if (zBottom < worldTop || zTop > worldBottom) continue;

      const sTop    = camera.worldToScreen(0, zTop).sy;
      const sBottom = camera.worldToScreen(0, zBottom).sy;
      const dstTop    = Math.max(sTop, -2);
      const dstBottom = Math.min(sBottom, camera.viewportH + 2);
      if (dstBottom <= dstTop) continue;

      const tex = this._zoneCanvases[z];
      const vTop    = (dstTop - sTop)    / (sBottom - sTop);
      const vBottom = (dstBottom - sTop) / (sBottom - sTop);

      ctx.drawImage(
        tex,
        uLeft * tex.width, vTop * tex.height,
        Math.max(1, (uRight - uLeft) * tex.width), Math.max(1, (vBottom - vTop) * tex.height),
        drawX, dstTop, drawW, dstBottom - dstTop,
      );
    }

    // Golden zone boundaries with a soft glow (screen space, cheap)
    ctx.save();
    ctx.strokeStyle = '#C8A84B';
    ctx.shadowColor = 'rgba(200,168,75,0.9)';
    ctx.shadowBlur = 4;
    ctx.lineWidth = 2;
    for (let z = 1; z < ZONE_STYLE.length; z++) {
      const by = camera.worldToScreen(0, z * zoneH).sy;
      if (by < -2 || by > camera.viewportH + 2) continue;
      ctx.beginPath();
      ctx.moveTo(drawX, by);
      ctx.lineTo(drawX + drawW, by);
      ctx.stroke();
    }
    ctx.restore();

    // Subtle vignette over the board — darkens the corners for a focused,
    // premium look (entities are drawn afterwards, so they stay bright).
    const cx = camera.viewportW / 2, cy = camera.viewportH / 2;
    const vig = ctx.createRadialGradient(
      cx, cy, Math.min(camera.viewportW, camera.viewportH) * 0.35,
      cx, cy, Math.max(camera.viewportW, camera.viewportH) * 0.72,
    );
    vig.addColorStop(0, 'rgba(0,0,0,0)');
    vig.addColorStop(1, 'rgba(0,0,0,0.3)');
    ctx.fillStyle = vig;
    ctx.fillRect(0, 0, camera.viewportW, camera.viewportH);
  }

  // ---------------------------------------------------------------------------
  // Gears
  // ---------------------------------------------------------------------------

  _drawGears(gears, camera) {
    const ctx = this.ctx;
    // Slow throb on the danger halo so gears read as "live" hazards
    const pulse = 0.12 + 0.06 * (0.5 + 0.5 * Math.sin(this._time * 0.004));

    for (const gear of gears) {
      if (!camera.isVisible(gear.x, gear.y, gear.radius * 1.4)) continue;
      const { sx, sy } = camera.worldToScreen(gear.x, gear.y);
      const r = gear.radius * camera.zoom;
      const N = Math.max(8, Math.floor(gear.radius / 8));

      ctx.save();
      ctx.translate(sx, sy);

      // Pulsing red danger halo at the collision radius (drawn un-rotated)
      const halo = ctx.createRadialGradient(0, 0, r * 0.6, 0, 0, r * 1.12);
      halo.addColorStop(0, 'rgba(200,30,20,0)');
      halo.addColorStop(1, `rgba(220,40,28,${pulse.toFixed(3)})`);
      ctx.beginPath();
      ctx.arc(0, 0, r * 1.12, 0, TAU);
      ctx.fillStyle = halo;
      ctx.fill();

      ctx.rotate(gear.angle);

      // Buzzsaw blade body — metallic radial gradient, sharp raked teeth
      this._buildGearPath(ctx, r, N);
      const steel = ctx.createRadialGradient(-r * 0.35, -r * 0.35, r * 0.1, 0, 0, r * 1.28);
      steel.addColorStop(0,    '#d2d5da');
      steel.addColorStop(0.45, '#9296a0');
      steel.addColorStop(0.8,  '#5c606a');
      steel.addColorStop(1,    '#3a3d44');
      ctx.fillStyle = steel;
      ctx.fill();
      ctx.lineJoin = 'miter';
      ctx.miterLimit = 4;
      ctx.strokeStyle = '#2c2f35';
      ctx.lineWidth = Math.max(1, r * 0.04);
      ctx.stroke();

      // Inner hub ring
      ctx.beginPath();
      ctx.arc(0, 0, r * 0.52, 0, TAU);
      ctx.fillStyle = '#41444b';
      ctx.fill();
      ctx.strokeStyle = '#23262b';
      ctx.lineWidth = Math.max(1, r * 0.05);
      ctx.stroke();

      // Rivets around the hub
      const rivets = 6;
      for (let i = 0; i < rivets; i++) {
        const a = (i / rivets) * TAU;
        ctx.beginPath();
        ctx.arc(Math.cos(a) * r * 0.38, Math.sin(a) * r * 0.38, Math.max(1, r * 0.05), 0, TAU);
        ctx.fillStyle = '#1f2125';
        ctx.fill();
      }

      // Center bolt with a soft metal highlight
      const bolt = ctx.createRadialGradient(-r * 0.05, -r * 0.05, r * 0.02, 0, 0, r * 0.2);
      bolt.addColorStop(0, '#7a7e87');
      bolt.addColorStop(1, '#1b1d21');
      ctx.beginPath();
      ctx.arc(0, 0, r * 0.18, 0, TAU);
      ctx.fillStyle = bolt;
      ctx.fill();

      ctx.restore();
    }
  }

  /**
   * Builds a circular-saw blade path in the current rotated/translated context.
   * Each tooth has a curved leading edge sweeping out to a forward-raked sharp
   * tip, then a steep straight trailing edge back to the next valley (the hook).
   */
  _buildGearPath(ctx, radius, N) {
    const period = TAU / N;
    const innerR = radius * 0.72;
    const tipR   = radius * 1.28;

    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const valleyA = i * period;
      const cos = Math.cos(valleyA) * innerR;
      const sin = Math.sin(valleyA) * innerR;
      if (i === 0) ctx.moveTo(cos, sin);
      else         ctx.lineTo(cos, sin);

      // Curved leading edge → forward-leaning tip (sickle/buzzsaw look)
      const ctrlA = valleyA + period * 0.15;
      const tipA  = valleyA + period * 0.62;
      ctx.quadraticCurveTo(
        Math.cos(ctrlA) * radius * 1.0, Math.sin(ctrlA) * radius * 1.0,
        Math.cos(tipA)  * tipR,         Math.sin(tipA)  * tipR,
      );
      // Trailing edge is the straight line to the next valley (next iteration),
      // forming a sharp backward hook.
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
    // Batch every stone inside one save/restore (perf budget)
    ctx.save();
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
        ctx.arc(sx, sy, r + t * 18, 0, TAU);
        ctx.strokeStyle = `rgba(255,255,255,${(0.4 * (1 - t)).toFixed(3)})`;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // --- Premium marble body --------------------------------------------
      // Soft drop shadow beneath the stone (scales with size)
      ctx.shadowColor = 'rgba(0,0,0,0.35)';
      ctx.shadowBlur = Math.max(4, r * 0.25);
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = Math.max(2, r * 0.12);

      // Spherical shading: lit from upper-left, falling to a dark lower-right rim
      const lx = sx - r * 0.32, ly = sy - r * 0.36;
      const body = ctx.createRadialGradient(lx, ly, r * 0.1, sx, sy, r * 1.08);
      body.addColorStop(0,   this._lighten(stone.color, 0.5));
      body.addColorStop(0.5, stone.color);
      body.addColorStop(1,   this._darken(stone.color, 0.5));
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, TAU);
      ctx.fillStyle = body;
      ctx.fill();

      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.shadowOffsetY = 0;

      // Broad glossy sheen (soft, fading toward the centre)
      const gloss = ctx.createRadialGradient(
        sx - r * 0.25, sy - r * 0.3, 0,
        sx - r * 0.25, sy - r * 0.3, r * 0.95,
      );
      gloss.addColorStop(0,   'rgba(255,255,255,0.45)');
      gloss.addColorStop(0.5, 'rgba(255,255,255,0.08)');
      gloss.addColorStop(1,   'rgba(255,255,255,0)');
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, TAU);
      ctx.fillStyle = gloss;
      ctx.fill();

      // Tight specular glint
      ctx.beginPath();
      ctx.arc(sx - r * 0.34, sy - r * 0.38, Math.max(1, r * 0.12), 0, TAU);
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.fill();

      // Polished rim — dark edge just inside the silhouette for definition
      ctx.beginPath();
      ctx.arc(sx, sy, r - Math.max(0.5, r * 0.03), 0, TAU);
      ctx.strokeStyle = this._darken(stone.color, 0.45);
      ctx.lineWidth = Math.max(1, r * 0.06);
      ctx.stroke();

      // Nickname — hidden on small stones (world radius < 14)
      if (stone.radius >= 14) {
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.shadowColor = 'rgba(0,0,0,0.75)';
        ctx.shadowBlur = 3;
        ctx.fillStyle = '#ffffff';
        ctx.fillText(stone.nickname, sx, sy);
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
      }
    }
    ctx.restore();
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
    const deltaMS = gameState._deltaMS != null ? gameState._deltaMS.toFixed(2) : '?';
    ctx.fillText(`r: ${myStone.radius.toFixed(1)}   spd: ${speed}   rank: ${rank} / ${aliveStones.length}   fragments: ${frg_cnt}   step: ${deltaMS}ms`, 12, 12);
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
