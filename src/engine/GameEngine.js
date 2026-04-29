import { Stone } from './entities/Stone.js';
import { Fragment } from './entities/Fragment.js';
import { MapGenerator } from './MapGenerator.js';
import * as Physics from './Physics.js';
import { RuleBasedBot } from '../bots/RuleBasedBot.js';
import { TrainedBot } from '../bots/TrainedBot.js';

const COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#e91e63'];

export class GameEngine {
  constructor(config) {
    this.config = config;
    this._idCounter = 0;
    this._mapSeed = 12345;
    this._bots = new Map();
    this._botType = 'rule-based';
    this._trainedWeights = null;
    this._lastState = null;
    this.reset();
  }

  reset() {
    this.tick = 0;
    this._totalTime = 0;
    this.stones = new Map();
    this.fragments = [];
    this._inputs = new Map();
    this._boostCooldowns = new Map();
    this._spawnTimer = 0;
    this._events = [];
    this.gears = new MapGenerator(this._mapSeed).generateGears(this.config);
    this._initFragGrid();
  }

  // ---------------------------------------------------------------------------
  // Fragment spatial grid
  // Cell size must exceed max(stone.radius + max_fragment_drift) for correctness.
  // Death fragments drift at most ~30 world units before stopping (friction 0.9/frame).
  // ---------------------------------------------------------------------------

  _initFragGrid() {
    const cs = this.config.FRAG_CELL_SIZE;
    this._cellSize = cs;
    this._gridCols = Math.ceil(this.config.MAP_WIDTH  / cs);
    this._gridRows = Math.ceil(this.config.MAP_HEIGHT / cs);
    // _fragCells[row][col] = Fragment[]  (indexed by spawn position, never updated)
    this._fragCells = Array.from({ length: this._gridRows }, () =>
      Array.from({ length: this._gridCols }, () => [])
    );
  }

  _addFragment(frag) {
    const cs = this._cellSize;
    frag._gridRow = Math.min(this._gridRows - 1, Math.max(0, Math.floor(frag.y / cs)));
    frag._gridCol = Math.min(this._gridCols - 1, Math.max(0, Math.floor(frag.x / cs)));
    this.fragments.push(frag);
    this._fragCells[frag._gridRow][frag._gridCol].push(frag);
  }

  _removeFragFromGrid(frag) {
    const cell = this._fragCells[frag._gridRow][frag._gridCol];
    const idx = cell.indexOf(frag);
    if (idx !== -1) cell.splice(idx, 1);
  }

  /** Returns all fragments in the 3×3 cell neighbourhood around world position (wx, wy). */
  getFragmentsNear(wx, wy) {
    const cs = this._cellSize;
    const col = Math.min(this._gridCols - 1, Math.max(0, Math.floor(wx / cs)));
    const row = Math.min(this._gridRows - 1, Math.max(0, Math.floor(wy / cs)));
    const result = [];
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        const r = row + dr, c = col + dc;
        if (r >= 0 && r < this._gridRows && c >= 0 && c < this._gridCols) {
          const cell = this._fragCells[r][c];
          for (let i = 0; i < cell.length; i++) result.push(cell[i]);
        }
      }
    }
    return result;
  }

  // ---------------------------------------------------------------------------
  // Player / bot management
  // ---------------------------------------------------------------------------

  /** Returns the stoneId assigned to this player. */
  addPlayer(id, nickname) {
    const stoneId = ++this._idCounter;
    const { MAP_WIDTH, MAP_HEIGHT, ZONES, STONE_INIT_RADIUS } = this.config;
    const zoneHeight = MAP_HEIGHT / ZONES.length;
    const zone4Top = zoneHeight * (ZONES.length - 1);
    const x = 100 + Math.random() * (MAP_WIDTH - 200);
    const y = zone4Top + 50 + Math.random() * (zoneHeight - 100);
    this.stones.set(stoneId, new Stone(stoneId, x, y, STONE_INIT_RADIUS, COLORS[stoneId % COLORS.length], nickname));
    return stoneId;
  }

  removePlayer(stoneId) {
    this.stones.delete(stoneId);
    this._inputs.delete(stoneId);
  }

  addBot(nickname) {
    const stoneId = this.addPlayer(null, nickname);
    const bot = (this._botType === 'trained' && this._trainedWeights)
      ? new TrainedBot(stoneId, this._trainedWeights)
      : new RuleBasedBot(stoneId);
    this._bots.set(stoneId, bot);
    return stoneId;
  }

  removeBot(stoneId) {
    this.removePlayer(stoneId);
    this._bots.delete(stoneId);
  }

  /**
   * Swap all bots to a different implementation.
   * type: 'rule-based' | 'trained'
   * weightsJson: parsed JSON from bot.json (required when type === 'trained')
   */
  setBotType(type, weightsJson = null) {
    this._botType = type;
    this._trainedWeights = weightsJson;
    for (const [stoneId] of this._bots) {
      if (type === 'trained' && weightsJson) {
        this._bots.set(stoneId, new TrainedBot(stoneId, weightsJson));
      } else {
        this._bots.set(stoneId, new RuleBasedBot(stoneId));
      }
    }
  }

  /**
   * Instant boost toward the cursor. Costs 10 area; 1-second cooldown.
   * Returns true if the boost was applied.
   */
  boost(stoneId) {
    const cooldownEnd = this._boostCooldowns.get(stoneId) ?? 0;
    if (this._totalTime < cooldownEnd) return false;

    const stone = this.stones.get(stoneId);
    if (!stone || !stone.alive) return false;

    const { STONE_INIT_RADIUS, BOOST_IMPULSE, BOOST_COOLDOWN, BOOST_AREA_COST } = this.config;
    if (stone.radius <= STONE_INIT_RADIUS) return false;

    const inp = this._inputs.get(stoneId);
    if (inp) {
      const dx = inp.mouseX - inp.viewportW / 2;
      const dy = inp.mouseY - inp.viewportH / 2;
      const dist = Math.hypot(dx, dy);
      if (dist > 0) {
        stone.vx += (dx / dist) * BOOST_IMPULSE;
        stone.vy += (dy / dist) * BOOST_IMPULSE;
      }
    }

    stone.radius = Math.sqrt(Math.max(0, stone.area - BOOST_AREA_COST) / Math.PI);

    this._boostCooldowns.set(stoneId, this._totalTime + BOOST_COOLDOWN);
    this._events.push({ type: 'boost', stoneId });
    return true;
  }

  /** Store mouse intent; applied at next step(). Coordinates are relative to the player's viewport. */
  setInput(stoneId, mouseX, mouseY, viewportW, viewportH) {
    this._inputs.set(stoneId, { mouseX, mouseY, viewportW, viewportH });
  }

  /** Merge partial config values into the running simulation without a full reset. */
  updateConfig(partial) {
    Object.assign(this.config, partial);
  }

  /** Returns per-zone entity counts for balance inspection. */
  getZoneStats() {
    const { ZONES, MAP_HEIGHT } = this.config;
    const zoneH = MAP_HEIGHT / ZONES.length;
    const stats = ZONES.map((_, i) => ({ zoneIndex: i, stoneCount: 0, fragmentCount: 0, gearCount: 0 }));
    for (const stone of this.stones.values()) {
      if (!stone.alive) continue;
      const z = Math.min(ZONES.length - 1, Math.floor(stone.y / zoneH));
      if (z >= 0) stats[z].stoneCount++;
    }
    for (const frag of this.fragments) {
      const z = Math.min(ZONES.length - 1, Math.floor(frag.y / zoneH));
      if (z >= 0) stats[z].fragmentCount++;
    }
    for (const gear of this.gears) {
      stats[gear.zoneIndex].gearCount++;
    }
    return stats;
  }

  // ---------------------------------------------------------------------------
  // Main simulation step
  // ---------------------------------------------------------------------------

  /** Advance simulation by deltaMs milliseconds. Returns a GameState snapshot. */
  step(deltaMs) {
    this.tick++;
    this._totalTime += deltaMs;
    this._events = [];

    // Bots decide using previous frame's state (1-tick perception delay)
    if (this._lastState) {
      for (const bot of this._bots.values()) {
        bot.update(deltaMs, this._lastState, this);
      }
    }

    // Respawn dead stones after their 2-second delay
    for (const stone of this.stones.values()) {
      if (!stone.alive && stone.respawnAt !== null && this._totalTime >= stone.respawnAt) {
        this._respawnStone(stone);
      }
    }

    for (const gear of this.gears) gear.update(deltaMs);

    // Apply input + friction + movement + groggy
    for (const [stoneId, stone] of this.stones) {
      if (!stone.alive) continue;
      const inp = this._inputs.get(stoneId);
      if (inp) {
        const dx = inp.mouseX - inp.viewportW / 2;
        const dy = inp.mouseY - inp.viewportH / 2;
        const dist = Math.hypot(dx, dy);
        if (stone.groggyUntil > 0) stone.groggyUntil -= this.config.GROGGY_COUNTDOWN;
        if (stone.groggyUntil <= 0)  stone.groggyUntil = 0;
        if (dist >= this.config.DEAD_ZONE_RADIUS) {
          Physics.applyAcceleration(stone, Math.atan2(dy, dx), dist, this.config);
        }
      }
      const fric_x = this.config.FRICTION + Math.abs(stone.vx) / 450;
      const fric_y = this.config.FRICTION + Math.abs(stone.vy) / 450;
      stone.vx *= (fric_x + (1 - fric_x) * stone.groggyUntil / stone.last_impulse);
      stone.vy *= (fric_y + (1 - fric_y) * stone.groggyUntil / stone.last_impulse);
      if (Math.hypot(stone.vx, stone.vy) > this.config.MAX_SPEED) {
        stone.vx *= this.config.MAX_SPEED / Math.hypot(stone.vx, stone.vy);
        stone.vy *= this.config.MAX_SPEED / Math.hypot(stone.vx, stone.vy);
      }
      stone.x += stone.vx;
      stone.y += stone.vy;

      // Wall collision
      if (stone.x - stone.radius * 0.5 < 0) {
        this._killStone(stone);
      } else if (stone.x + stone.radius * 0.5 > this.config.MAP_WIDTH) {
        this._killStone(stone);
      }
      if (stone.y - stone.radius * 0.5 < 0) {
        this._killStone(stone);
      } else if (stone.y + stone.radius * 0.5 > this.config.MAP_HEIGHT) {
        this._killStone(stone);
      }
    }

    // Stone-stone collisions
    const alive = [...this.stones.values()].filter(s => s.alive);
    for (let i = 0; i < alive.length - 1; i++) {
      for (let j = i + 1; j < alive.length; j++) {
        const a = alive[i], b = alive[j];
        if (Math.hypot(b.x - a.x, b.y - a.y) < a.radius + b.radius) {
          Physics.resolveStoneCollision(a, b, this.config.RESTITUTION, this.config.COLLISION_FRICTION);
          this._events.push({ type: 'collision' });
        }
      }
    }

    // Gear collisions — invincible stones are immune
    for (const stone of alive) {
      if (!stone.alive) continue;
      if (this._totalTime < stone.invincibleUntil) continue;
      for (const gear of this.gears) {
        if (Physics.checkGearCollision(stone, gear)) {
          this._killStone(stone);
          break;
        }
      }
    }

    // Fragment physics + expiry (iterate flat list; remove expired from grid)
    {
      const live = [];
      for (const f of this.fragments) {
        f.vx *= this.config.FRICTION;
        f.vy *= this.config.FRICTION;
        f.x += f.vx;
        f.y += f.vy;
        f.ttl -= deltaMs;
        if (f.ttl > 0) {
          live.push(f);
        } else {
          this._removeFragFromGrid(f);
        }
      }
      this.fragments = live;
    }

    // Fragment absorption — only check 3×3 grid neighbourhood per stone
    const aliveNow = [...this.stones.values()].filter(s => s.alive);
    const absorbed = new Set();
    for (const stone of aliveNow) {
      for (const frag of this.getFragmentsNear(stone.x, stone.y)) {
        if (absorbed.has(frag)) continue;
        if (Math.hypot(stone.x - frag.x, stone.y - frag.y) < stone.radius + frag.radius) {
          stone.absorb(frag.area);
          this._events.push({ type: 'absorb', x: frag.x, y: frag.y });
          absorbed.add(frag);
        }
      }
    }
    if (absorbed.size > 0) {
      this.fragments = this.fragments.filter(f => !absorbed.has(f));
      for (const f of absorbed) this._removeFragFromGrid(f);
    }

    // Natural fragment spawn
    this._spawnTimer += deltaMs;
    while (this._spawnTimer >= this.config.SPAWN_INTERVAL) {
      this._spawnTimer -= this.config.SPAWN_INTERVAL;
      this._spawnNaturalFragments();
    }

    const state = this.getState();
    this._lastState = state;
    return state;
  }

  getState() {
    return {
      stones: [...this.stones.values()].map(s => ({
        id: s.id, x: s.x, y: s.y, vx: s.vx, vy: s.vy,
        radius: s.radius, color: s.color, nickname: s.nickname, alive: s.alive,
        invincible: this._totalTime < s.invincibleUntil,
      })),
      gears: this.gears.map(g => ({
        x: g.x, y: g.y, radius: g.radius, collisionRadius: g.collisionRadius,
        zoneIndex: g.zoneIndex, angle: g.angle,
      })),
      fragments: this.fragments.map(f => ({
        x: f.x, y: f.y, radius: f.radius, ttl: f.ttl, maxTtl: f.maxTtl, color: f.color,
      })),
      events: [...this._events],
      tick: this.tick,
    };
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  _killStone(stone) {
    stone.alive = false;
    stone.respawnAt = this._totalTime + 2000;
    let fragment_spawned = 1;
    const basic_radius = stone.radius ** (3/4);
    const target = Math.ceil(stone.radius);
    let attempts = 0;
    while (fragment_spawned <= Math.sqrt(stone.radius * 0.8)) {
      if (++attempts > target * 20) break;

      const angle = Math.random() * 2 * Math.PI;
      const speed = 1 + Math.random() * 2;
      const radius = basic_radius + basic_radius * 0.2 * (Math.random() * 2 - 1);
      const fx = stone.x + Math.cos(angle) * stone.radius;
      const fy = stone.y + Math.sin(angle) * stone.radius;

      const { MAP_WIDTH, MAP_HEIGHT } = this.config;
      if (fx - radius < 0 || fx + radius > MAP_WIDTH ||
          fy - radius < 0 || fy + radius > MAP_HEIGHT) continue;
      const tooClose = this.gears.some(
        gear => Math.hypot(fx - gear.x, fy - gear.y) < gear.collisionRadius + radius + 20
      );
      if (tooClose) continue;
      fragment_spawned++;

      // Grid cell is assigned from spawn position and never updated.
      // Death fragments drift at most ~30 world units (speed 1-3, friction 0.9/frame).
      this._addFragment(new Fragment(fx, fy, radius,
        Math.cos(angle) * speed, Math.sin(angle) * speed,
        this.config.FRAGMENT_LIFETIME, stone.color,
      ));
    }
  }

  _respawnStone(stone) {
    const { MAP_WIDTH, MAP_HEIGHT, ZONES, STONE_INIT_RADIUS } = this.config;
    const zoneH = MAP_HEIGHT / ZONES.length;
    const zone4Top = zoneH * (ZONES.length - 1);
    stone.x = 100 + Math.random() * (MAP_WIDTH - 200);
    stone.y = zone4Top + 50 + Math.random() * (zoneH - 100);
    stone.vx = 0;
    stone.vy = 0;
    stone.radius = STONE_INIT_RADIUS;
    stone.alive = true;
    stone.respawnAt = null;
    stone.invincibleUntil = this._totalTime + 1500;
  }

  _spawnNaturalFragments() {
    const { MAP_WIDTH, MAP_HEIGHT, ZONES,
            MIN_FRAGMENT_SPAWN, MAX_FRAGMENT_SPAWN, FRAGMENT_LIFETIME } = this.config;
    const zoneH = MAP_HEIGHT / ZONES.length;
    const count = MIN_FRAGMENT_SPAWN +
      Math.floor(Math.random() * (MAX_FRAGMENT_SPAWN - MIN_FRAGMENT_SPAWN + 1));
    for (let i = 0; i < count; i++) {
      let x, y, radius;
      let attempts = 0;
      do {
        x = 50 + Math.random() * (MAP_WIDTH - 100);
        y = 50 + Math.random() * (MAP_HEIGHT - 100);
        const zone = Math.min(ZONES.length - 1, Math.floor(y / zoneH));
        radius = zone <= 1
          ? 8  + Math.random() * 4   // upper (hard) zones: 8–12 — big reward for risk
          : zone >= 3
            ? 3  + Math.random() * 3   // lower (safe) zones: 3–6
            : 5  + Math.random() * 3;  // middle zone: 5–8
        attempts++;
      } while (
        attempts < 10 &&
        this.gears.some(g => Math.hypot(x - g.x, y - g.y) < g.collisionRadius + radius + 20)
      );
      // Natural fragments have vx=vy=0, so spawn position = forever position
      if (attempts < 10) this._addFragment(new Fragment(x, y, radius, 0, 0, FRAGMENT_LIFETIME));
    }
  }

  _spawnInitialFragments() {
    const { MAP_WIDTH, MAP_HEIGHT, ZONES, FRAGMENT_LIFETIME } = this.config;
    const zoneH = MAP_HEIGHT / ZONES.length;
    let x, y, radius;
    let attempts = 0;
    do {
      x = 50 + Math.random() * (MAP_WIDTH - 100);
      y = 50 + Math.random() * (MAP_HEIGHT - 100);
      const zone = Math.min(ZONES.length - 1, Math.floor(y / zoneH));
      radius = zone <= 1
        ? 8  + Math.random() * 4   // upper (hard) zones: 8–12 — big reward for risk
        : zone >= 3
          ? 3  + Math.random() * 3   // lower (safe) zones: 3–6
          : 5  + Math.random() * 3;  // middle zone: 5–8
      attempts++;
    } while (
      attempts < 10 &&
      this.gears.some(g => Math.hypot(x - g.x, y - g.y) < g.collisionRadius + radius + 20)
    );
    if (attempts < 10) this._addFragment(new Fragment(x, y, radius, 0, 0, Math.random() * FRAGMENT_LIFETIME));
  }
}
