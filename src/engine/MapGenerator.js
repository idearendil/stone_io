import { Gear } from './entities/Gear.js';

export class MapGenerator {
  constructor(seed = 42) {
    // Non-zero seed required for xorshift32
    this._seed = (seed >>> 0) || 1;
  }

  _rand() {
    // xorshift32 — period 2^32-1, passes BigCrush
    this._seed ^= this._seed << 13;
    this._seed ^= this._seed >>> 17;
    this._seed ^= this._seed << 5;
    return (this._seed >>> 0) / 0x100000000;
  }

  /**
   * Place gears per zone using Poisson disk sampling.
   * Zone 0 (index 0) is y=0..MAP_HEIGHT/5 (top, hardest).
   * Zone 4 is y=MAP_HEIGHT*4/5..MAP_HEIGHT (bottom, easiest).
   */
  generateGears(config) {
    const { ZONES, MAP_WIDTH, MAP_HEIGHT } = config;
    const zoneHeight = MAP_HEIGHT / ZONES.length;
    const gears = [];

    for (let z = 0; z < ZONES.length; z++) {
      const zone = ZONES[z];
      const yOffset = z * zoneHeight;
      const points = this._poissonDisk(MAP_WIDTH, zoneHeight, zone.gearSpacing);
      for (const [x, y] of points) {
        const radius = zone.gearRadiusMin + this._rand() * (zone.gearRadiusMax - zone.gearRadiusMin);
        gears.push(new Gear(x, y + yOffset, radius, z, zone.rpm));
      }
    }

    return gears;
  }

  /**
   * Grid-accelerated Poisson disk sampling (Bridson 2007).
   * Returns array of [x, y] points with minimum pairwise distance >= minDist.
   */
  _poissonDisk(width, height, minDist) {
    const cellSize = minDist / Math.SQRT2;
    const cols = Math.ceil(width / cellSize);
    const rows = Math.ceil(height / cellSize);
    const grid = new Array(cols * rows).fill(null);
    const active = [];
    const result = [];
    const K = 30;

    const tryInsert = (x, y) => {
      const col = Math.floor(x / cellSize);
      const row = Math.floor(y / cellSize);
      // Check 5x5 neighbourhood of grid cells for proximity violations
      for (let dr = -2; dr <= 2; dr++) {
        for (let dc = -2; dc <= 2; dc++) {
          const nr = row + dr;
          const nc = col + dc;
          if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
          const nb = grid[nr * cols + nc];
          if (!nb) continue;
          const ddx = x - nb[0];
          const ddy = y - nb[1];
          if (ddx * ddx + ddy * ddy < minDist * minDist) return false;
        }
      }
      grid[row * cols + col] = [x, y];
      active.push([x, y]);
      result.push([x, y]);
      return true;
    };

    // Seed with one random point
    tryInsert(this._rand() * width, this._rand() * height);

    while (active.length > 0) {
      const idx = Math.floor(this._rand() * active.length);
      const [ax, ay] = active[idx];
      let found = false;

      for (let k = 0; k < K; k++) {
        const angle = this._rand() * Math.PI * 2;
        const r = minDist * (1 + this._rand()); // sample annulus [minDist, 2*minDist]
        const x = ax + Math.cos(angle) * r;
        const y = ay + Math.sin(angle) * r;
        if (x < 0 || x >= width || y < 0 || y >= height) continue;
        if (tryInsert(x, y)) { found = true; break; }
      }

      if (!found) active.splice(idx, 1);
    }

    return result;
  }
}
