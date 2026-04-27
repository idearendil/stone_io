const VIEWPORT_SIZE = 200;
const HALF_VP = VIEWPORT_SIZE / 2;
const PUSH_DIST = 120; // > DEAD_ZONE_RADIUS (60) so input always registers

export class RuleBasedBot {
  constructor(stoneId) {
    this.stoneId = stoneId;
    this._wanderAngle = Math.random() * Math.PI * 2;
    this._wanderTimer = 0;
    this._wanderInterval = 2000 + Math.random() * 2000;
  }

  update(deltaMs, state, engine) {
    const stone = state.stones.find(s => s.id === this.stoneId);
    if (!stone || !stone.alive) return;

    const angle = this._decide(deltaMs, stone, state, engine);
    const mouseX = HALF_VP + Math.cos(angle) * PUSH_DIST;
    const mouseY = HALF_VP + Math.sin(angle) * PUSH_DIST;
    engine.setInput(this.stoneId, mouseX, mouseY, VIEWPORT_SIZE, VIEWPORT_SIZE);

    if (this._shouldBoost(stone, state)) engine.boost(this.stoneId);
  }

  _shouldBoost(stone, state) {
    const { x, y, radius } = stone;
    for (const other of state.stones) {
      if (other.id === this.stoneId || !other.alive) continue;
      if (other.radius > radius * 1.3) {
        if (Math.hypot(other.x - x, other.y - y) < 250) return true;
      }
    }
    return false;
  }

  _decide(deltaMs, stone, state, engine) {
    const { x, y, radius } = stone;

    // Priority 1: FLEE_GEAR — flee if within 80px of gear collision edge
    let closestGear = null;
    let closestGearDist = Infinity;
    for (const gear of state.gears) {
      const edgeDist = Math.hypot(gear.x - x, gear.y - y) - gear.collisionRadius;
      if (edgeDist < radius + 80 && edgeDist < closestGearDist) {
        closestGearDist = edgeDist;
        closestGear = gear;
      }
    }
    if (closestGear) {
      return Math.atan2(y - closestGear.y, x - closestGear.x);
    }

    // Priority 2: FLEE_BIG — run from stones >1.3× my radius within 400px
    let bigThreat = null;
    let bigDist = Infinity;
    for (const other of state.stones) {
      if (other.id === this.stoneId || !other.alive) continue;
      if (other.radius > radius * 1.3) {
        const d = Math.hypot(other.x - x, other.y - y);
        if (d < 400 && d < bigDist) { bigDist = d; bigThreat = other; }
      }
    }
    if (bigThreat) {
      return Math.atan2(y - bigThreat.y, x - bigThreat.x);
    }

    // Priority 3: CHASE_SMALL — chase stones <0.8× my radius within 500px
    let smallTarget = null;
    let smallDist = Infinity;
    for (const other of state.stones) {
      if (other.id === this.stoneId || !other.alive) continue;
      if (other.radius < radius * 0.8) {
        const d = Math.hypot(other.x - x, other.y - y);
        if (d < 500 && d < smallDist) { smallDist = d; smallTarget = other; }
      }
    }
    if (smallTarget) {
      return Math.atan2(smallTarget.y - y, smallTarget.x - x);
    }

    // Priority 4: CHASE_FRAGMENT — best score fragment in 3×3 nearby grid cells
    let bestFrag = null;
    let bestScore = -Infinity;
    for (const frag of engine.getFragmentsNear(x, y)) {
      const d = Math.hypot(frag.x - x, frag.y - y);
      if (d > 600) continue;
      const score = frag.radius / (d + 1);
      if (score > bestScore) { bestScore = score; bestFrag = frag; }
    }
    if (bestFrag) {
      return Math.atan2(bestFrag.y - y, bestFrag.x - x);
    }

    // Priority 5: WANDER — random direction, reroll every 2–4 seconds
    this._wanderTimer += deltaMs;
    if (this._wanderTimer >= this._wanderInterval) {
      this._wanderTimer = 0;
      this._wanderInterval = 2000 + Math.random() * 2000;
      this._wanderAngle = Math.random() * Math.PI * 2;
    }
    return this._wanderAngle;
  }
}
