/**
 * accel = BASE_ACCEL / (1 + MASS_ACCEL_FACTOR * area)
 * soft reduction so large stones feel heavier but still controllable
 */
export function applyAcceleration(stone, targetAngle, dist, config) {
  const accel = config.BASE_ACCEL * (1 + config.MASS_ACCEL_FACTOR * stone.radius) * (stone.last_impulse - stone.groggyUntil) / stone.last_impulse * Math.min(1, (dist - config.DEAD_ZONE_RADIUS) / config.MAX_ACCEL_RADIUS);
  stone.vx += Math.cos(targetAngle) * accel;
  stone.vy += Math.sin(targetAngle) * accel;
}

/**
 * Impulse-based 2D collision. Uses area as mass proxy (mass = pi*r^2).
 * Conserves momentum for any restitution value.
 *
 * collisionFriction blends relative-velocity direction into the collision
 * normal, simulating surface friction on a glancing hit.
 */
export function resolveStoneCollision(s1, s2, restitution, collisionFriction) {
  const dx = s2.x - s1.x;
  const dy = s2.y - s1.y;
  const dist = Math.hypot(dx, dy);
  if (dist === 0) return;

  // Geometric normal (center-to-center)
  const gnx = dx / dist;
  const gny = dy / dist;

  // Blend with relative-velocity direction to simulate surface friction
  let nx, ny;
  const rvx = s1.vx - s2.vx;
  const rvy = s1.vy - s2.vy;
  const rvLen = Math.hypot(rvx, rvy);
  if (rvLen > 0 && collisionFriction > 0) {
    const bx = gnx + collisionFriction * (rvx / rvLen);
    const by = gny + collisionFriction * (rvy / rvLen);
    const bLen = Math.hypot(bx, by);
    nx = bx / bLen;
    ny = by / bLen;
  } else {
    nx = gnx;
    ny = gny;
  }

  const m1 = s1.area;
  const m2 = s2.area;

  // Relative velocity along collision normal (positive = approaching)
  const relVn = (s1.vx - s2.vx) * nx + (s1.vy - s2.vy) * ny;
  if (relVn <= 0) return;

  const impulse = (1 + restitution * Math.sqrt((m1 + m2) / 2 / 256)) * relVn / (1 / m1 + 1 / m2);

  s1.vx -= (impulse / m1) * nx;
  s1.vy -= (impulse / m1) * ny;
  s2.vx += (impulse / m2) * nx;
  s2.vy += (impulse / m2) * ny;

  s1.groggyUntil = impulse / ((m1 + m2) / 2) ** 1.6;
  s2.groggyUntil = impulse / ((m1 + m2) / 2) ** 1.6;
  s1.last_impulse = impulse / ((m1 + m2) / 2) ** 1.6;
  s2.last_impulse = impulse / ((m1 + m2) / 2) ** 1.6;

  // Positional correction — use geometric normal to push overlapping stones apart
  const overlap = s1.radius + s2.radius - dist;
  if (overlap > 0) {
    const half = overlap * 0.5;
    s1.x -= half * gnx;
    s1.y -= half * gny;
    s2.x += half * gnx;
    s2.y += half * gny;
  }
}

export function checkGearCollision(stone, gear) {
  return Math.hypot(stone.x - gear.x, stone.y - gear.y) < stone.radius + gear.collisionRadius;
}

export function resolveWallCollision(stone, mapW, mapH) {
  if (stone.x - stone.radius * 0.5 < 0) {
    this._killStone(stone);
  } else if (stone.x + stone.radius * 0.5 > mapW) {
    this._killStone(stone);
  }
  if (stone.y - stone.radius * 0.5 < 0) {
    this._killStone(stone);
  } else if (stone.y + stone.radius * 0.5 > mapH) {
    this._killStone(stone);
  }
}
