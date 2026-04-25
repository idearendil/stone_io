/**
 * accel = BASE_ACCEL / (1 + MASS_ACCEL_FACTOR * area)
 * soft reduction so large stones feel heavier but still controllable
 */
export function applyAcceleration(stone, targetAngle, config) {
  const accel = config.BASE_ACCEL / (1 + config.MASS_ACCEL_FACTOR * stone.radius) * (stone.last_impulse - stone.groggyUntil) / stone.last_impulse;
  stone.vx += Math.cos(targetAngle) * accel;
  stone.vy += Math.sin(targetAngle) * accel;
}

/**
 * Impulse-based 2D collision. Uses area as mass proxy (mass = pi*r^2).
 * Conserves momentum for any restitution value.
 */
export function resolveStoneCollision(s1, s2, restitution) {
  const dx = s2.x - s1.x;
  const dy = s2.y - s1.y;
  const dist = Math.hypot(dx, dy);
  if (dist === 0) return;

  const nx = dx / dist;
  const ny = dy / dist;

  const m1 = s1.area;
  const m2 = s2.area;

  // Relative velocity along collision normal (positive = approaching)
  const relVn = (s1.vx - s2.vx) * nx + (s1.vy - s2.vy) * ny;
  if (relVn <= 0) return;

  const impulse = (1 + restitution) * relVn / (1 / m1 + 1 / m2);

  s1.vx -= (impulse / m1) * nx;
  s1.vy -= (impulse / m1) * ny;
  s2.vx += (impulse / m2) * nx;
  s2.vy += (impulse / m2) * ny;

  s1.groggyUntil = impulse / m1;
  s2.groggyUntil = impulse / m2;
  s1.last_impulse = impulse / m1;
  s2.last_impulse = impulse / m2;

  // Positional correction — push overlapping stones apart equally
  const overlap = s1.radius + s2.radius - dist;
  if (overlap > 0) {
    const half = overlap * 0.5;
    s1.x -= half * nx;
    s1.y -= half * ny;
    s2.x += half * nx;
    s2.y += half * ny;
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
