export class Stone {
  constructor(id, x, y, radius, color, nickname) {
    this.id = id;
    this.x = x;
    this.y = y;
    this.vx = 0;
    this.vy = 0;
    this.radius = radius;
    this.color = color;
    this.nickname = nickname;
    this.alive = true;
    this.respawnAt = null;       // engine totalTime when stone should respawn (null = alive)
    this.invincibleUntil = 0;   // engine totalTime until invincibility expires
    this.groggyUntil = 0;       // engine totalTime until acceleration fully applied
    this.last_impulse = 1;      // last impulse applied to this stone (by collision)
  }

  get area() {
    return Math.PI * this.radius * this.radius;
  }

  absorb(fragmentArea) {
    this.radius = Math.sqrt((this.area + fragmentArea) / Math.PI);
  }
}
