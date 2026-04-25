export class Gear {
  constructor(x, y, radius, zoneIndex, rpm) {
    this.x = x;
    this.y = y;
    this.radius = radius;
    this.zoneIndex = zoneIndex;
    this.rpm = rpm;
    this.angle = 0;
  }

  get collisionRadius() {
    return this.radius * 1.05;
  }

  update(deltaMs) {
    this.angle = (this.angle + (this.rpm * 2 * Math.PI / 60000) * deltaMs) % (2 * Math.PI);
  }
}
