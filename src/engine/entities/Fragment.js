export class Fragment {
  constructor(x, y, radius, vx, vy, ttl, color = '#90d469') {
    this.x = x;
    this.y = y;
    this.radius = radius;
    this.area = Math.PI * radius * radius;
    this.vx = vx;
    this.vy = vy;
    this.ttl = ttl;
    this.maxTtl = ttl;
    this.color = color;
  }
}
