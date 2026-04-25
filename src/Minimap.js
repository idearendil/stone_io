const W = 200;
const H = 160;
const MARGIN = 10;
const ZONE_COLORS = ['#1A0F05', '#3D250F', '#7A4F20', '#B07A3A', '#D4A860'];

export class Minimap {
  constructor(config) {
    this.config = config;
    this._fullMap = false;
  }

  /** Toggle between nearby-range view and full-map view. */
  toggleZoom() {
    this._fullMap = !this._fullMap;
  }

  draw(ctx, gameState, myStoneId, camera, canvasW, canvasH) {
    const { MAP_WIDTH, MAP_HEIGHT, MINIMAP_VIEW_RANGE } = this.config;
    const myStone = gameState.stones.find(s => s.id === myStoneId && s.alive);
    if (!myStone) return;
    const viewRange = this._fullMap ? Infinity : MINIMAP_VIEW_RANGE;

    const ox = canvasW - W - MARGIN;
    const oy = MARGIN;

    ctx.save();

    // Clip to minimap rect
    ctx.beginPath();
    ctx.rect(ox, oy, W, H);
    ctx.clip();

    // Zone bands
    const bandH = H / ZONE_COLORS.length;
    for (let i = 0; i < ZONE_COLORS.length; i++) {
      ctx.fillStyle = ZONE_COLORS[i];
      ctx.fillRect(ox, oy + i * bandH, W, bandH);
    }

    // Darken overlay
    ctx.fillStyle = 'rgba(0,0,0,0.38)';
    ctx.fillRect(ox, oy, W, H);

    const toMini = (wx, wy) => ({
      mx: ox + (wx / MAP_WIDTH) * W,
      my: oy + (wy / MAP_HEIGHT) * H,
    });

    // Gears — gray triangles
    ctx.fillStyle = '#999';
    for (const gear of gameState.gears) {
      if (Math.hypot(gear.x - myStone.x, gear.y - myStone.y) > viewRange) continue;
      const { mx, my } = toMini(gear.x, gear.y);
      ctx.beginPath();
      ctx.moveTo(mx, my - 3);
      ctx.lineTo(mx + 2.5, my + 2.5);
      ctx.lineTo(mx - 2.5, my + 2.5);
      ctx.closePath();
      ctx.fill();
    }

    // Other stones
    for (const stone of gameState.stones) {
      if (!stone.alive || stone.id === myStoneId) continue;
      if (Math.hypot(stone.x - myStone.x, stone.y - myStone.y) > viewRange) continue;
      const { mx, my } = toMini(stone.x, stone.y);
      ctx.fillStyle = stone.color;
      ctx.beginPath();
      ctx.arc(mx, my, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Own stone — white, 4px
    {
      const { mx, my } = toMini(myStone.x, myStone.y);
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(mx, my, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Viewport rectangle — divide by zoom to get the visible world extent
    const worldW = camera.viewportW / camera.zoom;
    const worldH = camera.viewportH / camera.zoom;
    const vpLeft = camera.x - worldW * 0.5;
    const vpTop  = camera.y - worldH * 0.5;
    const { mx: rx, my: ry } = toMini(vpLeft, vpTop);
    const rw = (worldW / MAP_WIDTH) * W;
    const rh = (worldH / MAP_HEIGHT) * H;
    ctx.strokeStyle = 'rgba(255,255,255,0.55)';
    ctx.lineWidth = 1;
    ctx.strokeRect(rx, ry, rw, rh);

    ctx.restore();

    // Border drawn outside clip so it's crisp
    ctx.strokeStyle = 'rgba(255,255,255,0.22)';
    ctx.lineWidth = 1;
    ctx.strokeRect(ox, oy, W, H);
  }
}
