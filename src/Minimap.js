const W = 200;
const H = 160;
const MARGIN = 10;
const ZONE_COLORS = ['#1A0F05', '#3D250F', '#7A4F20', '#B07A3A', '#D4A860'];

export class Minimap {
  constructor(config) {
    this.config = config;
    this._fullMap = true;
  }

  /** Toggle between nearby-range view and full-map view. */
  toggleZoom() {
    this._fullMap = !this._fullMap;
  }

  draw(ctx, gameState, myStoneId, camera, canvasW, canvasH) {
    const { MAP_WIDTH, MAP_HEIGHT } = this.config;
    const myStone = gameState.stones.find(s => s.id === myStoneId && s.alive);
    if (!myStone) return;

    const ox = canvasW - W - MARGIN;
    const oy = MARGIN;

    // World extent shown by the minimap
    const vpW = camera.viewportW / camera.zoom;
    const vpH = camera.viewportH / camera.zoom;
    const shownW = this._fullMap ? MAP_WIDTH  : vpW * 2;
    const shownH = this._fullMap ? MAP_HEIGHT : vpH * 2;
    const viewCX   = this._fullMap ? MAP_WIDTH  * 0.5 : myStone.x;
    const viewCY   = this._fullMap ? MAP_HEIGHT * 0.5 : myStone.y;
    const viewLeft = viewCX - shownW * 0.5;
    const viewTop  = viewCY - shownH * 0.5;
    const viewRight  = viewLeft + shownW;
    const viewBottom = viewTop  + shownH;

    const toMini = (wx, wy) => ({
      mx: ox + (wx - viewLeft) / shownW * W,
      my: oy + (wy - viewTop)  / shownH * H,
    });

    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, W, H);
    ctx.clip();

    // Out-of-map base fill
    ctx.fillStyle = '#0a0808';
    ctx.fillRect(ox, oy, W, H);

    // Zone bands — only the portion overlapping the view
    const zoneH = MAP_HEIGHT / ZONE_COLORS.length;
    for (let i = 0; i < ZONE_COLORS.length; i++) {
      const zWorldTop    = i * zoneH;
      const zWorldBottom = (i + 1) * zoneH;
      const overlapTop    = Math.max(zWorldTop,    viewTop);
      const overlapBottom = Math.min(zWorldBottom, viewBottom);
      if (overlapBottom <= overlapTop) continue;
      const { my: sTop    } = toMini(0, overlapTop);
      const { my: sBottom } = toMini(0, overlapBottom);
      ctx.fillStyle = ZONE_COLORS[i];
      ctx.fillRect(ox, sTop, W, sBottom - sTop);
    }

    // Darken overlay
    ctx.fillStyle = 'rgba(0,0,0,0.38)';
    ctx.fillRect(ox, oy, W, H);

    // Gears — gray triangles
    ctx.fillStyle = '#999';
    for (const gear of gameState.gears) {
      if (gear.x < viewLeft || gear.x > viewRight || gear.y < viewTop || gear.y > viewBottom) continue;
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
      if (stone.x < viewLeft || stone.x > viewRight || stone.y < viewTop || stone.y > viewBottom) continue;
      const { mx, my } = toMini(stone.x, stone.y);
      ctx.fillStyle = stone.color;
      ctx.beginPath();
      ctx.arc(mx, my, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Own stone — white, 4px (center of minimap in local view)
    {
      const { mx, my } = toMini(myStone.x, myStone.y);
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(mx, my, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Viewport rectangle
    const { mx: rx, my: ry } = toMini(camera.x - vpW * 0.5, camera.y - vpH * 0.5);
    const rw = (vpW / shownW) * W;
    const rh = (vpH / shownH) * H;
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
