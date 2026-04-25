export class DebugOverlay {
  constructor(config) {
    this.config = config;
    this._fps = 0;
    this._frameCount = 0;
    this._fpsLastTime = 0;
  }

  draw(ctx, gameState, camera, timestamp) {
    this._drawFps(ctx, timestamp);
    this._drawZoneLabels(ctx, camera);
    this._drawGearCollisionRadii(ctx, gameState.gears, camera);
    this._drawStoneDebug(ctx, gameState.stones, camera);
  }

  _drawFps(ctx, timestamp) {
    this._frameCount++;
    const elapsed = timestamp - this._fpsLastTime;
    if (elapsed >= 500) {
      this._fps = Math.round(this._frameCount * 1000 / elapsed);
      this._frameCount = 0;
      this._fpsLastTime = timestamp;
    }
    ctx.save();
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#00ff44';
    ctx.shadowColor = 'rgba(0,0,0,0.85)';
    ctx.shadowBlur = 3;
    ctx.fillText(`FPS: ${this._fps}`, 12, 34);
    ctx.restore();
  }

  _drawZoneLabels(ctx, camera) {
    const { MAP_HEIGHT, ZONES } = this.config;
    const zoneH = MAP_HEIGHT / ZONES.length;
    ctx.save();
    ctx.font = 'bold 11px monospace';
    ctx.fillStyle = 'rgba(255,200,100,0.75)';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = 'rgba(0,0,0,0.7)';
    ctx.shadowBlur = 3;
    for (let z = 0; z < ZONES.length; z++) {
      const { sy } = camera.worldToScreen(0, (z + 0.5) * zoneH);
      if (sy < 0 || sy > camera.viewportH) continue;
      ctx.fillText(`Zone ${z}  (${ZONES[z].rpm} rpm)`, 5, sy);
    }
    ctx.restore();
  }

  _drawGearCollisionRadii(ctx, gears, camera) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255,0,0,0.55)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    for (const gear of gears) {
      if (!camera.isVisible(gear.x, gear.y, gear.radius * 1.5)) continue;
      const { sx, sy } = camera.worldToScreen(gear.x, gear.y);
      ctx.beginPath();
      ctx.arc(sx, sy, gear.radius * 1.05, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.restore();
  }

  _drawStoneDebug(ctx, stones, camera) {
    ctx.save();
    for (const stone of stones) {
      if (!stone.alive) continue;
      if (!camera.isVisible(stone.x, stone.y, stone.radius + 60)) continue;
      const { sx, sy } = camera.worldToScreen(stone.x, stone.y);
      const speed = Math.hypot(stone.vx, stone.vy);

      // Velocity arrow
      if (speed > 0.05) {
        const arrowLen = speed * 10;
        const ux = stone.vx / speed;
        const uy = stone.vy / speed;
        const ex = sx + ux * arrowLen;
        const ey = sy + uy * arrowLen;
        const headA = Math.atan2(stone.vy, stone.vx);
        const headLen = 7;

        ctx.strokeStyle = '#ffff00';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - Math.cos(headA - 0.45) * headLen, ey - Math.sin(headA - 0.45) * headLen);
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - Math.cos(headA + 0.45) * headLen, ey - Math.sin(headA + 0.45) * headLen);
        ctx.stroke();
      }

      // Radius + speed label
      ctx.font = '10px monospace';
      ctx.fillStyle = 'rgba(255,255,150,0.92)';
      ctx.shadowColor = 'rgba(0,0,0,0.8)';
      ctx.shadowBlur = 2;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`r:${stone.radius.toFixed(1)} v:${speed.toFixed(2)}`, sx + stone.radius + 5, sy);
    }
    ctx.shadowBlur = 0;
    ctx.restore();
  }
}
