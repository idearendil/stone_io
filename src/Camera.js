const BASE_RADIUS = 16;   // zoom = 1 at this stone radius
const ZOOM_MIN    = 0.20; // max 5× zoom-out
const ZOOM_MAX    = 1.00;
const ZOOM_LERP   = 0.06; // smoothing factor per frame (~60 fps)

export class Camera {
  constructor(viewportW, viewportH) {
    this.viewportW = viewportW;
    this.viewportH = viewportH;
    this.x = 0;
    this.y = 0;
    this.zoom = ZOOM_MAX;
    this._targetZoom = ZOOM_MAX;
  }

  /** Lock camera to target and smoothly adjust zoom based on stone radius. */
  update(targetX, targetY, stoneRadius = BASE_RADIUS) {
    this.x = targetX;
    this.y = targetY;
    this._targetZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, BASE_RADIUS * 4 / (stoneRadius + BASE_RADIUS * 3)));
    this.zoom += (this._targetZoom - this.zoom) * ZOOM_LERP;
  }

  worldToScreen(wx, wy) {
    return {
      sx: (wx - this.x) * this.zoom + this.viewportW * 0.5,
      sy: (wy - this.y) * this.zoom + this.viewportH * 0.5,
    };
  }

  screenToWorld(sx, sy) {
    return {
      wx: (sx - this.viewportW * 0.5) / this.zoom + this.x,
      wy: (sy - this.viewportH * 0.5) / this.zoom + this.y,
    };
  }

  /** True if a circle at world (wx, wy) with given radius intersects the viewport. */
  isVisible(wx, wy, radius) {
    const { sx, sy } = this.worldToScreen(wx, wy);
    const sr = radius * this.zoom;
    return (
      sx + sr > 0 && sx - sr < this.viewportW &&
      sy + sr > 0 && sy - sr < this.viewportH
    );
  }
}
