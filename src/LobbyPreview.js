/**
 * Decorative lobby animation: three marble stones slowly orbiting a spinning
 * gear over a maple-wood background. Pure canvas, runs behind the lobby card.
 * Returns a stop() function that halts the rAF loop and frees the canvas.
 */

const STONE_COLORS = ['#2C3E6E', '#6E2C3E', '#2C6E3E'];
const TAU = Math.PI * 2;

export function startLobbyPreview(canvas, config) {
  if (!canvas) return () => {};
  const ctx = canvas.getContext('2d');
  let raf = 0;
  let running = true;

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  const start = performance.now();

  function drawGear(cx, cy, r, angle) {
    const N = 14;
    const period = TAU / N;
    const innerR = r * 0.72;
    const tipR = r * 1.28;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);

    // Buzzsaw blade with raked teeth
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const valleyA = i * period;
      const fn = i === 0 ? 'moveTo' : 'lineTo';
      ctx[fn](Math.cos(valleyA) * innerR, Math.sin(valleyA) * innerR);
      const ctrlA = valleyA + period * 0.15;
      const tipA = valleyA + period * 0.62;
      ctx.quadraticCurveTo(
        Math.cos(ctrlA) * r, Math.sin(ctrlA) * r,
        Math.cos(tipA) * tipR, Math.sin(tipA) * tipR,
      );
    }
    ctx.closePath();
    const steel = ctx.createRadialGradient(-r * 0.35, -r * 0.35, r * 0.1, 0, 0, r * 1.28);
    steel.addColorStop(0, '#d2d5da');
    steel.addColorStop(0.45, '#9296a0');
    steel.addColorStop(0.8, '#5c606a');
    steel.addColorStop(1, '#3a3d44');
    ctx.fillStyle = steel;
    ctx.fill();
    ctx.lineJoin = 'miter';
    ctx.strokeStyle = '#2c2f35';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Hub + bolt
    ctx.beginPath();
    ctx.arc(0, 0, r * 0.52, 0, TAU);
    ctx.fillStyle = '#41444b';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(0, 0, r * 0.18, 0, TAU);
    ctx.fillStyle = '#1b1d21';
    ctx.fill();
    ctx.restore();
  }

  function drawStone(x, y, r, color) {
    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.3)';
    ctx.shadowBlur = 8;
    ctx.shadowOffsetY = 3;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, TAU);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();
    ctx.beginPath();
    ctx.arc(x - r * 0.25, y - r * 0.28, r * 0.35, 0, TAU);
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.fill();
  }

  function frame(now) {
    if (!running) return;
    const t = (now - start) / 1000;
    const w = canvas.width, h = canvas.height;

    // Maple wood background (zone 4 tone) with faint grain
    ctx.fillStyle = '#D4A860';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(224,184,112,0.35)';
    ctx.lineWidth = 1;
    for (let y = 0; y < h; y += 22) {
      const off = Math.sin((y + t * 4) * 0.05) * 6;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.bezierCurveTo(w * 0.33, y + off, w * 0.66, y - off, w, y + off * 0.4);
      ctx.stroke();
    }
    // Darken toward edges so the centred card pops
    const vg = ctx.createRadialGradient(w / 2, h / 2, h * 0.2, w / 2, h / 2, h * 0.75);
    vg.addColorStop(0, 'rgba(0,0,0,0)');
    vg.addColorStop(1, 'rgba(0,0,0,0.45)');
    ctx.fillStyle = vg;
    ctx.fillRect(0, 0, w, h);

    const cx = w / 2, cy = h / 2;
    drawGear(cx, cy, Math.min(w, h) * 0.09, t * 0.5);

    const orbit = Math.min(w, h) * 0.22;
    for (let i = 0; i < 3; i++) {
      const a = t * 0.4 + (i / 3) * TAU;
      const x = cx + Math.cos(a) * orbit;
      const y = cy + Math.sin(a) * orbit * 0.6; // slight perspective squash
      drawStone(x, y, 22, STONE_COLORS[i]);
    }

    raf = requestAnimationFrame(frame);
  }
  raf = requestAnimationFrame(frame);

  return function stop() {
    running = false;
    cancelAnimationFrame(raf);
    window.removeEventListener('resize', resize);
  };
}
