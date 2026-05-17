import http from 'node:http';
import { GameEngine } from '../engine/GameEngine.js';
import { CONFIG } from '../config.js';

const PORT       = Number(process.env.PORT       ?? 7777);
const NUM_AGENTS = Number(process.env.NUM_AGENTS  ?? 1);
const NUM_BOTS   = Number(process.env.NUM_BOTS    ?? 0);

const VEC_SIZE    = 5;
const IMG_CHANNELS = 8;
const IMG_SIZE    = 32;
const BASE_RADIUS = 16;
const ZOOM_MIN    = 0.20;
const ZOOM_MAX    = 1.00;
const VP          = 200;

// Mutable config shared with engine (engine stores the same reference)
const config = { ...CONFIG, ZONES: CONFIG.ZONES };

/** @type {GameEngine|null} */
let engine = null;
/** @type {number[]} stoneIds for the RL agents */
let agentIds = [];
/** stoneId -> area at start of current step */
const prevAreas = new Map();
/** stoneId -> {dx, dy} direction from previous step */
const prevDirs = new Map();

// ---------------------------------------------------------------------------
// Observation builder
// ---------------------------------------------------------------------------

function getZoom(r) {
  return Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, BASE_RADIUS * 4 / (r + BASE_RADIUS * 3)));
}

function buildObs(stoneId) {
  const stone = engine.stones.get(stoneId);
  const vec = new Array(VEC_SIZE).fill(0);
  const img = new Float32Array(IMG_CHANNELS * IMG_SIZE * IMG_SIZE);

  if (!stone || !stone.alive) return { vec, img };

  const { x, y, vx, vy, radius } = stone;
  const { MAP_WIDTH, MAP_HEIGHT } = config;

  // vec[0-4]: vx, vy, log(radius), zone_index, invincibility_flag
  vec[0] = Math.log(Math.abs(vx) + 1) * Math.sign(vx);
  vec[1] = Math.log(Math.abs(vy) + 1) * Math.sign(vy);
  vec[2] = Math.log(radius + 1);
  vec[3] = Math.floor((MAP_HEIGHT - y) / MAP_HEIGHT * 5);
  vec[4] = engine._totalTime < stone.invincibleUntil ? 1.0 : 0.0;

  // Viewport/pixel mapping
  const zoom       = getZoom(radius);
  const pixelScale = zoom * IMG_SIZE / VP;   // world units → image pixels

  function worldToPixel(wx, wy) {
    const px = Math.floor(IMG_SIZE / 2 + (wx - x) * pixelScale);
    const py = Math.floor(IMG_SIZE / 2 + (wy - y) * pixelScale);
    return { px, py };
  }
  function inBounds(px, py) { return px >= 0 && px < IMG_SIZE && py >= 0 && py < IMG_SIZE; }
  function setPixel(ch, px, py, val) { img[ch * IMG_SIZE * IMG_SIZE + py * IMG_SIZE + px] = val; }

  // Channel 0 (fragment area) + Channel 1 (fragment dist): 3 small + 2 large fragments
  const allFrags = engine.getFragmentsNear(x, y)
    .sort((a, b) => (a.x - x) ** 2 + (a.y - y) ** 2 - ((b.x - x) ** 2 + (b.y - y) ** 2));
  const fragSlots = [
    ...allFrags.filter(f => f.radius <= 8).slice(0, 3),
    ...allFrags.filter(f => f.radius >= 9).slice(0, 2),
  ];
  for (const f of fragSlots) {
    const { px, py } = worldToPixel(f.x, f.y);
    if (!inBounds(px, py)) continue;
    setPixel(0, px, py, Math.log(f.area + 1));
    setPixel(1, px, py, Math.log(Math.max(0, Math.hypot(f.x - x, f.y - y) - radius) + 1));
  }

  // Channels 2-5: all visible stones (ch2=radius_ratio, ch3=dvx, ch4=dvy, ch5=dist)
  for (const s of engine.stones.values()) {
    if (s.id === stoneId || !s.alive) continue;
    const { px, py } = worldToPixel(s.x, s.y);
    if (!inBounds(px, py)) continue;
    setPixel(2, px, py, Math.log(s.radius / radius));
    setPixel(3, px, py, Math.log(Math.abs(s.vx - vx) + 1) * Math.sign(s.vx - vx));
    setPixel(4, px, py, Math.log(Math.abs(s.vy - vy) + 1) * Math.sign(s.vy - vy));
    setPixel(5, px, py, Math.log(Math.max(0, Math.hypot(s.x - x, s.y - y) - s.radius - radius) + 1));
  }

  // Channel 6: gear distance (sparse, center pixel only)
  for (const g of engine.gears) {
    const { px, py } = worldToPixel(g.x, g.y);
    if (!inBounds(px, py)) continue;
    const dist = Math.max(0.01, Math.hypot(x - g.x, y - g.y) - radius - g.collisionRadius);
    setPixel(6, px, py, Math.log(dist + 1));
  }

  // Channel 7: danger zone (dense)
  // Map boundary strips
  const leftEnd   = Math.ceil(IMG_SIZE / 2 - x * pixelScale);
  const rightStart = Math.floor(IMG_SIZE / 2 + (MAP_WIDTH - x) * pixelScale);
  const topEnd    = Math.ceil(IMG_SIZE / 2 - y * pixelScale);
  const botStart  = Math.floor(IMG_SIZE / 2 + (MAP_HEIGHT - y) * pixelScale);
  const ch7base   = 7 * IMG_SIZE * IMG_SIZE;
  for (let py = 0; py < IMG_SIZE; py++) {
    for (let px = 0; px < IMG_SIZE; px++) {
      if (px < leftEnd || px >= rightStart || py < topEnd || py >= botStart) {
        img[ch7base + py * IMG_SIZE + px] = 1.0;
      }
    }
  }
  // Gear danger circles
  for (const g of engine.gears) {
    const dangerPxR = (radius + g.collisionRadius) * pixelScale;
    const { px: cx, py: cy } = worldToPixel(g.x, g.y);
    const minPx = Math.max(0,         Math.floor(cx - dangerPxR));
    const maxPx = Math.min(IMG_SIZE - 1, Math.ceil(cx  + dangerPxR));
    const minPy = Math.max(0,         Math.floor(cy - dangerPxR));
    const maxPy = Math.min(IMG_SIZE - 1, Math.ceil(cy  + dangerPxR));
    for (let py = minPy; py <= maxPy; py++) {
      for (let px = minPx; px <= maxPx; px++) {
        if ((px - cx) ** 2 + (py - cy) ** 2 <= dangerPxR ** 2) {
          img[ch7base + py * IMG_SIZE + px] = 1.0;
        }
      }
    }
  }

  return { vec, img };
}

function buildObsAll() {
  const observations = {};
  for (let i = 0; i < agentIds.length; i++) {
    const { vec, img } = buildObs(agentIds[i]);
    observations[`agent_${i}`] = { vec: Array.from(vec), img: Array.from(img) };
  }
  return observations;
}

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------

function resetEngine() {
  engine = new GameEngine(config);
  agentIds = [];
  prevAreas.clear();

  for (let i = 0; i < NUM_AGENTS; i++) {
    const id = engine.addPlayer(null, `rl_${i}`);
    agentIds.push(id);
  }
  for (let i = 0; i < NUM_BOTS; i++) {
    engine.addBot(`bot_${i}`);
  }

  // Pre-populate fragments to steady-state density
  const { FRAGMENT_LIFETIME, SPAWN_INTERVAL, MAX_FRAGMENT_SPAWN, MIN_FRAGMENT_SPAWN } = config;
  const initCount = Math.floor(
    FRAGMENT_LIFETIME / SPAWN_INTERVAL * (MAX_FRAGMENT_SPAWN + MIN_FRAGMENT_SPAWN) / 2
  );
  for (let i = 0; i < initCount; i++) engine._spawnInitialFragments();

  for (const id of agentIds) {
    const stone = engine.stones.get(id);
    prevAreas.set(id, stone ? stone.area : 0);
    prevDirs.set(id, { dx: 0, dy: 0 });
  }
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

const server = http.createServer((req, res) => {
  res.setHeader('Content-Type', 'application/json');

  let body = '';
  req.on('data', chunk => { body += chunk; });
  req.on('end', () => {
    try {
      const { method } = req;
      const path = new URL(req.url, `http://localhost`).pathname;

      if (method === 'GET' && path === '/ping') {
        res.writeHead(200);
        res.end('{"ok":true}');

      } else if (method === 'POST' && path === '/reset') {
        resetEngine();
        res.writeHead(200);
        res.end(JSON.stringify({ observations: buildObsAll(), info: {} }));

      } else if (method === 'POST' && path === '/step') {
        if (!engine) { res.writeHead(400); res.end('{"error":"call /reset first"}'); return; }

        const { actions } = JSON.parse(body);
        const VP = 200;

        // Snapshot alive states before step
        const wasAlive = new Map();
        for (const id of agentIds) {
          const s = engine.stones.get(id);
          wasAlive.set(id, s ? s.alive : false);
        }

        // Apply actions: [dir_x, dir_y, boost] → setInput + optional boost
        const currDirs = new Map();
        for (let i = 0; i < agentIds.length; i++) {
          const key = `agent_${i}`;
          const act = actions[key] ?? [0, 0, 0];
          const [dx, dy, boostVal] = act;
          currDirs.set(agentIds[i], { dx, dy });
          engine.setInput(agentIds[i], VP / 2 + dx * 120, VP / 2 + dy * 120, VP, VP);
          if (boostVal > 0.5) engine.boost(agentIds[i]);
        }

        // Random step 15–18 ms (mirrors real browser variance)
        const deltaMs = 15 + Math.random() * 3;
        engine.step(deltaMs);

        // Build response
        const observations = {};
        const rewards      = {};
        const terminated   = {};
        const truncated    = {};

        for (let i = 0; i < agentIds.length; i++) {
          const key  = `agent_${i}`;
          const id   = agentIds[i];
          const stone = engine.stones.get(id);
          const alive = stone ? stone.alive : false;
          const prevArea = prevAreas.get(id) ?? 0;
          const currArea = stone ? stone.area : 0;
          const died = wasAlive.get(id) && !alive;

          let reward;
          if (died) {
            reward = -1000.0;
          } else if (alive) {
            reward = Math.sign(currArea - prevArea) * Math.log(Math.abs(currArea - prevArea) + 1) * 0.4;

            // Penalty: direction change proportional to Euclidean distance between actions
            // Compensate for huge velocity
            // const prev = prevDirs.get(id) ?? { dx: 0, dy: 0 };
            // const curr = currDirs.get(id) ?? { dx: 0, dy: 0 };
            // const dirDist = Math.hypot(curr.dx - prev.dx, curr.dy - prev.dy);
            // const velocity = Math.min(1, Math.hypot(curr.dx, curr.dy));
            // reward += ((velocity - 0.8) * 0.0 - dirDist * 0.0);
            const velocity = Math.hypot(stone.vx, stone.vy);
            reward += Math.log(velocity + 1) * 0.01;
          } else {
            reward = 0.0;
          }

          const { vec: ov, img: oi } = buildObs(id);
          observations[key] = { vec: Array.from(ov), img: Array.from(oi) };
          rewards[key]      = Math.max(-10, Math.min(10, reward));
          terminated[key]   = died;
          truncated[key]    = false;

          prevAreas.set(id, alive ? currArea : 0);
          prevDirs.set(id, alive ? (currDirs.get(id) ?? { dx: 0, dy: 0 }) : { dx: 0, dy: 0 });
        }

        res.writeHead(200);
        res.end(JSON.stringify({ observations, rewards, terminated, truncated, info: {} }));

      } else if (method === 'GET' && path === '/state') {
        if (!engine) { res.writeHead(400); res.end('{"error":"call /reset first"}'); return; }
        res.writeHead(200);
        res.end(JSON.stringify(engine.getState()));

      } else if (method === 'GET' && path === '/radii') {
        if (!engine) { res.writeHead(400); res.end('{"error":"call /reset first"}'); return; }
        const radii = [];
        for (const stone of engine.stones.values()) {
          if (stone.alive) radii.push(stone.radius);
        }
        res.writeHead(200);
        res.end(JSON.stringify({ radii }));

      } else if (method === 'POST' && path === '/config') {
        const partial = JSON.parse(body);
        Object.assign(config, partial);
        if (engine) engine.updateConfig(partial);
        res.writeHead(200);
        res.end('{"ok":true}');

      } else {
        res.writeHead(404);
        res.end('{"error":"not found"}');
      }
    } catch (err) {
      res.writeHead(500);
      res.end(JSON.stringify({ error: String(err) }));
    }
  });
});

server.listen(PORT, '127.0.0.1', () => {
  process.stdout.write(`HeadlessServer listening on 127.0.0.1:${PORT}\n`);
});
