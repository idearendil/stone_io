import http from 'node:http';
import { GameEngine } from '../engine/GameEngine.js';
import { CONFIG } from '../config.js';

const PORT       = Number(process.env.PORT       ?? 7777);
const NUM_AGENTS = Number(process.env.NUM_AGENTS  ?? 1);
const NUM_BOTS   = Number(process.env.NUM_BOTS    ?? 0);

const OBS_SIZE = 42;

// Mutable config shared with engine (engine stores the same reference)
const config = { ...CONFIG, ZONES: CONFIG.ZONES };

/** @type {GameEngine|null} */
let engine = null;
/** @type {number[]} stoneIds for the RL agents */
let agentIds = [];
/** stoneId -> area at start of current step */
const prevAreas = new Map();

// ---------------------------------------------------------------------------
// Observation builder
// ---------------------------------------------------------------------------

function buildObs(stoneId) {
  const obs = new Array(OBS_SIZE).fill(0);
  const stone = engine.stones.get(stoneId);
  if (!stone || !stone.alive) return obs;

  const { x, y, vx, vy, radius } = stone;
  const { MAP_WIDTH, MAP_HEIGHT, MAX_SPEED } = config;

  // [0-5] self
  obs[0] = x / MAP_WIDTH;
  obs[1] = y / MAP_HEIGHT;
  obs[2] = vx / MAX_SPEED;
  obs[3] = vy / MAX_SPEED;
  obs[4] = radius / 1000;
  obs[5] = 1.0;

  // [6-20] 5 nearest fragments (from grid neighbourhood — same as bot logic)
  const nearFrags = engine.getFragmentsNear(x, y)
    .sort((a, b) => (a.x - x) ** 2 + (a.y - y) ** 2 - ((b.x - x) ** 2 + (b.y - y) ** 2));
  for (let i = 0; i < 5 && i < nearFrags.length; i++) {
    const f = nearFrags[i];
    const base = 6 + i * 3;
    obs[base]     = (f.x - x) / 1200;
    obs[base + 1] = (f.y - y) / 1200;
    obs[base + 2] = f.area   / 200;
  }

  // [21-32] 4 nearest other alive stones
  const nearStones = [...engine.stones.values()]
    .filter(s => s.id !== stoneId && s.alive)
    .sort((a, b) => Math.hypot(a.x - x, a.y - y) - Math.hypot(b.x - x, b.y - y));
  for (let i = 0; i < 4 && i < nearStones.length; i++) {
    const s = nearStones[i];
    const base = 21 + i * 3;
    obs[base]     = (s.x - x)   / 1200;
    obs[base + 1] = (s.y - y)   / 1200;
    obs[base + 2] = s.radius / radius;
  }

  // [33-41] 3 nearest gears
  const nearGears = [...engine.gears]
    .sort((a, b) => Math.hypot(a.x - x, a.y - y) - Math.hypot(b.x - x, b.y - y));
  for (let i = 0; i < 3 && i < nearGears.length; i++) {
    const g = nearGears[i];
    const base = 33 + i * 3;
    obs[base]     = (g.x - x)            / 1200;
    obs[base + 1] = (g.y - y)            / 1200;
    obs[base + 2] = g.collisionRadius    / 1200;
  }

  return obs;
}

function buildObsAll() {
  const observations = {};
  for (let i = 0; i < agentIds.length; i++) {
    observations[`agent_${i}`] = buildObs(agentIds[i]);
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

        // Apply actions: [dir_x, dir_y] → mouseX/Y in a 200×200 virtual viewport
        for (let i = 0; i < agentIds.length; i++) {
          const key = `agent_${i}`;
          const act = actions[key] ?? [0, 0];
          const [dx, dy] = act;
          engine.setInput(agentIds[i], VP / 2 + dx * 120, VP / 2 + dy * 120, VP, VP);
        }

        // Random step 15–50 ms (mirrors real browser variance)
        const deltaMs = 15 + Math.random() * 35;
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
            reward = -10.0;
          } else if (alive) {
            reward = (currArea - prevArea) * 0.1 + 0.005;
          } else {
            reward = 0.0;
          }

          observations[key] = buildObs(id);
          rewards[key]      = Math.max(-10, Math.min(10, reward));
          terminated[key]   = died;
          truncated[key]    = false;

          prevAreas.set(id, alive ? currArea : 0);
        }

        res.writeHead(200);
        res.end(JSON.stringify({ observations, rewards, terminated, truncated, info: {} }));

      } else if (method === 'GET' && path === '/state') {
        if (!engine) { res.writeHead(400); res.end('{"error":"call /reset first"}'); return; }
        res.writeHead(200);
        res.end(JSON.stringify(engine.getState()));

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
