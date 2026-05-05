import http from 'node:http';
import { GameEngine } from '../engine/GameEngine.js';
import { CONFIG } from '../config.js';

const PORT       = Number(process.env.PORT       ?? 7777);
const NUM_AGENTS = Number(process.env.NUM_AGENTS  ?? 1);
const NUM_BOTS   = Number(process.env.NUM_BOTS    ?? 0);

const OBS_SIZE = 62;

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
/** stoneId -> obs[] rolling buffer for {OBS_DELAYxACTION_REPEAT}-step perception delay */
const obsBuffers = new Map();
/** stoneId -> [[dx,dy,boost]×OBS_DELAYxACTION_REPEAT] rolling action history */
const actionBuffers = new Map();
const OBS_DELAY = 3;
const ACTION_REPEAT = 3;

// ---------------------------------------------------------------------------
// Observation builder
// ---------------------------------------------------------------------------

function buildObs(stoneId) {
  const obs = new Array(OBS_SIZE).fill(0);
  const stone = engine.stones.get(stoneId);
  if (!stone || !stone.alive) return obs;

  const { x, y, vx, vy, radius } = stone;
  const { MAP_WIDTH, MAP_HEIGHT, MAX_SPEED } = config;

  // [0-3] log distances to map edges (normalized by log of map dimensions)
  const logW = Math.log(MAP_WIDTH);
  const logH = Math.log(MAP_HEIGHT);
  obs[0] = Math.log(Math.max(0, Math.min(x - radius * 0.5, 300)) + 1);
  obs[1] = Math.log(Math.max(0, Math.min(MAP_WIDTH - x - radius * 0.5, 300)) + 1);
  obs[2] = Math.log(Math.max(0, Math.min(y - radius * 0.5, 300)) + 1);
  obs[3] = Math.log(Math.max(0, Math.min(MAP_HEIGHT - y - radius * 0.5, 300)) + 1);

  // [4-7] self
  obs[4] = Math.log(Math.abs(vx) + 1) * Math.sign(vx);
  obs[5] = Math.log(Math.abs(vy) + 1) * Math.sign(vy);
  obs[6] = Math.log(Math.abs(radius) + 1);
  obs[7] = Math.floor((MAP_HEIGHT - y) / MAP_HEIGHT * 5);

  // [8-27] 5 nearest fragments (dx, dy, area, dist)
  const nearFrags = engine.getFragmentsNear(x, y)
    .sort((a, b) => (a.x - x) ** 2 + (a.y - y) ** 2 - ((b.x - x) ** 2 + (b.y - y) ** 2));
  for (let i = 0; i < 5 && i < nearFrags.length; i++) {
    const f = nearFrags[i];
    const base = 8 + i * 4;
    obs[base]     = Math.log(Math.abs(f.x - x) + 1) * Math.sign(f.x - x);
    obs[base + 1] = Math.log(Math.abs(f.y - y) + 1) * Math.sign(f.y - y);
    obs[base + 2] = Math.log(f.area + 1);
    obs[base + 3] = Math.log(Math.max(0, Math.hypot(f.x - x, f.y - y) - radius) + 1);
  }

  // [28-51] 4 nearest other alive stones (dx, dy, radius_ratio, dvx, dvy, dist)
  const nearStones = [...engine.stones.values()]
    .filter(s => s.id !== stoneId && s.alive)
    .sort((a, b) => Math.hypot(a.x - x, a.y - y) - Math.hypot(b.x - x, b.y - y));
  for (let i = 0; i < 4 && i < nearStones.length; i++) {
    const s = nearStones[i];
    const base = 28 + i * 6;
    obs[base]     = Math.log(Math.abs(s.x - x) + 1) * Math.sign(s.x - x);
    obs[base + 1] = Math.log(Math.abs(s.y - y) + 1) * Math.sign(s.y - y);
    obs[base + 2] = Math.log(s.radius / radius);
    obs[base + 3] = Math.log(Math.abs(s.vx - vx) + 1) * Math.sign(s.vx - vx);
    obs[base + 4] = Math.log(Math.abs(s.vy - vy) + 1) * Math.sign(s.vy - vy);
    obs[base + 5] = Math.log(Math.max(0, Math.hypot(s.x - x, s.y - y) - s.radius - radius) + 1);
  }

  // [52-60] 3 nearest gears
  const nearGears = [...engine.gears]
    .sort((a, b) => Math.hypot(a.x - x, a.y - y) - Math.hypot(b.x - x, b.y - y));
  for (let i = 0; i < 3 && i < nearGears.length; i++) {
    const g = nearGears[i];
    const base = 52 + i * 3;
    obs[base]     = Math.log(Math.abs(g.x - x) + 1) * Math.sign(g.x - x);
    obs[base + 1] = Math.log(Math.abs(g.y - y) + 1) * Math.sign(g.y - y);
    obs[base + 2] = Math.log(Math.max(0, Math.hypot(x - g.x, y - g.y) - radius - g.collisionRadius) + 1);
  }

  // [61] spawn invincibility flag
  obs[61] = engine._totalTime < stone.invincibleUntil ? 1.0 : 0.0;

  return obs;
}

function buildObsAll() {
  const observations = {};
  for (let i = 0; i < agentIds.length; i++) {
    const id = agentIds[i];
    const actHistory = (actionBuffers.get(id) ?? []).flat();
    observations[`agent_${i}`] = buildObs(id).concat(actHistory);
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

  obsBuffers.clear();
  actionBuffers.clear();
  for (const id of agentIds) {
    const stone = engine.stones.get(id);
    prevAreas.set(id, stone ? stone.area : 0);
    prevDirs.set(id, { dx: 0, dy: 0 });
    const initObs = buildObs(id);
    obsBuffers.set(id, Array.from({ length: OBS_DELAY * ACTION_REPEAT }, () => initObs.slice()));
    actionBuffers.set(id, Array.from({ length: OBS_DELAY * ACTION_REPEAT }, () => [0, 0, 0]));
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
          const abuf = actionBuffers.get(agentIds[i]) ?? [];
          abuf.push([dx, dy, boostVal > 0.5 ? 1.0 : 0.0]);
          abuf.shift();
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
            reward = -10.0;
          } else if (alive) {
            reward = Math.sign(currArea - prevArea) * Math.log(Math.abs(currArea - prevArea) + 1) * 0.6;

            // Penalty: direction change proportional to Euclidean distance between actions
            const prev = prevDirs.get(id) ?? { dx: 0, dy: 0 };
            const curr = currDirs.get(id) ?? { dx: 0, dy: 0 };
            const dirDist = Math.hypot(curr.dx - prev.dx, curr.dy - prev.dy);
            reward -= dirDist * 0.002;

            // Penalty: idle (nearly stationary)
            if (Math.hypot(stone.vx, stone.vy) < 1.0) reward -= 0.01;
          } else {
            reward = 0.0;
          }

          if (died) {
            obsBuffers.set(id, Array.from({ length: OBS_DELAY * ACTION_REPEAT }, () => new Array(OBS_SIZE).fill(0)));
            actionBuffers.set(id, Array.from({ length: OBS_DELAY * ACTION_REPEAT }, () => [0, 0, 0]));
          }
          const buf = obsBuffers.get(id) ?? [];
          buf.push(buildObs(id));
          const actHistory = (actionBuffers.get(id) ?? []).flat();
          observations[key] = buf.shift().concat(actHistory);
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
