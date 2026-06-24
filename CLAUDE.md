# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`stone.io` — browser-based multiplayer .io game inspired by Korean 알까기 (flicking marbles). Stones accelerate toward the mouse cursor, collide with each other and spinning gears, and absorb fragments to grow. The player can fire a short **boost** (z key) toward the cursor. The goal is to knock rival stones into gears or off the board, then absorb the fragments they drop to grow larger. The engine is intentionally decoupled from rendering so it can run headless for reinforcement learning.

## Commands

```bash
npm install                          # install dependencies (only vite)
npm run dev                          # Vite dev server (browser client)
npm run build                        # production bundle
npm test                             # run engine tests with Node's built-in runner (Node ≥18)
node --test tests/engine.test.js     # same, explicit path

# Headless RL server (one process per training env, auto-assigned port)
PORT=7777 NUM_AGENTS=1 NUM_BOTS=0 node src/headless/HeadlessServer.js

# Reinforcement learning (from rl/)
pip install -r rl/requirements.txt
python rl/train.py                   # PPO self-play; spawns Node servers per env
python rl/eval.py                    # evaluate a checkpoint
python rl/export_bot.py              # export trained weights → bot.json for the browser client
```

## Architecture

```
src/
  config.js              # single CONFIG export — all tunable numbers live here
  main.js                # Vite entry point (rendering hooks go here)
  Renderer.js            # canvas rendering: pre-rendered per-zone wood textures, marble stones,
                         #   routes engine events → particles + sound
  Camera.js              # viewport follow
  Minimap.js             # zone/position minimap
  ConfigPanel.js         # live CONFIG tweaking UI
  DebugOverlay.js        # debug HUD
  ParticleSystem.js      # cosmetic effects (absorb/death/collision/zoneEntry); world-space, capped 200
  LobbyPreview.js        # animated lobby backdrop (stones orbiting a gear)
  audio/
    SoundEngine.js       # procedural Web Audio SFX (no files); resume() needs a user gesture
  engine/
    GameEngine.js        # simulation loop; no DOM, safe to import in Node
    Physics.js           # pure functions: applyAcceleration, resolveStoneCollision,
                         #   checkGearCollision, resolveWallCollision
    MapGenerator.js      # deterministic Poisson disk gear placement (seeded xorshift32)
    entities/
      Stone.js           # player entity; area getter, absorb(); groggy + invincibility state
      Gear.js            # spinning obstacle; collisionRadius = radius*1.05
      Fragment.js        # collectible debris; fixed area, decays via ttl
  bots/
    RuleBasedBot.js      # heuristic bot
    TrainedBot.js        # runs exported policy weights (bot.json)
    bot.json             # exported trained policy weights
  headless/
    HeadlessServer.js    # HTTP wrapper around GameEngine for RL (obs/reward builder lives here)
tests/
  engine.test.js         # Node built-in test runner (node:test)
rl/                       # PyTorch PPO self-play training pipeline (see "Reinforcement learning")
```

### Engine contract

`GameEngine` is the only stateful object. Typical headless usage:

```js
import { GameEngine } from './src/engine/GameEngine.js';
import { CONFIG } from './src/config.js';

const engine = new GameEngine(CONFIG);
const stoneId = engine.addPlayer(id, nickname);

// each frame / RL step:
engine.setInput(stoneId, mouseX, mouseY, vpW, vpH); // store intent (cursor relative to viewport)
engine.boost(stoneId);                               // optional: fire boost toward cursor
const state = engine.step(deltaMs);                  // advance + return snapshot
```

`step()` returns a plain-object snapshot (`{ stones, gears, fragments, events, tick }`) — no class
instances, safe to serialize for network or RL observation vectors. `events` carries per-step
notifications for rendering/audio hooks: `boost` `{stoneId}`, `collision` `{x,y}`,
`absorb` `{x,y,color}`, `death` `{stoneId,x,y,color,cause}` (cause: `'gear'` | `'wall'`).

Bots are owned by the engine (`addBot` / `setBotType`) and decide using the **previous** frame's
state (1-tick perception delay). `setBotType('trained', weightsJson)` swaps all bots to the exported
policy.

### Map layout

The map is **8000×16000**. Five horizontal zones stack top-to-bottom; zone 0 (top) has the largest,
fastest gears (few of them) and zone 4 (bottom) the smallest, slowest gears (many of them). Players
always spawn in zone 4. `MapGenerator` uses a seeded xorshift32 so `reset()` always produces the same
gear layout. Per-zone gear radius/spacing/count/rpm live in `CONFIG.ZONES`.

### Physics notes

- Mass proxy: `area = π·r²`; used in the impulse formula — larger stones transfer less velocity.
- `applyAcceleration`: acceleration scales **up** with radius (`1 + MASS_ACCEL_FACTOR*radius`), ramps
  in over `MAX_ACCEL_RADIUS` past the dead zone, and is suppressed while groggy.
- `resolveStoneCollision` applies a velocity impulse (restitution scales with combined mass),
  positional correction (prevents overlap tunnelling), and sets a **groggy** value on both stones
  proportional to impulse. Groggy temporarily weakens acceleration and reduces friction recovery,
  decaying by `GROGGY_COUNTDOWN` per frame.
- Friction (`FRICTION = 0.8`) is applied per-frame, but is velocity-dependent (faster stones keep
  more of their speed) and modulated by the groggy state.
- Velocity is hard-capped at `MAX_SPEED`.
- Dead zone: cursor must be `DEAD_ZONE_RADIUS` pixels from viewport centre before acceleration applies.
- Boost (z key): instant impulse toward the cursor, costs `BOOST_AREA_COST` area, on `BOOST_COOLDOWN`,
  and is refused while the stone is at its initial radius.
- Death: a stone dies on gear contact (`checkGearCollision`) or when its centre-ish point crosses a
  wall. On death it scatters fragments (count ∝ √radius) and respawns after 2s with ~1.5s invincibility.
- Gear collisions skip stones that are still spawn-invincible.

### Fragment spatial grid

Fragments are bucketed into a `FRAG_CELL_SIZE` grid by spawn position (never re-indexed — death
fragments drift only ~30 units before friction stops them). `getFragmentsNear(x, y)` returns the 3×3
cell neighbourhood, so stone↔fragment absorption is local rather than O(stones·fragments). The RL
observation builder reuses this for nearby-fragment features.

## Reinforcement learning (`rl/`)

PPO **self-play** with a **distributional (quantile) critic** and GAE. Active research area —
recent work has focused on making bots play more like humans (reward shaping, observation design).

```
rl/
  train.py               # main loop: ParallelEnv workers, self-play opponent pool, PPO update
  ppo.py                 # PPOAgent, RolloutBuffer, quantile-Huber loss, distributional GAE returns
  network.py             # Actor (shared MLP→move Normal + boost Bernoulli) / Critic (51 quantiles)
  stone_env/
    env.py               # Gymnasium wrapper (single- or multi-agent); OBS_SIZE = 110, action = 3
    bridge.py            # HTTP client to the Node HeadlessServer
  eval.py                # checkpoint evaluation
  export_bot.py          # checkpoint → bot.json for TrainedBot
  checkpoints/           # ckpt_*.pt + final.pt
```

- **Observation (110-dim)**, built in `HeadlessServer.js::buildObs`: log distances to the 4 map
  edges, own velocity/radius/zone, 5 nearest fragments (3 small + 2 large), 6 nearest other stones,
  15 nearest gears, and a spawn-invincibility flag. Distances are log-compressed and signed.
- **Action (3-dim)** `[dx, dy, boost]`: move direction (Normal, tanh-squashed mean, clamped to
  ±1, mapped to a cursor 120px from centre) plus a Bernoulli boost gate (fires when > 0.5).
- **Reward** (`HeadlessServer.js::/step`): log-scaled area gain, small speed bonus, an anti-camp
  penalty (distance from position 100 frames ago), and −1000 on death; clamped to ±10.
- **Training**: `ACTION_REPEAT = 3`, opponent pool of 5 snapshots refreshed every 5 updates,
  parallel env workers each running their own Node `HeadlessServer`. Logs to Weights & Biases.

> The observation/reward logic lives in **HeadlessServer.js**, not in Python — change it there. Keep
> `OBS_SIZE` in `env.py` and `HeadlessServer.js`, and `OBS_DIM` in `train.py`/`ppo.py`, in sync.
