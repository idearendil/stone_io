# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`stone.io` — browser-based multiplayer .io game inspired by Korean 알까기 (flicking marbles). Stones accelerate toward the mouse cursor, collide with each other and spinning gears, and absorb fragments to grow. The engine is intentionally decoupled from rendering so it can run headless for reinforcement learning.

## Commands

```bash
npm install                          # install dependencies (only vite)
npm run dev                          # Vite dev server (browser client)
npm run build                        # production bundle
npm test                             # run engine tests with Node's built-in runner (Node ≥18)
node --test tests/engine.test.js     # same, explicit path
```

## Architecture

```
src/
  config.js              # single CONFIG export — all tunable numbers live here
  main.js                # Vite entry point (rendering hooks go here)
  engine/
    GameEngine.js        # simulation loop; no DOM, safe to import in Node
    Physics.js           # pure functions: applyAcceleration, resolveStoneCollision,
                         #   checkGearCollision, resolveWallCollision
    MapGenerator.js      # deterministic Poisson disk gear placement (seeded xorshift32)
    entities/
      Stone.js           # player entity; area getter, absorb()
      Gear.js            # spinning obstacle; collisionRadius = radius*1.05
      Fragment.js        # collectible debris; fixed area, decays via ttl
tests/
  engine.test.js         # Node built-in test runner (node:test)
```

### Engine contract

`GameEngine` is the only stateful object. Typical headless usage:

```js
import { GameEngine } from './src/engine/GameEngine.js';
import { CONFIG } from './src/config.js';

const engine = new GameEngine(CONFIG);
const stoneId = engine.addPlayer(id, nickname);

// each frame / RL step:
engine.setInput(stoneId, mouseX, mouseY, vpW, vpH); // store intent
const state = engine.step(deltaMs);                  // advance + return snapshot
```

`step()` returns a plain-object snapshot (`{ stones, gears, fragments, tick }`) — no class instances, safe to serialize for network or RL observation vectors.

### Map layout

The map is 4000×8000. Five horizontal zones stack top-to-bottom; zone 0 (top) has the largest, fastest gears and zone 4 (bottom) the smallest, slowest. Players always spawn in zone 4. `MapGenerator` uses a seeded xorshift32 so `reset()` always produces the same gear layout.

### Physics notes

- Mass proxy: `area = π·r²`; used in impulse formula — larger stones transfer less velocity.
- `resolveStoneCollision` applies both velocity impulse and positional correction (prevents overlap tunnelling).
- Friction (`FRICTION = 0.98`) is applied per-frame to velocity before movement.
- Dead zone: mouse must be `DEAD_ZONE_RADIUS` pixels from viewport centre before acceleration applies.
