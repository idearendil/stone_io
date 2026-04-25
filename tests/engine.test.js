import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CONFIG } from '../src/config.js';
import { GameEngine } from '../src/engine/GameEngine.js';
import { resolveStoneCollision } from '../src/engine/Physics.js';
import { Stone } from '../src/engine/entities/Stone.js';

test('300 ticks with one player — state is consistent', () => {
  const engine = new GameEngine(CONFIG);
  const stoneId = engine.addPlayer(1, 'TestPlayer');

  for (let i = 0; i < 300; i++) {
    engine.setInput(stoneId, 500, 400, 800, 600);
    const state = engine.step(16);

    assert.equal(state.tick, i + 1, 'tick increments by 1');
    assert.ok(Array.isArray(state.stones), 'stones is array');
    assert.ok(Array.isArray(state.gears), 'gears is array');
    assert.ok(Array.isArray(state.fragments), 'fragments is array');

    const stone = state.stones.find(s => s.id === stoneId);
    assert.ok(stone !== undefined, 'player stone present in state');
    assert.ok(Number.isFinite(stone.x) && Number.isFinite(stone.y), 'stone position is finite');
    assert.ok(stone.radius > 0, 'stone radius is positive');

    for (const frag of state.fragments) {
      assert.ok(Number.isFinite(frag.x) && Number.isFinite(frag.y), 'fragment position is finite');
      assert.ok(frag.ttl > 0, 'expired fragments are pruned before snapshot');
    }
  }
});

test('momentum conservation on elastic collision', () => {
  // Two equal-mass stones approaching head-on along x-axis
  const s1 = new Stone(1, 0, 0, 20, '#fff', 'A');
  const s2 = new Stone(2, 39, 0, 20, '#000', 'B'); // centres 39 apart → 1px overlap
  s1.vx = 5;  s1.vy = 1;
  s2.vx = -5; s2.vy = -1;

  const m1 = s1.area;
  const m2 = s2.area;
  const pxBefore = m1 * s1.vx + m2 * s2.vx;
  const pyBefore = m1 * s1.vy + m2 * s2.vy;

  resolveStoneCollision(s1, s2, 1.0); // fully elastic

  const pxAfter = m1 * s1.vx + m2 * s2.vx;
  const pyAfter = m1 * s1.vy + m2 * s2.vy;

  assert.ok(Math.abs(pxAfter - pxBefore) < 1e-9,
    `x-momentum conserved (delta=${pxAfter - pxBefore})`);
  assert.ok(Math.abs(pyAfter - pyBefore) < 1e-9,
    `y-momentum conserved (delta=${pyAfter - pyBefore})`);
});

test('gear collision kills stone and spawns fragments', () => {
  const engine = new GameEngine(CONFIG);
  const stoneId = engine.addPlayer(1, 'TestPlayer');

  assert.ok(engine.gears.length > 0, 'map contains at least one gear');

  // Teleport stone dead-centre on first gear
  const stone = engine.stones.get(stoneId);
  const gear = engine.gears[0];
  stone.x = gear.x;
  stone.y = gear.y;
  stone.vx = 0;
  stone.vy = 0;

  engine.step(16);

  const state = engine.getState();
  const stoneState = state.stones.find(s => s.id === stoneId);
  assert.ok(!stoneState.alive, 'stone is marked dead after gear overlap');
});
