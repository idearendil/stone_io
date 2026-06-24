import { GameEngine } from './engine/GameEngine.js';
import { Renderer } from './Renderer.js';
import { ConfigPanel } from './ConfigPanel.js';
import { SoundEngine } from './audio/SoundEngine.js';
import { startLobbyPreview } from './LobbyPreview.js';
import { CONFIG } from './config.js';

const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('game'));
const engine = new GameEngine(CONFIG);
const renderer = new Renderer(canvas, CONFIG);
const soundEngine = new SoundEngine();
renderer.setSoundEngine(soundEngine);
const configPanel = new ConfigPanel(
  CONFIG,
  partial => engine.updateConfig(partial),
  (type, weights) => engine.setBotType(type, weights),
);

// Death-cause presentation
const DEATH_CAUSE = {
  gear: { icon: '⚙', label: 'crushed by a gear' },
  wall: { icon: '🧱', label: 'pushed off the board' },
};

function resize() {
  renderer.resize(window.innerWidth, window.innerHeight);
}
resize();
window.addEventListener('resize', resize);

// --- UI elements ---
const lobbyOverlay   = /** @type {HTMLElement} */ (document.getElementById('lobby-overlay'));
const deadOverlay    = /** @type {HTMLElement} */ (document.getElementById('dead-overlay'));
const nicknameInput  = /** @type {HTMLInputElement} */ (document.getElementById('nickname-input'));
const playBtn        = document.getElementById('play-btn');
const deadRadiusEl   = document.getElementById('dead-radius');
const deadRankEl     = document.getElementById('dead-rank');
const deadCountdownEl = document.getElementById('dead-countdown');
const deadIconEl     = document.getElementById('dead-icon');
const deadCauseEl    = document.getElementById('dead-cause');
const soundBtn       = document.getElementById('sound-btn');

// Animated orbiting-stones preview behind the lobby card
const stopLobbyPreview = startLobbyPreview(document.getElementById('lobby-canvas'), CONFIG);

// Speaker toggle (top-right of canvas)
soundBtn.addEventListener('click', () => {
  const muted = soundEngine.toggle();
  soundBtn.textContent = muted ? '🔇' : '🔊';
  soundBtn.classList.toggle('muted', muted);
});

const BOT_NAMES = [];
for(let i=0; i<100; i++) BOT_NAMES.push(i.toString());

let myId = null;
let lastTs = null;
let statsTimer = 0;

// Per-life tracking for the death screen
let peakRadius = 0;
let lastRank = 1;
let wasAlive = false;
let deadAt = 0;

function startGame() {
  soundEngine.resume(); // unlock Web Audio from this user gesture
  const nickname = nicknameInput.value.trim() || 'You';
  myId = engine.addPlayer('p1', nickname);
  for (const name of BOT_NAMES) engine.addBot(name);
  for (let i=0; i<CONFIG.FRAGMENT_LIFETIME / CONFIG.SPAWN_INTERVAL * (CONFIG.MAX_FRAGMENT_SPAWN + CONFIG.MIN_FRAGMENT_SPAWN) / 2; i++) engine._spawnInitialFragments();
  lobbyOverlay.style.display = 'none';
  stopLobbyPreview();
  requestAnimationFrame(loop);
}

playBtn.addEventListener('click', startGame);
nicknameInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') startGame();
});
// Auto-focus nickname input
nicknameInput.focus();

// Mouse → engine input (viewport-relative coords match canvas pixel coords at 1:1 DPR)
canvas.addEventListener('mousemove', e => {
  if (myId === null) return;
  engine.setInput(myId, e.clientX, e.clientY, canvas.width, canvas.height);
});

canvas.addEventListener('touchmove', e => {
  e.preventDefault();
  if (myId === null) return;
  const t = e.touches[0];
  engine.setInput(myId, t.clientX, t.clientY, canvas.width, canvas.height);
}, { passive: false });

// Keyboard shortcuts
window.addEventListener('keydown', e => {
  switch (e.key) {
    case '`':
      renderer.toggleDebug();
      e.preventDefault();
      break;
    case 'Tab':
      configPanel.toggle();
      e.preventDefault();
      break;
    case 'm':
    case 'M':
      renderer.minimap.toggleZoom();
      break;
    case 'z':
    case 'Z':
      if (myId !== null) engine.requestBoost(myId);
      break;
  }
});

function loop(ts) {
  if (lastTs === null) lastTs = ts;
  const delta = Math.min(ts - lastTs, 100); // cap prevents huge jump after tab switch
  lastTs = ts;

  const gameState = engine.step(delta);
  gameState._deltaMS = delta;
  renderer.render(gameState, myId, ts);

  // Dead overlay logic
  const myStone = gameState.stones.find(s => s.id === myId);
  if (myStone) {
    if (myStone.alive) {
      if (myStone.radius > peakRadius) peakRadius = myStone.radius;

      const rank = [...gameState.stones]
        .filter(s => s.alive)
        .sort((a, b) => b.radius - a.radius)
        .findIndex(s => s.id === myId) + 1;
      lastRank = rank;

      if (!wasAlive) {
        // Stone just respawned — hide dead overlay
        deadOverlay.style.display = 'none';
      }
      wasAlive = true;
    } else {
      if (wasAlive) {
        // Stone just died — show dead overlay with stats
        wasAlive = false;
        deadAt = ts;
        const deathEv = gameState.events && gameState.events.find(e => e.type === 'death' && e.stoneId === myId);
        const cause = DEATH_CAUSE[deathEv?.cause] || DEATH_CAUSE.gear;
        deadIconEl.textContent = cause.icon;
        deadCauseEl.textContent = cause.label;
        deadRadiusEl.textContent = peakRadius.toFixed(1);
        deadRankEl.textContent = String(lastRank);
        deadCountdownEl.textContent = '2.0';
        deadOverlay.style.display = 'flex';
        peakRadius = 0;
      } else if (deadOverlay.style.display !== 'none') {
        // Update countdown while dead
        const remaining = Math.max(0, 2 - (ts - deadAt) / 1000);
        deadCountdownEl.textContent = remaining.toFixed(1);
      }
    }
  }

  // Zone balance stats every 5 seconds (debug)
  statsTimer += delta;
  if (statsTimer >= 5000) {
    statsTimer -= 5000;
    // console.table(engine.getZoneStats());
  }

  requestAnimationFrame(loop);
}
