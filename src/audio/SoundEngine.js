/**
 * Procedural sound effects via the Web Audio API — no external audio files.
 * Every sound is synthesized on the fly from oscillators / filtered noise.
 *
 * Browsers require a user gesture before audio can start, so the AudioContext
 * is created lazily and resume() should be called from a click handler (the
 * lobby Play button). High-frequency events (absorb/collision) are rate-limited
 * to avoid a machine-gun effect.
 */

// Note frequencies (Hz)
const C4 = 261.63;
const E4 = 329.63;
const G4 = 392.0;

export class SoundEngine {
  constructor() {
    /** @type {AudioContext|null} */
    this._ctx = null;
    /** @type {GainNode|null} */
    this._master = null;
    this._muted = false;
    this._lastPlay = { absorb: 0, collision: 0 };
  }

  get muted() {
    return this._muted;
  }

  _ensure() {
    if (this._ctx) return this._ctx;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    this._ctx = new AC();
    this._master = this._ctx.createGain();
    this._master.gain.value = this._muted ? 0 : 1;
    this._master.connect(this._ctx.destination);
    return this._ctx;
  }

  /** Call from a user-gesture handler (e.g. Play button) to unlock audio. */
  resume() {
    const ctx = this._ensure();
    if (ctx && ctx.state === 'suspended') ctx.resume();
  }

  /** Returns the new muted state. */
  toggle() {
    this._muted = !this._muted;
    if (this._master) {
      const now = this._ctx.currentTime;
      this._master.gain.cancelScheduledValues(now);
      this._master.gain.setValueAtTime(this._muted ? 0 : 1, now);
    }
    return this._muted;
  }

  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  /** Rate-limit a named effect; returns false if it fired too recently. */
  _gate(name, minMs) {
    const ctx = this._ctx;
    if (!ctx) return true;
    const now = ctx.currentTime * 1000;
    if (now - this._lastPlay[name] < minMs) return false;
    this._lastPlay[name] = now;
    return true;
  }

  /** Single oscillator with a frequency glide and gain envelope. */
  _tone(type, f0, f1, durMs, peak) {
    const ctx = this._ensure();
    if (!ctx || this._muted) return;
    const t = ctx.currentTime;
    const dur = durMs / 1000;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(f0, t);
    osc.frequency.exponentialRampToValueAtTime(Math.max(1, f1), t + dur);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(peak, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    osc.connect(g).connect(this._master);
    osc.start(t);
    osc.stop(t + dur + 0.02);
  }

  // ---------------------------------------------------------------------------
  // Effects
  // ---------------------------------------------------------------------------

  /** A burst of lowpass-filtered white noise — thuds and impacts. */
  _noiseBurst(cutoff, durMs, peak) {
    const ctx = this._ensure();
    if (!ctx || this._muted) return;
    const t = ctx.currentTime;
    const dur = durMs / 1000;
    const frames = Math.floor(ctx.sampleRate * dur);
    const buf = ctx.createBuffer(1, frames, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < frames; i++) data[i] = Math.random() * 2 - 1;

    const src = ctx.createBufferSource();
    src.buffer = buf;
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = cutoff;
    const g = ctx.createGain();
    g.gain.setValueAtTime(peak, t);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(lp).connect(g).connect(this._master);
    src.start(t);
    src.stop(t + dur);
  }

  absorb() {
    if (!this._gate('absorb', 45)) return;
    this._tone('sine', 440, 880, 80, 0.15);
  }

  collision() {
    if (!this._gate('collision', 70)) return;
    this._noiseBurst(200, 120, 0.2);
  }

  /** Generic death (another stone) — descending tone + heavy low thud. */
  death() {
    this._tone('sine', 300, 80, 400, 0.3);
    this._tone('triangle', 200, 55, 420, 0.12); // sub layer for body
    this._noiseBurst(120, 260, 0.35);           // impact thud
  }

  /**
   * The player's own death — bigger, lower and more dramatic than a generic
   * death so it clearly stands out (different timbre + louder).
   */
  playerDeath() {
    this._tone('sawtooth', 260, 55, 560, 0.32); // gritty descending lead
    this._tone('sine',     170, 40, 580, 0.42); // deep body
    this._tone('triangle',  90, 28, 600, 0.20); // sub-bass rumble
    this._noiseBurst(180, 380, 0.5);            // big impact
  }

  zoneCross() {
    const ctx = this._ensure();
    if (!ctx || this._muted) return;
    [C4, E4, G4].forEach((f, i) => {
      const t = ctx.currentTime + i * 0.05;
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(f, t);
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(0.1, t + 0.01);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.05);
      osc.connect(g).connect(this._master);
      osc.start(t);
      osc.stop(t + 0.07);
    });
  }
}
