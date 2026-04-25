const SLIDER_DEFS = [
  { key: 'BASE_ACCEL',        min: 0.1,   max: 1.0,   step: 0.05   },
  { key: 'MASS_ACCEL_FACTOR', min: 0,     max: 0.001, step: 0.00005 },
  { key: 'RESTITUTION',       min: 0,     max: 1,     step: 0.05   },
  { key: 'FRICTION',          min: 0.9,   max: 1.0,   step: 0.001  },
  { key: 'FRAGMENT_COUNT',    min: 4,     max: 30,    step: 1      },
];

export class ConfigPanel {
  constructor(config, onUpdate) {
    this.config = config;
    this.onUpdate = onUpdate;
    this._visible = false;
    this._el = this._build();
  }

  toggle() {
    this._visible = !this._visible;
    this._el.style.display = this._visible ? 'block' : 'none';
  }

  _build() {
    const panel = document.createElement('div');
    panel.style.cssText = [
      'position:absolute', 'top:10px', 'right:220px', 'width:260px',
      'background:rgba(12,8,6,0.93)', 'border:1px solid rgba(200,168,75,0.45)',
      'border-radius:6px', 'padding:12px 14px', 'font:12px/1.5 monospace',
      'color:#ccc', 'display:none', 'user-select:none', 'z-index:100',
    ].join(';');

    const title = document.createElement('div');
    title.textContent = 'CONFIG  [Tab to hide]';
    title.style.cssText = 'color:#C8A84B;font-weight:bold;margin-bottom:10px;letter-spacing:1px;font-size:11px';
    panel.appendChild(title);

    for (const def of SLIDER_DEFS) panel.appendChild(this._row(def));

    const hint = document.createElement('div');
    hint.textContent = '` debug   M minimap zoom';
    hint.style.cssText = 'margin-top:8px;color:#666;font-size:10px';
    panel.appendChild(hint);

    document.body.appendChild(panel);
    return panel;
  }

  _row({ key, min, max, step }) {
    // Derive display decimal places from step size
    const decimals = step < 1 ? (String(step).split('.')[1] ?? '').length : 0;

    const wrap = document.createElement('div');
    wrap.style.cssText = 'margin-bottom:9px';

    const header = document.createElement('div');
    header.style.cssText = 'display:flex;justify-content:space-between;margin-bottom:2px';

    const lbl = document.createElement('span');
    lbl.textContent = key;

    const valDisplay = document.createElement('span');
    valDisplay.style.color = '#C8A84B';
    valDisplay.textContent = Number(this.config[key]).toFixed(decimals);

    header.appendChild(lbl);
    header.appendChild(valDisplay);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(this.config[key]);
    slider.style.cssText = 'width:100%;accent-color:#C8A84B;cursor:pointer';

    slider.addEventListener('input', () => {
      const n = parseFloat(slider.value);
      this.config[key] = n;
      valDisplay.textContent = n.toFixed(decimals);
      this.onUpdate({ [key]: n });
    });

    wrap.appendChild(header);
    wrap.appendChild(slider);
    return wrap;
  }
}
