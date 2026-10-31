// Shared helpers: smell definitions, DOM builder, number formatting, glyphs.

export const SMELLS = [
  { key: 'long_method', label: 'Long Method', level: 'method', unit: ['method', 'methods'],
    hex: '#B06D12', glyph: 'bar',
    blurb: 'A method that does too much in one place.' },
  { key: 'data_class', label: 'Data Class', level: 'class', unit: ['class', 'classes'],
    hex: '#2B6A9E', glyph: 'grid',
    blurb: 'A class that mostly holds fields and accessors, with little behavior of its own.' },
  { key: 'blob', label: 'Blob', level: 'class', unit: ['class', 'classes'],
    hex: '#A04336', glyph: 'dot',
    blurb: 'A large class that centralizes too much behavior and knowledge.' },
  { key: 'feature_envy', label: 'Feature Envy', level: 'method', unit: ['method', 'methods'],
    hex: '#6A4C9A', glyph: 'arrow',
    blurb: "A method that works with another object's data more than its own." },
];
export const BY_KEY = Object.fromEntries(SMELLS.map((s) => [s.key, s]));

export const OUTCOME = { caught: 'Caught', missed: 'Missed', false_alarm: 'False alarm' };
export const MODEL_NAME = {
  dummy: 'Flag everything', stump: 'One-metric rule', logreg: 'Logistic regression',
  random_forest: 'Random forest', hist_gb: 'Gradient boosting',
};

export function el(tag, props = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'text') node.textContent = v;
    else if (k === 'class') node.className = v;
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) node.append(kid);
  return node;
}

export const n = (v) => (v == null ? '–' : Number(v).toLocaleString('en-US'));
export const f2 = (v) => (v == null ? '–' : Number(v).toFixed(2));
export const f3 = (v) => (v == null ? '–' : Number(v).toFixed(3));
export const pct = (v, d = 0) => (v == null ? '–' : `${(v * 100).toFixed(d)}%`);
export const plural = (k, one, many) => `${n(k)} ${k === 1 ? one : many}`;

const GLYPHS = {
  bar: '<rect x="5.5" y="1" width="5" height="14" rx="1.5"/>',
  grid: '<rect x="1.5" y="1.5" width="5.5" height="5.5" rx="1"/><rect x="9" y="1.5" width="5.5" height="5.5" rx="1"/>'
    + '<rect x="1.5" y="9" width="5.5" height="5.5" rx="1"/><rect x="9" y="9" width="5.5" height="5.5" rx="1"/>',
  dot: '<circle cx="8" cy="8" r="6.5"/>',
  arrow: '<path d="M2.5 6.5v7h7" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>'
    + '<path d="M7 9l7-7M8.2 2H14v5.8" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>',
};

/** A small shape that identifies a smell without relying on color alone. */
export function glyph(key, size = 16) {
  const s = BY_KEY[key];
  const span = el('span', { class: 'glyph', 'aria-hidden': 'true', style: `color:${s.hex};width:${size}px;height:${size}px` });
  span.innerHTML = `<svg viewBox="0 0 16 16" width="${size}" height="${size}" fill="currentColor">${GLYPHS[s.glyph]}</svg>`;
  return span;
}

/** Ten dots; `filled` of them are solid. Used to show "n in 10 flags are right". */
export function dotRow(filled, color) {
  const wrap = el('span', { class: 'dots', style: `--c:${color}`, 'aria-hidden': 'true' });
  for (let i = 0; i < 10; i++) wrap.append(el('i', { class: i < filled ? 'on' : '' }));
  return wrap;
}

export function errorBlock(e) {
  return el('div', { class: 'wrap page' },
    el('div', { class: 'notice error', role: 'alert' },
      el('strong', { text: 'This page could not load. ' }),
      el('span', { text: e && e.message ? e.message : String(e) })));
}

export function sectionHead(title, sub) {
  return el('div', { class: 'section-head' },
    el('h2', { class: 'h2', text: title }),
    sub ? el('p', { class: 'sub', text: sub }) : null);
}

export function svgEl(tag, attrs = {}, ...kids) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) node.setAttribute(k, v);
  for (const kid of kids.flat()) if (kid != null) node.append(kid);
  return node;
}
