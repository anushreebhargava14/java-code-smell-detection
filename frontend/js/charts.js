// Small SVG charts and the metric math behind the threshold explorer.
import { svgEl } from './ui.js';

const INK = '#1F2630';
const MUTED = '#5F6873';
const RULE = '#DDD6C7';
const TEAL = '#0B5D63';
const TICKS = { 0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1' };

/** Confusion counts and derived metrics when flagging every item with score >= t. */
export function metricsAt(y, p, t) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (let i = 0; i < y.length; i++) {
    if (p[i] >= t) { if (y[i]) tp++; else fp++; } else if (y[i]) fn++; else tn++;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  const denom = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  const mcc = denom ? (tp * tn - fp * fn) / denom : 0;
  return { tp, fp, fn, tn, precision, recall, f1, mcc, flagged: tp + fp };
}

/** Precision-recall points, starting at (recall 0, precision 1). */
export function prPoints(y, p) {
  const idx = p.map((_, i) => i).sort((a, b) => p[b] - p[a]);
  const positives = y.reduce((a, b) => a + b, 0);
  const pts = [[0, 1]];
  let tp = 0, fp = 0;
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k];
    if (y[i]) tp++; else fp++;
    const last = k === idx.length - 1 || p[idx[k + 1]] !== p[i];
    if (last && positives) pts.push([tp / positives, tp / (tp + fp)]);
  }
  return pts;
}

/** Average precision with a 95% range, drawn on a 0 to 1 axis; dashed tick = random guessing. */
export function rangeChart({ value, lo, hi, base, color, w = 190, h = 28 }) {
  const x = (v) => 6 + v * (w - 12);
  const svg = svgEl('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: 'img',
    'aria-label': `Average precision ${value.toFixed(2)}, 95% range ${lo.toFixed(2)} to ${hi.toFixed(2)}` });
  const mid = h / 2;
  svg.append(svgEl('line', { x1: x(0), x2: x(1), y1: mid, y2: mid, stroke: RULE, 'stroke-width': 1.5 }));
  for (const t of [0, 0.5, 1]) {
    svg.append(svgEl('line', { x1: x(t), x2: x(t), y1: mid - 3, y2: mid + 3, stroke: RULE, 'stroke-width': 1.5 }));
  }
  svg.append(svgEl('rect', { x: x(lo), y: mid - 5, width: Math.max(2, x(hi) - x(lo)), height: 10, rx: 5,
    fill: color, 'fill-opacity': 0.22 }));
  if (base != null) {
    svg.append(svgEl('line', { x1: x(base), x2: x(base), y1: 3, y2: h - 3, stroke: MUTED,
      'stroke-width': 1.4, 'stroke-dasharray': '2 2.5' }));
  }
  svg.append(svgEl('circle', { cx: x(value), cy: mid, r: 5.5, fill: color, stroke: '#FCFAF5', 'stroke-width': 1.5 }));
  return svg;
}

/** Precision-recall curve with a movable threshold marker. Returns { node, setThreshold }. */
export function prChart({ y, p, color, threshold, w = 560, h = 380 }) {
  const m = { l: 52, r: 18, t: 18, b: 50 };
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const X = (r) => m.l + r * iw;
  const Y = (v) => m.t + (1 - v) * ih;
  const svg = svgEl('svg', { viewBox: `0 0 ${w} ${h}`, class: 'chart', role: 'img',
    'aria-label': 'Precision-recall curve on held-out test data' });

  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(svgEl('line', { x1: m.l, x2: w - m.r, y1: Y(t), y2: Y(t), stroke: RULE, 'stroke-width': t === 0 ? 1.4 : 0.8 }));
    svg.append(svgEl('line', { x1: X(t), x2: X(t), y1: m.t, y2: h - m.b, stroke: RULE, 'stroke-width': 0.8 }));
    const ty = svgEl('text', { x: m.l - 10, y: Y(t) + 4, 'text-anchor': 'end', class: 'ax' });
    ty.textContent = TICKS[t];
    svg.append(ty);
    const tx = svgEl('text', { x: X(t), y: h - m.b + 18, 'text-anchor': 'middle', class: 'ax' });
    tx.textContent = TICKS[t];
    svg.append(tx);
  }
  const xl = svgEl('text', { x: m.l + iw / 2, y: h - 8, 'text-anchor': 'middle', class: 'ax-title' });
  xl.textContent = 'Recall: share of real cases found';
  const yl = svgEl('text', { x: 14, y: m.t + ih / 2, 'text-anchor': 'middle', class: 'ax-title',
    transform: `rotate(-90 14 ${m.t + ih / 2})` });
  yl.textContent = 'Precision: share of flags that are right';
  svg.append(xl, yl);

  const prevalence = y.reduce((a, b) => a + b, 0) / y.length;
  svg.append(svgEl('line', { x1: X(0), x2: X(1), y1: Y(prevalence), y2: Y(prevalence), stroke: MUTED,
    'stroke-width': 1.4, 'stroke-dasharray': '4 4' }));
  const bl = svgEl('text', { x: X(1) - 4, y: Y(prevalence) - 7, 'text-anchor': 'end', class: 'ax' });
  bl.textContent = 'random guessing';
  svg.append(bl);

  const pts = prPoints(y, p);
  const line = pts.map(([r, v], i) => `${i ? 'L' : 'M'}${X(r).toFixed(1)} ${Y(v).toFixed(1)}`).join(' ');
  const last = pts[pts.length - 1];
  svg.append(svgEl('path', { d: `${line} L${X(last[0]).toFixed(1)} ${Y(0)} L${X(0)} ${Y(0)} Z`,
    fill: color, 'fill-opacity': 0.1 }));
  svg.append(svgEl('path', { d: line, fill: 'none', stroke: color, 'stroke-width': 2.2, 'stroke-linejoin': 'round' }));

  const at = (t) => { const mt = metricsAt(y, p, t); return { r: mt.recall, v: mt.flagged ? mt.precision : 1 }; };
  const dep = at(threshold);
  const deployed = svgEl('circle', { cx: X(dep.r), cy: Y(dep.v), r: 9, fill: 'none', stroke: INK, 'stroke-width': 1.6 });
  const dl = svgEl('text', { x: X(dep.r) + 14, y: Y(dep.v) - 12, class: 'ax-strong' });
  dl.textContent = 'deployed';
  const cur = svgEl('circle', { cx: X(dep.r), cy: Y(dep.v), r: 5.5, fill: TEAL, stroke: '#FCFAF5', 'stroke-width': 1.5 });
  svg.append(deployed, dl, cur);

  return {
    node: svg,
    setThreshold(t) {
      const a = at(t);
      cur.setAttribute('cx', X(a.r));
      cur.setAttribute('cy', Y(a.v));
    },
  };
}

/** Where a value sits relative to flagged and unflagged training cases (log-compressed scale). */
export function compareChart({ value, ref, color, w = 250, h = 30 }) {
  const top = Math.max(value, ref.pos_p75 * 1.25, ref.neg_p90 * 1.1, 1);
  const x = (v) => 8 + (Math.log1p(Math.max(v, 0)) / Math.log1p(top)) * (w - 16);
  const mid = h / 2;
  const svg = svgEl('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: 'img',
    'aria-label': `This value ${value}; flagged training cases typically ${ref.pos_median}; unflagged ${ref.neg_median}` });
  svg.append(svgEl('line', { x1: 8, x2: w - 8, y1: mid, y2: mid, stroke: RULE, 'stroke-width': 1.5 }));
  svg.append(svgEl('rect', { x: x(ref.pos_p25), y: mid - 6, width: Math.max(2, x(ref.pos_p75) - x(ref.pos_p25)),
    height: 12, rx: 3, fill: color, 'fill-opacity': 0.22 }));
  svg.append(svgEl('circle', { cx: x(ref.pos_median), cy: mid, r: 4, fill: color }));
  svg.append(svgEl('circle', { cx: x(ref.neg_median), cy: mid, r: 4, fill: '#FCFAF5', stroke: MUTED, 'stroke-width': 1.5 }));
  svg.append(svgEl('rect', { x: x(value) - 1.5, y: 3, width: 3, height: h - 6, rx: 1.5, fill: TEAL }));
  return svg;
}
