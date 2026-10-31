// Model intelligence: how each model performs, what it relies on, and what was tried.
import { get } from '../api.js';
import { el, SMELLS, BY_KEY, glyph, f2, f3, pct, n, MODEL_NAME, sectionHead } from '../ui.js';
import { prChart, metricsAt } from '../charts.js';

const MODEL_ORDER = ['dummy', 'stump', 'logreg', 'random_forest', 'hist_gb'];

export async function mount(root, { params }) {
  const [res, sc, ds, tun] = await Promise.all([
    get('/api/results'), get('/api/scores'), get('/api/dataset'),
    get('/api/tuning').catch(() => ({ rows: null })),
  ]);
  let key = SMELLS.some((s) => s.key === params.get('smell')) ? params.get('smell') : SMELLS[0].key;

  const rail = el('nav', { class: 'rail', 'aria-label': 'Choose a smell' });
  const panel = el('div', { class: 'panel' });
  const railButtons = {};
  for (const s of SMELLS) {
    const m = res.summary.find((r) => r.key === s.key);
    const b = el('button', { class: 'rail-item', type: 'button' },
      glyph(s.key, 18),
      el('span', {}, el('span', { class: 'name', text: s.label }),
        el('span', { class: 'sub', text: m ? `Average precision ${f2(m.AP)}` : '' })));
    b.addEventListener('click', () => show(s.key));
    railButtons[s.key] = b;
    rail.append(b);
  }

  function show(k) {
    key = k;
    for (const [kk, b] of Object.entries(railButtons)) {
      if (kk === k) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current');
    }
    history.replaceState(null, '', `#/models?smell=${k}`);
    panel.replaceChildren(buildPanel(k));
  }

  // ---- panel for one smell
  function buildPanel(k) {
    const s = BY_KEY[k];
    const summary = res.summary.find((r) => r.key === k);
    const score = sc[k];
    const d = ds.smells[k];
    if (!summary || !score || !d) return el('div', { class: 'notice error', text: 'Results for this smell are not available yet.' });
    const frag = el('div', {});

    frag.append(el('header', { class: 'panel-head' },
      el('h1', { class: 'h1' }, glyph(k, 24), el('span', { text: s.label })),
      el('p', { class: 'lede small', text: s.blurb }),
      el('p', { class: 'meta-line',
        text: `Deployed model: ${MODEL_NAME[summary.model] || summary.model}. Tested on ${n(summary.n_test)} held-out code locations; `
          + `${n(summary.n_positive)} of them (${pct(d.prevalence_test, 1)}) are real ${s.label} cases.` })));

    frag.append(prSection(s, summary, score));
    frag.append(compareSection(s, k, summary));
    frag.append(reliesSection(s, k));
    if (k === 'feature_envy') frag.append(tuningSection(s));
    return frag;
  }

  // ---- precision and recall, with a threshold explorer
  function prSection(s, summary, score) {
    const chart = prChart({ y: score.y, p: score.p, color: s.hex, threshold: score.threshold });
    const slider = el('input', { type: 'range', min: '0', max: '1', step: '0.01', value: String(score.threshold),
      'aria-label': 'Flag threshold' });
    const label = el('p', { class: 'slider-label' });
    const cells = {};
    const cell = (id, title, hint) => {
      cells[id] = el('span', { class: 'big', text: '–' });
      return el('td', { class: `c-${id}` }, cells[id], el('span', { class: 'sub', text: title }), hint ? el('span', { class: 'sub', text: hint }) : null);
    };
    const confusion = el('table', { class: 'confusion' },
      el('thead', {}, el('tr', {}, el('td'), el('th', { scope: 'col', text: 'Model flags it' }), el('th', { scope: 'col', text: 'Model stays quiet' }))),
      el('tbody', {},
        el('tr', {}, el('th', { scope: 'row', text: 'Real smell' }), cell('tp', 'found'), cell('fn', 'missed')),
        el('tr', {}, el('th', { scope: 'row', text: 'Not a smell' }), cell('fp', 'false alarm'), cell('tn', 'correctly quiet'))));
    const mets = {};
    const metric = (id, name) => { mets[id] = el('strong', { text: '–' }); return el('div', {}, mets[id], el('span', { class: 'sub', text: name })); };
    const metricRow = el('div', { class: 'metric-row' }, metric('precision', 'Flags that are right'),
      metric('recall', 'Real cases found'), metric('f1', 'F1'), metric('mcc', 'MCC'));
    const reset = el('button', { class: 'btn', type: 'button', text: `Reset to deployed (${f2(score.threshold)})` });

    function update() {
      const t = Number(slider.value);
      const m = metricsAt(score.y, score.p, t);
      label.textContent = `Flag anything scored at or above ${f2(t)}. That flags ${n(m.flagged)} of ${n(score.y.length)} test cases.`;
      cells.tp.textContent = n(m.tp); cells.fn.textContent = n(m.fn);
      cells.fp.textContent = n(m.fp); cells.tn.textContent = n(m.tn);
      mets.precision.textContent = pct(m.precision); mets.recall.textContent = pct(m.recall);
      mets.f1.textContent = f2(m.f1); mets.mcc.textContent = f2(m.mcc);
      chart.setThreshold(t);
    }
    slider.addEventListener('input', update);
    reset.addEventListener('click', () => { slider.value = String(score.threshold); update(); });
    update();

    return el('section', { class: 'section' },
      sectionHead('Precision and recall',
        `Average precision ${f2(summary.AP)} (95% range ${f2(summary.AP_ci_low)} to ${f2(summary.AP_ci_high)}). The curve shows the trade-off at every possible threshold.`),
      el('div', { class: 'pr-grid' },
        el('figure', { class: 'figure' }, chart.node,
          el('figcaption', { class: 'caption', text: 'Moving right finds more real cases but lets in more false alarms. The ring marks the deployed threshold; the filled dot follows the slider.' })),
        el('div', { class: 'explorer' },
          el('h3', { class: 'h3', text: 'Move the threshold' }),
          slider, label, confusion, metricRow, el('div', { class: 'row-actions' }, reset),
          el('p', { class: 'caption',
            text: 'For exploration only. The deployed threshold was fixed from training data before the test set was scored, and this slider changes nothing in the analyzer.' }))));
  }

  // ---- candidate models compared
  function compareSection(s, k, summary) {
    const cv = res.cv.filter((r) => r.smell === k);
    const test = Object.fromEntries(res.test.filter((r) => r.smell === k).map((r) => [r.model, r]));
    const maxAP = Math.max(...cv.map((r) => r.oof_ap), 0.05) * 1.12;
    const rows = MODEL_ORDER.map((m) => cv.find((r) => r.model === m)).filter(Boolean).map((r) => {
      const t = test[r.model];
      const selected = r.model === summary.model;
      return el('tr', { class: selected ? 'selected' : '' },
        el('th', { scope: 'row' }, el('span', { class: 'name', text: MODEL_NAME[r.model] || r.model }),
          selected ? el('span', { class: 'tag', text: 'deployed' }) : null),
        el('td', {}, el('div', { class: 'bar-cell' },
          el('span', { class: 'bar' }, el('i', { style: `width:${(r.oof_ap / maxAP) * 100}%;background:${s.hex}` })),
          el('span', { class: 'val', text: f2(r.oof_ap) }),
          el('span', { class: 'sub', text: `±${f2(r.fold_ap_std)}` }))),
        el('td', { class: 'num', text: f2(r.oof_f1) }),
        el('td', { class: 'num', text: f2(r.oof_mcc) }),
        el('td', { class: 'num' }, t
          ? [el('strong', { text: f2(t.test_ap) }), el('span', { class: 'sub', text: ` ${f2(t.ap_ci_low)} to ${f2(t.ap_ci_high)}` })]
          : el('span', { class: 'sub', text: 'not evaluated', title: 'The test set was scored once, for the baselines and the deployed model only.' })));
    });
    return el('section', { class: 'section' },
      sectionHead('Which model, and why',
        'Models were compared with repository-grouped cross-validation on training repositories only. The test set was scored once, for the baselines and the model that cross-validation picked.'),
      el('div', { class: 'table-wrap' },
        el('table', { class: 'ledger compact' },
          el('thead', {}, el('tr', {},
            el('th', { scope: 'col', text: 'Model' }),
            el('th', { scope: 'col', text: 'Cross-validation average precision, ± spread across folds' }),
            el('th', { scope: 'col', class: 'num', text: 'F1' }),
            el('th', { scope: 'col', class: 'num', text: 'MCC' }),
            el('th', { scope: 'col', class: 'num', text: 'Held-out average precision' }))),
          el('tbody', {}, rows))),
      el('p', { class: 'caption',
        text: '“Flag everything” marks every case as a smell and shows what the base rate alone achieves. “One-metric rule” is the best single threshold on one metric, the way a traditional rule-based checker works.' }));
  }

  // ---- what the model relies on
  function reliesSection(s, k) {
    const imp = (res.importance[k] || []).slice(0, 10);
    if (!imp.length) return el('div');
    const kind = imp[0].kind;
    const maxAbs = Math.max(...imp.map((r) => Math.abs(r.value)), 1e-9);
    const diverging = imp.some((r) => r.value < 0);
    const rows = imp.map((r) => {
      const w = (Math.abs(r.value) / maxAbs) * (diverging ? 50 : 100);
      const side = diverging ? (r.value >= 0 ? `left:50%;background:${s.hex}` : 'right:50%;background:#8A929B') : `left:0;background:${s.hex}`;
      return el('div', { class: 'imp-row' },
        el('span', { class: 'imp-name', text: r.feature }),
        el('span', { class: `imp-bar${diverging ? ' div' : ''}` }, el('i', { style: `width:${w}%;${side}` })),
        el('span', { class: 'imp-val', text: f3(r.value) }));
    });
    return el('section', { class: 'section' },
      sectionHead('What the model relies on',
        kind === 'importance'
          ? 'Relative importance of each metric inside the random forest. Higher means the forest splits on it more usefully.'
          : 'Weights in the logistic model, on log-scaled and standardized metrics. Positive raises the score, negative lowers it. Metrics move together, so read signs with care.'),
      el('div', { class: 'imp' }, rows));
  }

  // ---- Feature Envy tuning study
  function tuningSection(s) {
    const rows = (tun && tun.rows) || null;
    const findings = el('ul', { class: 'plain' },
      el('li', { text: 'Flagging the top 5% of training methods by score (85 methods) found 17 of the 49 real cases, about seven times better than random.' }),
      el('li', { text: 'Missed cases are often short methods that read several members of one parameter object. The available metrics do not count that kind of access, so the model cannot see it.' }),
      el('li', { text: 'Confident false alarms are mostly large methods, which the model treats as a sign of envy.' }),
      el('li', { text: '35 of the 49 positive training examples had a reviewer who disagreed, so the labels themselves are noisy.' }));
    const blocks = [];
    if (rows) {
      const ks = ['1', '2', '3', '5', '8', 'all'];
      const cs = [0.03, 0.1, 0.3, 1];
      const lr = rows.filter((r) => r.family === 'logreg');
      const vals = lr.map((r) => r.fold_ap_mean);
      const lo = Math.min(...vals), hi = Math.max(...vals);
      const cur = lr.find((r) => String(r.k) === 'all' && Number(r.reg) === 1);
      const heat = el('table', { class: 'heat' },
        el('thead', {}, el('tr', {}, el('th', { scope: 'col', text: 'Metrics kept' }),
          cs.map((c) => el('th', { scope: 'col', text: `C = ${c}` })))),
        el('tbody', {}, ks.map((kk) => el('tr', {},
          el('th', { scope: 'row', text: kk === 'all' ? 'All 29' : kk }),
          cs.map((c) => {
            const r = lr.find((x) => String(x.k) === kk && Number(x.reg) === c);
            if (!r) return el('td', { text: '–' });
            const a = 0.1 + 0.4 * ((r.fold_ap_mean - lo) / (hi - lo || 1));
            const isCur = cur && r.id === cur.id;
            return el('td', { class: isCur ? 'current' : '', style: `background:rgba(11,93,99,${a.toFixed(2)})`,
              title: isCur ? 'The deployed model' : '', text: f3(r.fold_ap_mean) });
          })))));
      const rf = rows.filter((r) => r.family === 'rf');
      blocks.push(
        el('h3', { class: 'h3', text: 'Logistic regression: metrics kept against regularization strength' }),
        el('div', { class: 'table-wrap' }, heat),
        el('p', { class: 'caption',
          text: `Each cell is mean average precision over 5 repeats of 5-fold repository-grouped cross-validation, on training repositories only. The outlined cell is the deployed model (${f3(cur ? cur.fold_ap_mean : NaN)}). `
            + `A standard error is about ${f3(cur ? cur.fold_ap_se : NaN)}, so the best cells are within roughly one standard error of the deployed model. `
            + 'The rule fixed before the run said to keep the current model unless a candidate beat it by more than that, so the model stayed.' }),
        rf.length ? el('p', { class: 'caption',
          text: `Random forest by minimum leaf size: ${rf.map((r) => `${r.reg} → ${f3(r.fold_ap_mean)}`).join(', ')}. Stronger regularization helped both model families a little, but not enough to be distinguishable from noise.` }) : null);
    }
    return el('section', { class: 'section' },
      sectionHead('Feature Envy: tuning study and error analysis',
        'The weakest model got the most scrutiny. Everything here uses training data only.'),
      ...blocks,
      el('h3', { class: 'h3', text: 'What the error analysis found' }), findings);
  }

  root.append(el('div', { class: 'wrap page' },
    el('div', { class: 'models-grid' }, rail, panel)));
  show(key);
}
