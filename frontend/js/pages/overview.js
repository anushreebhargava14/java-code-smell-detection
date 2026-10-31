// Overview: what the system found, how far to trust it, and where the data came from.
import { get } from '../api.js';
import { el, SMELLS, glyph, dotRow, pct, f2, n, OUTCOME, MODEL_NAME, sectionHead } from '../ui.js';
import { rangeChart } from '../charts.js';

export async function mount(root) {
  const [res, ds, ex] = await Promise.all([
    get('/api/results'), get('/api/dataset'), get('/api/examples').catch(() => []),
  ]);
  const rows = SMELLS.map((s) => ({ s, m: res.summary.find((r) => r.key === s.key), d: ds.smells[s.key] }))
    .filter((r) => r.m && r.d);
  if (!rows.length) throw new Error('No evaluation results found yet. Run final_evaluation.py first.');
  rows.sort((a, b) => b.m.AP - a.m.AP);
  const best = rows[0], worst = rows[rows.length - 1];
  const examples = rows.reduce((sum, r) => sum + r.d.n_train + r.d.n_test, 0);
  const testRepos = ds.repos && ds.repos.test;
  const fe = rows.find((r) => r.s.key === 'feature_envy');

  // ---- hero
  const hero = el('section', { class: 'hero' },
    el('div', {},
      el('h1', { class: 'display' },
        el('span', { text: `${best.s.label} is detected well.` }), ' ',
        el('span', { class: 'dim', text: `${worst.s.label} is the weak spot.` })),
      el('p', { class: 'lede',
        text: `Four models, one per smell, learned from ${n(examples)} human-reviewed Java code locations in the MLCQ dataset. `
          + `They were then tested on ${testRepos ? `${n(testRepos)} open-source repositories` : 'open-source repositories'} that played no part in training.` }),
      el('div', { class: 'actions' },
        el('a', { class: 'btn primary', href: '#/analyzer', text: 'Analyze your own code' }),
        el('a', { class: 'textlink', href: '#/method', text: 'How the models were built' }))),
    el('dl', { class: 'smell-defs' },
      SMELLS.map((s) => el('div', {},
        el('dt', {}, glyph(s.key, 16), el('span', { text: s.label })),
        el('dd', { text: s.blurb })))));

  // ---- results ledger
  const head = el('tr', {},
    el('th', { scope: 'col', text: 'Smell' }),
    el('th', { scope: 'col', text: 'Average precision, with 95% range' }),
    el('th', { scope: 'col', text: 'Flags that are right' }),
    el('th', { scope: 'col', class: 'num', text: 'Real cases found' }),
    el('th', { scope: 'col', class: 'num', text: 'F1' }),
    el('th', { scope: 'col', class: 'num', text: 'MCC' }),
    el('th', { scope: 'col', class: 'num', text: 'Test cases' }));
  const body = rows.map(({ s, m, d }) => el('tr', {},
    el('th', { scope: 'row' },
      el('a', { class: 'cell-smell', href: `#/models?smell=${s.key}` },
        glyph(s.key, 18),
        el('span', {}, el('span', { class: 'name', text: s.label }),
          el('span', { class: 'sub', text: MODEL_NAME[m.model] || m.model })))),
    el('td', {}, el('div', { class: 'ap-cell' },
      rangeChart({ value: m.AP, lo: m.AP_ci_low, hi: m.AP_ci_high, base: d.prevalence_test, color: s.hex }),
      el('span', { class: 'ap-text' }, el('strong', { text: f2(m.AP) }),
        el('span', { class: 'sub', text: ` ${f2(m.AP_ci_low)} to ${f2(m.AP_ci_high)}` })))),
    el('td', {}, el('div', { class: 'dot-cell' },
      dotRow(Math.round(m.precision * 10), s.hex), el('span', { class: 'pct', text: pct(m.precision) }))),
    el('td', { class: 'num', text: pct(m.recall) }),
    el('td', { class: 'num', text: f2(m.F1) }),
    el('td', { class: 'num', text: f2(m.MCC) }),
    el('td', { class: 'num', text: `${n(m.n_positive)} of ${n(m.n_test)}` })));
  const ledger = el('section', { class: 'section' },
    sectionHead('Results on repositories the models never saw',
      'Each row is one model, ranked by average precision. Click a smell for its full evaluation.'),
    el('div', { class: 'table-wrap' },
      el('table', { class: 'ledger' }, el('thead', {}, head), el('tbody', {}, body))),
    el('p', { class: 'caption',
      text: 'Average precision scores how well a model ranks suspicious code above clean code. The dashed tick is what random guessing would score, '
        + 'the bar is a 95% range from resampling the test cases, and the dots show how many of every 10 flags are correct. '
        + (fe ? `A small number of real cases limits certainty: Feature Envy has only ${n(fe.m.n_positive)} in the test set.` : '') }));

  // ---- funnel + held-out cases
  const maxCount = Math.max(...ds.funnel.map((f) => f.count || 0), 1);
  const funnel = el('div', {},
    sectionHead('From reviews to training examples',
      ds.files ? `${n(ds.files.fetched)} of ${n(ds.files.unique)} referenced source files could be downloaded.` : null),
    ds.funnel.length
      ? el('ol', { class: 'funnel' }, ds.funnel.filter((f) => f.count != null).map((f) => el('li', {},
        el('span', { class: 'f-label', text: f.label }),
        el('span', { class: 'f-count', text: n(f.count) }),
        el('span', { class: 'f-bar' }, el('i', { style: `width:${Math.max(1.5, (f.count / maxCount) * 100)}%` })))))
      : el('p', { class: 'caption', text: 'Pipeline counts are not available yet.' }),
    el('p', { class: 'caption', text: 'Details of every exclusion are on the dataset and method page.' }));

  const byKey = {};
  ex.forEach((e) => { (byKey[e.smell] = byKey[e.smell] || {})[e.outcome] = e; });
  const cases = el('div', {},
    sectionHead('Try a held-out case',
      'Real code from test repositories, picked by rule: the best-scored hit, the lowest-scored miss and the highest-scored false alarm for each smell.'),
    ex.length
      ? el('table', { class: 'cases' }, el('tbody', {}, SMELLS.map((s) => el('tr', {},
        el('th', { scope: 'row' }, el('span', { class: 'cell-smell' }, glyph(s.key, 16), el('span', { class: 'name', text: s.label }))),
        ['caught', 'missed', 'false_alarm'].map((o) => el('td', {},
          byKey[s.key] && byKey[s.key][o]
            ? el('a', { class: 'pill', href: `#/analyzer?ex=${s.key}-${o}`, title: byKey[s.key][o].name, text: OUTCOME[o] })
            : el('span', { class: 'sub', text: 'none' })))))))
      : el('p', { class: 'caption', text: 'Example cases are not available yet.' }));

  root.append(el('div', { class: 'wrap page' }, hero, ledger,
    el('div', { class: 'split' }, funnel, cases),
    el('p', { class: 'caption foot',
      text: 'The models learned from Java projects only. Results do not carry over automatically to other languages or coding styles, and a flag is a prompt to read the code, not a verdict.' })));
}
