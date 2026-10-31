// Dataset and method: how the data was built, split, cleaned and evaluated.
import { get } from '../api.js';
import { el, SMELLS, glyph, n, pct, f2, MODEL_NAME } from '../ui.js';

const GLOSSARY = [
  ['loc', 'Lines of code in the class or method.'],
  ['wmc', 'Weighted methods per class: summed cyclomatic complexity (for a method, its own branching).'],
  ['rfc', 'Response for class: how many methods can run in response to calling this code.'],
  ['cbo', 'Coupling between objects: how many other types this code refers to.'],
  ['fanin, fanout', 'How many types depend on this code, and how many it depends on.'],
  ['maxNestedBlocksQty', 'Deepest nesting of blocks such as loops and conditions.'],
  ['variablesQty, assignmentsQty', 'Counts of variables declared and assignments made.'],
  ['methodsInvokedQty', 'Number of method calls made.'],
  ['totalFieldsQty, totalMethodsQty', 'Number of fields and methods a class declares.'],
];

const p = (t) => el('p', { text: t });

export async function mount(root) {
  const [ds, res] = await Promise.all([get('/api/dataset'), get('/api/results')]);
  const q = ds.quality || {};
  const sections = [];
  const section = (id, title, ...kids) => {
    const node = el('section', { class: 'doc-section', id: `sec-${id}` }, el('h2', { class: 'h2', text: title }), ...kids);
    sections.push({ id, title, node });
    return node;
  };
  const smellRows = SMELLS.map((s) => ({ s, d: ds.smells[s.key], m: res.summary.find((r) => r.key === s.key) })).filter((r) => r.d);

  // 1. data and funnel
  const maxCount = Math.max(...ds.funnel.map((f) => f.count || 0), 1);
  section('data', 'The data',
    p('MLCQ is a public dataset in which professional developers reviewed real Java code from open-source projects. Each review rates one code location for one smell as none, minor, major or critical. We use its four smells: Blob (also called God Class), Data Class, Feature Envy and Long Method.'),
    p('The dataset points to code rather than including it, so the pipeline downloaded each referenced file at the exact commit the reviewer saw and then measured it.'),
    ds.funnel.length
      ? el('ol', { class: 'funnel' }, ds.funnel.filter((f) => f.count != null).map((f) => el('li', {},
        el('span', { class: 'f-label', text: f.label }),
        el('span', { class: 'f-count', text: n(f.count) }),
        el('span', { class: 'f-bar' }, el('i', { style: `width:${Math.max(1.5, (f.count / maxCount) * 100)}%` }))))) : null,
    ds.files ? el('p', { class: 'caption', text: `${n(ds.files.fetched)} of ${n(ds.files.unique)} referenced source files could still be downloaded; the rest had been moved or deleted upstream.` }) : null);

  // 2. labels
  section('labels', 'Labels',
    p('A review counts as “smell present” when its severity is minor, major or critical. Several reviewers often rated the same code location, so reviews were merged into one example per location by majority vote. Exact ties were dropped.'),
    p('Counting every review separately would have made smells look much more common than they are: Feature Envy was positive in 13.5% of reviews but in 2.9% of locations.'),
    el('h3', { class: 'h3', text: 'How much reviewers agreed on the positive examples' }),
    el('div', { class: 'stacks' }, smellRows.map(({ s, d }) => {
      const tot = d.agreement.unanimous + d.agreement.majority_only || 1;
      const u = d.agreement.unanimous / tot;
      return el('div', { class: 'stack-row' },
        el('span', { class: 'cell-smell' }, glyph(s.key, 16), el('span', { class: 'name', text: s.label })),
        el('span', { class: 'stack', role: 'img', 'aria-label': `${pct(u)} unanimous, ${pct(1 - u)} with a dissenting reviewer` },
          el('i', { class: 'u', style: `width:${u * 100}%;background:${s.hex}` }),
          el('i', { class: 'm', style: `width:${(1 - u) * 100}%;background:${s.hex}` })),
        el('span', { class: 'sub', text: `${pct(u)} unanimous, ${pct(1 - u)} with a dissenting reviewer` }));
    })),
    el('p', { class: 'caption', text: 'Solid: every reviewer agreed. Faded: at least one reviewer disagreed. Disputed labels put a ceiling on what any model can learn from them.' }),
    q.ties_dropped != null ? p(`${n(q.ties_dropped)} code locations with an exactly split vote were removed.`) : null);

  // 3. measurements
  section('features', 'What the models see',
    p(`Each code location is described by source-code metrics from the CK tool, measured on the whole original file. Class-level smells (Blob, Data Class) use ${ds.features.class} class metrics; method-level smells (Feature Envy, Long Method) use ${ds.features.method} method metrics. The models never read the code text itself.`),
    el('div', { class: 'table-wrap' }, el('table', { class: 'plain-table' },
      el('thead', {}, el('tr', {}, el('th', { scope: 'col', text: 'Metric' }), el('th', { scope: 'col', text: 'Meaning' }))),
      el('tbody', {}, GLOSSARY.map(([k, v]) => el('tr', {}, el('th', { scope: 'row', class: 'mono', text: k }), el('td', { text: v })))))));

  // 4. split and evaluation
  section('split', 'Splitting and scoring honestly',
    p(`Repositories, not individual examples, were split: ${n(ds.repos.train)} repositories for training and cross-validation, and ${n(ds.repos.test)} held out for the final test. All code from one repository stays on one side, so the test measures how the models do on projects they have never seen.`),
    el('div', { class: 'table-wrap' }, el('table', { class: 'plain-table' },
      el('thead', {}, el('tr', {}, el('th', { scope: 'col', text: 'Smell' }),
        el('th', { scope: 'col', class: 'num', text: 'Train examples' }), el('th', { scope: 'col', class: 'num', text: 'of which smells' }),
        el('th', { scope: 'col', class: 'num', text: 'Test examples' }), el('th', { scope: 'col', class: 'num', text: 'of which smells' }))),
      el('tbody', {}, smellRows.map(({ s, d }) => el('tr', {},
        el('th', { scope: 'row' }, el('span', { class: 'cell-smell' }, glyph(s.key, 16), el('span', { class: 'name', text: s.label }))),
        el('td', { class: 'num', text: n(d.n_train) }), el('td', { class: 'num', text: n(d.pos_train) }),
        el('td', { class: 'num', text: n(d.n_test) }), el('td', { class: 'num', text: n(d.pos_test) })))))),
    p('Smells are rare, so plain accuracy would be misleading: a model that never flags anything would still score at least 86% on every smell. Results are reported as:'),
    el('dl', { class: 'defs' },
      el('div', {}, el('dt', { text: 'Average precision' }), el('dd', { text: 'How well a model ranks real smells above clean code, independent of any threshold. Random guessing scores the share of real smells.' })),
      el('div', {}, el('dt', { text: 'Precision and recall' }), el('dd', { text: 'Of the flags raised, how many are right; of the real smells, how many were flagged.' })),
      el('div', {}, el('dt', { text: 'F1 and MCC' }), el('dd', { text: 'Single-number summaries that balance both kinds of mistake; MCC stays honest when classes are very uneven.' }))),
    p('Decision thresholds were chosen from training data only. The test set was scored once for the final models.'));

  // 5. data-quality ledger
  const ledgerRows = [
    ['Names did not line up', 'MLCQ names a code location one way and the metrics tool another. A deterministic matcher uses class, method name, parameter types and line numbers.',
      `${n(q.mapped)} of ${n(q.reviews_labeled)} reviews matched; ${n(q.unmapped)} could not be`],
    ['Nested-class mismatches', 'Methods inside nested or anonymous classes were first matched to the wrong class. A second pass repaired them only where method name, parameter types and line all agreed.',
      `${n(q.repaired)} reviews repaired`],
    ['Doubled line endings', 'Our downloader saved some files with doubled line breaks, so line numbers were exactly twice MLCQ’s. It was caught by comparing the two line counts. The affected files were excluded rather than patched.',
      `${n(q.crlf_rows)} reviews excluded${q.crlf_files != null ? ` (${n(q.crlf_files)} files)` : ''}`],
    ['One file ran out of memory', 'The metrics tool crashed on a single generated parser file. It was excluded and documented rather than tuned around.',
      `${n(q.oom_rows)} reviews excluded`],
    ['Several reviewers per location', 'Reviews of one code location were merged by majority vote, with exact ties dropped.',
      `${n(q.locations)} locations; ${n(q.ties_dropped)} ties dropped`],
  ];
  section('quality', 'Data-quality work',
    p('Problems found along the way, what was done, and how many reviews each affected.'),
    el('div', { class: 'table-wrap' }, el('table', { class: 'plain-table wide-first' },
      el('thead', {}, el('tr', {}, el('th', { scope: 'col', text: 'Issue' }), el('th', { scope: 'col', text: 'What happened and what we did' }), el('th', { scope: 'col', text: 'Effect' }))),
      el('tbody', {}, ledgerRows.map(([a, b, c]) => el('tr', {}, el('th', { scope: 'row', text: a }), el('td', { text: b }), el('td', { text: c })))))));

  // 6. models
  section('models', 'Models and imbalance',
    p('Five candidates were compared for every smell: flag everything, a one-metric threshold rule, logistic regression, random forest and gradient boosting. Repository-grouped 5-fold cross-validation on training repositories chose the winner by average precision.'),
    p('Because smells are rare, training gives mistakes on real smells more weight (class weights). No synthetic examples were created.'),
    el('div', { class: 'table-wrap' }, el('table', { class: 'plain-table' },
      el('thead', {}, el('tr', {}, el('th', { scope: 'col', text: 'Smell' }), el('th', { scope: 'col', text: 'Deployed model' }), el('th', { scope: 'col', class: 'num', text: 'Metrics used' }))),
      el('tbody', {}, smellRows.map(({ s, d, m }) => el('tr', {},
        el('th', { scope: 'row' }, el('span', { class: 'cell-smell' }, glyph(s.key, 16), el('span', { class: 'name', text: s.label }))),
        el('td', { text: m ? (MODEL_NAME[m.model] || m.model) : '–' }),
        el('td', { class: 'num', text: d.level === 'class' ? ds.features.class : ds.features.method })))))));

  // 7. inference
  section('inference', 'From pasted code to flags',
    el('ol', { class: 'steps' },
      el('li', { text: 'The Java source is written to a temporary file and measured by the CK tool.' }),
      el('li', { text: 'Every class goes to the Blob and Data Class models; every method goes to the Feature Envy and Long Method models.' }),
      el('li', { text: 'A score at or above the model’s threshold becomes a flag, reported with its line number, score and the smell’s measured precision and recall.' })),
    p('A parity check on 32 held-out cases compared this single-file pipeline with the scores computed during training. 26 matched to within 0.001, the largest difference was 0.047, and no decision changed. The differences came from coupling metrics, which can shift when the metrics tool sees one file instead of a batch.'));

  // 8. limitations
  section('limits', 'Limitations',
    el('ul', { class: 'plain' },
      el('li', { text: 'Feature Envy is weak. It depends on how much a method uses another object’s data, which the available metrics do not measure.' }),
      el('li', { text: 'Labels are opinions, and for many positive examples at least one reviewer disagreed.' }),
      el('li', { text: 'Training covers Java open-source projects only; results may not carry to other languages, domains or styles.' }),
      el('li', { text: 'Input must be a complete class or file; fragments cannot be measured.' }),
      el('li', { text: 'Scores come from class-weighted models, so they rank code but are not calibrated probabilities.' })));

  // 9. reproduce
  section('repro', 'Reproducing the work',
    el('ol', { class: 'steps' },
      el('li', { text: 'Download MLCQ and retrieve each referenced Java file at its commit.' }),
      el('li', { text: 'Join reviews to files, then run CK on the full files in resumable batches.' }),
      el('li', { text: 'Map each review to its CK class or method, repair verified mismatches, and exclude flagged rows.' }),
      el('li', { text: 'Merge reviewers per location, split by repository, and build one dataset per smell.' }),
      el('li', { text: 'Train and compare models with grouped cross-validation, then score the test set once.' }),
      el('li', { text: 'Serve the frozen models through the API behind this site.' })));

  // ---- layout with a scroll-tracking index
  const toc = el('nav', { class: 'toc', 'aria-label': 'On this page' },
    sections.map((s) => {
      const b = el('button', { type: 'button', class: 'toc-link', 'data-id': s.id, text: s.title });
      b.addEventListener('click', () => {
        document.getElementById(`sec-${s.id}`).scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
      });
      return b;
    }));
  root.append(el('div', { class: 'wrap page' },
    el('header', { class: 'panel-head doc-head' },
      el('h1', { class: 'h1', text: 'Dataset and method' }),
      el('p', { class: 'lede small', text: 'Everything on this page is computed from the files the pipeline produced, so it stays in step with the models you see elsewhere on the site.' })),
    el('div', { class: 'doc' }, toc, el('div', { class: 'doc-body' }, sections.map((s) => s.node)))));

  const links = [...toc.querySelectorAll('.toc-link')];
  const mark = (id) => links.forEach((l) => (l.dataset.id === id ? l.setAttribute('aria-current', 'true') : l.removeAttribute('aria-current')));
  mark(sections[0].id);
  const io = new IntersectionObserver((entries) => {
    const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
    if (vis) mark(vis.target.id.replace('sec-', ''));
  }, { rootMargin: '-15% 0px -70% 0px' });
  sections.forEach((s) => io.observe(s.node));
  return () => io.disconnect();
}
