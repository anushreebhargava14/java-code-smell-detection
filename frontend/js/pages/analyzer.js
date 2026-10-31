// Code analyzer: paste or open Java code, see where each model flags it, and how far to trust the flag.
import { get, post } from '../api.js';
import { el, SMELLS, BY_KEY, OUTCOME, glyph, dotRow, f2, plural, pct } from '../ui.js';
import { compareChart } from '../charts.js';
import { highlightLines } from '../highlight.js';

const MAX_BYTES = 300000;
const DEFAULT_STATUS = 'Check some code, or open a held-out case, to see flagged classes and methods here.';

export async function mount(root, { params }) {
  const [examples, ref, res] = await Promise.all([
    get('/api/examples').catch(() => []),
    get('/api/reference').catch(() => ({})),
    get('/api/results').catch(() => null),
  ]);
  const stats = Object.fromEntries(((res && res.summary) || []).map((r) => [r.key, r]));
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let rowEls = [];
  let current = null; // held-out example currently loaded, if any

  // ---- toolbar
  const select = el('select', { 'aria-label': 'Open a held-out case' },
    el('option', { value: '', text: examples.length ? 'Open a held-out case…' : 'No held-out cases available' }),
    SMELLS.map((s) => {
      const list = examples.filter((e) => e.smell === s.key);
      return list.length ? el('optgroup', { label: s.label },
        list.map((e) => el('option', { value: e.id, text: `${OUTCOME[e.outcome]}: ${e.name}` }))) : null;
    }));
  select.disabled = !examples.length;
  const checkBtn = el('button', { class: 'btn primary', type: 'button', text: 'Check code' });
  const editBtn = el('button', { class: 'btn', type: 'button', hidden: '', text: 'Edit code' });
  const openBtn = el('button', { class: 'btn', type: 'button', text: 'Open .java file' });
  const fileInput = el('input', { type: 'file', accept: '.java,text/plain', hidden: '' });

  // ---- code pane
  const fileLabel = el('span', { class: 'file', text: 'Input.java' });
  const metaLabel = el('span', { class: 'meta', text: 'Ctrl+Enter to check' });
  const input = el('textarea', { class: 'code-input', spellcheck: 'false', 'aria-label': 'Java source code',
    placeholder: 'Paste a complete Java class here, including its class declaration.' });
  const view = el('div', { class: 'code-view', hidden: '', tabindex: '0',
    'aria-label': 'Checked code with flagged lines marked' });

  // ---- inspector
  const statusEl = el('p', { class: 'insp-status', role: 'status', 'aria-live': 'polite', text: DEFAULT_STATUS });
  const errorEl = el('div', { class: 'notice error', role: 'alert', hidden: '' });
  const reviewEl = el('div', { class: 'review', hidden: '' });
  const findingsEl = el('div', { class: 'findings' });

  const bench = el('div', { class: 'bench' },
    el('section', { class: 'pane-code', 'aria-label': 'Code' },
      el('div', { class: 'code-head' }, fileLabel, metaLabel), input, view),
    el('aside', { class: 'pane-inspect', 'aria-label': 'Findings' }, statusEl, errorEl, reviewEl, findingsEl));

  root.append(el('div', { class: 'wrap wide page analyzer' },
    el('div', { class: 'az-head' },
      el('div', {}, el('h1', { class: 'h1', text: 'Code analyzer' }),
        el('p', { class: 'lede small', text: 'Paste a complete Java class. Each flag shows its score, how often that smell’s flags are right, and how the code’s metrics compare with the training data.' })),
      el('div', { class: 'az-toolbar' }, checkBtn, editBtn, openBtn, fileInput, select)),
    bench));

  // ---- helpers
  const setStatus = (t) => { statusEl.textContent = t; };
  const clearError = () => { errorEl.hidden = true; };
  const showError = (m) => { errorEl.textContent = m; errorEl.hidden = false; };
  function setBusy(on) {
    checkBtn.disabled = on;
    checkBtn.textContent = on ? 'Checking…' : 'Check code';
  }
  function clearResults() {
    findingsEl.replaceChildren();
    reviewEl.hidden = true;
    clearError();
  }
  function toEdit() {
    view.hidden = true;
    input.hidden = false;
    editBtn.hidden = true;
    metaLabel.textContent = 'Ctrl+Enter to check';
    clearResults();
    setStatus(DEFAULT_STATUS);
  }
  function scrollToLine(line, smooth) {
    const row = line != null ? rowEls[line - 1] : null;
    if (!row) return null;
    view.scrollTo({ top: Math.max(0, row.offsetTop - view.clientHeight / 3),
      behavior: smooth && !reduceMotion ? 'smooth' : 'auto' });
    return row;
  }
  function focusLine(line) {
    const row = scrollToLine(line, true);
    if (!row) return;
    row.classList.remove('flash');
    void row.offsetWidth;
    row.classList.add('flash');
  }

  // ---- rendering
  function renderView(code, data) {
    const lines = code.replace(/\r\n/g, '\n').split('\n');
    const frags = highlightLines(lines);
    const marks = new Map();
    for (const f of data.findings) {
      if (f.line == null) continue;
      if (!marks.has(f.line)) marks.set(f.line, new Set());
      marks.get(f.line).add(f.smell);
    }
    const wrap = el('div', { class: 'rows' });
    rowEls = lines.map((text, i) => {
      const ln = i + 1;
      const set = marks.get(ln);
      const inRange = current && ln >= current.start_line && ln <= current.end_line;
      const dots = el('span', { class: 'gdots' });
      if (set) for (const key of set) dots.append(el('span', { class: 'gdot', style: `background:${BY_KEY[key].hex}`, title: BY_KEY[key].label }));
      const src = el('span', { class: 'src' });
      src.append(text ? frags[i] : document.createTextNode(' '));
      const row = el('div', { class: `row${set ? ' flagged' : ''}${inRange ? ' labeled' : ''}` },
        el('span', { class: 'num', text: String(ln) }), dots, src);
      wrap.append(row);
      return row;
    });
    view.replaceChildren(wrap);
    input.hidden = true;
    view.hidden = false;
    editBtn.hidden = false;
    metaLabel.textContent = plural(lines.length, 'line', 'lines');
    if (current) scrollToLine(current.start_line, false);
    else view.scrollTop = 0;
  }

  function scoreMeter(score, thr, color) {
    return el('div', { class: 'meter' },
      el('div', { class: 'track' },
        el('i', { class: 'fill', style: `width:${Math.min(100, score * 100)}%;background:${color}` }),
        el('i', { class: 'tick', style: `left:${thr * 100}%`, title: `flag threshold ${f2(thr)}` })),
      el('span', { class: 'sub', text: `Score ${f2(score)}. Flagged at ${f2(thr)} or higher.` }));
  }

  function compareRows(f, s) {
    const refs = ref[s.key] || {};
    const rows = Object.entries(f.key_metrics || {}).filter(([k, v]) => v != null && refs[k]);
    if (!rows.length) return null;
    return el('div', { class: 'compare' },
      el('p', { class: 'sub', text: 'This code against the training data:' }),
      rows.map(([k, v]) => el('div', { class: 'cmp-row' },
        el('span', { class: 'cmp-name', text: k }),
        compareChart({ value: v, ref: refs[k], color: s.hex }),
        el('span', { class: 'cmp-val', text: `${Number.isInteger(v) ? v : v.toFixed(2)}`,
          title: `Typical flagged: ${refs[k].pos_median}. Typical unflagged: ${refs[k].neg_median}.` }))),
      el('p', { class: 'sub key',
        text: 'Bar: this code. Colored band and dot: the middle half and median of flagged training cases. Hollow dot: median of unflagged cases. Large values are compressed.' }));
  }

  function findingCard(f, s) {
    const body = el('div', { class: 'finding-body', hidden: '' },
      scoreMeter(f.probability, f.threshold, s.hex), compareRows(f, s));
    const head = el('button', { class: 'finding-head', type: 'button', 'aria-expanded': 'false' },
      el('span', { class: 'f-name', text: f.name, title: f.name }),
      el('span', { class: 'f-line', text: f.line != null ? `line ${f.line}` : '' }),
      el('span', { class: 'f-score', text: f2(f.probability) }));
    head.addEventListener('click', () => {
      const open = body.hidden;
      body.hidden = !open;
      head.setAttribute('aria-expanded', String(open));
      focusLine(f.line);
    });
    return el('div', { class: 'finding' }, head, body);
  }

  function renderFindings(data) {
    findingsEl.replaceChildren();
    const order = [...SMELLS].sort((a, b) => ((stats[b.key] || {}).precision || 0) - ((stats[a.key] || {}).precision || 0));
    for (const s of order) {
      const scored = data.scores.filter((x) => x.smell === s.key);
      if (!scored.length) continue;
      const flagged = scored.filter((x) => x.flagged).sort((a, b) => b.probability - a.probability);
      const precision = (stats[s.key] || {}).precision;
      const right = precision != null ? Math.round(precision * 10) : null;
      const group = el('section', { class: 'group' },
        el('div', { class: 'group-head' }, glyph(s.key, 16), el('h3', { text: s.label }),
          el('span', { class: 'count', text: `${flagged.length} of ${plural(scored.length, ...s.unit)}` })),
        right != null ? el('div', { class: 'trust' }, dotRow(right, s.hex),
          el('span', { class: 'sub', text: `${right} in 10 flags are right` })) : null,
        flagged.length && precision != null && precision < 0.25
          ? el('p', { class: 'caution', text: `Only about ${right} in 10 ${s.label} flags are right, so read these as places to look.` }) : null,
        flagged.length ? flagged.map((f) => findingCard(f, s)) : el('p', { class: 'none', text: 'Nothing flagged.' }));
      findingsEl.append(group);
    }
  }

  function renderReview(data) {
    if (!current) { reviewEl.hidden = true; return; }
    const s = BY_KEY[current.smell];
    const a = current.start_line, b = current.end_line;
    const hits = data.scores.filter((x) => x.smell === current.smell && x.line != null && x.line >= a - 2 && x.line <= b)
      .sort((p, q) => Math.abs(p.line - a) - Math.abs(q.line - a));
    const hit = hits[0];
    const said = Math.round(current.label_mean * current.n_reviews);
    let verdict;
    if (current.label === 1) verdict = hit && hit.flagged ? 'Agreement: reviewers and model both marked it.' : 'Miss: reviewers marked it and the model did not.';
    else verdict = hit && hit.flagged ? 'False alarm: the model flagged it and most reviewers did not.' : 'Agreement: neither marked it.';
    reviewEl.replaceChildren(
      el('div', { class: 'review-head' }, glyph(s.key, 16),
        el('strong', { text: `Held-out case: ${current.name}` })),
      el('p', { text: `${said} of ${plural(current.n_reviews, 'reviewer', 'reviewers')} marked this a ${s.label} (lines ${a} to ${b}, outlined in the code).` }),
      el('p', { text: hit
        ? `The model scored it ${f2(hit.probability)}, ${hit.flagged ? 'above' : 'below'} its ${f2(hit.threshold)} threshold.`
        : 'The model gave this location no score; it may not be an entity the metrics tool reports.' }),
      el('p', { class: 'verdict', text: verdict }));
    reviewEl.hidden = false;
  }

  async function analyze() {
    const code = input.value;
    clearError();
    if (!code.trim()) { setStatus('Paste a Java class first.'); return; }
    if (new Blob([code]).size > MAX_BYTES) { showError('This code is larger than 300 KB. Check a smaller class or split the file.'); return; }
    setBusy(true);
    setStatus('Checking code. This takes a few seconds.');
    try {
      const data = await post('/analyze', { code });
      renderView(code, data);
      renderFindings(data);
      renderReview(data);
      const notes = [...new Set(data.warnings || [])];
      setStatus(`Checked ${plural(data.n_classes, 'class', 'classes')} and ${plural(data.n_methods, 'method', 'methods')}; ${data.findings.length} flagged.`
        + (notes.length ? ` Note: ${notes.join('; ')}.` : ''));
    } catch (e) {
      let msg = e.status ? e.message : 'Cannot reach the server. Start it with "python -m uvicorn api.main:app --port 8000" and open http://127.0.0.1:8000/.';
      if (e.status === 422) msg += ' Paste a complete class, including its class declaration.';
      showError(msg);
      setStatus(DEFAULT_STATUS);
      findingsEl.replaceChildren();
      reviewEl.hidden = true;
    } finally {
      setBusy(false);
    }
  }

  async function loadExample(id) {
    const e = examples.find((x) => x.id === id);
    if (!e) return;
    toEdit();
    setStatus('Loading the held-out case…');
    try {
      const d = await get(`/api/example/${e.file_id}`);
      current = e;
      input.value = d.source;
      fileLabel.textContent = `${e.name}.java`;
      select.value = id;
      await analyze();
    } catch (err) {
      showError(err.message);
      setStatus(DEFAULT_STATUS);
    }
  }

  // ---- events
  checkBtn.addEventListener('click', analyze);
  editBtn.addEventListener('click', () => { toEdit(); input.focus(); });
  openBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const text = await f.text();
    current = null;
    select.value = '';
    toEdit();
    input.value = text;
    fileLabel.textContent = f.name;
    setStatus(`Loaded ${f.name}. Press Check code.`);
    e.target.value = '';
  });
  select.addEventListener('change', () => { if (select.value) loadExample(select.value); });
  input.addEventListener('input', () => {
    if (current) { current = null; select.value = ''; fileLabel.textContent = 'Input.java'; reviewEl.hidden = true; }
  });
  input.addEventListener('keydown', (e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyze(); });

  if (params.get('ex')) loadExample(params.get('ex'));
  else input.focus({ preventScroll: true });
}
