// App shell: hash router, navigation state, backend status.
import { get } from './api.js';
import { el, errorBlock } from './ui.js';

const ROUTES = {
  overview: { title: 'Overview', load: () => import('./pages/overview.js') },
  analyzer: { title: 'Code analyzer', load: () => import('./pages/analyzer.js') },
  models: { title: 'Model intelligence', load: () => import('./pages/models.js') },
  method: { title: 'Dataset and method', load: () => import('./pages/method.js') },
};

function parseHash() {
  const raw = location.hash.replace(/^#\/?/, '');
  const [path, query = ''] = raw.split('?');
  return { route: ROUTES[path] ? path : 'overview', params: new URLSearchParams(query) };
}

let cleanup = null;
let token = 0;

async function render() {
  const { route, params } = parseHash();
  const my = ++token;
  const main = document.getElementById('main');
  document.querySelectorAll('.nav a').forEach((a) => {
    if (a.dataset.route === route) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
  if (cleanup) { try { cleanup(); } catch (e) { /* ignore */ } cleanup = null; }
  main.replaceChildren(el('div', { class: 'wrap page' }, el('p', { class: 'loading', text: 'Loading…' })));
  document.title = `${ROUTES[route].title} | Smell check`;
  try {
    const mod = await ROUTES[route].load();
    if (my !== token) return;
    main.replaceChildren();
    cleanup = (await mod.mount(main, { params })) || null;
    if (my !== token && cleanup) { cleanup(); cleanup = null; }
  } catch (e) {
    if (my === token) main.replaceChildren(errorBlock(e));
  }
  if (my === token) window.scrollTo({ top: 0 });
}

async function showStatus() {
  const box = document.getElementById('status');
  try {
    const h = await get('/health');
    const ok = h.java && h.ck_jar && h.models.length === 4;
    box.className = `status ${ok ? 'ok' : 'warn'}`;
    box.querySelector('.label').textContent = ok ? 'Analyzer ready' : 'Analyzer needs setup';
    box.title = `Java: ${h.java ? 'found' : 'missing'}. Metrics tool: ${h.ck_jar ? 'found' : 'missing'}. Models loaded: ${h.models.length} of 4.`;
  } catch (e) {
    box.className = 'status down';
    box.querySelector('.label').textContent = 'Backend offline';
  }
}

window.addEventListener('hashchange', render);
render();
showStatus();
