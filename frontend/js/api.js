// Small fetch helpers with caching for read-only data.
const cache = new Map();

async function parse(r) {
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const err = new Error(data.detail || `Request failed (${r.status})`);
    err.status = r.status;
    throw err;
  }
  return data;
}

export function get(url) {
  if (!cache.has(url)) {
    cache.set(url, fetch(url).then(parse).catch((e) => { cache.delete(url); throw e; }));
  }
  return cache.get(url);
}

export async function post(url, body) {
  const r = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  return parse(r);
}
