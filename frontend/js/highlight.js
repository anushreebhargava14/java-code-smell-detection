// Tiny Java syntax highlighter (keywords, strings, comments, numbers, types, annotations).
const KW = new Set(('abstract assert boolean break byte case catch char class const continue default do double else '
  + 'enum extends final finally float for goto if implements import instanceof int interface long native new package '
  + 'private protected public return short static strictfp super switch synchronized this throw throws transient try '
  + 'void volatile while var record sealed permits yield null true false').split(' '));

function tokenizeLine(line, state) {
  const frag = document.createDocumentFragment();
  let plain = '';
  const flush = () => { if (plain) { frag.append(document.createTextNode(plain)); plain = ''; } };
  const push = (cls, text) => {
    flush();
    const s = document.createElement('span');
    s.className = `tk-${cls}`;
    s.textContent = text;
    frag.append(s);
  };
  const len = line.length;
  let i = 0;
  while (i < len) {
    if (state.block) {
      const e = line.indexOf('*/', i);
      if (e < 0) { push('c', line.slice(i)); i = len; } else { push('c', line.slice(i, e + 2)); i = e + 2; state.block = false; }
      continue;
    }
    const ch = line[i];
    if (ch === '/' && line[i + 1] === '/') { push('c', line.slice(i)); i = len; continue; }
    if (ch === '/' && line[i + 1] === '*') {
      const e = line.indexOf('*/', i + 2);
      if (e < 0) { push('c', line.slice(i)); i = len; state.block = true; } else { push('c', line.slice(i, e + 2)); i = e + 2; }
      continue;
    }
    if (ch === '"' || ch === "'") {
      let j = i + 1;
      while (j < len && line[j] !== ch) { if (line[j] === '\\') j++; j++; }
      push('s', line.slice(i, Math.min(j + 1, len)));
      i = j + 1;
      continue;
    }
    if (ch >= '0' && ch <= '9') {
      const m = /^(0[xX][0-9a-fA-F_]+|\d[\d_]*(\.\d+)?([eE][+-]?\d+)?[lLfFdD]?)/.exec(line.slice(i));
      push('n', m[0]);
      i += m[0].length;
      continue;
    }
    if (/[A-Za-z_$@]/.test(ch)) {
      const m = /^@?[A-Za-z_$][\w$]*/.exec(line.slice(i));
      if (!m) { plain += ch; i++; continue; }
      const w = m[0];
      if (w[0] === '@') push('a', w);
      else if (KW.has(w)) push('k', w);
      else if (/^[A-Z]/.test(w)) push('t', w);
      else if (/^\s*\(/.test(line.slice(i + w.length))) push('f', w);
      else plain += w;
      i += w.length;
      continue;
    }
    plain += ch;
    i++;
  }
  flush();
  return frag;
}

/** Returns one DocumentFragment per input line. */
export function highlightLines(lines) {
  const state = { block: false };
  return lines.map((l) => tokenizeLine(l, state));
}
