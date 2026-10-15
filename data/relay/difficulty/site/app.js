'use strict';
// Student difficulty viewer. Static: reads data/index.json and data/traj/<id>.json (build_site.py); never runs a model.

const CLASSES = ['reasoning', 'prose', 'command', 'think_tag', 'call_fmt', 'eot'];
const CONTENT = [0, 1, 2];
const FORMAT = [3, 4, 5];
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const SPECIAL = { '<|start_think|>': '⟨think⟩', '<|end_think|>': '⟨/think⟩', '<|eot_id|>': '⟨eot⟩' };
const showSpecial = s => s.replace(/<\|(start_think|end_think|eot_id)\|>/g, m => SPECIAL[m]);
const fmt = (x, d = 2) => (x == null || Number.isNaN(x)) ? '–' : Number(x).toFixed(d);
const pct = x => (x == null || Number.isNaN(x)) ? '–' : (100 * x).toFixed(1) + '%';

const store = {
  get(k, d) { try { const v = localStorage.getItem('sdiff.' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('sdiff.' + k, JSON.stringify(v)); } catch (e) { /* private window */ } },
};

const S = Object.assign({
  tag: null, dA: null, dB: null, mode: 'global', floor: 1, smax: 8, pctTop: 15, dmax: 3, tau: 2.3, gap: 3, smooth: 1,
  show: { reasoning: true, prose: true, command: true, format: false },
  ws: false, bounds: false, foldTools: true, metric: 'mean', trajSort: ['content_mean', -1], turnSort: ['k', 1],
}, store.get('settings', {}));
let INDEX = null, CFG = null, TRAJ = null, G = null, SPANS = [], spanPtr = -1, pinned = null, rangeA = null, range = null;
let globalSpans = null;

// ---------- colors ----------
const STOPS = [[255, 244, 179], [255, 196, 87], [244, 112, 60], [205, 40, 72], [110, 24, 120]];
function seqColor(t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (STOPS.length - 1), i = Math.min(STOPS.length - 2, Math.floor(x)), f = x - i;
  const c = STOPS[i].map((v, j) => Math.round(v + f * (STOPS[i + 1][j] - v)));
  const a = 0.35 + 0.65 * Math.min(1, t * 1.6);
  return `rgba(${c[0]},${c[1]},${c[2]},${a.toFixed(3)})`;
}
function divColor(t) {  // t in [-1, 1]: blue = easier for the second checkpoint, red = harder
  t = Math.max(-1, Math.min(1, t));
  const c = t > 0 ? [214, 40, 40] : [34, 102, 214];
  return `rgba(${c[0]},${c[1]},${c[2]},${(0.08 + 0.75 * Math.abs(t)).toFixed(3)})`;
}
const isDelta = () => S.tag === 'delta';
const tags = () => CFG.tags.map(t => t.tag);
const tagLabel = t => t === 'delta' ? `Δ ${tagLabel(S.dB)} − ${tagLabel(S.dA)}` : (CFG.tags.find(x => x.tag === t) || {}).label || t;

// ---------- data ----------
async function getJSON(u) { const r = await fetch(u); if (!r.ok) throw new Error(u + ': ' + r.status); return r.json(); }

function flatten(traj) {
  let n = 0;
  for (const t of traj.turns) n += t.ids.length;
  const g = { n, turn: new Int32Array(n), off: new Int32Array(n), abs: new Int32Array(n), cls: new Uint8Array(n),
    mixed: new Uint8Array(n), id: new Int32Array(n), piece: new Array(n), lp: {}, rk: {}, top: {}, tlp: {},
    tStart: [], tEnd: [] };
  for (const tag of tags()) { g.lp[tag] = new Float32Array(n); g.rk[tag] = new Int32Array(n); g.top[tag] = new Array(n); g.tlp[tag] = new Array(n); }
  let i = 0;
  for (const t of traj.turns) {
    g.tStart.push(i);
    const mixed = new Set(t.mixed);
    for (let j = 0; j < t.ids.length; j++, i++) {
      g.turn[i] = t.k; g.off[i] = j; g.abs[i] = t.tok_start + j; g.cls[i] = t.cls.charCodeAt(j) - 48;
      g.mixed[i] = mixed.has(j) ? 1 : 0; g.id[i] = t.ids[j]; g.piece[i] = t.pieces[j];
      for (const tag of tags()) {
        const s = t.s[tag];
        g.lp[tag][i] = s.lp[j]; g.rk[tag][i] = s.rk[j]; g.top[tag][i] = s.top[j]; g.tlp[tag][i] = s.tlp[j];
      }
    }
    g.tEnd.push(i);
  }
  return g;
}

const nllOf = (tag, i) => -G.lp[tag][i];
function visible(i) {
  const c = G.cls[i];
  return c < 3 ? S.show[CLASSES[c]] : S.show.format;
}
function rawVal(i) {
  if (isDelta()) return nllOf(S.dB, i) - nllOf(S.dA, i);
  return nllOf(S.tag, i);
}
const spanTag = () => isDelta() ? S.dA : S.tag;

let VAL = null;    // displayed value per token (smoothed if asked); NaN for tokens not scored under the class filter
function computeValues() {
  VAL = new Float32Array(G.n);
  const w = +S.smooth, h = (w - 1) >> 1;
  for (let k = 0; k < G.tStart.length; k++) {
    const a = G.tStart[k], b = G.tEnd[k];
    const idx = [];
    for (let i = a; i < b; i++) { if (visible(i)) idx.push(i); else VAL[i] = NaN; }
    if (w <= 1) { for (const i of idx) VAL[i] = rawVal(i); continue; }
    const pre = new Float64Array(idx.length + 1);
    for (let j = 0; j < idx.length; j++) pre[j + 1] = pre[j] + rawVal(idx[j]);
    for (let j = 0; j < idx.length; j++) {
      const lo = Math.max(0, j - h), hi = Math.min(idx.length, j + h + 1);
      VAL[idx[j]] = (pre[hi] - pre[lo]) / (hi - lo);
    }
  }
}

// per-turn percentile of each visible token's displayed value
let PCT = null;
function computePct() {
  PCT = new Float32Array(G.n).fill(NaN);
  for (let k = 0; k < G.tStart.length; k++) {
    const idx = [];
    for (let i = G.tStart[k]; i < G.tEnd[k]; i++) if (!Number.isNaN(VAL[i])) idx.push(i);
    idx.sort((x, y) => VAL[x] - VAL[y]);
    idx.forEach((i, r) => { PCT[i] = idx.length > 1 ? r / (idx.length - 1) : 1; });
  }
}

function colorOf(i) {
  const v = VAL[i];
  if (Number.isNaN(v)) return null;
  if (isDelta()) return Math.abs(v) < 0.05 ? null : divColor(v / S.dmax);
  if (S.mode === 'pct') {
    const p = PCT[i], cut = 1 - S.pctTop / 100;
    return p < cut ? null : seqColor((p - cut) / Math.max(1e-6, 1 - cut));
  }
  if (v < S.floor) return null;
  return seqColor((v - S.floor) / Math.max(1e-6, S.smax - S.floor));
}
function hotness(i) {   // 0..1, for the white-ink threshold
  const v = VAL[i];
  if (Number.isNaN(v)) return 0;
  if (isDelta()) return Math.abs(v) / S.dmax;
  if (S.mode === 'pct') { const cut = 1 - S.pctTop / 100; return PCT[i] < cut ? 0 : (PCT[i] - cut) / (1 - cut); }
  return (v - S.floor) / (S.smax - S.floor);
}

// ---------- turn stats & spans (current settings) ----------
function turnStats(k, tag = spanTag()) {
  let n = 0, sum = 0, high = 0, max = 0;
  const byc = CLASSES.map(() => [0, 0]);
  for (let i = G.tStart[k]; i < G.tEnd[k]; i++) {
    const x = nllOf(tag, i);
    byc[G.cls[i]][0] += x; byc[G.cls[i]][1]++;
    if (!visible(i)) continue;
    n++; sum += x; if (x >= S.tau) high++; if (x > max) max = x;
  }
  return { n, mean: n ? sum / n : NaN, sum, high: n ? high / n : NaN, max, byc: byc.map(([s, c]) => c ? s / c : NaN) };
}
function turnSpans(k, tag = spanTag()) {
  const out = []; let cur = null, last = -1;
  for (let i = G.tStart[k]; i < G.tEnd[k]; i++) {
    if (!visible(i) || G.cls[i] >= 3) continue;   // spans are made of content tokens
    if (nllOf(tag, i) >= S.tau) {
      if (cur && i - last <= S.gap + 1) cur.b = i; else { if (cur) out.push(cur); cur = { a: i, b: i }; }
      last = i;
    }
  }
  if (cur) out.push(cur);
  for (const s of out) {
    let sum = 0, nh = 0, max = 0; const cc = {};
    for (let i = s.a; i <= s.b; i++) {
      const x = nllOf(tag, i); sum += x; max = Math.max(max, x);
      if (x >= S.tau && G.cls[i] < 3) nh++;
      cc[G.cls[i]] = (cc[G.cls[i]] || 0) + 1;
    }
    Object.assign(s, { k, sum, max, nh, len: s.b - s.a + 1, type: nh === 1 ? 'isolated' : nh <= 3 ? 'burst' : 'extended',
      cls: CLASSES[+Object.keys(cc).sort((x, y) => cc[y] - cc[x])[0]] });
  }
  return out;
}

// ---------- rendering: trajectory table ----------
function trajMetric(t, key) {
  const pick = tag => t.summary[tag];
  const m = (tag) => {
    const s = pick(tag);
    return { content_mean: s.content.mean, all_mean: s.all.mean, content_high: s.content.high, spans: s.n_spans / s.content.n * 100 }[key];
  };
  if (key === 'traj_id') return t.traj_id;
  if (key === 'domain') return t.domain;
  if (key === 'n_turns') return t.n_turns;
  if (key === 'n_targets') return t.n_targets;
  if (isDelta()) return m(S.dB) - m(S.dA);
  return m(S.tag);
}
function renderTrajTable() {
  const cols = [['traj_id', 'id'], ['domain', 'domain'], ['n_turns', 'turns', 1], ['n_targets', 'targets', 1],
    ['content_mean', isDelta() ? 'Δμ' : 'μ NLL', 1], ['content_high', isDelta() ? 'Δhigh pp' : 'high', 1], ['spans', 'sp/100', 1]];
  const [sk, sd] = S.trajSort;
  const rows = INDEX.trajectories.slice().sort((x, y) => {
    const a = trajMetric(x, sk), b = trajMetric(y, sk);
    return (typeof a === 'string' ? a.localeCompare(b) : a - b) * sd;
  });
  const vals = INDEX.trajectories.map(t => trajMetric(t, 'content_mean'));
  const vmax = Math.max(...vals.map(Math.abs));
  let h = '<tr>' + cols.map(([k, l, num]) => `<th data-k="${k}" class="${num ? 'num' : ''} ${k === sk ? 'sorted' + (sd > 0 ? ' asc' : '') : ''}" title="${{
    content_mean: 'mean NLL over content tokens (reasoning, prose, command)', content_high: 'fraction of content tokens with NLL ≥ the build τ (' + CFG.tau + ')',
    spans: 'high-loss spans (build τ / gap) per 100 content tokens', n_targets: 'scored teacher tokens' }[k] || ''}">${l}</th>`).join('') + '</tr>';
  for (const t of rows) {
    const v = trajMetric(t, 'content_mean');
    const sw = isDelta() ? divColor(v / Math.max(0.01, vmax)) : seqColor(v / Math.max(0.01, vmax));
    h += `<tr class="row ${TRAJ && TRAJ.meta.traj_id === t.traj_id ? 'sel' : ''}" data-id="${t.traj_id}">
      <td><span class="swatch" style="background:${sw}"></span>${t.traj_id}</td><td><span class="dom" title="${esc(t.task)}">${esc(t.domain)}</span></td>
      <td class="num">${t.n_turns}</td><td class="num">${(t.n_targets / 1000).toFixed(1)}k</td>
      <td class="num">${fmt(v, 3)}</td><td class="num">${isDelta() ? fmt(trajMetric(t, 'content_high') * 100, 1) : pct(trajMetric(t, 'content_high'))}</td>
      <td class="num">${fmt(trajMetric(t, 'spans'), 2)}</td></tr>`;
  }
  $('#traj-table').innerHTML = h;
}

// ---------- rendering: conversation ----------
function wsMarks(s) { return s.replace(/ /g, '·').replace(/\t/g, '→'); }
function tokenHTML(i) {
  const c = G.cls[i], p = G.piece[i];
  const cl = `t c${c}${c >= 3 ? ' f' : ''}`;
  if (c === 3 || c === 5) {
    const lab = p === '<|start_think|>' ? '⟨think⟩' : p === '<|end_think|>' ? '⟨/think⟩' : p === '<|eot_id|>' ? '⟨eot⟩' : esc(p);
    return `<span class="${cl} sp" data-g="${i}">${lab}</span>`;
  }
  let txt = esc(p);
  if (p.includes('\n')) txt = txt.replace(/\n/g, '<span class="nl">↵</span>\n');
  return `<span class="${cl}" data-g="${i}">${txt}</span>`;
}
function turnBodyHTML(k) {
  const a = G.tStart[k], b = G.tEnd[k];
  let h = '', cur = null, buf = '', inThink = false;
  const open = g => {
    if (cur === g) return;
    if (cur) h += buf + '</div></div>';
    buf = ''; cur = g;
    const lab = { think: 'thinking', prose: 'prose', call: 'tool call' }[g];
    h += `<div class="blk blk-${g}"><div class="blk-label">${lab}</div><div class="tx">`;
  };
  for (let i = a; i < b; i++) {
    const c = G.cls[i];
    if (c === 3) { if (G.piece[i] === '<|start_think|>') { inThink = true; open('think'); } buf += tokenHTML(i); if (G.piece[i] === '<|end_think|>') inThink = false; continue; }
    if (c === 5) { if (!cur) open('prose'); buf += tokenHTML(i); continue; }
    const g = inThink ? 'think' : (c === 2 || c === 4) ? 'call' : 'prose';
    open(g); buf += tokenHTML(i);
  }
  if (cur) h += buf + '</div></div>';
  return h;
}
function toolHTML(text) {
  const m = text.match(/<tool_response[^>]*>([\s\S]*)<\/tool_response>/);
  const inner = m ? m[1] : text;
  try {
    const o = JSON.parse(inner);
    if (o && typeof o === 'object' && 'output' in o) {
      const rc = o.returncode;
      return `<div class="hint">returncode <span class="rc ${rc === 0 ? '' : 'bad'}">${esc(String(rc))}</span>${o.exception_info ? ' · ' + esc(String(o.exception_info)) : ''}</div>
        <pre class="plain tool-out ${S.foldTools ? 'folded' : ''}">${esc(String(o.output))}</pre>`;
    }
  } catch (e) { /* not JSON: show as sent */ }
  return `<pre class="plain tool-out ${S.foldTools ? 'folded' : ''}">${esc(inner)}</pre>`;
}
function stripHeader(text) {
  return text.replace(/^<\|begin_of_text\|>/, '').replace(/^<\|start_header_id\|>\w+<\|end_header_id\|>\n?/, '').replace(/<\|eot_id\|>\n?$/, '');
}
function turnHeadStats(k) {
  const st = turnStats(k), sp = SPANS.filter(s => s.k === k).length, t = TRAJ.turns[k];
  const beyond = t.history_tokens >= CFG.eval_context ? ` <span class="beyond-tag" title="history longer than the MSA2 eval's ${CFG.eval_context}-token context">beyond eval context</span>` : '';
  const bc = st.byc;
  return `<span class="stats">${st.n.toLocaleString()} tok · μ <b>${fmt(st.mean, 3)}</b> · high <b>${pct(st.high)}</b> · ${sp} span${sp === 1 ? '' : 's'} ·
    <span title="mean NLL by class (all tokens of the class)">think ${fmt(bc[0], 2)} / prose ${fmt(bc[1], 2)} / cmd ${fmt(bc[2], 2)} / fmt ${fmt((bc[3] + bc[4] + bc[5]) / 3, 2)}</span>
    · history ${t.history_tokens.toLocaleString()} tok${beyond}</span>`;
}
function renderConvo() {
  let h = '';
  for (const m of TRAJ.messages) {
    if (m.role === 'assistant') {
      const k = m.turn;
      h += `<div class="msg assistant" id="turn-${k}" data-k="${k}"><div class="msg-head"><span class="role">Turn ${k}</span>
        <span class="hs">${turnHeadStats(k)}</span><span class="acts"><button class="small bm-turn" data-k="${k}" title="bookmark this turn">★</button></span></div>
        <div class="msg-body">${turnBodyHTML(k)}</div></div>`;
    } else if (m.role === 'tool') {
      h += `<div class="msg tool"><div class="msg-head"><span class="role">tool output</span><span class="hint">context only (not scored) · ${(m.tok_end - m.tok_start).toLocaleString()} tok</span>
        <span class="acts"><button class="small unfold">unfold</button></span></div><div class="msg-body">${toolHTML(m.text)}</div></div>`;
    } else {
      const body = m.role === 'template' ? m.text.replace(/^<\|begin_of_text\|>/, '') : stripHeader(m.text);
      const fold = m.role === 'template' || body.length > 2500;
      const lab = m.role === 'template' ? 'chat template header (tools block)' : m.role;
      h += `<div class="msg ${m.role}"><div class="msg-head"><span class="role">${lab}</span><span class="hint">context only · ${(m.tok_end - m.tok_start).toLocaleString()} tok</span>
        ${fold ? '<span class="acts"><button class="small unfold">unfold</button></span>' : ''}</div>
        <div class="msg-body"><pre class="plain ${fold ? 'tool-out folded' : ''}">${esc(body)}</pre></div></div>`;
    }
  }
  const el = $('#convo');
  el.innerHTML = h;
  el.classList.toggle('ws', S.ws);
  el.classList.toggle('bounds', S.bounds);
}
function recolor() {
  computeValues(); computePct();
  const showSet = new Set(CONTENT.filter(c => S.show[CLASSES[c]]).concat(S.show.format ? FORMAT : []));
  for (const el of $$('#convo .t')) {
    const i = +el.dataset.g, col = colorOf(i);
    el.style.backgroundColor = col || '';
    el.classList.toggle('hot', !!col);
    el.classList.toggle('ink', !!col && hotness(i) > 0.72);
    el.classList.toggle('off', !showSet.has(G.cls[i]));
    el.classList.toggle('hi', showSet.has(G.cls[i]) && nllOf(spanTag(), i) >= S.tau && !isDelta());
  }
  SPANS = [];
  for (let k = 0; k < G.tStart.length; k++) SPANS.push(...turnSpans(k));
  spanPtr = -1;
  $('#span-pos').textContent = `${SPANS.length} spans`;
  for (const el of $$('#convo .msg.assistant')) $('.hs', el).innerHTML = turnHeadStats(+el.dataset.k);
  renderOverview(); renderLocalSpans(); drawLegend();
}

// ---------- overview ----------
function metricOf(k) {
  const st = turnStats(k), n = Math.max(1, st.n);
  const sp = SPANS.filter(s => s.k === k).length;
  return { mean: st.mean, high: st.high, spans: 100 * sp / n, max: st.max, sum: st.sum }[S.metric];
}
function renderOverview() {
  const K = G.tStart.length, vals = [], means = [];
  for (let k = 0; k < K; k++) { vals.push(metricOf(k)); means.push(turnStats(k).mean); }
  const vmax = Math.max(...vals.filter(x => !Number.isNaN(x)), 1e-6);
  const mmax = Math.max(...means.filter(x => !Number.isNaN(x)), 1e-6);
  let h = '';
  for (let k = 0; k < K; k++) {
    const v = vals[k], ht = Number.isNaN(v) ? 0 : Math.max(3, 100 * v / vmax);
    const beyond = TRAJ.turns[k].history_tokens >= CFG.eval_context;
    h += `<div class="bar ${beyond ? 'beyond' : ''}" data-k="${k}" style="height:${ht}%;background:${seqColor(means[k] / mmax)}"
      title="turn ${k}: ${S.metric} ${fmt(v, 3)} · mean ${fmt(means[k], 3)} · ${G.tEnd[k] - G.tStart[k]} tokens${beyond ? ' · beyond eval context' : ''}"></div>`;
  }
  $('#bars').innerHTML = h;
  drawHeat();
  renderTurnTable();
}
function drawHeat() {
  const cv = $('#heat'), K = G.tStart.length;
  const W = Math.max(200, Math.floor(cv.clientWidth || 800)), rowH = 6, H = K * rowH;
  cv.width = W; cv.height = H; cv.style.height = H + 'px';
  const ctx = cv.getContext('2d');
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--sys') || '#eee';
  let maxLen = 0;
  for (let k = 0; k < K; k++) maxLen = Math.max(maxLen, G.tEnd[k] - G.tStart[k]);
  const per = maxLen / W;
  cv._per = per; cv._rowH = rowH;
  for (let k = 0; k < K; k++) {
    const a = G.tStart[k], len = G.tEnd[k] - a;
    ctx.fillStyle = 'rgba(128,128,128,.12)';
    ctx.fillRect(0, k * rowH, Math.ceil(len / per), rowH - 1);
    for (let x = 0; x * per < len; x++) {
      let best = null, bh = -1;
      for (let i = a + Math.floor(x * per); i < a + Math.min(len, Math.floor((x + 1) * per) + 1); i++) {
        const c = colorOf(i); if (!c) continue;
        const hv = hotness(i); if (hv > bh) { bh = hv; best = c; }
      }
      if (best) { ctx.fillStyle = best; ctx.fillRect(x, k * rowH, 1, rowH - 1); }
    }
  }
}
function renderTurnTable() {
  if ($('#turn-table-wrap').hidden) return;
  const cols = [['k', 'turn'], ['n', 'tokens'], ['mean', 'μ'], ['high', 'high'], ['spans', 'sp/100'], ['max', 'max'], ['sum', 'total'],
    ['r', 'μ think'], ['p', 'μ prose'], ['c', 'μ cmd'], ['hist', 'history']];
  const rows = [];
  for (let k = 0; k < G.tStart.length; k++) {
    const st = turnStats(k), sp = SPANS.filter(s => s.k === k).length;
    rows.push({ k, n: st.n, mean: st.mean, high: st.high, spans: 100 * sp / Math.max(1, st.n), max: st.max, sum: st.sum,
      r: st.byc[0], p: st.byc[1], c: st.byc[2], hist: TRAJ.turns[k].history_tokens });
  }
  const [sk, sd] = S.turnSort;
  rows.sort((x, y) => ((Number.isNaN(x[sk]) ? -1e9 : x[sk]) - (Number.isNaN(y[sk]) ? -1e9 : y[sk])) * sd);
  let h = '<tr>' + cols.map(([k, l]) => `<th class="num ${k === sk ? 'sorted' + (sd > 0 ? ' asc' : '') : ''}" data-k="${k}">${l}</th>`).join('') + '</tr>';
  for (const r of rows) {
    h += `<tr class="row" data-k="${r.k}">` + cols.map(([k]) => `<td class="num">${
      k === 'high' ? pct(r[k]) : ['k', 'n', 'hist'].includes(k) ? r[k].toLocaleString() : fmt(r[k], k === 'sum' ? 0 : 3)}</td>`).join('') + '</tr>';
  }
  $('#turn-table').innerHTML = h;
}

// ---------- inspector ----------
function visWS(s) {
  return esc(s).replace(/ /g, '<span class="wsm">·</span>').replace(/\t/g, '<span class="wsm">→</span>')
    .replace(/\n/g, '<span class="wsm">↵</span>\n').replace(/\r/g, '<span class="wsm">␍</span>');
}
function vocabPiece(id) { const v = TRAJ.vocab[id]; return v ? v[0] : '#' + id; }
function inspect(i) {
  const k = G.turn[i], t = TRAJ.turns[k];
  const piece = G.piece[i], raw = (TRAJ.vocab[G.id[i]] || [])[1];
  const tagRows = tags().map(tag => {
    const lp = G.lp[tag][i];
    return `<tr><td>${esc(tagLabel(tag))}</td><td class="ckcol">p ${fmt(Math.exp(lp), 4)} · NLL <b>${fmt(-lp, 3)}</b> · rank ${G.rk[tag][i]}</td></tr>`;
  }).join('');
  const dl = isDelta() ? `<tr><td>${esc(tagLabel('delta'))}</td><td class="ckcol"><b>${fmt(nllOf(S.dB, i) - nllOf(S.dA, i), 3)}</b> nats</td></tr>` : '';
  const atag = spanTag();
  const alts = G.top[atag][i].map((id, j) => {
    const p = Math.exp(G.tlp[atag][i][j]);
    return `<div class="alt ${id === G.id[i] ? 'teacher' : ''}"><div class="bar"><i style="width:${(100 * p).toFixed(1)}%"></i><span>${visWS(vocabPiece(id))}${id === G.id[i] ? '  ← teacher' : ''}</span></div><div class="p">${(100 * p).toFixed(1)}%</div></div>`;
  }).join('');
  const inTop = G.top[atag][i].includes(G.id[i]);
  const a = Math.max(G.tStart[k], i - 40), b = Math.min(G.tEnd[k], i + 25);
  let ctx = '';
  for (let j = a; j < b; j++) ctx += j === i ? `<mark>${esc(G.piece[j])}</mark>` : esc(G.piece[j]);
  const smoothed = S.smooth > 1 && !Number.isNaN(VAL[i]) ? `<tr><td>smoothed (${S.smooth})</td><td>${fmt(VAL[i], 3)}</td></tr>` : '';
  const rangeBtn = range ? `<button id="bm-range">★ bookmark selected range (${range.b - range.a + 1} tok)</button>` : '';
  $('#inspector').innerHTML = `
    <div class="tok">${visWS(piece) || '<span class="hint">(empty piece: byte of a split character)</span>'}</div>
    <table>
      <tr><td>token</td><td>id ${G.id[i]} · BPE <code>${esc(raw || '')}</code>${G.mixed[i] ? ' · <b>spans two classes</b>' : ''}</td></tr>
      <tr><td>JSON</td><td><code>${esc(JSON.stringify(piece))}</code></td></tr>
      <tr><td>class</td><td>${CLASSES[G.cls[i]]}</td></tr>
      <tr><td>position</td><td>turn ${k}, token ${G.off[i]} of ${G.tEnd[k] - G.tStart[k]} · sequence position ${G.abs[i].toLocaleString()}${G.abs[i] >= CFG.eval_context ? ' <span class="beyond-tag">beyond eval context</span>' : ''}</td></tr>
      ${tagRows}${dl}${smoothed}
    </table>
    <div class="alts"><div class="hint">top-${G.top[atag][i].length} of ${esc(tagLabel(atag))}${inTop ? '' : ' (teacher token not among them)'}</div>${alts}</div>
    <div class="ctx">${ctx}</div>
    <div style="margin-top:6px;display:flex;gap:6px;flex-wrap:wrap"><button id="bm-tok">★ bookmark token</button>${rangeBtn}</div>`;
  const bt = $('#bm-tok'); if (bt) bt.onclick = () => addBookmark({ k, a: i, b: i });
  const br = $('#bm-range'); if (br) br.onclick = () => addBookmark({ k: G.turn[range.a], a: range.a, b: range.b });
}

// ---------- spans: local list & navigation ----------
function spanText(s) { let x = ''; for (let i = s.a; i <= s.b; i++) x += G.piece[i]; return showSpecial(x); }
function spanCtx(s) {
  let pre = '', post = '';
  for (let i = Math.max(G.tStart[s.k], s.a - 30); i < s.a; i++) pre += G.piece[i];
  for (let i = s.b + 1; i < Math.min(G.tEnd[s.k], s.b + 15); i++) post += G.piece[i];
  return [showSpecial(pre.slice(-160)), showSpecial(post.slice(0, 90))];
}
function renderLocalSpans() {
  const list = SPANS.slice().sort((x, y) => y.sum - x.sum).slice(0, 60);
  $('#local-span-count').textContent = `${SPANS.length} at τ ${S.tau}, gap ${S.gap}; top ${list.length} by total NLL`;
  $('#local-spans').innerHTML = list.map(s => {
    const [pre, post] = spanCtx(s);
    return `<div class="sp-row" data-a="${s.a}"><div class="sp-meta"><span>turn ${s.k}</span><span class="type-${s.type}">${s.type}</span><span>${s.cls}</span>
      <span>${s.len} tok · ${s.nh} high</span><span>Σ ${fmt(s.sum, 1)} · max ${fmt(s.max, 1)}</span></div>
      <div class="sp-text"><span class="ctxt">…${esc(pre)}</span><span class="core" style="background:${seqColor(s.max / 10)}">${esc(spanText(s))}</span><span class="ctxt">${esc(post)}…</span></div></div>`;
  }).join('');
}
function scrollToToken(i, flash = true) {
  const el = $(`#convo .t[data-g="${i}"]`);
  if (!el) return;
  const fold = el.closest('.folded'); if (fold) fold.classList.remove('folded');
  el.scrollIntoView({ block: 'center' });
  if (flash) { el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); }
  if (!pinned) inspect(i);
}
function gotoSpan(d) {
  if (!SPANS.length) return;
  spanPtr = (spanPtr + d + SPANS.length) % SPANS.length;
  const s = SPANS[spanPtr];
  scrollToToken(s.a);
  $('#span-pos').textContent = `span ${spanPtr + 1} / ${SPANS.length}`;
}
function scrollToTurn(k) {
  const el = $('#turn-' + k); if (!el) return;
  history.replaceState(null, '', '#' + TRAJ.meta.traj_id + '.' + k);
  el.scrollIntoView({ block: 'start' });
  $$('#convo .msg.focus').forEach(e => e.classList.remove('focus'));
  el.classList.add('focus');
}

// ---------- trajectory header ----------
function renderHead() {
  const m = TRAJ.meta, tm = m.task_meta || {};
  const s = INDEX.trajectories.find(t => t.traj_id === m.traj_id);
  const sumRows = tags().map(tag => `<span>${esc(tagLabel(tag))}: μ content <b>${fmt(s.summary[tag].content.mean, 3)}</b>, all <b>${fmt(s.summary[tag].all.mean, 3)}</b></span>`).join('');
  $('#traj-head').innerHTML = `<h2>${m.traj_id} · ${esc(m.task)}</h2>
    <div class="meta-row"><span>domain <b>${esc(m.domain)}</b></span><span>generator <b>${esc(m.generator || '')}</b></span>
      <span>difficulty <b>${esc(String(tm.difficulty || '–'))}</b></span><span>reward <b>${m.reward}</b></span>
      <span>turns <b>${TRAJ.turns.length}</b></span><span>sequence <b>${TRAJ.n_tokens.toLocaleString()}</b> tok, <b>${TRAJ.n_targets.toLocaleString()}</b> scored</span>
      <span>expert est. <b>${tm.expert_time_estimate_min ?? '–'}</b> min</span></div>
    <div class="meta-row">${(tm.tags || []).map(x => `<span class="pill">${esc(x)}</span>`).join('')}</div>
    <div class="meta-row">${sumRows}</div>
    <div class="meta-row"><span>run <b>${esc(m.run)}</b></span><span>trial <b>${esc(m.trial)}</b></span><span>sid <span class="copy" title="click to copy">${esc(m.sid)}</span></span></div>
    <div class="meta-row"><span class="copy" title="MSA trajectory (click to copy)">${esc(m.msa_trajectory)}</span></div>
    ${tm.instruction ? `<details class="instr"><summary>task instruction (instruction.md)</summary><pre>${esc(tm.instruction)}</pre></details>` : ''}`;
}

// ---------- global spans view ----------
function renderGlobalSpans() {
  const rows0 = globalSpans || INDEX.spans.filter(r => r.tag === spanTag());
  const ty = $('#sf-type').value, cl = $('#sf-cls').value, q = $('#sf-q').value.toLowerCase(), so = $('#sf-sort').value;
  let rows = rows0.filter(r => (!ty || r.type === ty) && (!cl || r.cls === cl) && (!q || r.text.toLowerCase().includes(q)));
  const key = r => so === 'mean' ? r.sum_nll / r.n_tokens : r[so];
  rows = rows.slice().sort((x, y) => key(y) - key(x));
  $('#sf-count').textContent = `${rows.length} spans`;
  $('#spans-note').textContent = globalSpans
    ? `Recomputed in this browser with τ ${S.tau}, gap ${S.gap}, scored classes ${CONTENT.filter(c => S.show[CLASSES[c]]).map(c => CLASSES[c]).join(', ')}, for ${tagLabel(spanTag())}.`
    : `Precomputed by build_site.py at τ ${CFG.tau} nats (p < ${Math.exp(-CFG.tau).toFixed(2)}), gap ${CFG.gap}, content tokens, the ${INDEX.spans.length / tags().length} largest per checkpoint by total NLL; showing ${tagLabel(spanTag())}. “recompute” uses the current controls.`;
  $('#span-list').innerHTML = rows.slice(0, 500).map(r => `<div class="sp-row" data-traj="${r.traj_id}" data-turn="${r.turn}" data-first="${r.first}">
    <div class="sp-meta"><b>${r.traj_id}</b><span>turn ${r.turn}</span><span class="type-${r.type}">${r.type}</span><span>${r.cls}</span>
    <span>${r.n_tokens} tok · ${r.n_high} high</span><span>Σ ${fmt(r.sum_nll, 1)} · max ${fmt(r.max_nll, 1)}</span>
    ${r.beyond_eval_context ? '<span class="beyond-tag">beyond eval context</span>' : ''}<span>${esc((INDEX.trajectories.find(t => t.traj_id === r.traj_id) || {}).domain || '')}</span></div>
    <div class="sp-text"><span class="ctxt">…${esc(showSpecial(r.before.slice(-160)))}</span><span class="core" style="background:${seqColor(r.max_nll / 10)}">${esc(showSpecial(r.text))}</span><span class="ctxt">${esc(showSpecial(r.after))}…</span></div></div>`).join('');
}
async function recomputeGlobalSpans() {
  const btn = $('#recompute-spans'); btn.disabled = true;
  const keep = { TRAJ, G };
  const out = [];
  try {
    for (const t of INDEX.trajectories) {
      btn.textContent = `loading ${t.traj_id}…`;
      TRAJ = await getJSON(`data/traj/${t.traj_id}.json`); G = flatten(TRAJ);
      for (let k = 0; k < G.tStart.length; k++) for (const s of turnSpans(k)) {
        const [pre, post] = spanCtx(s);
        out.push({ traj_id: t.traj_id, turn: k, first: s.a - G.tStart[k], last: s.b - G.tStart[k], n_tokens: s.len, n_high: s.nh,
          type: s.type, sum_nll: s.sum, max_nll: s.max, cls: s.cls, text: spanText(s), before: pre, after: post,
          beyond_eval_context: G.abs[s.a] >= CFG.eval_context });
      }
    }
  } finally { TRAJ = keep.TRAJ; G = keep.G; btn.disabled = false; btn.textContent = 'recompute with current settings'; }
  globalSpans = out;
  renderGlobalSpans();
}

// ---------- bookmarks ----------
const bms = () => store.get('bookmarks.v1', []);
function addBookmark({ k, a, b }) {
  const list = bms();
  let text = ''; for (let i = a; i <= b; i++) text += G.piece[i];
  const st = (() => { let s = 0; for (let i = a; i <= b; i++) s += nllOf(spanTag(), i); return s; })();
  list.push({ id: Date.now(), traj_id: TRAJ.meta.traj_id, task: TRAJ.meta.task, turn: k, first: a - G.tStart[k], last: b - G.tStart[k],
    tag: spanTag(), sum_nll: +st.toFixed(3), text: text.slice(0, 2000), note: '', created: new Date().toISOString() });
  store.set('bookmarks.v1', list); renderBookmarks();
}
function renderBookmarks() {
  const list = bms();
  $('#bm-count').textContent = list.length ? `(${list.length})` : '';
  $('#bm-list').innerHTML = list.length ? list.slice().reverse().map(b => `<div class="bm" data-id="${b.id}">
    <div class="sp-meta"><b class="jump" data-traj="${b.traj_id}" data-turn="${b.turn}" data-first="${b.first}" style="cursor:pointer;text-decoration:underline">${b.traj_id} turn ${b.turn}${b.first != null ? ' tok ' + b.first + (b.last !== b.first ? '–' + b.last : '') : ''}</b>
      <span>${esc(b.tag)}</span><span>Σ NLL ${b.sum_nll ?? '–'}</span><span>${esc(b.created.slice(0, 16))}</span>
      <button class="small del">delete</button></div>
    <div class="sp-text">${esc(b.text || '')}</div>
    <textarea placeholder="note: what is surprising here, and why?">${esc(b.note || '')}</textarea></div>`).join('') : '<p class="hint">No bookmarks yet.</p>';
}

// ---------- about ----------
function renderAbout() {
  const c = CFG, s = c.sample;
  const ck = c.tags.map(t => `<tr><td><b>${esc(t.label)}</b> (${t.tag})</td><td>${esc(t.model_id || '')} @ ${esc(t.revision || '')}<br><code>${esc(t.model_path)}</code><br>
    vLLM ${esc(t.vllm_version || '')} · prompt_logprobs ${t.prompt_logprobs_k} · job ${esc(String(t.slurm_job || ''))}<br><code>${esc(t.serve_args || '')}</code></td></tr>`).join('');
  const checks = c.tags.map(t => `<b>${esc(t.label)}</b>\n${esc(JSON.stringify(t.checks, null, 1) || 'none')}`).join('\n\n');
  $('#about').innerHTML = `
    <h3>What the colors mean</h3>
    <p>Each teacher (Qwen3.8-27B) token is colored by the student's negative log-probability of it, given the whole conversation before it and the teacher's earlier tokens in the same turn (teacher forcing). High NLL means the token surprised the student under this history. It is a hypothesis generator, not proof of missing knowledge: formatting conventions, arbitrary choices (file names, exact numbers) and wording the student would phrase differently all score high too.</p>
    <h3>History scored (two conditions)</h3><p><b>full history</b>: ${esc(c.conditions.full_history)}.<br><b>no-refeed</b>: ${esc(c.conditions.norefeed)}.<br>
    Tool outputs, system and task messages are context only in both. The same teacher tokens are scored in both, so Δ(no-refeed − full history) per token is how much the teacher's own earlier reasoning made that token predictable: the context a student at eval time does not have.</p>
    <h3>Noise floor</h3><p>vLLM serving of this MoE is not bit-deterministic: repeating the identical request moves a token's log-prob by median ~0.0005, mean ~0.02, p99 ~0.3, max ~1.5–2 nats (MoE routing near ties). Treat per-token differences below ~0.3 nats, and Δ maps at small ±range, as noise; turn and span means average it out.</p>
    <h3>Sample</h3><p>${esc(s.method)}</p><p>Pool: <code>${esc(s.pool)}</code>, ${s.pool_rows} episodes, ${s.pool_tasks} tasks; seed ${s.seed}. Exclusions: ${s.exclusions.length ? esc(JSON.stringify(s.exclusions)) : 'none'}.</p>
    <p>Teacher: ${esc(s.teacher)}. Harness: ${esc(s.harness)}.</p>
    <h3>Checkpoints</h3><table>${ck}</table>
    <h3>Classes</h3><p><b>reasoning</b> = inside ⟨think⟩; <b>prose</b> = text before the tool calls; <b>command</b> = the bash command string inside the tool-call JSON (JSON-escaped, as the student writes it); <b>format</b> = ⟨think⟩/⟨/think⟩ tags, tool-call JSON scaffolding and ⟨eot⟩. Format tokens are excluded from scores and spans by default. Tool calls are the student's own <code>&lt;tool_call&gt;{json}</code> form: the teacher wrote XML, so scaffolding tokens measure the conversion, not the teacher.</p>
    <h3>Spans</h3><p>Content tokens with NLL ≥ τ are high. High tokens with at most <i>gap</i> other tokens between them join one span. Isolated = 1 high token, burst = 2–3, extended = 4+. The smoothing control colors by a moving mean so sustained regions stand out from single spikes.</p>
    <h3>Correctness checks (run on the first episode before the rest)</h3><pre>${checks}</pre>
    <h3>Eval context</h3><p>Turns whose history passes ${c.eval_context.toLocaleString()} tokens (hatched bars) are beyond what the student's MSA2 eval could ever see; they were scored with a longer serve window.</p>`;
}

// ---------- legend ----------
function drawLegend() {
  const cv = $('#legend'), ctx = cv.getContext('2d'), W = cv.width;
  ctx.clearRect(0, 0, W, cv.height);
  for (let x = 0; x < W; x++) {
    const t = x / (W - 1);
    ctx.fillStyle = isDelta() ? divColor(2 * t - 1) : seqColor(t);
    ctx.fillRect(x, 0, 1, 18);
  }
  let txt;
  if (isDelta()) txt = `−${S.dmax} … 0 … +${S.dmax} nats of ${tagLabel('delta')} (red: higher NLL under ${tagLabel(S.dB)})`;
  else if (S.mode === 'pct') txt = `top ${S.pctTop}% of each turn, by within-turn rank`;
  else txt = `NLL ${S.floor} … ${S.smax}+ nats (p ${Math.exp(-S.floor).toFixed(2)} … ${Math.exp(-S.smax).toExponential(0)}); underline = NLL ≥ τ ${S.tau}`;
  $('#legend-text').textContent = txt + (S.smooth > 1 ? ` · smoothed over ${S.smooth} tokens` : '');
}

// ---------- controls ----------
function syncControls() {
  $('#floor').value = S.floor; $('#floor-v').textContent = S.floor;
  $('#smax').value = S.smax; $('#smax-v').textContent = S.smax;
  $('#pct').value = S.pctTop; $('#pct-v').textContent = S.pctTop + '%';
  $('#dmax').value = S.dmax; $('#dmax-v').textContent = S.dmax;
  if (S.dA) { $('#dA').value = S.dA; $('#dB').value = S.dB; }
  $('#tau').value = S.tau; $('#tau-v').textContent = `${S.tau} (p<${Math.exp(-S.tau).toFixed(2)})`;
  $('#gap').value = S.gap; $('#gap-v').textContent = S.gap;
  $('#smooth').value = S.smooth; $('#metric').value = S.metric;
  $('#ws').checked = S.ws; $('#bounds').checked = S.bounds; $('#tools-collapsed').checked = S.foldTools;
  $$('#controls input[data-cls]').forEach(el => { el.checked = S.show[el.dataset.cls]; });
  $$('#mode button').forEach(b => b.classList.toggle('on', b.dataset.mode === S.mode));
  $$('#ckpt button').forEach(b => b.classList.toggle('on', b.dataset.tag === S.tag));
  $$('#controls [data-show]').forEach(el => {
    const w = el.dataset.show;
    el.hidden = isDelta() ? w !== 'delta' : (w === 'delta' || w !== S.mode);
  });
  $('#mode').parentElement.hidden = isDelta();
  store.set('settings', S);
}
let rq = null;
function changed(full = true) { syncControls(); clearTimeout(rq); rq = setTimeout(() => { if (G) { recolor(); if (pinned != null) inspect(pinned); } renderTrajTable(); if ($('#view-spans').classList.contains('on')) renderGlobalSpans(); }, full ? 60 : 0); }

function bind() {
  const num = (id, key) => $('#' + id).addEventListener('input', e => { S[key] = +e.target.value; changed(); });
  num('floor', 'floor'); num('smax', 'smax'); num('pct', 'pctTop'); num('dmax', 'dmax'); num('tau', 'tau'); num('gap', 'gap');
  $('#smooth').onchange = e => { S.smooth = +e.target.value; changed(); };
  $('#metric').onchange = e => { S.metric = e.target.value; store.set('settings', S); renderOverview(); };
  $$('#controls input[data-cls]').forEach(el => el.onchange = () => { S.show[el.dataset.cls] = el.checked; changed(); });
  $('#ws').onchange = e => { S.ws = e.target.checked; $('#convo').classList.toggle('ws', S.ws); syncControls(); if (G) { $$('#convo .tx .t').forEach(t => { const i = +t.dataset.g; if (G.cls[i] !== 3 && G.cls[i] !== 5) t.innerHTML = S.ws ? wsMarks(esc(G.piece[i])).replace(/\n/g, '<span class="nl">↵</span>\n') : esc(G.piece[i]).replace(/\n/g, '<span class="nl">↵</span>\n'); }); } };
  $('#bounds').onchange = e => { S.bounds = e.target.checked; $('#convo').classList.toggle('bounds', S.bounds); syncControls(); };
  $('#tools-collapsed').onchange = e => { S.foldTools = e.target.checked; syncControls(); $$('#convo .msg.tool .tool-out').forEach(el => el.classList.toggle('folded', S.foldTools)); };
  $$('#mode button').forEach(b => b.onclick = () => { S.mode = b.dataset.mode; changed(); });
  $('#prev-span').onclick = () => gotoSpan(-1);
  $('#next-span').onclick = () => gotoSpan(1);
  $('#hardest').onclick = () => {
    let best = -1, bv = -Infinity;
    for (let k = 0; k < G.tStart.length; k++) { const v = metricOf(k); if (v > bv) { bv = v; best = k; } }
    if (best >= 0) scrollToTurn(best);
  };
  $('#toggle-turn-table').onclick = () => { $('#turn-table-wrap').hidden = !$('#turn-table-wrap').hidden; renderTurnTable(); };
  $('#turn-table').addEventListener('click', e => {
    const th = e.target.closest('th');
    if (th) { const k = th.dataset.k; S.turnSort = [k, S.turnSort[0] === k ? -S.turnSort[1] : -1]; store.set('settings', S); renderTurnTable(); return; }
    const tr = e.target.closest('tr.row'); if (tr) scrollToTurn(+tr.dataset.k);
  });
  $('#bars').addEventListener('click', e => { const b = e.target.closest('.bar'); if (b) scrollToTurn(+b.dataset.k); });
  $('#heat').addEventListener('click', e => {
    const cv = e.currentTarget, r = cv.getBoundingClientRect();
    const k = Math.floor((e.clientY - r.top) / r.height * G.tStart.length);
    const x = (e.clientX - r.left) / r.width * cv.width;
    const i = Math.min(G.tEnd[k] - 1, G.tStart[k] + Math.floor(x * cv._per));
    if (k >= 0 && k < G.tStart.length) scrollToToken(i);
  });
  $('#traj-table').addEventListener('click', e => {
    const th = e.target.closest('th');
    if (th) { const k = th.dataset.k; S.trajSort = [k, S.trajSort[0] === k ? -S.trajSort[1] : -1]; store.set('settings', S); renderTrajTable(); return; }
    const tr = e.target.closest('tr.row'); if (tr) openTraj(tr.dataset.id);
  });
  const convo = $('#convo');
  convo.addEventListener('mouseover', e => { const t = e.target.closest('.t'); if (t && pinned == null) inspect(+t.dataset.g); });
  convo.addEventListener('click', e => {
    if (e.target.closest('.unfold')) { const b = e.target.closest('.msg').querySelector('.tool-out'); if (b) b.classList.toggle('folded'); return; }
    if (e.target.closest('.bm-turn')) { const k = +e.target.closest('.bm-turn').dataset.k; addBookmark({ k, a: G.tStart[k], b: G.tStart[k] }); return; }
    const t = e.target.closest('.t'); if (!t) return;
    const i = +t.dataset.g;
    if (e.shiftKey) {
      if (rangeA == null || G.turn[rangeA] !== G.turn[i]) { rangeA = i; range = { a: i, b: i }; }
      else { range = { a: Math.min(rangeA, i), b: Math.max(rangeA, i) }; rangeA = null; }
      $$('#convo .t.inrange').forEach(x => x.classList.remove('inrange'));
      for (let j = range.a; j <= range.b; j++) { const el = $(`#convo .t[data-g="${j}"]`); if (el) el.classList.add('inrange'); }
      inspect(i); return;
    }
    $$('#convo .t.pinned').forEach(x => x.classList.remove('pinned'));
    if (pinned === i) { pinned = null; $('#pin-hint').textContent = 'hover a token · click to pin'; return; }
    pinned = i; t.classList.add('pinned'); $('#pin-hint').textContent = 'pinned · click again to release'; inspect(i);
  });
  $('#traj-head').addEventListener('click', e => { const c = e.target.closest('.copy'); if (c) { navigator.clipboard && navigator.clipboard.writeText(c.textContent); } });
  $('#local-spans').addEventListener('click', e => { const r = e.target.closest('.sp-row'); if (r) scrollToToken(+r.dataset.a); });
  $$('#tabs button').forEach(b => b.onclick = () => showView(b.dataset.view));
  ['sf-type', 'sf-cls', 'sf-sort'].forEach(id => $('#' + id).onchange = renderGlobalSpans);
  $('#sf-q').oninput = renderGlobalSpans;
  $('#recompute-spans').onclick = recomputeGlobalSpans;
  const jump = async (traj, turn, first) => { showView('traj'); await openTraj(traj); const i = G.tStart[turn] + (first || 0); scrollToToken(i); };
  $('#span-list').addEventListener('click', e => { const r = e.target.closest('.sp-row'); if (r) jump(r.dataset.traj, +r.dataset.turn, +r.dataset.first); });
  $('#bm-list').addEventListener('click', e => {
    const j = e.target.closest('.jump'); if (j) { jump(j.dataset.traj, +j.dataset.turn, +j.dataset.first); return; }
    if (e.target.closest('.del')) { const id = +e.target.closest('.bm').dataset.id; store.set('bookmarks.v1', bms().filter(b => b.id !== id)); renderBookmarks(); }
  });
  $('#bm-list').addEventListener('change', e => {
    if (e.target.tagName !== 'TEXTAREA') return;
    const id = +e.target.closest('.bm').dataset.id, list = bms();
    const b = list.find(x => x.id === id); if (b) { b.note = e.target.value; store.set('bookmarks.v1', list); }
  });
  $('#bm-export').onclick = () => {
    const blob = new Blob([JSON.stringify(bms(), null, 1)], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'student_difficulty_bookmarks.json'; a.click();
  };
  $('#bm-import').onchange = async e => {
    const f = e.target.files[0]; if (!f) return;
    try { const got = JSON.parse(await f.text()); const have = bms(), ids = new Set(have.map(b => b.id)); store.set('bookmarks.v1', have.concat(got.filter(b => !ids.has(b.id)))); renderBookmarks(); } catch (err) { alert('not a bookmarks file: ' + err); }
  };
  document.addEventListener('keydown', e => {
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
    if (e.key === 'n') gotoSpan(1); else if (e.key === 'p') gotoSpan(-1);
    else if (e.key === 'Escape') { pinned = null; $$('#convo .t.pinned').forEach(x => x.classList.remove('pinned')); }
  });
  window.addEventListener('resize', () => { if (G) drawHeat(); });
}
function showView(v) {
  $$('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.view === v));
  $$('.view').forEach(el => el.classList.toggle('on', el.id === 'view-' + v));
  if (v === 'spans') renderGlobalSpans();
  if (v === 'bookmarks') renderBookmarks();
  if (v === 'about') renderAbout();
  store.set('view', v);
}

async function openTraj(id) {
  if (TRAJ && TRAJ.meta.traj_id === id && G) return;
  $('#loading').classList.add('on');
  try {
    TRAJ = await getJSON(`data/traj/${id}.json`);
    G = flatten(TRAJ);
    pinned = null; range = null; rangeA = null;
    renderHead(); renderConvo(); recolor(); renderTrajTable();
    $('#main').scrollTop = 0;
    store.set('traj', id);
    history.replaceState(null, '', '#' + id);
  } finally { $('#loading').classList.remove('on'); }
}

async function main() {
  INDEX = await getJSON('data/index.json');
  CFG = INDEX.config;
  const ts = tags();
  if (!S.tag || (S.tag !== 'delta' && !ts.includes(S.tag)) || (S.tag === 'delta' && ts.length < 2)) S.tag = ts[0];
  if (!ts.includes(S.dA)) S.dA = ts[0];
  if (!ts.includes(S.dB) || S.dB === S.dA) S.dB = ts.find(t => t !== S.dA);
  const opts = ts.map(t => `<option value="${t}">${esc(tagLabel(t))}</option>`).join('');
  $('#dA').innerHTML = opts; $('#dB').innerHTML = opts;
  $('#dA').onchange = e => { S.dA = e.target.value; changed(); };
  $('#dB').onchange = e => { S.dB = e.target.value; changed(); };
  $('#ckpt').innerHTML = CFG.tags.map(t => `<button data-tag="${t.tag}" title="${esc(t.model_id || t.model_path)}">${esc(t.label)}</button>`).join('')
    + (ts.length > 1 ? `<button data-tag="delta" title="difference between two score sets (choose the pair in the controls)">Δ</button>` : '');
  $$('#ckpt button').forEach(b => b.onclick = () => { S.tag = b.dataset.tag; changed(); });
  bind(); syncControls(); renderTrajTable(); renderBookmarks();
  // deep links: #t05 opens a trajectory, #t05.12 (or #t05:12) also scrolls to turn 12
  // #spans / #bookmarks / #about open a view
  const h = location.hash.slice(1), views = ['spans', 'bookmarks', 'about'];
  const [want, turn] = (h && !views.includes(h) ? h : store.get('traj', null) || INDEX.trajectories[0].traj_id).split(/[:.]/);
  await openTraj(INDEX.trajectories.some(t => t.traj_id === want) ? want : INDEX.trajectories[0].traj_id);
  showView(views.includes(h) ? h : h ? 'traj' : store.get('view', 'traj'));
  if (turn != null && G && +turn < G.tStart.length) { scrollToTurn(+turn); history.replaceState(null, '', '#' + want + '.' + turn); }
}
main().catch(e => { document.body.insertAdjacentHTML('afterbegin', `<pre style="color:#c00;padding:10px">${esc(String(e.stack || e))}\n\nServe this directory over HTTP (python3 -m http.server), not file://.</pre>`); });
