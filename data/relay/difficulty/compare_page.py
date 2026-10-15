#!/usr/bin/env python3
"""benchmark.json (+ judged window labels) + written text -> compare.html, the method-benchmark report page.

    python compare_page.py --bench <dir with benchmark.json, windows.jsonl, labels*.jsonl> --text report_text.html \
        --out compare.html

The text file is the report prose (written after reading the results), with figure placeholders that this script
fills from the data: <!--FIG:turn-->, <!--FIG:spans-->, <!--FIG:sftcurve-->, <!--FIG:hard-->, <!--FIG:predict-->,
<!--FIG:examples_all--> and <!--EX:<window id>--> for one judged example. Figures are HTML/CSS so their text stays
readable at phone width.
"""
import argparse
import collections
import glob
import html
import json
import os
import re

MEASURE_LABEL = dict(
    nll='Raw NLL', gap="Gap to the student's top choice", excess='Surprisal − entropy', z='Standardized log-prob (z)',
    pval='Surprise p-value', rank='Rank of the teacher token', conflict='Confident conflict', entropy='Student entropy',
    D='Turn curvature D', rsr='Rank / surprisal ratio', sub1='Share of tokens under 1 %', length='Turn length',
    dnr='NLL change without earlier reasoning', xstudent='Mean NLL over both students',
    position='First tokens of a turn', sft_stays_hard='Tokens SFT failed to learn')
STUDENTS = [('dk0921', '09-21'), ('rlvr1s12', 'RLVR1')]
CONDITIONS = [('full_history', 'Full history'), ('norefeed', 'Earlier reasoning dropped')]
TURN_ROWS = ['D', 'excess', 'z', 'gap', 'nll', 'length']
GROUPS = [('good', 'Worth teaching', 'KSOR'), ('plain', 'Formatting or wording', 'TW'),
          ('odd', 'Invented specifics', 'A'), ('bad', 'Teacher confusion', 'C')]
CAT_LABEL = dict(K='Missing knowledge', S='Strategy or action choice', O='Reading the tool output',
                 R='Relies on earlier reasoning', T='Transition or format', W='Surface wording',
                 A='Invented specifics', C='Teacher confusion or error')
CAT_GROUP = {c: g for g, _, cs in GROUPS for c in cs}


def esc(s):
    return html.escape(str(s))


def pct(v, lo, hi):
    return 100.0 * (max(lo, min(hi, v)) - lo) / (hi - lo)


def dotplot(rows, series, lo=0.0, hi=1.0, ref=0.5, ref_label='0.5 · chance', ticks=None, emph=()):
    """rows: [(key, label, {series: (value, ci or None)})]; series: [(key, label, css class)]."""
    ticks = ticks or [lo + (hi - lo) * k / 4 for k in range(5)]
    out = ['<div class="dp" role="table">',
           '<div class="dp-lab"></div><div class="dp-axis" aria-hidden="true">'
           + ''.join(f'<span style="left:{pct(t, lo, hi):.2f}%">{t:.2f}</span>' for t in ticks)
           + '</div><div class="dp-val dp-vh">' + ' '.join(f'<span class="{c}">{esc(lab)}</span>' for _, lab, c in series)
           + '</div>']
    for key, label, vals in rows:
        dots, nums = [], []
        for i, (sk, slab, c) in enumerate(series):
            v = vals.get(sk)
            if v is None or v[0] is None:
                nums.append('<span>–</span>')
                continue
            val, ci = v
            y = (i + 1) * 100 / (len(series) + 1)
            if ci and ci[0] is not None:
                dots.append(f'<i class="ci {c}" style="left:{pct(ci[0], lo, hi):.2f}%;'
                            f'width:{pct(ci[1], lo, hi) - pct(ci[0], lo, hi):.2f}%;top:{y:.0f}%"></i>')
            dots.append(f'<i class="dot {c}" style="left:{pct(val, lo, hi):.2f}%;top:{y:.0f}%" '
                        f'title="{esc(slab)}: {val:.2f}"></i>')
            nums.append(f'<span class="{c}">{val:.2f}</span>')
        cls = ' class="em"' if key in emph else ''
        out.append(f'<div class="dp-lab"{cls}>{esc(label)}</div>'
                   f'<div class="dp-trk"><i class="ref" style="left:{pct(ref, lo, hi):.2f}%"></i>{"".join(dots)}</div>'
                   f'<div class="dp-val">{" ".join(nums)}</div>')
    out.append(f'<div class="dp-lab"></div><div class="dp-reflab"><span style="left:{pct(ref, lo, hi):.2f}%">'
               f'{esc(ref_label)}</span></div><div></div></div>')
    return ''.join(out)


def legend(series):
    return '<div class="legend">' + ''.join(f'<span><i class="sw {c}"></i>{esc(lab)}</span>' for _, lab, c in series) + '</div>'


def fig_turn(B):
    series = [(k, lab, f's{i}') for i, (k, lab) in enumerate(STUDENTS)]
    tabs, panes = [], []
    for j, (cond, clab) in enumerate(CONDITIONS):
        rows = []
        for m in TURN_ROWS:
            vals = {}
            for stu, _ in STUDENTS:
                r = B['b1'].get(f'{stu}/{cond}', {}).get(m)
                vals[stu] = (r['auroc'], r['ci']) if r else None
            rows.append((m, MEASURE_LABEL[m], vals))
        tabs.append(f'<button type="button" id="turn-tab-{cond}" data-pane="turn-{cond}" aria-pressed="{str(j == 0).lower()}">{esc(clab)}</button>')
        panes.append(f'<div id="turn-{cond}" class="pane"{"" if j == 0 else " hidden"}>'
                     + dotplot(rows, series, emph=('D', 'excess', 'z')) + '</div>')
    return ('<div class="fig-head">' + legend(series) + '<div class="seg" role="group" aria-label="History the student sees">'
            + ''.join(tabs) + '</div></div>' + ''.join(panes))


def fig_spans(per):
    rows = sorted(per.items(), key=lambda kv: (-sum(kv[1][c] for c in 'KSOR') / max(1, sum(kv[1].values())), kv[0]))
    out = [legend([(g, lab, g) for g, lab, _ in GROUPS]), '<div class="sb">']
    for m, cnt in rows:
        n = sum(cnt.values()) or 1
        segs = []
        for g, glab, cs in GROUPS:
            k = sum(cnt[c] for c in cs)
            if k:
                detail = ', '.join(f'{CAT_LABEL[c].lower()} {cnt[c]}' for c in cs if cnt[c])
                segs.append(f'<i class="{g}" style="flex-basis:{100 * k / n:.2f}%" title="{esc(glab)}: {k} of {n} ({esc(detail)})"></i>')
        good = sum(cnt[c] for c in 'KSOR')
        em = ' class="em"' if m == 'gap' else ''
        out.append(f'<div class="sb-lab"{em}>{esc(MEASURE_LABEL.get(m, m))}</div><div class="sb-bar">{"".join(segs)}</div>'
                   f'<div class="sb-val"><b>{good}</b> of {n}</div>')
    out.append('</div>')
    return ''.join(out)


def fig_sftcurve(b2, sft):
    xs = ['Before SFT', 'Pass 1', 'Pass 2', 'Pass 3']
    trained = [b2['mean_nll_by_checkpoint'][k] for k in ['base'] + sft]
    ctrl = b2['untrained_control']['mean_nll']
    untrained = [ctrl[k] for k in [next(k for k in ctrl if k not in sft)] + sft]
    n_tr, n_un = b2['n_tokens'], b2['untrained_control']['n']
    W, H, L, R, T, Bm = 640, 300, 52, 172, 24, 44
    ymax = 1.2
    X = lambda i: L + (W - L - R) * i / (len(xs) - 1)  # noqa: E731
    Y = lambda v: T + (H - T - Bm) * (1 - v / ymax)  # noqa: E731
    o = [f'<svg viewBox="0 0 {W} {H}" class="lc" role="img" aria-label="Mean NLL of trained and untrained teacher tokens across SFT passes">']
    for v in (0, 0.3, 0.6, 0.9, 1.2):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{L - 10}" y="{Y(v) + 5:.1f}" class="ax" text-anchor="end">{v:.1f}</text>')
    for i, xl in enumerate(xs):
        o.append(f'<text x="{X(i):.1f}" y="{H - 14}" class="ax" text-anchor="middle">{xl}</text>')
    for vals, cls, lab, n in ((untrained, 'un', 'Never trained', n_un), (trained, 'tr', 'Trained', n_tr)):
        o.append(f'<polyline points="{" ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))}" class="ln {cls}"/>')
        for i, v in enumerate(vals):
            o.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4.5" class="pt {cls}"><title>{lab}, {xs[i]}: {v:.3f}</title></circle>')
        o.append(f'<text x="{X(0) - 2:.1f}" y="{Y(vals[0]) - 12:.1f}" class="vl {cls}">{vals[0]:.2f}</text>')
        o.append(f'<text x="{X(3) + 12:.1f}" y="{Y(vals[-1]) - 4:.1f}" class="sl {cls}">{lab}</text>'
                 f'<text x="{X(3) + 12:.1f}" y="{Y(vals[-1]) + 15:.1f}" class="sn">{vals[-1]:.2f} nats · {n / 1000:.0f}k tokens</text>')
    o.append(f'<text x="{L}" y="{T - 8}" class="ax">mean NLL, nats</text></svg>')
    return ''.join(o)


def fig_hard(b2):
    n = b2['n_tokens']
    c = b2['categories']
    hh, hl, lh, ll = (round(c[k] * n) for k in ('HH', 'HL', 'LH', 'LL'))
    rows = [('Started hard', hh + hl, [('bad', 'Still hard', hh), ('good', 'Learned', hl)]),
            ('Started easy', lh + ll, [('plain', 'Still easy', ll), ('odd', 'Became hard', lh)])]
    out = ['<div class="hb">']
    for lab, tot, segs in rows:
        bar = ''.join(f'<i class="{cls}" style="flex-basis:{100 * k / tot:.2f}%">'
                      + (f'<span>{esc(sl)} {100 * k / tot:.0f}%</span>' if k / tot > 0.2 else '') + '</i>'
                      for cls, sl, k in segs)
        small = ' · '.join(f'{esc(sl).lower()} {100 * k / tot:.1f}% ({k:,})' for _, sl, k in segs)
        out.append(f'<div class="hb-row"><div class="hb-lab"><b>{esc(lab)}</b> {tot:,} tokens ({100 * tot / n:.1f}%)</div>'
                   f'<div class="hb-bar">{bar}</div><div class="hb-note">{small}</div></div>')
    out.append('</div>')
    return ''.join(out)


def fig_predict(b2):
    bm = b2.get('predict_stay_hard_base_matched', {})
    series = [('all', 'Among all hard tokens', 'p0'), ('m', 'At equal starting NLL', 'p1')]
    rows = sorted(((p, MEASURE_LABEL.get(p, p), {'all': (v['auroc'], v['ci']), 'm': (bm.get(p), None)})
                   for p, v in b2['predict_stay_hard'].items()), key=lambda r: -r[2]['all'][0])
    rows = [(k, 'Starting NLL' if k == 'nll' else lab, v) for k, lab, v in rows]
    return legend(series) + dotplot(rows, series, lo=0.3, hi=0.9, ref=0.5, ref_label='0.5 · no signal',
                                    ticks=[0.3, 0.45, 0.6, 0.75, 0.9])


def example(w, lab, viewer='index.html'):
    g = CAT_GROUP.get(lab['cat'], 'plain')
    why = re.sub(r';?\s*labelled by main session', '', lab.get('why', '')).strip()
    by = ', '.join(MEASURE_LABEL[m] if m in MEASURE_LABEL else str(m) for m in w['methods'])
    return (f'<div class="ex"><div class="ex-head"><span class="chip {g}">{esc(CAT_LABEL.get(lab["cat"], lab["cat"]))}</span>'
            f'<span class="ex-why">{esc(why)}</span></div>'
            f'<div class="ex-txt">…{esc(w["before"][-240:])}<mark>{esc(w["token"])}</mark>{esc(w["after"][:110])}…</div>'
            f'<div class="ex-foot"><span>Flagged by {esc(by)}</span><a href="{viewer}#{w["traj"]}.{w["turn"]}">'
            f'Open {esc(w["traj"])} turn {w["turn"]} in the viewer</a></div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', required=True)
    ap.add_argument('--text', required=True, help='report prose with figure placeholders, written after reading results')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    B = json.load(open(os.path.join(a.bench, 'benchmark.json')))
    wins = {}
    for line in open(os.path.join(a.bench, 'windows.jsonl')):
        w = json.loads(line)
        wins[w['id']] = w
    labels = {}
    for f in glob.glob(os.path.join(a.bench, 'labels*.jsonl')):
        for line in open(f):
            if line.strip():
                lab = json.loads(line)
                labels[lab['id']] = lab
    per = collections.defaultdict(collections.Counter)
    for wid, w in wins.items():
        if wid in labels:
            for m in w['methods']:
                per[m][labels[wid]['cat']] += 1

    order = sorted(per, key=lambda m: -sum(per[m][c] for c in 'KSOR') / max(1, sum(per[m].values())))
    all_ex = ''.join(
        f'<details><summary>{esc(MEASURE_LABEL.get(m, m))} <span>{sum(per[m][c] for c in "KSOR")} of '
        f'{sum(per[m].values())} worth teaching</span></summary>'
        + ''.join(example(w, labels[w['id']]) for w in [w for w in wins.values() if m in w['methods'] and w['id'] in labels][:4])
        + '</details>' for m in order)
    figs = dict(turn=fig_turn(B), spans=fig_spans(per), sftcurve=fig_sftcurve(B['b2'], B['config']['sft']),
                hard=fig_hard(B['b2']), predict=fig_predict(B['b2']), examples_all=all_ex)

    text = open(a.text).read()
    text = re.sub(r'<!--FIG:(\w+)-->', lambda m: figs[m.group(1)], text)
    text = re.sub(r'<!--EX:([\w:]+)-->', lambda m: example(wins[m.group(1)], labels[m.group(1)]), text)
    page = TEMPLATE.replace('{{BODY}}', text)
    open(a.out, 'w').write(page)
    print('wrote', a.out, len(page))


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>SFTA Method Benchmark</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:ital,wght@0,400;0,600;0,700;1,400&display=swap">
<style>
/* Layout: one reading column (44rem) that figures widen to 60rem; a slim sticky bar; sections divided by space and
   one hairline, cards only for examples. Student colors stay fixed: 09-21 blue, RLVR1 orange. */
:root{
  --bg:#f4f6f8; --surface:#ffffff; --ink:#15202b; --ink-2:#3a4754; --muted:#62707e; --rule:#d9e0e7;
  --accent:#0d6b5e; --accent-soft:#e2f0ec;
  --s0:#2f5bd3; --s1:#d1551b; --p0:#0d6b5e; --p1:#9fc9c0;
  --good:#2f8f46; --plain:#a7b1bb; --odd:#e2a21b; --bad:#d0412f;
  --mark:#ffd08f; --mark-ink:#1d1206; --code-bg:#f1f3f6; --on-solid:#ffffff;
  --display:"Newsreader","Iowan Old Style",Georgia,serif;
  --body:"Public Sans",-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0e1318; --surface:#151c23; --ink:#e6ebf0; --ink-2:#c2ccd6; --muted:#8e9baa; --rule:#27313b;
  --accent:#5cc4b2; --accent-soft:#132d29; --s0:#7f9dff; --s1:#ff8b55; --p0:#5cc4b2; --p1:#2e5d55;
  --good:#4cbf69; --plain:#56616d; --odd:#eeb43c; --bad:#ff6a5a; --mark:#7d5214; --mark-ink:#fff4e2; --code-bg:#10161c; --on-solid:#0e1318;
  color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0e1318; --surface:#151c23; --ink:#e6ebf0; --ink-2:#c2ccd6; --muted:#8e9baa; --rule:#27313b;
  --accent:#5cc4b2; --accent-soft:#132d29; --s0:#7f9dff; --s1:#ff8b55; --p0:#5cc4b2; --p1:#2e5d55;
  --good:#4cbf69; --plain:#56616d; --odd:#eeb43c; --bad:#ff6a5a; --mark:#7d5214; --mark-ink:#fff4e2; --code-bg:#10161c; --on-solid:#0e1318;
  color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16.5px/1.65 var(--body);-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-underline-offset:2px}
a:focus-visible,button:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}
.bar{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);border-bottom:1px solid var(--rule)}
.bar-in{max-width:60rem;margin:0 auto;padding:.55rem 16px;display:flex;gap:1rem;align-items:baseline;justify-content:space-between}
.bar b{font:600 1rem/1.2 var(--display);letter-spacing:.01em}
.bar a{font-size:.88rem;text-decoration:none;white-space:nowrap}
main{max-width:60rem;margin:0 auto;padding-inline:16px;padding-block:2.2rem 4rem}
.col{max-width:44rem}
.kicker{font:600 .74rem/1 var(--body);letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
h1{font:600 2.5rem/1.12 var(--display);margin:.6rem 0 .9rem;text-wrap:balance;letter-spacing:-.01em}
h2{font:600 1.75rem/1.2 var(--display);margin:.4rem 0 .8rem;text-wrap:balance;letter-spacing:-.005em}
h3{font:600 1.05rem/1.3 var(--body);margin:1.8rem 0 .6rem}
p{margin:0 0 1rem}
.lead{font-size:1.14rem;line-height:1.55;color:var(--ink)}
.tldr{font-size:1.2rem;line-height:1.5;margin-bottom:1rem}
.tldr-list{margin:0 0 1.6rem;padding-left:1.2rem;color:var(--ink-2)}
.tldr-list li{margin:.35rem 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:1rem;margin:1.6rem 0 2rem}
.tile{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:1rem 1.1rem}
.tile b{display:block;font:600 2.3rem/1 var(--display);color:var(--accent);margin-bottom:.45rem;font-variant-numeric:tabular-nums}
.tile span{font-size:.92rem;line-height:1.45;color:var(--ink-2)}
.outline{margin:0 0 1rem;padding:0;list-style:none;counter-reset:o;border-top:1px solid var(--rule)}
.outline li{counter-increment:o;border-bottom:1px solid var(--rule)}
.outline a{display:flex;gap:.9rem;padding:.55rem 0;color:var(--ink);text-decoration:none;line-height:1.4}
.outline a::before{content:counter(o);font:600 .85rem/1.6 var(--mono);color:var(--accent);min-width:1.1rem}
.outline a:hover{color:var(--accent)}
section{border-top:1px solid var(--rule);padding-top:2.2rem;margin-top:2.6rem;scroll-margin-top:3.5rem}
figure{margin:1.4rem 0 1.4rem;padding:1.1rem 1.2rem 1rem;background:var(--surface);border:1px solid var(--rule);border-radius:10px}
figcaption{font-size:.9rem;line-height:1.5;color:var(--muted);margin-top:.9rem;max-width:44rem}
.note{font-size:.92rem;line-height:1.55;color:var(--muted);border-left:2px solid var(--rule);padding:.1rem 0 .1rem .9rem;margin:1.2rem 0 0}
.note p{margin:0 0 .5rem}.note p:last-child{margin:0}
b.k{font-weight:700;color:var(--ink)}
code{font:.86em var(--mono);background:var(--code-bg);padding:.08em .32em;border-radius:4px}
.fig-head{display:flex;flex-wrap:wrap;gap:.8rem 1.4rem;align-items:center;justify-content:space-between;margin-bottom:.8rem}
.legend{display:flex;flex-wrap:wrap;gap:.4rem 1.1rem;font-size:.86rem;color:var(--ink-2);margin-bottom:.6rem}
.fig-head .legend{margin:0}
.legend span{display:inline-flex;align-items:center;gap:.4rem}
.sw{display:inline-block;width:.75rem;height:.75rem;border-radius:50%}
.seg{display:inline-flex;border:1px solid var(--rule);border-radius:7px;overflow:hidden}
.seg button{font:inherit;font-size:.84rem;padding:.3rem .7rem;border:0;background:transparent;color:var(--ink-2);cursor:pointer}
.seg button+button{border-left:1px solid var(--rule)}
.seg button[aria-pressed="true"]{background:var(--accent-soft);color:var(--ink);font-weight:600}
/* dot plot */
.dp{display:grid;grid-template-columns:minmax(9rem,15rem) minmax(0,1fr) 6.5rem;column-gap:1rem;row-gap:.15rem;align-items:center}
.dp-lab{font-size:.92rem;text-align:right;line-height:1.25;color:var(--ink-2)}
.dp-lab.em{color:var(--ink);font-weight:600}
.dp-trk{position:relative;height:2.1rem;border-left:1px solid var(--rule);border-right:1px solid var(--rule);
  background:linear-gradient(to right,transparent calc(25% - .5px),var(--rule) calc(25% - .5px),var(--rule) calc(25% + .5px),transparent calc(25% + .5px),
    transparent calc(50% - .5px),var(--rule) calc(50% - .5px),var(--rule) calc(50% + .5px),transparent calc(50% + .5px),
    transparent calc(75% - .5px),var(--rule) calc(75% - .5px),var(--rule) calc(75% + .5px),transparent calc(75% + .5px))}
.dp-trk .ref{position:absolute;top:-.15rem;bottom:-.15rem;width:0;border-left:1.5px dashed var(--muted)}
.dp-trk .dot{position:absolute;width:.8rem;height:.8rem;border-radius:50%;transform:translate(-50%,-50%);border:2px solid var(--surface)}
.dp-trk .ci{position:absolute;height:2px;transform:translateY(-50%);opacity:.55}
.dp-axis,.dp-reflab{position:relative;height:1.3rem;font:.74rem/1 var(--mono);color:var(--muted)}
.dp-axis span,.dp-reflab span{position:absolute;top:.25rem;transform:translateX(-50%);white-space:nowrap}
.dp-reflab span{top:.3rem}
.dp-axis span:first-child{transform:none}.dp-axis span:last-child{transform:translateX(-100%)}
.dp-val{font:600 .86rem/1.2 var(--mono);display:flex;gap:.6rem;font-variant-numeric:tabular-nums}
.dp-vh{font:400 .72rem/1.2 var(--body);color:var(--muted)}
.dot.s0,.ci.s0,.sw.s0{background:var(--s0)}.dot.s1,.ci.s1,.sw.s1{background:var(--s1)}
.dot.p0,.ci.p0,.sw.p0{background:var(--p0)}.dot.p1,.ci.p1,.sw.p1{background:var(--p1)}
.dp-val .s0{color:var(--s0)}.dp-val .s1{color:var(--s1)}.dp-val .p0{color:var(--p0)}.dp-val .p1{color:var(--muted)}
/* stacked bars */
.sb{display:grid;grid-template-columns:minmax(9rem,15rem) minmax(0,1fr) 4.6rem;column-gap:1rem;row-gap:.45rem;align-items:center}
.sb-lab{font-size:.92rem;text-align:right;line-height:1.25;color:var(--ink-2)}.sb-lab.em{color:var(--ink);font-weight:600}
.sb-bar{display:flex;height:1.15rem;border-radius:4px;overflow:hidden;background:var(--rule)}
.sb-bar i{display:block;height:100%}
.sb-bar i+i{box-shadow:inset 1.5px 0 0 var(--surface)}
.sb-val{font-size:.84rem;color:var(--muted);white-space:nowrap}.sb-val b{font:600 .95rem var(--mono);color:var(--ink)}
.sw.good,.sb-bar .good,.hb-bar .good{background:var(--good)}
.sw.plain,.sb-bar .plain,.hb-bar .plain{background:var(--plain)}
.sw.odd,.sb-bar .odd,.hb-bar .odd{background:var(--odd)}
.sw.bad,.sb-bar .bad,.hb-bar .bad{background:var(--bad)}
.legend .sw.good,.legend .sw.plain,.legend .sw.odd,.legend .sw.bad{border-radius:3px}
/* hard-token bars */
.hb{display:grid;gap:1.1rem}
.hb-lab{font-size:.95rem;margin-bottom:.35rem;color:var(--ink-2)}.hb-lab b{color:var(--ink)}
.hb-bar{display:flex;height:2rem;border-radius:5px;overflow:hidden}
.hb-bar i{display:flex;align-items:center;padding:0 .6rem;min-width:3px;font:600 .84rem var(--body);color:var(--on-solid);font-style:normal}
.hb-bar .plain span{color:var(--ink)}
.hb-bar i+i{box-shadow:inset 1.5px 0 0 var(--surface)}
.hb-note{font-size:.84rem;color:var(--muted);margin-top:.3rem}
/* line chart */
svg.lc{display:block;width:100%;height:auto;max-width:44rem}
svg .grid{stroke:var(--rule);stroke-width:1}
svg .ax{font:13px var(--mono);fill:var(--muted)}
svg .ln{fill:none;stroke-width:3}svg .ln.tr{stroke:var(--accent)}svg .ln.un{stroke:var(--plain)}
svg .pt.tr{fill:var(--accent)}svg .pt.un{fill:var(--plain)}
svg .vl{font:600 14px var(--mono)}svg .vl.tr{fill:var(--accent)}svg .vl.un{fill:var(--muted)}
svg .sl{font:600 15px var(--body);fill:var(--ink)}svg .sn{font:13px var(--body);fill:var(--muted)}
/* tables */
.tbl-wrap{overflow-x:auto;margin:1.2rem 0}
table.t{border-collapse:collapse;width:100%;font-size:.94rem;min-width:30rem}
table.t th{font:600 .74rem/1.3 var(--body);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);text-align:left;padding:.5rem .7rem;border-bottom:1px solid var(--rule)}
table.t td{padding:.6rem .7rem;border-bottom:1px solid var(--rule)}
table.t td.n{font:600 1rem var(--mono);font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:.1rem .55rem;border-radius:999px;font-size:.84rem;font-weight:600}
.pill.good{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.pill.bad{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}
/* examples */
.exs{display:grid;gap:.9rem;margin:1rem 0}
.ex{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:.85rem 1rem}
.ex-head{display:flex;flex-wrap:wrap;gap:.4rem .7rem;align-items:baseline;margin-bottom:.55rem}
.chip{display:inline-block;padding:.12rem .55rem;border-radius:5px;font:600 .76rem/1.4 var(--body);white-space:nowrap}
.chip.good{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}
.chip.plain{background:color-mix(in srgb,var(--plain) 30%,transparent);color:var(--ink-2)}
.chip.odd{background:color-mix(in srgb,var(--odd) 22%,transparent);color:var(--ink)}
.chip.bad{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}
.ex-why{font-size:.95rem;color:var(--ink)}
.ex-txt{font:.8rem/1.6 var(--mono);color:var(--ink-2);background:var(--code-bg);border-radius:6px;padding:.6rem .75rem;white-space:pre-wrap;overflow-wrap:anywhere}
.ex-txt mark{background:var(--mark);color:var(--mark-ink);border-radius:3px;padding:0 .1em;font-weight:600}
.ex-foot{display:flex;flex-wrap:wrap;gap:.3rem 1rem;justify-content:space-between;font-size:.8rem;color:var(--muted);margin-top:.5rem}
details{border-top:1px solid var(--rule)}details:last-of-type{border-bottom:1px solid var(--rule)}
summary{cursor:pointer;padding:.6rem 0;font-weight:600}summary span{font-weight:400;color:var(--muted);font-size:.88rem;margin-left:.4rem}
details .ex{margin-bottom:.8rem}
.next{padding-left:1.2rem}.next li{margin:.6rem 0}
dl.gl{display:grid;grid-template-columns:minmax(10rem,14rem) minmax(0,1fr);gap:.55rem 1.2rem;font-size:.92rem;margin:1rem 0 1.6rem}
dl.gl dt{font-weight:600}dl.gl dd{margin:0;color:var(--ink-2)}
dl.gl code{font-size:.82em}
.method{font-size:.9rem;color:var(--muted)}
@media (max-width:600px){
  body{font-size:16px}h1{font-size:2rem}h2{font-size:1.45rem}
  .dp,.sb{grid-template-columns:minmax(0,1fr) auto}
  .dp-lab,.sb-lab{grid-column:1/-1;text-align:left;margin-top:.45rem}
  .dp>.dp-lab:first-child,.dp>.dp-lab:nth-last-child(3){display:none}
  .dp-vh{display:none}
  figure{padding:.9rem .8rem}
  dl.gl{grid-template-columns:1fr}dl.gl dd{margin-bottom:.4rem}
}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
</style></head><body>
<div class="bar"><div class="bar-in"><b>SFTA method benchmark</b><a href="index.html">Token viewer →</a></div></div>
<main>
{{BODY}}
</main>
<script>
document.querySelectorAll('.seg').forEach(function (g) {
  g.addEventListener('click', function (e) {
    var b = e.target.closest('button'); if (!b) return;
    g.querySelectorAll('button').forEach(function (x) {
      var on = x === b; x.setAttribute('aria-pressed', String(on));
      document.getElementById(x.dataset.pane).hidden = !on;
    });
  });
});
</script>
</body></html>"""


if __name__ == '__main__':
    main()
