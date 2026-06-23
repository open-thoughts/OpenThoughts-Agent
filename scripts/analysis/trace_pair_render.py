#!/usr/bin/env python3
"""Render same-task trace pairs side-by-side for human inspection.

Answers research question (3): **Do behavioral changes persist in post-RL
eval traces?** The macro deltas from ``behavioral_delta.py`` say *what*
shifted; this script surfaces *what those shifts look like in practice*.

For each task present in both datasets, picks one representative trial
from each side (highest reward, then most recent) and renders both
conversations as columns in a single HTML page.

Layout features (vs. the original raw-dump renderer):
- Each pair starts with a **diff summary card** — reward delta, turn
  delta, tool-call delta, think-token delta — laid out side by side.
- Messages render inside ``<details>`` blocks (collapsible). The
  ``<summary>`` shows role, message index, token count, and a
  first-line preview.
- Default open state: only the first user message and the last
  assistant message are expanded per side. Reviewers click through
  the rest.
- Role-coded left border (user / assistant / tool / system each its
  own color). The ``after`` column gets a green wash, ``before`` red.
- ``<think>...</think>`` content is split out into its own block with
  a distinct background.
- Triple-backtick fenced code blocks render in a ``<pre>`` with light
  Pygments highlighting (graceful fallback to plain pre when pygments
  is not installed).
- ``<tool_call>`` JSON blocks (the OT-Agent embedded-tool format) get a
  distinct chip header with the tool name.

Usage:
    python -m scripts.analysis.trace_pair_render \\
        --before  penfever/pre-rl-eval-traces \\
        --after   penfever/post-rl-eval-traces \\
        --output  /path/trace_pairs.html \\
        --top-n   25
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.utils import (  # noqa: E402
    Trace,
    count_tokens,
    get_tiktoken_encoder,
    group_by_task,
    load_traces,
)

# Optional pygments for syntax highlighting. We import lazily-via-try so the
# script still runs without it.
try:
    from pygments import highlight  # type: ignore[import-not-found]
    from pygments.formatters import HtmlFormatter  # type: ignore[import-not-found]
    from pygments.lexers import get_lexer_by_name, guess_lexer  # type: ignore[import-not-found]
    from pygments.util import ClassNotFound  # type: ignore[import-not-found]
    _PYGMENTS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    highlight = None  # type: ignore[assignment]
    HtmlFormatter = None  # type: ignore[assignment]
    get_lexer_by_name = None  # type: ignore[assignment]
    guess_lexer = None  # type: ignore[assignment]
    ClassNotFound = Exception  # type: ignore[assignment, misc]
    _PYGMENTS_AVAILABLE = False


_THINK_BLOCK_RE = re.compile(r"<think(?:ing)?\s*>(.*?)</think(?:ing)?\s*>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+\-]*)\n?(.*?)```", re.DOTALL)


def _pick_representative(rows: Sequence[Trace]) -> Trace:
    """Pick the highest-reward trial, then the most-recent within ties."""
    return max(
        rows,
        key=lambda t: (
            t.reward if t.reward is not None else -1.0,
            t.date.timestamp() if t.date else 0.0,
        ),
    )


def _messages_of(raw: dict) -> List[dict]:
    msgs = raw.get("messages") or raw.get("conversations") or []
    if not isinstance(msgs, list):
        return []
    return msgs


def _stringify_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", "") or "")
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if isinstance(content, dict):
        return content.get("text", "") or str(content)
    return str(content)


def _highlight_code(code: str, lang_hint: Optional[str] = None) -> str:
    """Render code with pygments syntax highlighting; fall back to ``<pre>``.

    The fallback path still html-escapes the content so it's always safe to
    embed inline. We use Pygments' inline-style output (``noclasses=True``)
    so the resulting HTML doesn't need an external stylesheet.
    """
    if not _PYGMENTS_AVAILABLE:
        return f'<pre class="code-block">{html.escape(code)}</pre>'
    try:
        lexer = None
        if lang_hint:
            try:
                lexer = get_lexer_by_name(lang_hint, stripall=False)  # type: ignore[arg-type]
            except ClassNotFound:
                lexer = None
        if lexer is None:
            try:
                lexer = guess_lexer(code)  # type: ignore[misc]
            except (ClassNotFound, ValueError):
                # Fall back to plain pre.
                return f'<pre class="code-block">{html.escape(code)}</pre>'
        formatter = HtmlFormatter(  # type: ignore[misc]
            noclasses=True,
            nowrap=False,
            style="default",
            cssclass="code-block",
        )
        return highlight(code, lexer, formatter)  # type: ignore[misc]
    except Exception:
        return f'<pre class="code-block">{html.escape(code)}</pre>'


def _render_tool_call_block(payload: str) -> str:
    """Render a ``<tool_call>{json}</tool_call>`` block as a labelled chip."""
    name = None
    pretty = payload.strip()
    try:
        data = json.loads(pretty)
        if isinstance(data, dict):
            name = data.get("name")
            pretty = json.dumps(data, indent=2)
    except json.JSONDecodeError:
        name = None
    chip = (
        f'<span class="tool-call-chip">tool_call'
        f'{": " + html.escape(name) if name else ""}</span>'
    )
    body = _highlight_code(pretty, "json")
    return f'<div class="tool-call-wrapper">{chip}{body}</div>'


def _render_message_content(content: str) -> str:
    """Render a message body with think / tool_call / code-fence highlighting.

    We process in two passes:
      1. Splice out ``<think>...</think>`` blocks and render them as inline
         "thinking" panels (greyed-out, distinct background).
      2. Splice out ``<tool_call>{json}</tool_call>`` blocks and render as
         labeled JSON chips.
      3. Splice out triple-backtick fenced blocks and syntax-highlight them.
      4. Everything else stays in a single escaped ``<pre>``.

    All splice operations preserve order via a single linear pass over the
    string, so even with overlapping syntax (rare in practice), the output
    stays well-formed.
    """
    # Use a token list: each token is (kind, payload). We scan the string,
    # peeling off the earliest match of any pattern at each step.
    tokens: list[Tuple[str, str]] = []
    cursor = 0
    text = content
    while cursor < len(text):
        # Find the earliest of: <think>, <tool_call>, ```fence
        candidates: list[Tuple[int, str, "re.Match"]] = []
        m_think = _THINK_BLOCK_RE.search(text, cursor)
        if m_think:
            candidates.append((m_think.start(), "think", m_think))
        m_tc = _TOOL_CALL_BLOCK_RE.search(text, cursor)
        if m_tc:
            candidates.append((m_tc.start(), "tool_call", m_tc))
        m_code = _CODE_FENCE_RE.search(text, cursor)
        if m_code:
            candidates.append((m_code.start(), "code", m_code))
        if not candidates:
            tail = text[cursor:]
            if tail:
                tokens.append(("text", tail))
            break
        candidates.sort(key=lambda c: c[0])
        start, kind, match = candidates[0]
        if start > cursor:
            tokens.append(("text", text[cursor:start]))
        if kind == "think":
            tokens.append(("think", match.group(1)))
        elif kind == "tool_call":
            tokens.append(("tool_call", match.group(1)))
        elif kind == "code":
            lang = match.group(1) or ""
            code = match.group(2)
            tokens.append(("code", code if not lang else f"{lang}\x00{code}"))
        cursor = match.end()

    parts: List[str] = []
    for kind, payload in tokens:
        if kind == "text":
            parts.append(f'<pre class="content-text">{html.escape(payload)}</pre>')
        elif kind == "think":
            parts.append(
                '<details class="think-block" open>'
                '<summary>thinking</summary>'
                f'<pre class="think-text">{html.escape(payload.strip())}</pre>'
                '</details>'
            )
        elif kind == "tool_call":
            parts.append(_render_tool_call_block(payload))
        elif kind == "code":
            lang, _, code = payload.partition("\x00")
            parts.append(_highlight_code(code, lang or None))
    return "".join(parts)


def _preview(content: str, max_chars: int = 90) -> str:
    """First non-empty line, truncated. Used in <summary> previews."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            return (stripped[:max_chars] + ("…" if len(stripped) > max_chars else ""))
    return ""


def _format_message_html(
    msg: dict,
    idx: int,
    total: int,
    open_by_default: bool,
    encoder,
) -> str:
    role = str(msg.get("role", "?"))
    role_safe = html.escape(role)
    content = _stringify_content(msg.get("content"))
    tok = count_tokens(content, encoder)
    preview = html.escape(_preview(content))

    # Has-structured-tool-calls flag for the badge.
    has_tc_struct = bool(msg.get("tool_calls"))
    has_tc_xml = "<tool_call>" in content.lower()
    badges = []
    if has_tc_struct or has_tc_xml:
        badges.append('<span class="badge tc">tool_call</span>')
    if "<think>" in content.lower() or "<thinking>" in content.lower():
        badges.append('<span class="badge think">think</span>')
    if "```" in content:
        badges.append('<span class="badge code">code</span>')
    badge_html = "".join(badges)

    rendered = _render_message_content(content) if content else (
        '<pre class="content-text"><em>(empty)</em></pre>'
    )

    open_attr = " open" if open_by_default else ""
    return (
        f'<details class="msg role-{role_safe}"{open_attr}>'
        f'<summary class="msg-summary">'
        f'<span class="msg-idx">#{idx + 1}/{total}</span>'
        f'<span class="msg-role">{role_safe}</span>'
        f'{badge_html}'
        f'<span class="msg-toks">{tok} tok</span>'
        f'<span class="msg-preview">{preview}</span>'
        f'</summary>'
        f'<div class="msg-body">{rendered}</div>'
        f"</details>"
    )


def _trial_panel_html(t: Trace, encoder) -> str:
    msgs = _messages_of(t.raw)
    reward = "—" if t.reward is None else f"{t.reward:.3f}"
    err = f" · err: <code>{html.escape(t.error_type)}</code>" if t.error_type else ""
    fm = f" · fm: <code>{html.escape(t.failure_mode)}</code>" if t.failure_mode else ""
    b = t.behavioral
    tool_summary = f" · {b.tool_calls_total} tool calls"
    if b.tool_responses:
        tool_summary += f" ({b.tool_error_rate:.0%} errors)" if b.tool_error_rate is not None else ""
    header = (
        '<div class="trial-header">'
        f'reward: <b>{reward}</b> · turns: <b>{len(msgs)}</b>{tool_summary}{err}{fm}'
        f'<div class="trial-source">{html.escape(t.source or "")}</div>'
        '</div>'
    )

    # Decide which messages start expanded. Default policy: first user
    # message, last assistant message. Reviewer can click any other open.
    user_idxs = [i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "user"]
    asst_idxs = [i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "assistant"]
    open_idxs: set[int] = set()
    if user_idxs:
        open_idxs.add(user_idxs[0])
    if asst_idxs:
        open_idxs.add(asst_idxs[-1])

    body_parts: List[str] = []
    n = len(msgs)
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        body_parts.append(_format_message_html(m, i, n, i in open_idxs, encoder))

    # Quick controls to expand/collapse all messages in this panel.
    controls = (
        '<div class="panel-controls">'
        '<button type="button" class="ctrl-btn" data-action="expand-all">Expand all</button>'
        '<button type="button" class="ctrl-btn" data-action="collapse-all">Collapse all</button>'
        '</div>'
    )
    return header + controls + "".join(body_parts)


def _diff_card(before: Trace, after: Trace) -> str:
    """Compact 4-metric diff summary rendered above each pair."""
    bb = before.behavioral
    ab = after.behavioral

    def _delta_cell(label: str, b: Optional[float], a: Optional[float], fmt: str = ".2f") -> str:
        if b is None or a is None:
            return (
                f'<div class="metric-cell">'
                f'<div class="metric-label">{html.escape(label)}</div>'
                f'<div class="metric-values">— → —</div>'
                f'</div>'
            )
        delta = a - b
        cls = "up" if delta > 0 else ("dn" if delta < 0 else "flat")
        delta_str = f"{delta:+{fmt}}"
        b_str = f"{b:{fmt}}"
        a_str = f"{a:{fmt}}"
        return (
            f'<div class="metric-cell">'
            f'<div class="metric-label">{html.escape(label)}</div>'
            f'<div class="metric-values">{b_str} → {a_str}</div>'
            f'<div class="metric-delta {cls}">Δ {delta_str}</div>'
            f'</div>'
        )

    return (
        '<div class="diff-card">'
        + _delta_cell("reward",          before.reward,     after.reward,     ".3f")
        + _delta_cell("turns",           float(before.turns) if before.turns else None,
                                          float(after.turns) if after.turns else None, ".0f")
        + _delta_cell("tool calls",      float(bb.tool_calls_total), float(ab.tool_calls_total), ".0f")
        + _delta_cell("tool errors",     float(bb.tool_errors),      float(ab.tool_errors),      ".0f")
        + _delta_cell("asst tokens",     float(bb.assistant_tokens), float(ab.assistant_tokens), ".0f")
        + _delta_cell("think tokens",    float(bb.think_tokens),     float(ab.think_tokens),     ".0f")
        + _delta_cell("self-corr",       float(bb.self_correction_hits), float(ab.self_correction_hits), ".0f")
        + '</div>'
    )


CSS = """
* { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
body { margin: 0; padding: 1.2rem 1.5rem; background: #f7f7f8; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 1.35rem; margin: 0 0 .25rem 0; }
.intro { color: #555; margin-bottom: 1.5rem; font-size: .9rem; }
.pair-block { margin-bottom: 2.5rem; padding: .8rem; background: white; border: 1px solid #e0e0e3;
              border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.pair-block > h2 { font-size: 1.0rem; margin: 0 0 .6rem 0; padding: .4rem .55rem;
                   background: linear-gradient(to right, #eef4ff, #fff); border-left: 4px solid #36c;
                   border-radius: 3px; }
.pair-block > h2 code { font-size: .95rem; }
.diff-card { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: .4rem;
             padding: .55rem; background: #fafbff; border: 1px solid #e3e8f0;
             border-radius: 5px; margin-bottom: .9rem; }
.metric-cell { text-align: center; padding: .3rem .25rem; background: white;
               border: 1px solid #ecedf2; border-radius: 4px; }
.metric-label { font-size: .65rem; color: #888; text-transform: uppercase; letter-spacing: .04em; }
.metric-values { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .8rem;
                 margin: .15rem 0; color: #222; }
.metric-delta { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .75rem; font-weight: 600; }
.metric-delta.up   { color: #198754; }
.metric-delta.dn   { color: #b02a37; }
.metric-delta.flat { color: #6c757d; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; }
.col { background: white; padding: .4rem .55rem .55rem .55rem; border: 1px solid #ddd;
       border-radius: 5px; overflow: hidden; min-width: 0; }
.col.before { background: #fff5f5; border-color: #f3d6d6; }
.col.after  { background: #f3fbf4; border-color: #c9e8cf; }
.col > h3 { margin: 0 0 .35rem 0; padding: .15rem .35rem; font-size: .8rem; color: #444; }
.col.before > h3 { background: #ffe0e0; }
.col.after  > h3 { background: #d6efdc; }
.trial-header { font-size: .78rem; color: #444; padding: .25rem .35rem; background: rgba(255,255,255,.6);
                border: 1px solid #eaeaea; border-radius: 4px; margin-bottom: .3rem; }
.trial-source { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .68rem;
                color: #888; margin-top: .15rem; word-break: break-all; }
.panel-controls { display: flex; gap: .35rem; margin-bottom: .4rem; }
.ctrl-btn { font-size: .7rem; padding: .2rem .55rem; cursor: pointer; background: #f3f5f9;
            border: 1px solid #d4d8e0; border-radius: 3px; color: #444; }
.ctrl-btn:hover { background: #e7ebf2; }
details.msg { margin: .35rem 0; border-left: 3px solid #ccc; padding: .1rem .1rem .25rem .55rem;
              background: rgba(255,255,255,.78); border-radius: 3px; }
details.msg[open] { background: rgba(255,255,255,1.0); }
details.msg.role-user      { border-left-color: #3a73d3; }
details.msg.role-assistant { border-left-color: #2c8a5e; }
details.msg.role-system    { border-left-color: #7e7e7e; }
details.msg.role-tool      { border-left-color: #b8860b; }
.msg-summary { cursor: pointer; padding: .25rem .15rem; font-size: .76rem; display: flex;
               align-items: center; gap: .45rem; flex-wrap: wrap; color: #444; }
.msg-idx { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #999; min-width: 3.5rem; }
.msg-role { font-weight: 600; text-transform: uppercase; letter-spacing: .04em; font-size: .7rem;
            background: #f0f0f3; padding: .07rem .35rem; border-radius: 2px; color: #555; }
details.msg.role-user      .msg-role { background: #e0eaf8; color: #2a5093; }
details.msg.role-assistant .msg-role { background: #dcf0e3; color: #1f6440; }
details.msg.role-system    .msg-role { background: #e5e5e8; color: #555; }
details.msg.role-tool      .msg-role { background: #fdebc8; color: #8a6310; }
.msg-toks { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #888; font-size: .7rem; }
.msg-preview { color: #777; font-size: .72rem; flex: 1; min-width: 0;
               overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.badge { font-size: .6rem; padding: .05rem .3rem; border-radius: 2px; text-transform: uppercase; letter-spacing: .04em; }
.badge.tc    { background: #fdebc8; color: #8a6310; }
.badge.think { background: #ece1f5; color: #5a3a8a; }
.badge.code  { background: #d8e9f5; color: #1f4d75; }
.msg-body { padding: .2rem .15rem .3rem .15rem; }
.content-text, .think-text { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                              font-size: .76rem; margin: .15rem 0; line-height: 1.45; color: #1a1a1a; }
details.think-block { margin: .25rem 0; padding: .3rem .4rem; background: #f1ecf7; border: 1px dashed #c6b4dc;
                      border-radius: 3px; }
details.think-block > summary { cursor: pointer; font-size: .7rem; color: #5a3a8a; font-weight: 600; }
.think-text { color: #4b3273; }
.tool-call-wrapper { margin: .35rem 0; }
.tool-call-chip { display: inline-block; background: #fdebc8; color: #8a6310; padding: .05rem .4rem;
                  border-radius: 2px; font-size: .65rem; font-weight: 600; margin-bottom: .15rem;
                  text-transform: uppercase; letter-spacing: .04em; }
.code-block { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .73rem; margin: .25rem 0;
              padding: .35rem .5rem; background: #1e2230; color: #d8dee9; border-radius: 3px;
              overflow-x: auto; line-height: 1.42; }
.code-block pre { margin: 0; }
"""


JS = """
// Per-panel "Expand all" / "Collapse all" buttons.
document.addEventListener('click', function(e) {
    if (!(e.target instanceof HTMLElement)) return;
    if (!e.target.classList.contains('ctrl-btn')) return;
    const action = e.target.getAttribute('data-action');
    const panel = e.target.closest('.col');
    if (!panel) return;
    const open = (action === 'expand-all');
    panel.querySelectorAll('details.msg').forEach(d => { d.open = open; });
});
"""


def render_html(
    pairs: List[Tuple[str, Trace, Trace]],
    before_label: str,
    after_label: str,
    output_path: Path,
) -> None:
    encoder = get_tiktoken_encoder()
    parts: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Trace pair render</title>",
        f"<style>{CSS}</style>",
        "</head><body>",
        f"<h1>Trace pair render: <code>{html.escape(before_label)}</code> vs "
        f"<code>{html.escape(after_label)}</code></h1>",
        f'<p class="intro">{len(pairs)} task pairs rendered '
        '(highest-reward trial from each side). Each pair starts with a '
        'diff summary card; messages are collapsed by default — click the '
        '<code>&gt;</code> arrow on any message to expand it. Only the '
        'first user message and the last assistant message are open by '
        'default.</p>',
    ]
    for task, before, after in pairs:
        diff = _diff_card(before, after)
        parts.append('<section class="pair-block">')
        parts.append(f'<h2>Task: <code>{html.escape(task)}</code></h2>')
        parts.append(diff)
        parts.append('<div class="pair">')
        parts.append(f'<div class="col before"><h3>Before — {html.escape(before_label)}</h3>')
        parts.append(_trial_panel_html(before, encoder))
        parts.append("</div>")
        parts.append(f'<div class="col after"><h3>After — {html.escape(after_label)}</h3>')
        parts.append(_trial_panel_html(after, encoder))
        parts.append("</div></div></section>")
    parts.append(f"<script>{JS}</script>")
    parts.append("</body></html>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _pair_rank_key(tup: Tuple[str, Trace, Trace], prefer: str) -> Tuple:
    """Sort key for ranking pairs by 'most interesting'. Larger = first."""
    _, b, a = tup
    if prefer == "flips":
        bp = (b.reward or 0.0) > 0
        ap = (a.reward or 0.0) > 0
        flipped = 1 if bp != ap else 0
        return (flipped, abs((a.reward or 0.0) - (b.reward or 0.0)))
    if prefer == "reward-delta":
        return (abs((a.reward or 0.0) - (b.reward or 0.0)),)
    if prefer == "behavior-delta":
        # Sum of normalized absolute deltas across the four behavioral
        # features that are most likely to be load-bearing. Largest =
        # most-shifted pair, surfacing the richest material for review.
        bb, ab = b.behavioral, a.behavioral
        score = 0.0
        for bv, av in [
            (bb.tool_calls_total, ab.tool_calls_total),
            (bb.tool_errors, ab.tool_errors),
            (bb.assistant_tokens, ab.assistant_tokens),
            (bb.think_tokens, ab.think_tokens),
            (bb.self_correction_hits, ab.self_correction_hits),
        ]:
            scale = max(abs(bv), abs(av), 1.0)
            score += abs(av - bv) / scale
        return (score,)
    return ()


def select_pairs(
    before: Dict[str, List[Trace]],
    after: Dict[str, List[Trace]],
    top_n: int,
    prefer: str,
    exclude: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> List[Tuple[str, Trace, Trace]]:
    """Pick representative pairs, ordered by what's most interesting.

    ``prefer`` choices:
      * ``flips`` — tasks where pass/fail status flipped (most interesting
        for "did behavior change matter?")
      * ``reward-delta`` — tasks with the largest absolute reward swing
      * ``behavior-delta`` — tasks with the largest behavioral-feature
        deltas (helps the LLM-judge step pick the most informative pairs)
      * ``random`` — a uniform sample WITHOUT replacement of the common
        tasks, seeded by ``seed`` for reproducibility. Use this to estimate
        "did behavior change *in general*", as opposed to on the
        cherry-picked most-shifted tasks that ``behavior-delta`` surfaces.
      * ``any`` — first ``top_n`` common tasks sorted alphabetically

    ``exclude`` — task ids to drop from the candidate set before selecting.
    Pass the ``behavior-delta`` winners here so a ``random`` draw is disjoint
    from them (sampling the *rest* of the distribution, without replacement).
    """
    excluded = set(exclude or ())
    common = [t for t in sorted(set(before) & set(after)) if t not in excluded]
    pairs: List[Tuple[str, Trace, Trace]] = []
    for task in common:
        b = _pick_representative(before[task])
        a = _pick_representative(after[task])
        pairs.append((task, b, a))
    if prefer == "random":
        import random as _random

        _random.Random(seed).shuffle(pairs)
        return pairs[:top_n]
    if prefer in ("flips", "reward-delta", "behavior-delta"):
        pairs.sort(key=lambda t: _pair_rank_key(t, prefer), reverse=True)
    return pairs[:top_n]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--before", required=True, help="Pre-RL trace source")
    parser.add_argument("--after", required=True, help="Post-RL trace source")
    parser.add_argument("--output", type=Path, required=True, help="HTML output path")
    parser.add_argument("--top-n", type=int, default=20, help="Number of task pairs to render")
    parser.add_argument(
        "--prefer",
        choices=("flips", "reward-delta", "behavior-delta", "any"),
        default="flips",
        help="How to rank which task pairs to render first",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows loaded per side")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    before = load_traces(args.before, max_rows=args.max_rows)
    after = load_traces(args.after, max_rows=args.max_rows)
    bb = group_by_task(before)
    aa = group_by_task(after)
    pairs = select_pairs(bb, aa, args.top_n, args.prefer)
    if not pairs:
        print("[trace-pair-render] no common tasks between before/after", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_html(pairs, args.before, args.after, args.output)
    print(f"[trace-pair-render] wrote {args.output} ({len(pairs)} pairs)")
    return 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
