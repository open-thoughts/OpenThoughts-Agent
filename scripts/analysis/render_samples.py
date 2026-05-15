#!/usr/bin/env python3
"""Render N rows of sera-4.6-lite-t2_v4_1000.jsonl as colored HTML.

Each row → one .html file with:
- Metadata header (instance_id, source, problem_statement preview).
- One block per message, color-coded by role.
- Inside assistant content: <think>...</think> and <tool_call>...</tool_call>
  blocks highlighted with their own background.
- White-space preserved (pre-wrap), no JSON pretty-printing.
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #1c1c1e; color: #ececec; max-width: 1100px; margin: 24px auto;
       padding: 0 16px; line-height: 1.45; }
h1 { color: #f5f5f7; border-bottom: 2px solid #444; padding-bottom: 8px; }
h2 { color: #f5f5f7; margin-top: 24px; }
.meta { background: #2c2c2e; padding: 12px 16px; border-radius: 8px;
        margin-bottom: 18px; border: 1px solid #3a3a3c; }
.meta .key { color: #98989d; font-weight: 600; }
.meta .val { color: #f5f5f7; }
.problem { background: #1f1f21; padding: 12px 14px; border-radius: 6px;
           border-left: 3px solid #007aff; margin: 8px 0;
           white-space: pre-wrap; font-family: ui-monospace, SF Mono, monospace;
           font-size: 12.5px; max-height: 240px; overflow-y: auto; }
.msg { border-radius: 10px; padding: 10px 14px; margin: 12px 0;
       border-left: 4px solid; white-space: pre-wrap;
       font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12.5px; }
.msg .role-tag { display: inline-block; font-weight: 700; padding: 2px 8px;
                 border-radius: 4px; margin-right: 8px; font-size: 11px;
                 font-family: -apple-system, sans-serif; letter-spacing: 0.04em; }
.msg .train-tag { display: inline-block; font-weight: 600; padding: 1px 6px;
                  border-radius: 3px; margin-left: 6px; font-size: 10px;
                  background: #444; color: #ccc; font-family: sans-serif; }
.msg.system    { background: #2c2c2e; border-color: #8e8e93; }
.msg.user      { background: #1c2842; border-color: #0a84ff; }
.msg.tool-resp { background: #3a2e1c; border-color: #ff9f0a; }
.msg.assistant { background: #1c3324; border-color: #30d158; }
.msg.tool      { background: #3a2e1c; border-color: #ff9f0a; }
.role-tag.system    { background: #8e8e93; color: #fff; }
.role-tag.user      { background: #0a84ff; color: #fff; }
.role-tag.tool-resp { background: #ff9f0a; color: #1c1c1e; }
.role-tag.assistant { background: #30d158; color: #1c1c1e; }
.role-tag.tool      { background: #ff9f0a; color: #1c1c1e; }

.think  { display: block; background: #1f2a3a; border-left: 3px solid #5e9eff;
          padding: 6px 10px; margin: 6px 0; border-radius: 4px;
          color: #c4d3eb; }
.think::before { content: "▼ THINK"; display: block; font-size: 10px;
                 color: #5e9eff; font-weight: 700; margin-bottom: 4px;
                 letter-spacing: 0.05em; font-family: sans-serif; }
.toolcall { display: block; background: #2c1a32; border-left: 3px solid #c97cff;
            padding: 6px 10px; margin: 6px 0; border-radius: 4px;
            color: #e7c8f7; }
.toolcall::before { content: "▶ TOOL_CALL"; display: block; font-size: 10px;
                    color: #c97cff; font-weight: 700; margin-bottom: 4px;
                    letter-spacing: 0.05em; font-family: sans-serif; }
.toolresp-inner { display: block; background: #3a2e1c; border-left: 3px solid #ff9f0a;
                  padding: 6px 10px; margin: 6px 0; border-radius: 4px;
                  color: #fbe5bc; }
.toolresp-inner::before { content: "← TOOL_RESPONSE"; display: block; font-size: 10px;
                          color: #ff9f0a; font-weight: 700; margin-bottom: 4px;
                          letter-spacing: 0.05em; font-family: sans-serif; }
.tag-bracket { color: #888; }
"""


THINK_RE     = re.compile(r"(<think>)(.*?)(</think>)", re.DOTALL)
TOOLCALL_RE  = re.compile(r"(<tool_call>)\s*(.*?)\s*(</tool_call>)", re.DOTALL)
TOOLRESP_RE  = re.compile(r"(<tool_response>)\s*(.*?)\s*(</tool_response>)", re.DOTALL)


def highlight_inline(text: str) -> str:
    """Escape HTML, then wrap <think>/<tool_call>/<tool_response> blocks
    with their own styled containers. Order matters: handle each block type
    on the escaped text, replacing escaped tag literals with HTML wrappers."""
    s = html.escape(text)

    def think_sub(m):
        body = m.group(2)
        return f'<span class="think">{body.strip()}</span>'

    def toolcall_sub(m):
        body = m.group(2)
        return f'<span class="toolcall">{body.strip()}</span>'

    def toolresp_sub(m):
        body = m.group(2)
        return f'<span class="toolresp-inner">{body.strip()}</span>'

    # Escaped tags look like &lt;think&gt; etc.
    THINK_E = re.compile(r"&lt;think&gt;(.*?)&lt;/think&gt;", re.DOTALL)
    TC_E    = re.compile(r"&lt;tool_call&gt;\s*(.*?)\s*&lt;/tool_call&gt;", re.DOTALL)
    TR_E    = re.compile(r"&lt;tool_response&gt;\s*(.*?)\s*&lt;/tool_response&gt;", re.DOTALL)

    s = THINK_E.sub(lambda m: f'<span class="think">{m.group(1).strip()}</span>', s)
    s = TC_E.sub(lambda m: f'<span class="toolcall">{m.group(1).strip()}</span>', s)
    s = TR_E.sub(lambda m: f'<span class="toolresp-inner">{m.group(1).strip()}</span>', s)
    return s


def classify(role: str, content: str) -> str:
    """user-role messages that carry <tool_response> get a distinct class."""
    if role == "user" and "<tool_response>" in content:
        return "tool-resp"
    return role


def render_row(row: dict, idx: int) -> str:
    msgs = row.get("messages")
    if isinstance(msgs, str):
        msgs = json.loads(msgs)

    instance_id = row.get("instance_id", "?")
    source = row.get("source", "?")
    problem = row.get("problem_statement", "")
    func_name = row.get("func_name", "")
    func_path = row.get("func_path", "")

    # Stats
    n_msgs = len(msgs)
    role_counts = {}
    for m in msgs:
        r = m.get("role", "?")
        role_counts[r] = role_counts.get(r, 0) + 1

    parts = [
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Sera v4-1000 sample {idx} — {html.escape(instance_id)}</title>",
        f"<style>{CSS}</style></head><body>",
        f"<h1>Sample {idx}: {html.escape(instance_id)}</h1>",
        "<div class='meta'>",
        f"<div><span class='key'>instance_id:</span> <span class='val'>{html.escape(instance_id)}</span></div>",
        f"<div><span class='key'>source:</span> <span class='val'>{html.escape(source)}</span></div>",
        f"<div><span class='key'>func_name:</span> <span class='val'>{html.escape(func_name)}</span></div>",
        f"<div><span class='key'>func_path:</span> <span class='val'>{html.escape(func_path)}</span></div>",
        f"<div><span class='key'>messages:</span> <span class='val'>{n_msgs}  ({', '.join(f'{r}={c}' for r,c in role_counts.items())})</span></div>",
        "</div>",
        "<h2>Problem statement</h2>",
        f"<div class='problem'>{html.escape(problem)}</div>",
        "<h2>Trajectory</h2>",
    ]

    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        content = m.get("content", "") or ""
        if isinstance(content, list):
            content = "".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content)
        train = m.get("train")
        cls = classify(role, content)
        train_tag = ""
        if train is not None:
            train_tag = f"<span class='train-tag'>train={train}</span>"
        body = highlight_inline(content)
        parts.append(
            f"<div class='msg {cls}'>"
            f"<span class='role-tag {cls}'>#{i:02d} {role.upper()}</span>"
            f"{train_tag}"
            f"<br>{body}</div>"
        )

    parts.append("</body></html>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/Users/benjaminfeuer/Documents/agent-traces-analysis/sera_v4_training_data/sera-4.6-lite-t2_v4_1000.jsonl")
    ap.add_argument("--out_dir", default="/Users/benjaminfeuer/Documents/agent-traces-analysis/sera_v4_training_data")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.input) as f:
        for idx in range(args.n):
            line = f.readline()
            if not line:
                break
            row = json.loads(line)
            html_str = render_row(row, idx)
            inst = (row.get("instance_id") or f"row{idx}").replace("/", "_").replace(":", "_")
            outpath = out / f"sample_{idx:02d}_{inst}.html"
            outpath.write_text(html_str)
            print(f"wrote {outpath}  ({len(html_str)//1024} KB)")


if __name__ == "__main__":
    main()
