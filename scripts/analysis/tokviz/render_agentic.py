"""render_agentic.py -- render captured shim prompts into agentic_*.html.

The OpenAI shim (serve_hf_openai.py) logs every request's fully-templated
prompt + response to a JSONL file. After running swe-agent against the shim,
this module picks a representative generation step (the FIRST model call for a
given run) and renders it via render_tokens.render_to_html, with the response
appended as the generated continuation.

This is also usable standalone to re-render an existing log without re-running
swe-agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from config import INSTRUCT_MODEL_ID, OUTPUT_DIR, SHIM_LOG
from render_tokens import render_to_html


def load_log(path: Path = SHIM_LOG) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def render_record(
    tok, record: Dict, idx: int, which: str, suffix: str = "", parser_key: str = ""
) -> str:
    """Render one logged shim record (one model call) to an HTML file.

    ``suffix`` is appended before ``.html`` to distinguish parser variants:
      ""          -> agentic_NN_<which>.html         (thought_action)
      "_funccall" -> agentic_NN_<which>_funccall.html (function_calling)
    """
    parser_label = parser_key or ("function_calling" if suffix else "thought_action")
    label = (
        f"Agentic #{idx:02d} — {which.upper()} / {parser_label} "
        f"(swe-agent step, via OpenAI shim)"
    )
    note = (
        f"Captured from the OpenAI shim log: the exact templated prompt swe-agent "
        f"sent to the {which} model on a representative step using the "
        f"{parser_label} parser. "
    )
    if parser_label == "function_calling":
        note += (
            "The tools are rendered by the chat template as a <tools>…</tools> "
            "JSON array in the system message (structured tool-calling form). "
        )
    else:
        note += (
            "The tools are documented as TEXT under COMMANDS: in the system "
            "prompt (no structured tool array is sent). "
        )
    if which == "base":
        note += "Base-model agentic output is expected to be incoherent — that is part of the comparison."
    html = render_to_html(
        tok,
        model_id=record.get("model_id", INSTRUCT_MODEL_ID),
        setting_label=label,
        prompt_ids=record["prompt_ids"],
        generated_ids=record.get("generated_ids") or None,
        extra_note=note,
    )
    out = OUTPUT_DIR / f"agentic_{idx:02d}_{which}{suffix}.html"
    out.write_text(html, encoding="utf-8")
    return str(out)


def is_first_call(record: Dict) -> bool:
    """True if this record is the FIRST model call of a swe-agent task.

    The first call of a task is the only one whose templated prompt contains the
    system message (tool docs) + the instance message (issue) but NO prior
    assistant turn -- i.e. exactly one ``<|im_start|>assistant`` (the trailing
    generation prompt swe-agent asks the model to fill in). On subsequent calls
    the model's previous replies are folded back into history, so there are 2+
    assistant turns. Selecting first-calls guarantees the rendered prompt
    contains BOTH the tool docs and the problem statement, not a mid-trajectory
    step (which the last_n_observations / our no-history setup may have trimmed).
    """
    prompt = record.get("templated_prompt", "")
    return prompt.count("<|im_start|>assistant") <= 1


def render_first_calls_per_which(tok_loader) -> List[str]:
    """Render the first model call of each task, split by parser variant.

    Records are bucketed by (which, parser) where parser is derived from the
    shim's ``has_tools`` flag:
      * has_tools=False -> thought_action -> agentic_NN_<which>.html
      * has_tools=True  -> function_calling -> agentic_NN_<which>_funccall.html

    Within each bucket we render the FIRST model call of each distinct swe-agent
    task (see is_first_call) — the steps carrying both the tool docs/array and
    the issue text. Falls back to the first N records if none are flagged as
    first-calls.
    """
    records = load_log()
    produced: List[str] = []

    # (which, parser_key, suffix) -> records
    def parser_of(r: Dict) -> tuple:
        if r.get("has_tools"):
            return ("function_calling", "_funccall")
        return ("thought_action", "")

    buckets: Dict[tuple, List[Dict]] = {}
    for r in records:
        pk, suffix = parser_of(r)
        buckets.setdefault((r["which"], pk, suffix), []).append(r)

    for (which, parser_key, suffix), recs in sorted(buckets.items()):
        firsts = [r for r in recs if is_first_call(r)]
        chosen = firsts[:5] if firsts else recs[:5]
        if not chosen:
            continue
        tok = tok_loader(chosen[0]["model_id"])
        for i, rec in enumerate(chosen, start=1):
            produced.append(
                render_record(tok, rec, i, which, suffix=suffix, parser_key=parser_key)
            )
    return produced
