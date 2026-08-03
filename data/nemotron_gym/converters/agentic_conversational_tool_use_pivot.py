"""Convert nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1.

Single-step "pivot" task: the agent is dropped into the middle of a
customer-service conversation (a policy lives in the system prompt) and must
produce the *next* action. We grade against the dataset's `expected_action`.

Schema (one row):
  responses_create_params.input  : list of OpenAI Responses API items —
       {role, content} dicts for system/user/assistant turns, plus
       {type: "reasoning", summary} items, {type: "function_call", name,
       arguments, call_id} items, and {type: "function_call_output", call_id,
       output} items. We render ALL of these into a transcript so the agent
       sees prior tool calls + their results, which it needs to decide the
       next action.
  responses_create_params.tools  : list of tool definitions (name, description,
       JSON-schema parameters). We render these as an available-tools section.
  expected_action                : the gold next action, one of
       - {"type": "message",       "content": "<assistant reply text>"}
       - {"type": "function_call", "name": "<tool>", "arguments": "<json str>"}

Per-action-type grading (one parquet, per-row verifier choice):
  * function_call  -> TOOL_CALL_VERIFIER_PY (self-contained). The agent writes
       a single JSON object {"name", "arguments"} to /app/answer.txt; the
       verifier checks name + args (case/whitespace-normalized) against the
       single gold call. Deterministic; no network.
  * message        -> LLM_JUDGE_VERIFIER_PY (litellm). Free-form replies can't
       be exact-matched; we route to an LLM judge for *semantic adequacy* —
       does the agent's reply convey the same decision/information as the gold
       reply, consistent with the policy. The gold reply + the policy context
       go into verifier_data; the agent writes /app/response.txt.

These rows are HARD (qwen_235b_info / pass_rate skew low); we grade faithfully
and do not tune to inflate pass rate. Malformed expected_action rows are
skipped (counted by run.py as returned_None).
"""

from __future__ import annotations

import json

from ..adapter import (
    HarborTask,
    LLM_JUDGE_TASK_TOML,
    STANDARD_TEST_SH,
    answer_delivery_guidance,
    render_dockerfile,
    render_metadata,
    sanitize_text,
    task_id_for,
)
from ..verifiers import LLM_JUDGE_VERIFIER_PY, TOOL_CALL_VERIFIER_PY
from . import register

_DATASET = "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1"
_BASE_IMAGE = "python:3.11-slim-bookworm"

_TRANSCRIPT_MAX = 56 * 1024
_TOOLS_MAX = 24 * 1024
_GOLD_MSG_MAX = 16 * 1024
_INSTR_MAX = 128 * 1024


# ---------------------------------------------------------------------------
# Prompt / transcript rendering
# ---------------------------------------------------------------------------
def _as_text(content: object) -> str:
    """Flatten a Responses-API content value to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text") or part.get("content")
                if isinstance(t, str):
                    pieces.append(t)
            elif isinstance(part, str):
                pieces.append(part)
        return "\n".join(pieces)
    return ""


def _render_transcript(input_items: list) -> str:
    """Render the full conversation incl. prior tool calls + outputs.

    `extract_prompt` from _common only keeps role-based dicts and drops
    reasoning / function_call / function_call_output items — which carry the
    tool-use context the agent needs. So we render the transcript ourselves.
    """
    lines: list[str] = []
    for item in input_items:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role:
            text = _as_text(item.get("content"))
            if not text.strip():
                continue
            lines.append(f"[{role}]\n{text.strip()}")
            continue
        itype = item.get("type")
        if itype == "reasoning":
            summary = item.get("summary")
            text = _as_text(summary)
            if text.strip():
                lines.append(f"[assistant reasoning]\n{text.strip()}")
        elif itype == "function_call":
            name = item.get("name")
            args = item.get("arguments")
            if isinstance(name, str):
                args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                lines.append(f"[assistant tool_call] {name}({args_str})")
        elif itype == "function_call_output":
            out = item.get("output")
            out_str = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            if isinstance(out_str, str) and out_str.strip():
                lines.append(f"[tool result]\n{out_str.strip()}")
        elif itype == "message":
            # rare: a message item without a top-level role
            text = _as_text(item.get("content"))
            if text.strip():
                lines.append(f"[assistant]\n{text.strip()}")
    return "\n\n".join(lines)


def _render_tools(tools: object) -> str:
    if not isinstance(tools, list) or not tools:
        return ""
    out: list[str] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if not isinstance(name, str):
            continue
        desc = t.get("description") if isinstance(t.get("description"), str) else ""
        params = t.get("parameters")
        try:
            params_str = json.dumps(params, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            params_str = "{}"
        out.append(f"- {name}: {desc}\n  parameters: {params_str}")
    if not out:
        return ""
    return "Available tools:\n" + "\n".join(out)


# ---------------------------------------------------------------------------
# Per-type instruction headers
# ---------------------------------------------------------------------------
_HEADER_COMMON = (
    "You are the customer-service agent in the conversation below. A policy that "
    "governs how you must respond is included in the system message at the top of "
    "the transcript. Review the entire conversation — including any prior tool "
    "calls and their results — and decide the single next action to take.\n\n"
)

_HEADER_FUNCTION_CALL = (
    _HEADER_COMMON
    + "For THIS turn, the correct next action is to make a tool call. Choose the "
    "appropriate tool from the available tools and the correct arguments based on "
    "the conversation and the policy.\n\n"
    "Write ONLY a single JSON object to `/app/answer.txt` of the form:\n"
    '  {\"name\": \"<tool_name>\", \"arguments\": { ... }}\n'
    "where `arguments` is a JSON object of the tool's parameters. Do not write "
    "anything else to that file. Set `task_complete: true` once written.\n\n"
    "---\n\n"
)

_HEADER_MESSAGE = (
    _HEADER_COMMON
    + "For THIS turn, the correct next action is to send a message to the user "
    "(not a tool call). Write a single reply to the user that conveys the correct "
    "decision and information consistent with the policy and the conversation so "
    "far.\n\n"
    "Write ONLY your reply text to `/app/response.txt` (no JSON wrapper, no shell "
    "transcript, no scratch reasoning). Set `task_complete: true` once written.\n\n"
    "---\n\n"
)

_JUDGE_TEMPLATE = (
    "You are grading a customer-service agent's reply for SEMANTIC ADEQUACY "
    "against a reference reply, under the agent's policy.\n\n"
    "Conversation + policy the agent saw:\n{instruction}\n\n"
    "Reference (gold) reply that is known correct under the policy:\n"
    "{principle}\n\n"
    "Candidate reply produced by the agent:\n{response}\n\n"
    "Score how well the candidate conveys the SAME decision and the same "
    "policy-relevant information as the reference reply. Wording may differ; what "
    "matters is that the candidate (a) reaches the same outcome/decision, (b) does "
    "not contradict the reference or the policy, and (c) does not omit a "
    "policy-critical fact present in the reference. Minor extra pleasantries are "
    "fine. A reply that reaches a different decision, invents facts not supported "
    "by the conversation, or violates the policy should score low.\n\n"
    "Score 0.0 (wrong decision / contradicts reference or policy) to 1.0 "
    "(conveys the same decision and information). End with \\boxed{{<score>}} on "
    "the last line."
)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------
@register(_DATASET)
def convert_agentic_conversational_tool_use_pivot(row: dict, row_idx: int) -> HarborTask | None:
    rcp = row.get("responses_create_params")
    if not isinstance(rcp, dict):
        return None
    input_items = rcp.get("input")
    if not isinstance(input_items, list) or not input_items:
        return None

    ea = row.get("expected_action")
    if not isinstance(ea, dict):
        return None
    atype = ea.get("type")

    transcript = _render_transcript(input_items)
    if not transcript.strip():
        return None
    transcript = sanitize_text(transcript, field_name="transcript", max_len=_TRANSCRIPT_MAX)

    meta_info = row.get("meta_info") if isinstance(row.get("meta_info"), dict) else {}
    common_meta = {
        "row_index": row_idx,
        "trajectory_id": row.get("trajectory_id"),
        "num_unique_actions": row.get("num_unique_actions"),
        "turn": meta_info.get("turn"),
        "step": meta_info.get("step"),
        "assistant_depth": meta_info.get("assistant_depth"),
        "pass_rate": row.get("pass_rate"),
        "agent_ref": (row.get("agent_ref") or {}).get("name")
        if isinstance(row.get("agent_ref"), dict)
        else row.get("agent_ref"),
        "expected_action_type": atype,
    }

    if atype == "function_call":
        name = ea.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        raw_args = ea.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, ValueError):
                return None
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            return None
        if not isinstance(args, dict):
            return None
        try:
            json.dumps(args, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            return None

        tools_block = _render_tools(rcp.get("tools"))
        tools_block = sanitize_text(tools_block, field_name="tools", max_len=_TOOLS_MAX) if tools_block else ""
        instr_body = _HEADER_FUNCTION_CALL
        if tools_block:
            instr_body += tools_block + "\n\n---\n\n"
        delivery = answer_delivery_guidance(
            "/app/answer.txt", what="the JSON tool-call object"
        )
        instruction_md = sanitize_text(
            instr_body + transcript + delivery,
            field_name="instruction_md",
            max_len=_INSTR_MAX,
        )
        task_id = task_id_for(
            "agconv-tu-fc",
            transcript[:128] + "|" + name + "|" + json.dumps(args, sort_keys=True) + "|" + str(row_idx),
        )
        return HarborTask(
            task_id=task_id,
            instruction_md=instruction_md,
            dockerfile=render_dockerfile(base=_BASE_IMAGE),
            test_sh=STANDARD_TEST_SH,
            verifier_py=TOOL_CALL_VERIFIER_PY,
            verifier_data={"expected": [{"name": name, "arguments": args}]},
            metadata=render_metadata(
                source_dataset=_DATASET,
                source_uuid=None,
                extra={**common_meta, "family": "agentic_conv_tool_use_function_call"},
            ),
        )

    if atype == "message":
        content = ea.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        gold = sanitize_text(content, field_name="gold_message", max_len=_GOLD_MSG_MAX)
        delivery = answer_delivery_guidance("/app/response.txt", what="your reply")
        instruction_md = sanitize_text(
            _HEADER_MESSAGE + transcript + delivery,
            field_name="instruction_md",
            max_len=_INSTR_MAX,
        )
        task_id = task_id_for(
            "agconv-tu-msg",
            transcript[:128] + "|" + gold[:128] + "|" + str(row_idx),
        )
        return HarborTask(
            task_id=task_id,
            instruction_md=instruction_md,
            dockerfile=render_dockerfile(base=_BASE_IMAGE, pip_packages=("litellm==1.51.3",)),
            test_sh=STANDARD_TEST_SH,
            verifier_py=LLM_JUDGE_VERIFIER_PY,
            verifier_data={
                "instruction": transcript,
                "principle": gold,
                "judge_prompt_template": _JUDGE_TEMPLATE,
                "score_threshold": 0.5,
            },
            task_toml=LLM_JUDGE_TASK_TOML,
            metadata=render_metadata(
                source_dataset=_DATASET,
                source_uuid=None,
                extra={
                    **common_meta,
                    "family": "agentic_conv_tool_use_message",
                    "judge": "litellm:default(openai/gpt-4o-mini)",
                },
            ),
        )

    return None
