"""Convert nvidia/Nemotron-RL-agent-calendar_scheduling.

The expected output of the agent is a JSON list of events. The verifier checks
the exact event set, each event's local constraints, and pairwise non-overlap.
"""

from __future__ import annotations

import json

from ..adapter import (
    HarborTask,
    STANDARD_TEST_SH,
    render_dockerfile,
    render_metadata,
    sanitize_text,
    task_id_for,
)
from ..verifiers import CALENDAR_VERIFIER_PY
from . import register
from ._common import extract_prompt


_BASE_IMAGE = "python:3.11-slim-bookworm"
_INSTRUCTION_HEADER = (
    "You are scheduling events on a calendar. Read the conversation below and "
    "write your final calendar as a JSON list to `/app/answer.txt`. Each event "
    "must include `event_id` (int), `event_name` (str), `start_time` "
    '("HH:MM"), and `duration` (minutes). Events must not overlap. The verifier '
    "checks the exact event set, duration, time window, declared constraints, "
    "and pairwise overlap.\n\n"
    "---\n\n"
)
_MAX_EVENTS = 32


def _calendar_names_from_prompt(prompt: str) -> dict[int, str]:
    """Return the latest calendar name for each event ID in a transcript."""
    names: dict[int, str] = {}
    decoder = json.JSONDecoder()
    for start, char in enumerate(prompt):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(prompt[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, list):
            continue
        for event in value:
            if not isinstance(event, dict):
                continue
            event_id = event.get("event_id")
            event_name = event.get("event_name")
            if (
                isinstance(event_id, int)
                and not isinstance(event_id, bool)
                and isinstance(event_name, str)
                and event_name.strip()
            ):
                names[event_id] = event_name.strip()
    return names


def _coerce_expected_events(state: object, prompt: str) -> dict | None:
    if not isinstance(state, dict):
        return None
    names = _calendar_names_from_prompt(prompt)
    out: dict = {}
    for k, v in state.items():
        if not isinstance(k, str):
            try:
                k = str(int(k))
            except (TypeError, ValueError):
                continue
        if not isinstance(v, dict):
            continue
        try:
            json.dumps(v, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            continue
        event_id = v.get("event_id")
        if not isinstance(event_id, int) or isinstance(event_id, bool):
            continue
        event_name = names.get(event_id)
        if event_name is None:
            continue
        out[k] = {**v, "event_name": event_name}
        if len(out) >= _MAX_EVENTS:
            break
    return out or None


def _build(row: dict, source_dataset: str, *, row_idx: int) -> HarborTask | None:
    prompt = extract_prompt(row)
    expected = _coerce_expected_events(row.get("exp_cal_state"), prompt)
    if expected is None:
        return None
    uuid = row.get("uuid") if isinstance(row.get("uuid"), str) else None
    rid = row.get("id")
    task_id = task_id_for(
        "agent-calendar",
        (uuid or str(rid) or prompt[:128]) + "|" + json.dumps(expected, sort_keys=True),
    )
    instr = sanitize_text(
        _INSTRUCTION_HEADER + prompt, field_name="instruction", max_len=128 * 1024
    )
    return HarborTask(
        task_id=task_id,
        instruction_md=instr,
        dockerfile=render_dockerfile(base=_BASE_IMAGE),
        test_sh=STANDARD_TEST_SH,
        verifier_py=CALENDAR_VERIFIER_PY,
        verifier_data={"expected_events": expected},
        metadata=render_metadata(
            source_dataset=source_dataset,
            source_uuid=uuid,
            extra={
                "row_index": row_idx,
                "family": "agent_calendar",
                "n_events": len(expected),
            },
        ),
    )


@register("nvidia/Nemotron-RL-agent-calendar_scheduling")
def convert_agent_calendar(row: dict, row_idx: int) -> HarborTask | None:
    return _build(row, "nvidia/Nemotron-RL-agent-calendar_scheduling", row_idx=row_idx)


@register("nvidia/Nemotron-RL-Instruction-Following-Calendar-v2")
def convert_if_calendar(row: dict, row_idx: int) -> HarborTask | None:
    return _build(
        row, "nvidia/Nemotron-RL-Instruction-Following-Calendar-v2", row_idx=row_idx
    )
