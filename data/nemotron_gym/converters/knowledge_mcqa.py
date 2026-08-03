"""Convert MCQA datasets (knowledge-mcqa, web_search-mcqa)."""

from __future__ import annotations

from ..adapter import (
    HarborTask,
    STANDARD_TEST_SH,
    render_dockerfile,
    render_metadata,
    sanitize_text,
    task_id_for,
)
from ..verifiers import REGEX_LETTER_VERIFIER_PY
from . import register
from ._common import extract_prompt


_BASE_IMAGE = "python:3.11-slim-bookworm"
_DEFAULT_REGEX = r"Answer\s*:\s*(?!Answer)\s*([A-Za-z0-9])\s*"
_MAX_REGEX_LEN = 512


_INSTRUCTION_HEADER = (
    "You are answering a multiple-choice question. Read the question below and "
    "write your final answer to `/app/answer.txt`.\n\n"
    "The verifier extracts a single letter (A/B/C/...) from your answer file using "
    "a regex pattern; the simplest valid output is a file containing exactly\n"
    "`Answer: X` (where X is your chosen letter).\n\n"
    "---\n\n"
)


def _convert(row: dict, source_dataset: str, *, row_idx: int) -> HarborTask | None:
    prompt = extract_prompt(row)
    expected = row.get("expected_answer")
    if not isinstance(expected, str) or not expected.strip():
        return None
    expected = sanitize_text(expected, field_name="expected_answer", max_len=64)
    tmpl = row.get("template_metadata") or {}
    pattern = _DEFAULT_REGEX
    if isinstance(tmpl, dict):
        candidate = tmpl.get("output_regex")
        if isinstance(candidate, str) and 0 < len(candidate) <= _MAX_REGEX_LEN:
            pattern = sanitize_text(candidate, field_name="output_regex", max_len=_MAX_REGEX_LEN)
    dockerfile = render_dockerfile(base=_BASE_IMAGE)
    uuid = row.get("uuid") if isinstance(row.get("uuid"), str) else None
    task_id = task_id_for(source_dataset.split("/")[-1], (uuid or prompt[:128]) + "|" + expected)
    return HarborTask(
        task_id=task_id,
        instruction_md=_INSTRUCTION_HEADER + prompt,
        dockerfile=dockerfile,
        test_sh=STANDARD_TEST_SH,
        verifier_py=REGEX_LETTER_VERIFIER_PY,
        verifier_data={
            "expected_answer": expected,
            "output_regex": pattern,
        },
        metadata=render_metadata(
            source_dataset=source_dataset,
            source_uuid=uuid,
            extra={"row_index": row_idx, "family": "mcqa"},
        ),
    )


@register("nvidia/Nemotron-RL-knowledge-mcqa")
def convert_knowledge_mcqa(row: dict, row_idx: int) -> HarborTask | None:
    return _convert(row, "nvidia/Nemotron-RL-knowledge-mcqa", row_idx=row_idx)


@register("nvidia/Nemotron-RL-knowledge-web_search-mcqa")
def convert_web_search_mcqa(row: dict, row_idx: int) -> HarborTask | None:
    return _convert(row, "nvidia/Nemotron-RL-knowledge-web_search-mcqa", row_idx=row_idx)
