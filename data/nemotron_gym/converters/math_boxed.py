"""Convert math datasets (OpenMathReasoning, advanced_calculations, stack_overflow)."""

from __future__ import annotations

import re

from ..adapter import (
    HarborTask,
    STANDARD_TEST_SH,
    answer_delivery_guidance,
    render_dockerfile,
    render_metadata,
    sanitize_text,
    task_id_for,
)
from ..verifiers import MATH_BOXED_VERIFIER_PY, NUMERIC_COMPARE_VERIFIER_PY
from . import register
from ._common import extract_prompt


_BASE_IMAGE = "python:3.11-slim-bookworm"
_INSTRUCTION_HEADER = (
    "You are a careful mathematical reasoner. Read the problem below and "
    "show your work, then write your final answer at the path `/app/answer.txt`.\n\n"
    "Format requirement: the answer file should contain `\\boxed{<answer>}` on its\n"
    "last meaningful line. The verifier extracts the last `\\boxed{...}` it finds.\n\n"
    "---\n\n"
)

_NUMERIC_SCALAR = re.compile(
    r"^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+\-]?\d+(?:\.\d*)?)?(?:[eE][+\-]?\d+)?%?$"
)
_OUTER_MATH_DELIMITERS = ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"))


def _single_math_span(value: str) -> bool:
    stripped = value.strip()
    for left, right in _OUTER_MATH_DELIMITERS:
        if stripped.startswith(left) and stripped.endswith(right):
            interior = stripped[len(left) : -len(right)]
            return left not in interior and right not in interior
    return (
        stripped.startswith("$") and stripped.endswith("$") and stripped.count("$") == 2
    )


def _top_level_parts(value: str, separator: str) -> list[str]:
    closers = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    parts: list[str] = []
    start = 0
    for index, char in enumerate(value):
        if char in closers:
            stack.append(closers[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == separator and not stack:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return parts


def _single_container(value: str, left: str, right: str) -> bool:
    if not value.startswith(left) or not value.endswith(right):
        return False
    depth = 0
    for index, char in enumerate(value):
        if char == left:
            depth += 1
        elif char == right:
            depth -= 1
            if depth == 0 and index != len(value) - 1:
                return False
        if depth < 0:
            return False
    return depth == 0


def _interval_expression(value: str) -> bool:
    normalized = value.replace(r"\left", "").replace(r"\right", "")
    parts = re.split(r"\s*(?:\\cup|∪)\s*", normalized)
    if not parts:
        return False
    for part in parts:
        part = part.strip()
        if len(part) < 5 or part[0] not in "([" or part[-1] not in ")]":
            return False
        stack = ["interval"]
        for index, char in enumerate(part[1:], 1):
            if char in "([":
                stack.append(")" if char == "(" else "]")
            elif char in ")]":
                if not stack or (stack[-1] != "interval" and char != stack[-1]):
                    return False
                stack.pop()
                if not stack and index != len(part) - 1:
                    return False
        if stack:
            return False
        if len(_top_level_parts(part[1:-1], ",")) != 2:
            return False
    return True


def answer_type(expected: str) -> str:
    """Choose a conservative typed comparator from the packaged gold syntax."""
    single_math_span = _single_math_span(expected)
    value = _normalize_v3_gold(expected) if single_math_span else expected.strip()
    interval_marker = any(marker in value for marker in (r"\cup", "∪", r"\infty", "∞"))
    half_open = len(value) >= 2 and value[0] + value[-1] in {"(]", "[)"}
    if _interval_expression(value) and (interval_marker or half_open):
        return "interval"
    if (
        not single_math_span
        and _single_container(value, "{", "}")
        and not any(token in value for token in ('"', "'", "...", ":", " or "))
    ):
        return "set"
    equation_parts = _top_level_parts(value, "=")
    equation_has_container = any(
        _single_container(part, left, right)
        and len(_top_level_parts(part[1:-1], ",")) > 1
        for part in equation_parts
        for left, right in (("(", ")"), ("[", "]"), ("{", "}"))
    )
    if (
        len(equation_parts) == 2
        and all(equation_parts)
        and not equation_has_container
        and len(_top_level_parts(value, ",")) == 1
        and not any(token in value for token in ("!=", "<=", ">="))
        and not any(
            token in value
            for token in (
                r"\begin",
                r"\text",
                r"\mid",
                r"\subset",
                r"\in ",
                r"\Bbb",
                r"\mathbb",
                ":=",
            )
        )
        and (
            single_math_span
            or re.fullmatch(r"[A-Za-z0-9_+\-*/^().= ]+", value) is not None
        )
    ):
        return "equation"
    if (
        _single_container(value, "(", ")")
        and len(_top_level_parts(value[1:-1], ",")) > 1
    ):
        return "tuple"
    if (
        _single_container(value, "[", "]")
        and len(_top_level_parts(value[1:-1], ",")) > 1
    ):
        return "list"
    compact = re.sub(r"\s+", "", value)
    if _NUMERIC_SCALAR.fullmatch(compact):
        return "scalar"
    if single_math_span and (
        len(_top_level_parts(value, ",")) > 1
        or len(equation_parts) > 2
        or any(token in value for token in (r"\lor", r"\text", r"\begin"))
    ):
        return "text"
    if single_math_span and not any(
        token in value for token in (r"\text", r"\begin", r"\end")
    ):
        return "scalar"
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        return "scalar"
    if (
        " " not in value
        and re.fullmatch(r"[A-Za-z0-9_+\-*/^().]+", value)
        and any(char.isdigit() or char in "+-*/^()" for char in value)
    ):
        return "scalar"
    return "text"


def _convert(row: dict, source_dataset: str, *, row_idx: int) -> HarborTask | None:
    prompt = extract_prompt(row)
    expected = row.get("expected_answer") or row.get("answer")
    if not isinstance(expected, str) or not expected.strip():
        return None
    expected = sanitize_text(expected, field_name="expected_answer", max_len=8 * 1024)
    instruction_md = _INSTRUCTION_HEADER + prompt
    dockerfile = render_dockerfile(
        base=_BASE_IMAGE,
        pip_packages=("sympy==1.13.3", "antlr4-python3-runtime==4.11"),
    )
    task_id = task_id_for(source_dataset.split("/")[-1], prompt + "||" + expected)
    return HarborTask(
        task_id=task_id,
        instruction_md=instruction_md,
        dockerfile=dockerfile,
        test_sh=STANDARD_TEST_SH,
        verifier_py=MATH_BOXED_VERIFIER_PY,
        verifier_data={
            "expected_answer": expected,
            "answer_type": answer_type(expected),
        },
        metadata=render_metadata(
            source_dataset=source_dataset,
            source_uuid=row.get("uuid") if isinstance(row.get("uuid"), str) else None,
            extra={"row_index": row_idx, "family": "math_boxed"},
        ),
    )


@register("nvidia/Nemotron-RL-math-OpenMathReasoning")
def convert_openmathreasoning(row: dict, row_idx: int) -> HarborTask | None:
    return _convert(row, "nvidia/Nemotron-RL-math-OpenMathReasoning", row_idx=row_idx)


_ADVANCED_INSTRUCTION_HEADER = (
    "You are solving a calculation task. Read the problem below and write the "
    "final numeric answer (a single number) to `/app/answer.txt`.\n\n"
    "IMPORTANT: If the problem asks for **multiple** values (e.g. "
    '"Get me the values for X, Y, and Z"), the verifier only grades the '
    "LAST expression. Compute every expression if you wish, but write ONLY "
    "the value of the LAST one to `/app/answer.txt`. The verifier extracts "
    "the last numeric token from your answer file and compares it to the "
    "reference value within tolerance.\n\n"
    "---\n\n"
)


@register("nvidia/Nemotron-RL-math-advanced_calculations")
def convert_advanced_calculations(row: dict, row_idx: int) -> HarborTask | None:
    """Different schema from OpenMathReasoning — uses `simplified_values` (list[float]).

    For multi-value prompts (e.g. "Get me the values for X, Y, and Z"), we
    grade only the LAST value. This matches what the agent naturally does:
    the verifier extracts the *last* numeric token from /app/answer.txt, so
    the last `simplified_values` entry is the correct reference. (The
    original v1 of this converter used `sv[0]`, which mismatched the agent's
    answer on ~60% of rows and caused a 2.5% solve rate.)
    """
    prompt = extract_prompt(row)
    sv = row.get("simplified_values")
    if not isinstance(sv, list) or not sv:
        return None
    val = sv[-1]
    if not isinstance(val, (int, float)):
        return None
    val = float(val)
    dockerfile = render_dockerfile(base=_BASE_IMAGE)
    gt = row.get("ground_truth") if isinstance(row.get("ground_truth"), str) else None
    task_id = task_id_for(
        "math-advcalc",
        prompt[:128] + "||" + repr(val) + "|" + str(row_idx),
    )
    return HarborTask(
        task_id=task_id,
        instruction_md=_ADVANCED_INSTRUCTION_HEADER + prompt,
        dockerfile=dockerfile,
        test_sh=STANDARD_TEST_SH,
        verifier_py=NUMERIC_COMPARE_VERIFIER_PY,
        verifier_data={
            "expected_value": val,
            "tolerance_abs": 1e-4,
            "tolerance_rel": 1e-4,
        },
        metadata=render_metadata(
            source_dataset="nvidia/Nemotron-RL-math-advanced_calculations",
            source_uuid=None,
            extra={
                "row_index": row_idx,
                "family": "numeric_compare",
                "ground_truth_expr": gt,
                "breadth": row.get("breadth"),
                "max_depth": row.get("max_depth"),
            },
        ),
    )


@register("nvidia/Nemotron-RL-math-stack_overflow")
def convert_math_stack_overflow(row: dict, row_idx: int) -> HarborTask | None:
    return _convert(row, "nvidia/Nemotron-RL-math-stack_overflow", row_idx=row_idx)


# ---------------------------------------------------------------------------
# Nemotron-RL-Math-v2 (the v3-generation refresh of the math_boxed datasets).
#
# Schema: question / expected_answer / responses_create_params / verifier_type.
# `expected_answer` is gold, frequently wrapped in display/inline math
# delimiters (\[ ... \], \( ... \), $$ ... $$, $ ... $) and carrying \r\n.
# We strip a SINGLE surrounding delimiter pair (only when the whole string is
# one math span, never when it is prose with multiple inline spans) so the
# stored gold is the bare expression the agent will box. The verifier
# (MATH_BOXED_VERIFIER_PY) then string-matches / sympy-compares the agent's
# last \boxed{...} against this stored gold.
#
# Upstream `verifier_type == 'math_with_judge'` allows an LLM-judge equivalence
# fallback; we deliberately use the offline sympy/latex comparator instead
# (self-contained, no network at trial time).
# ---------------------------------------------------------------------------

_V3_DELIMS = (("\\[", "\\]"), ("$$", "$$"), ("\\(", "\\)"))


def _normalize_v3_gold(s: str) -> str:
    """Strip surrounding display/inline math delimiters + normalize whitespace.

    Conservative: only strips an outer delimiter pair when the matching opener
    does not reappear inside (so multi-span prose like
    `\\( a \\), \\( b \\)` is left untouched and stays balanced).
    """
    s = s.strip().replace("\r\n", "\n").replace("\r", "\n")
    changed = True
    while changed:
        changed = False
        t = s.strip()
        for open_d, close_d in _V3_DELIMS:
            if (
                len(t) >= len(open_d) + len(close_d)
                and t.startswith(open_d)
                and t.endswith(close_d)
            ):
                interior = t[len(open_d) : -len(close_d)]
                if open_d not in interior and close_d not in interior:
                    s = interior.strip()
                    changed = True
                    break
        else:
            # bare single-$ pair (not $$): strip only if exactly two $ in total
            t2 = s.strip()
            if (
                t2.startswith("$")
                and t2.endswith("$")
                and not t2.startswith("$$")
                and len(t2) >= 2
                and t2.count("$") == 2
            ):
                s = t2[1:-1].strip()
                changed = True
    return re.sub(r"\s*\n\s*", " ", s).strip()


# ---------------------------------------------------------------------------
# Pointer-row hydration.
#
# NVIDIA ships ~54% of Math-v2 as *masked* rows (licensing): `question` and
# `expected_answer` are empty, and the real content lives in an external public
# dataset referenced by `_hf_question_placeholder`. We replicate NVIDIA's own
# `fill_placeholders.py` reconstruction recipe exactly:
#
#   * `int(ph["row"])` is the RAW row offset into the named split (for BOTH
#     'exact' and 'canonical' modes — there is no index remap / dedup).
#   * mode 'exact'    : question = ph["prefix"] + bare + ph["suffix"]
#   * mode 'canonical': question = ph["lead"]   + bare + ph["trail"]
#     ('canonical' rows were reformatted upstream, so NVIDIA wraps the public
#      bare text with stored scaffolding under DIFFERENT key names.)
#   * bare text: for DAPO, strip the fixed instruction wrapper; Skywork is bare.
#   * gold: source row's reward_model.ground_truth — Skywork stores a JSON
#     list-string like '["5"]', DAPO stores it bare.
#
# The two source splits are large (DAPO 1.79M rows, Skywork-math 105k); we
# load+cache each split ONCE at module scope and index by raw offset.
# ---------------------------------------------------------------------------

_DAPO = "BytedTsinghua-SIA/DAPO-Math-17k"
_SKYWORK = "Skywork/Skywork-OR1-RL-Data"

_DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem."
)
_DAPO_SUFFIX = 'Remember to put your answer on its own line after "Answer:".'

# Cache: (dataset, split) -> loaded HF Dataset (lazy).
_HF_SOURCE_CACHE: dict = {}


def _get_source_split(dataset: str, split: str):
    key = (dataset, split)
    cached = _HF_SOURCE_CACHE.get(key)
    if cached is None:
        from datasets import load_dataset

        cached = load_dataset(dataset, split=split)
        _HF_SOURCE_CACHE[key] = cached
    return cached


def _strip_dapo_wrapper(text: str) -> str:
    t = text
    if _DAPO_PREFIX in t:
        t = t.split(_DAPO_PREFIX, 1)[1]
    if _DAPO_SUFFIX in t:
        t = t.rsplit(_DAPO_SUFFIX, 1)[0]
    return t.strip()


def _bare_question(dataset: str, content: str) -> str:
    if dataset == _DAPO:
        return _strip_dapo_wrapper(content)
    return content.strip()


def _unwrap_answer(raw) -> str:
    """Bare answer from a source row's reward_model.ground_truth."""
    import json

    if not isinstance(raw, str):
        if isinstance(raw, list) and raw:
            return str(raw[0])
        return str(raw)
    s = raw.strip()
    if (s.startswith("[") and s.endswith("]")) or (
        s.startswith("{") and s.endswith("}")
    ):
        try:
            v = json.loads(s)
        except Exception:
            return s
        if isinstance(v, list) and v:
            return str(v[0])
        return str(v)
    return s


def _reconstruct_question(ph: dict, bare: str) -> str:
    if ph.get("mode") == "canonical":
        return ph.get("lead", "") + bare + ph.get("trail", "")
    return ph.get("prefix", "") + bare + ph.get("suffix", "")


def _hydrate_pointer(ph: dict) -> tuple[str, str] | None:
    """Return (question, gold_answer) for a placeholder row, or None if unresolvable."""
    dataset = ph.get("dataset")
    split = ph.get("split")
    row_off = ph.get("row")
    if not isinstance(dataset, str) or not isinstance(split, str):
        return None
    try:
        off = int(row_off)
    except (TypeError, ValueError):
        return None
    src_ds = _get_source_split(dataset, split)
    if off < 0 or off >= len(src_ds):
        return None
    src = src_ds[off]
    prompt_field = src.get("prompt")
    if not isinstance(prompt_field, list) or not prompt_field:
        return None
    content = prompt_field[0].get("content")
    if not isinstance(content, str):
        return None
    question = _reconstruct_question(ph, _bare_question(dataset, content))
    gold = _unwrap_answer((src.get("reward_model") or {}).get("ground_truth"))
    if not question.strip() or not gold.strip():
        return None
    return question, gold


@register("nvidia/Nemotron-RL-Math-v2")
def convert_math_v3(row: dict, row_idx: int) -> HarborTask | None:
    """Refresh of the Nemotron-RL-math-* splits → v3 boxed-math tasks.

    Handles BOTH inlined rows (question/expected_answer populated) and masked
    POINTER rows (empty question/answer + `_hf_question_placeholder` referencing
    an external public dataset). Pointer rows are hydrated via NVIDIA's own
    fill_placeholders recipe (see helpers above).
    """
    source_dataset = "nvidia/Nemotron-RL-Math-v2"
    ph = row.get("_hf_question_placeholder")
    is_pointer = isinstance(ph, dict) and bool(ph.get("dataset"))

    if is_pointer:
        hydrated = _hydrate_pointer(ph)
        if hydrated is None:
            return None
        prompt, raw_gold = hydrated
        prompt = sanitize_text(prompt, field_name="prompt", max_len=64 * 1024)
        pointer_meta = {
            "hydrated": True,
            "hydration_mode": ph.get("mode"),
            "hydration_source": ph.get("dataset"),
            "hydration_split": ph.get("split"),
            "hydration_row": ph.get("row"),
        }
    else:
        prompt = extract_prompt(row)
        raw_gold = row.get("expected_answer")
        if not isinstance(raw_gold, str) or not raw_gold.strip():
            return None
        pointer_meta = {"hydrated": False}

    expected = _normalize_v3_gold(raw_gold)
    if not expected:
        return None
    expected = sanitize_text(expected, field_name="expected_answer", max_len=8 * 1024)
    instruction_md = (
        _INSTRUCTION_HEADER
        + prompt
        + answer_delivery_guidance(
            "/app/answer.txt",
            what="your solution, ending with the final answer in \\boxed{...}",
        )
    )
    dockerfile = render_dockerfile(
        base=_BASE_IMAGE,
        pip_packages=("sympy==1.13.3", "antlr4-python3-runtime==4.11"),
    )
    task_id = task_id_for("Nemotron-RL-Math-v2", prompt + "||" + expected)
    extra = {
        "row_index": row_idx,
        "family": "math_boxed",
        "verifier_type": row.get("verifier_type")
        if isinstance(row.get("verifier_type"), str)
        else None,
    }
    extra.update(pointer_meta)
    return HarborTask(
        task_id=task_id,
        instruction_md=instruction_md,
        dockerfile=dockerfile,
        test_sh=STANDARD_TEST_SH,
        verifier_py=MATH_BOXED_VERIFIER_PY,
        verifier_data={
            "expected_answer": expected,
            "answer_type": answer_type(expected),
        },
        metadata=render_metadata(
            source_dataset=source_dataset,
            source_uuid=row.get("uuid") if isinstance(row.get("uuid"), str) else None,
            extra=extra,
        ),
    )
