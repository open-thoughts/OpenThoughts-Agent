from __future__ import annotations

from dataclasses import dataclass

from bespokelabs import curator

from pydantic import BaseModel

from data.perturbed_docker.prompts import (
    build_baseline_prompt,
    build_judge_prompt,
    build_variant_prompt,
)
from data.perturbed_docker.schemas import (
    BaselineDockerfile,
    JudgmentResponse,
    VariantResponse,
)
from data.gcs_cache import gcs_cache

MODEL_NAME = "gpt-5-nano-2025-08-07"


def _coerce_model(model_cls: type[BaseModel], payload: object) -> BaseModel:
    if isinstance(payload, model_cls):
        return payload
    if isinstance(payload, dict):
        return model_cls.model_validate(payload)
    if isinstance(payload, str):
        return model_cls.model_validate_json(payload)
    raise TypeError(f"Unexpected payload type for {model_cls.__name__}: {type(payload)!r}")


class BaselineBlock(curator.LLM):
    def __init__(self, **kwargs):
        super().__init__(model_name=MODEL_NAME, response_format=BaselineDockerfile, **kwargs)

    def prompt(self, row: dict[str, str]) -> str:
        return build_baseline_prompt(row["instruction"])

    def parse(self, row: dict[str, str], response: str) -> dict[str, str]:
        parsed = _coerce_model(BaselineDockerfile, response)
        dockerfile = parsed.dockerfile.strip()
        return {
            "instruction": row["instruction"],
            "dockerfile_base": dockerfile,
            "base_rationale": (parsed.rationale or "").strip(),
        }


class VariantBlock(curator.LLM):
    def __init__(self, **kwargs):
        super().__init__(model_name=MODEL_NAME, response_format=VariantResponse, **kwargs)

    def prompt(self, row: dict[str, str]) -> str:
        return build_variant_prompt(row)

    def parse(self, row: dict[str, str], response: str) -> dict[str, object]:
        parsed = _coerce_model(VariantResponse, response)
        return {
            **row,
            "variants": [variant.model_dump() for variant in parsed.variants[:4]],
        }


class JudgeBlock(curator.LLM):
    def __init__(self, **kwargs):
        super().__init__(model_name=MODEL_NAME, response_format=JudgmentResponse, **kwargs)

    def prompt(self, row: dict[str, object]) -> str:
        return build_judge_prompt(row)

    def parse(self, row: dict[str, object], response: str) -> dict[str, object]:
        parsed = _coerce_model(JudgmentResponse, response)
        judgments = [j.model_dump() for j in parsed.judgments[: len(row["variants"])]]
        return {**row, "judgments": judgments}


@dataclass
class SelectionResult:
    dockerfile_final: str
    chosen_idx: int
    selection_reason: str


def _select_variant(row: dict[str, object]) -> SelectionResult:
    variants: list[dict[str, object]] = row.get("variants", [])
    judgments: list[dict[str, object]] = row.get("judgments", [])

    best_idx = -1
    best_score = float("-inf")
    reason = ""

    for idx, (variant, judgment) in enumerate(zip(variants, judgments)):
        if any(flag.strip() for flag in judgment.get("violation_flags", [])):
            continue

        solv = judgment.get("score_solvability", 0)
        diff = judgment.get("score_difficulty", 0)
        noise = judgment.get("score_noise", 5)
        score = solv * 100 + diff * 10 - noise

        if score > best_score:
            best_idx = idx
            best_score = score
            reason = judgment.get("overall") or judgment.get("reasoning", "")

    if best_idx == -1:
        return SelectionResult(
            dockerfile_final=row["dockerfile_base"],
            chosen_idx=-1,
            selection_reason="All variants flagged; fallback to baseline.",
        )

    return SelectionResult(
        dockerfile_final=variants[best_idx]["dockerfile"],
        chosen_idx=best_idx,
        selection_reason=reason or "Selected based on reviewer scores.",
    )

@gcs_cache()
def generate_dockerfiles_with_perturbations(instructions: list[str]) -> list[str]:
    """Run the baseline -> variants -> judge -> select pipeline and return final Dockerfiles."""

    rows: list[dict[str, object]] = [{"instruction": instruction} for instruction in instructions]

    baseline_block = BaselineBlock()
    variant_block = VariantBlock()
    judge_block = JudgeBlock()

    baseline_dataset = baseline_block(rows).dataset
    variant_dataset = variant_block(baseline_dataset).dataset
    judged_dataset = judge_block(variant_dataset).dataset

    finals: list[str] = []
    for row in judged_dataset:
        selection = _select_variant(row)
        finals.append(selection.dockerfile_final)

    return finals
