"""Prompt builders for the perturbed Docker pipeline."""

from __future__ import annotations

import json

from data.perturbed_docker.schemas import (
    BaselineDockerfile,
    JudgmentResponse,
    VariantResponse,
)

BASELINE_SCHEMA_JSON = json.dumps(BaselineDockerfile.model_json_schema(), indent=2)
VARIANT_SCHEMA_JSON = json.dumps(VariantResponse.model_json_schema(), indent=2)
JUDGMENT_SCHEMA_JSON = json.dumps(JudgmentResponse.model_json_schema(), indent=2)


def build_baseline_prompt(instruction: str) -> str:
    return f"""
Generate a minimal, correct Dockerfile to serve as the *environment* for solving a coding task down the line. Do NOT attempt to incorporate the solution of the task in the Dockerfile! And do NOT use heredocs in any way. The task specification is as follows:

Task:
{instruction}

Constraints:
- Base image must be ubuntu:24.04.
- Use non-interactive apt commands (apt-get update/install -y, no upgrade).
- Never use sudo, add-apt-repository, GUI/X11 deps, or imaginary packages.
- Only install packages essential to solve the task.
- Prefer --no-install-recommends and clean apt cache.
- Ensure Dockerfile builds successfully.

Return JSON matching this schema:
{BASELINE_SCHEMA_JSON}
"""


def build_variant_prompt(row: dict[str, object]) -> str:
    instruction = str(row["instruction"])
    dockerfile_base = str(row["dockerfile_base"])
    rationale = str(row.get("base_rationale", "")) or "n/a"

    return f"""
Task:
{instruction}

Baseline Dockerfile:
```dockerfile
{dockerfile_base}
```

Baseline rationale: {rationale}

Generate 4 variants of the baseline Dockerfile that are reasonably harder for solving the task yet still retain solvability. Ground each change in realistic developer friction, not sabotage. Focus on subtle, correct tweaks that force extra reasoning when running the task. Do NOT attempt to incorporate the solution of the task in the Dockerfile! And do NOT use heredocs in any way.

Motivating tweak patterns (pick different styles across the four variants):
- Remove or downgrade a non-obvious package so the solver must install or rebuild it manually.
- Swap installation channel (apt ↔ pip ↔ manual curl) for one key dependency, creating setup steps without breaking feasibility.
- Change runtime expectations (e.g., require `python3.10-venv`, add `systemctl` service enablement) to introduce environment nuance.
- Pin versions or add minimal config edits that require understanding rather than copy/paste (no fantasy packages, no superfluous churn).
- Inject a tiny service requirement (like ensuring `redis-server` runs) while keeping the Dockerfile lean and compliant with the guardrails.

Rules:
- Keep `ubuntu:24.04` base and valid Dockerfile syntax.
- Never add sudo, GUI/X11 deps, or imaginary packages.
- Minimise noise: every change must have a payoff for genuine debugging friction.
- Variants must build successfully

One-shot illustration (for guidance only — do NOT copy):
- Baseline installs `python3-pip` and `pytest` for a testing task.
- Variant example: remove `pytest`, install `python3-venv`, and add instructions to create a virtual env, summarised as "Forces user to set up pytest manually" with tags `['missing-dependency', 'env-setup']`.

Return JSON matching this schema:
{VARIANT_SCHEMA_JSON}
"""


def build_judge_prompt(row: dict[str, object]) -> str:
    instruction = str(row["instruction"])
    baseline = str(row["dockerfile_base"])
    variants = row["variants"]

    variants_text = "\n".join(
        f"Variant {idx} Dockerfile:\n```dockerfile\n{variant['dockerfile']}\n```\n"
        f"Summary: {variant['changes_summary']}\n"
        f"Tags: {', '.join(variant['difficulty_tags'])}"
        for idx, variant in enumerate(variants)
    )

    return f"""
Act as a strict reviewer selecting the best challenging Dockerfile variant.

Task:
{instruction}

Baseline Dockerfile:
```dockerfile
{baseline}
```

Variants:
{variants_text}

Score each variant (0-5 integer) on:
- solvability: certainty the Dockerfile builds, obeys guardrails, and supports task completion.
  * 0 = broken or violates hard rules; 1 = likely broken/flaky; 3 = likely builds; 5 = clearly builds cleanly and does not preclude task completion.
- difficulty: meaningful extra friction relative to baseline.
  * 0 = identical difficulty; 1 = trivial nuisance; 3 = requires non-trivial reasoning or setup; 5 = excellent deeply challenging-but-fair environment work.
- noise (lower is better): gratuitous churn, fantasy deps, or unrelated edits.
  * 0 = no noise; 2 = minor filler but acceptable; 4 = mostly noise; 5 = egregious bloat.

Flag any hard rule violations in violation_flags.

One-shot judging example (for intuition only):
- Variant removes `gcc` needed for building wheels → solvability 1, difficulty 2, noise 3, flag `['missing-critical-build-tool']`, overall: "Breaks builds via missing gcc".
- Variant replaces `apt-get install nodejs` with manual NodeSource setup → solvability 4, difficulty 4, noise 1.
- Variant adds unrelated desktop packages → solvability 3, difficulty 1, noise 5, flag `['gui-packages']`.

Return JSON matching this schema:
{JUDGMENT_SCHEMA_JSON}
"""
