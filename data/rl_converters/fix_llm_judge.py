#!/usr/bin/env python3
"""Rewire the three QA/instruction-following sandboxes with the LLM-judge pattern.

Takes the Harbor task datasets (``tasks.parquet`` = path + gzipped-tar
``task_binary``) for wizardlm-orca / staqc / qasper and installs the
``nemotron_gym`` LLM-judge verifier contract on every task:

  * ``instruction.md``      — original instruction, with submission guidance
                              appended telling the agent to write its answer to
                              ``/app/response.txt``.
  * ``environment/Dockerfile`` — ``ubuntu:24.04`` + python3 + pip + openai +
                              pytest + litellm (the judge runtime).
  * ``tests/verifier_data.json`` — the task instruction + a per-dataset rubric
                              (correctness/completeness/{relevance,clarity,
                              groundedness}) + judge prompts.
  * ``tests/test_state.py``  — pytest-runnable LLM judge: reads
                              ``/app/response.txt`` (``/app/answer.txt`` legacy
                              fallback), reads ``verifier_data.json``, calls
                              ``litellm.completion`` (default
                              ``openai/gpt-4o-mini``, temperature=0) with the
                              rubric, parses ``\\boxed{<score>}``, writes the
                              float score to ``/logs/verifier/reward.txt``.
  * ``tests/test.sh``        — defaults reward to 0, runs
                              ``python3 -m pytest /tests/test_state.py``.
  * ``task.toml``            — ``LLM_JUDGE_TASK_TOML`` so ``OPENAI_API_KEY``
                              / ``JUDGE_MODEL`` propagate via ``[verifier].env``.

Usage:
    python -m data.rl_converters.fix_llm_judge process  --workdir <dir>
    python -m data.rl_converters.fix_llm_judge smoke    --workdir <dir> [--source NAME]
    python -m data.rl_converters.fix_llm_judge upload   --workdir <dir>
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import tarfile
from pathlib import Path


# --------------------------------------------------------------------------- #
# LLM_JUDGE_TASK_TOML — mirrors data/nemotron_gym/adapter.LLM_JUDGE_TASK_TOML.
# [verifier].env propagates OPENAI_API_KEY (REQUIRED) + JUDGE_MODEL (optional)
# from the Harbor host into the verifier container at trial time.
# --------------------------------------------------------------------------- #
LLM_JUDGE_TASK_TOML = """\
version = "1.0"

[metadata]
adapter = "nemotron_gym"

[verifier]
timeout_sec = 600.0
env = { OPENAI_API_KEY = "${OPENAI_API_KEY}", JUDGE_MODEL = "${JUDGE_MODEL:-}" }

[agent]
timeout_sec = 900.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 4096
storage_mb = 10240
"""


# --------------------------------------------------------------------------- #
# Dockerfile — ubuntu:24.04 + python3 + pip + openai + pytest + litellm.
# --break-system-packages is required for pip on Ubuntu 24.04 (PEP 668).
# --------------------------------------------------------------------------- #
JUDGE_DOCKERFILE = """\
FROM ubuntu:24.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3 python3-pip python3-pytest \\
    && rm -rf /var/lib/apt/lists/* \\
    && pip3 install --no-cache-dir --break-system-packages litellm==1.51.3 openai
"""


# --------------------------------------------------------------------------- #
# test.sh — default-fail, runs the pytest-runnable judge, never fails the shell.
# The judge (test_state.py) writes the float score to reward.txt itself.
# --------------------------------------------------------------------------- #
JUDGE_TEST_SH = """\
#!/bin/bash
set -u

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

python3 -m pytest /tests/test_state.py -v --tb=short > /logs/verifier/test_output.txt 2>&1 || true
exit 0
"""


# --------------------------------------------------------------------------- #
# test_state.py — pytest-runnable LLM judge (logic ported from
# data/nemotron_gym/verifiers/llm_judge.py VERIFIER_PY).
# --------------------------------------------------------------------------- #
JUDGE_TEST_STATE_PY = r'''"""LLM-judge verifier (pytest-runnable) — generated from the nemotron_gym pattern.

Reads the agent response from /app/response.txt (legacy /app/answer.txt
fallback), reads /tests/verifier_data.json for the instruction + rubric, calls
litellm.completion (default openai/gpt-4o-mini, temperature=0), parses
\\boxed{<score>} from the judge reply, and writes the float score to
/logs/verifier/reward.txt.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from collections import defaultdict

REWARD = pathlib.Path("/logs/verifier/reward.txt")
RESPONSE_PRIMARY = pathlib.Path("/app/response.txt")
RESPONSE_LEGACY = pathlib.Path("/app/answer.txt")
DATA = pathlib.Path("/tests/verifier_data.json")

DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"
JUDGE_TIMEOUT_S = 300

DEFAULT_SYSTEM_PROMPT = (
    "You are an impartial evaluator. Read the task and the candidate response, "
    "score it from 0.0 to 1.0 on how well it satisfies the task. Provide your "
    "final score in \\boxed{...} on the last line."
)

DEFAULT_TEMPLATE = """Task / Instruction:
{instruction}

Candidate response:
{response}

Rubric (if any):
{rubric}

Score from 0.0 (does not satisfy) to 1.0 (fully satisfies). End with \\boxed{{<score>}}."""


def _read_response():
    for path in (RESPONSE_PRIMARY, RESPONSE_LEGACY):
        if path.exists():
            text = path.read_text(errors="replace")
            if text.strip():
                return text
    return None


def _format_rubric(items):
    if not isinstance(items, list) or not items:
        return "(none)"
    lines = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rid = it.get("id", "?")
        crit = it.get("criteria", "")
        lines.append(f"  [{rid}] {crit}")
    return "\n".join(lines) if lines else "(none)"


def _extract_score(text):
    m = re.search(r"\\boxed\{\s*([\d.]+)\s*\}", text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    for m2 in re.finditer(r"\b(0\.\d+|1\.0|0|1)\b", text):
        try:
            return max(0.0, min(1.0, float(m2.group(1))))
        except ValueError:
            continue
    return 0.0


def _resolve_model(data):
    cfg = data.get("judge_model")
    if isinstance(cfg, str) and cfg.strip():
        return cfg.strip()
    return os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL).strip() or DEFAULT_JUDGE_MODEL


def _write_reward(score):
    REWARD.parent.mkdir(parents=True, exist_ok=True)
    REWARD.write_text(str(score))


def _compute_score():
    """Run the LLM judge; return a float in [0, 1]. 0.0 on any failure."""
    if not DATA.exists():
        print("verifier_data.json missing", file=sys.stderr)
        return 0.0
    data = json.loads(DATA.read_text())
    response_text = _read_response()
    if response_text is None:
        print(
            "Agent response not found at /app/response.txt or /app/answer.txt",
            file=sys.stderr,
        )
        return 0.0
    instruction = data.get("instruction", "")
    rubric = data.get("rubric")
    principle = data.get("principle")
    template = data.get("judge_prompt_template") or DEFAULT_TEMPLATE
    system_prompt = data.get("judge_system_prompt") or DEFAULT_SYSTEM_PROMPT

    fmt_kwargs = defaultdict(str)
    fmt_kwargs["instruction"] = instruction
    fmt_kwargs["response"] = response_text
    fmt_kwargs["rubric"] = _format_rubric(rubric)
    if isinstance(principle, str):
        fmt_kwargs["principle"] = principle
    try:
        prompt = template.format_map(fmt_kwargs)
    except (KeyError, IndexError, ValueError) as e:
        print(f"template format error ({e}); using default")
        prompt = DEFAULT_TEMPLATE.format_map(fmt_kwargs)

    try:
        from litellm import completion
    except ImportError:
        print("litellm unavailable in container", file=sys.stderr)
        return 0.0

    model = _resolve_model(data)
    try:
        resp = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            timeout=JUDGE_TIMEOUT_S,
        )
    except Exception as e:
        print(f"judge API error ({model}): {e}", file=sys.stderr)
        return 0.0
    if not resp or not resp.choices:
        print(f"judge ({model}) returned no choices", file=sys.stderr)
        return 0.0
    judge_output = (resp.choices[0].message.content or "").strip()
    print(f"Judge model: {model}")
    print(f"Judge output:\n{judge_output[:2000]}")
    score = _extract_score(judge_output)
    print(f"Extracted score: {score}")
    return score


def test_llm_judge_score():
    """The LLM judge scores the agent response and writes the float reward.

    The score is always written to /logs/verifier/reward.txt. The pytest
    assertion additionally gives the pass_ratio reward shaper parseable output:
    a response that the judge scores below the rubric threshold fails the test.
    """
    score = 0.0
    try:
        score = float(_compute_score())
    except Exception as e:
        print(f"verifier exception: {e}", file=sys.stderr)
        score = 0.0
    _write_reward(score)

    threshold = 0.5
    if DATA.exists():
        try:
            threshold = float(json.loads(DATA.read_text()).get("score_threshold", 0.5))
        except Exception:
            threshold = 0.5
    assert score >= threshold, (
        f"response scored {score} below threshold {threshold}"
    )
'''


SUBMISSION_GUIDANCE = (
    "\n\n## Submitting your answer (IMPORTANT)\n"
    "You are a terminal agent. Your chat reply is NOT graded — the grader only "
    "reads the file `/app/response.txt` inside the sandbox. You MUST write your "
    "final answer to `/app/response.txt` by RUNNING A SHELL COMMAND, e.g. a "
    "heredoc:\n\n"
    "    cat > /app/response.txt <<'EOF'\n"
    "    <your answer here>\n"
    "    EOF\n\n"
    "An empty or missing `/app/response.txt` scores 0 regardless of what you "
    "wrote in your reply.\n"
)


# --------------------------------------------------------------------------- #
# Per-source configuration: rubric + judge system prompt.
# --------------------------------------------------------------------------- #
SOURCES = {
    "wizardlm-orca": {
        "src_repo": "DCAgent/wizardlm-orca-sandboxes",
        "dst_repo": "laion/wizardlm-orca-v2",
        "judge_system_prompt": (
            "You are an impartial grader for a general instruction-following "
            "response. Score the candidate response from 0.0 to 1.0 on how well "
            "it satisfies the task, weighing correctness, completeness, and "
            "relevance. Ignore stylistic flourish; reward substantive, accurate, "
            "complete, on-topic answers. Provide your final score in "
            "\\boxed{<score>} on the last line."
        ),
        "rubric": [
            {"id": "correctness", "criteria": "The response is factually correct and directly addresses what the task asks."},
            {"id": "completeness", "criteria": "The response is complete and covers all aspects of the task without omitting key information."},
            {"id": "relevance", "criteria": "The response is relevant and on-topic, free of unnecessary or off-topic content."},
        ],
        "notes": (
            "General instruction-following tasks (system + instruction from the\n"
            "WizardLM_Orca source). Graded by an LLM judge on correctness /\n"
            "completeness / relevance."
        ),
    },
    "staqc": {
        "src_repo": "mlfoundations-dev/staqc-sandboxes",
        "dst_repo": "laion/staqc-v2",
        "judge_system_prompt": (
            "You are an impartial grader for a Stack Overflow code answer. The "
            "task is a programming question; the candidate response should be a "
            "code answer (with any needed explanation). Score from 0.0 to 1.0 on "
            "correctness, completeness, and clarity of the code answer. A correct, "
            "complete, runnable solution scores 1.0; a wrong or missing solution "
            "scores 0.0. Provide your final score in \\boxed{<score>} on the last "
            "line."
        ),
        "rubric": [
            {"id": "correctness", "criteria": "The code answer is correct and would actually solve the stated programming problem."},
            {"id": "completeness", "criteria": "The answer is complete, including any necessary imports, context, or explanation needed to be usable."},
            {"id": "clarity", "criteria": "The code and any explanation are clear, readable, and well-presented."},
        ],
        "notes": (
            "Stack Overflow question->code pairs (STAQC). The agent is asked to\n"
            "answer the programming question; graded by an LLM judge on\n"
            "correctness / completeness / clarity of the code answer."
        ),
    },
    "qasper": {
        "src_repo": "mlfoundations-dev/qasper-sandboxes",
        "dst_repo": "laion/qasper-v2",
        "judge_system_prompt": (
            "You are an impartial grader for a research-paper QA task. The task "
            "names a paper and asks a question (the paper text may be included). "
            "Score the candidate answer from 0.0 to 1.0 on correctness, "
            "completeness, and groundedness in the provided paper text. An answer "
            "that is correct, complete, and supported by the paper scores 1.0; a "
            "wrong, unsupported, or fabricated answer scores 0.0. Provide your "
            "final score in \\boxed{<score>} on the last line."
        ),
        "rubric": [
            {"id": "correctness", "criteria": "The answer is correct given the paper and the question."},
            {"id": "completeness", "criteria": "The answer fully addresses the question, covering the relevant information from the paper."},
            {"id": "groundedness", "criteria": "The answer is grounded in and supported by the provided paper text, not fabricated or hallucinated."},
        ],
        "notes": (
            "QASPER paper-QA tasks. Each task names a paper and asks a question\n"
            "(the paper text is sometimes included). Graded by an LLM judge on\n"
            "correctness / completeness / groundedness in the paper."
        ),
    },
}


# --------------------------------------------------------------------------- #
# tar <-> dict helpers (same scheme as fix_exp_rpt.py)
# --------------------------------------------------------------------------- #
def read_task(task_binary: bytes) -> dict[str, bytes]:
    """Decompress a gzip-tar task binary into {path: content}."""
    try:
        raw = gzip.decompress(task_binary)
    except OSError:
        raw = task_binary
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        for m in tar.getmembers():
            if m.isfile():
                files[m.name] = tar.extractfile(m).read()
    return files


def write_task(files: dict[str, bytes]) -> bytes:
    """Re-encode {path: content} as a gzip-tar binary."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content in sorted(files.items()):
            if isinstance(content, str):
                content = content.encode("utf-8")
            ti = tarfile.TarInfo(name=name)
            ti.size = len(content)
            tar.addfile(ti, io.BytesIO(content))
    return gzip.compress(buf.getvalue())


# --------------------------------------------------------------------------- #
# transformer
# --------------------------------------------------------------------------- #
# Strip a trailing old "write your answer to (answer|response).txt" hint so the
# agent doesn't see conflicting submission guidance.
_OLD_HINT_RE = re.compile(
    r"\s*(?:provide|write|put|save)\b[^.\n]*(?:answer|response)\.?\s*txt[^.\n]*\n?",
    re.IGNORECASE,
)


def _clean_instruction(raw: str) -> str:
    """Drop a trailing legacy answer.txt hint; keep everything else intact."""
    txt = raw.rstrip()
    # Only strip a short trailing hint line, never the whole body.
    m = _OLD_HINT_RE.search(txt[-160:])
    if m:
        txt = (txt[: -160] + txt[-160:][: m.start()]).rstrip()
    return txt


def transform(files: dict[str, bytes], cfg: dict) -> dict[str, bytes]:
    """Install the LLM-judge verifier scaffolding on one task."""
    raw_instr = files.get("instruction.md", b"").decode("utf-8", "replace")
    instruction = _clean_instruction(raw_instr)

    out: dict[str, bytes] = {}
    # New instruction with submission guidance.
    out["instruction.md"] = (instruction + SUBMISSION_GUIDANCE).encode("utf-8")
    out["environment/Dockerfile"] = JUDGE_DOCKERFILE.encode("utf-8")
    out["task.toml"] = LLM_JUDGE_TASK_TOML.encode("utf-8")
    out["tests/test.sh"] = JUDGE_TEST_SH.encode("utf-8")
    out["tests/test_state.py"] = JUDGE_TEST_STATE_PY.encode("utf-8")
    out["tests/verifier_data.json"] = _verifier_data(instruction, cfg).encode("utf-8")
    out["metadata.json"] = _metadata(cfg).encode("utf-8")
    return out


def _verifier_data(instruction: str, cfg: dict) -> str:
    import json
    return json.dumps(
        {
            "instruction": instruction,
            "rubric": cfg["rubric"],
            "judge_system_prompt": cfg["judge_system_prompt"],
            "score_threshold": 0.5,
        },
        ensure_ascii=False,
        indent=2,
    )


def _metadata(cfg: dict) -> str:
    import json
    return json.dumps(
        {
            "source": "nemotron_gym llm_judge",
            "source_dataset": cfg["src_repo"],
            "verifier": "llm_judge",
            "judge": "litellm:default(openai/gpt-4o-mini)",
            "rubric": [r["id"] for r in cfg["rubric"]],
        },
        ensure_ascii=False,
        indent=2,
    )


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def _load_source(src_repo: str, cache: Path) -> list[tuple[str, bytes]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    pq_path = hf_hub_download(
        repo_id=src_repo,
        filename="tasks.parquet",
        repo_type="dataset",
        local_dir=str(cache / src_repo.replace("/", "__")),
    )
    df = pd.read_parquet(pq_path)
    return [(r["path"], r["task_binary"]) for _, r in df.iterrows()]


def process(workdir: Path) -> dict[str, Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    out_paths: dict[str, Path] = {}
    for key, cfg in SOURCES.items():
        rows = _load_source(cfg["src_repo"], workdir / "downloads")
        new_rows: list[tuple[str, bytes]] = []
        for path, blob in rows:
            files = read_task(blob)
            new_files = transform(files, cfg)
            new_rows.append((path, write_task(new_files)))
        out_path = workdir / "out" / f"{key}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(new_rows, out_path)
        out_paths[key] = out_path
        print(f"[{key}] {len(new_rows)} tasks -> {out_path}")
    return out_paths


def _write_parquet(rows: list[tuple[str, bytes]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "path": [p for p, _ in rows],
            "task_binary": [b for _, b in rows],
        }
    )
    pq.write_table(table, str(path))


# --------------------------------------------------------------------------- #
# docker smoke — empty /app -> reward ~0.0 (judge gives low score)
# --------------------------------------------------------------------------- #
def _materialize_task(files: dict[str, bytes], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


def _build_image(dockerfile_text: str, tag: str) -> str:
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "Dockerfile").write_text(dockerfile_text)
        subprocess.run(
            ["docker", "build", "-q", "-t", tag, d],
            check=True,
            capture_output=True,
        )
    return tag


def _docker_run_verifier(
    image: str, task_dir: Path, timeout: int = 240
) -> tuple[float, str]:
    import shutil
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as wd:
        wd = Path(wd)
        app = wd / "app"
        tests = wd / "tests"
        app.mkdir()
        tests.mkdir()
        src_tests = task_dir / "tests"
        if src_tests.exists():
            for f in src_tests.iterdir():
                shutil.copy2(f, tests / f.name)
        # empty /app (no response.txt) -> judge should score ~0.0
        cmd = (
            "mkdir -p /logs/verifier; cd /app; "
            "bash /tests/test.sh >/tmp/v.txt 2>&1; "
            "echo '---REWARD---'; cat /logs/verifier/reward.txt 2>/dev/null || echo NO_REWARD"
        )
        env = dict(os.environ)
        # Forward an OpenAI key if present (the empty-app path returns before any
        # API call, but keep parity with the runtime contract).
        try:
            r = subprocess.run(
                ["docker", "run", "--rm",
                 "-v", f"{app}:/app",
                 "-v", f"{tests}:/tests",
                 "-v", f"{wd}/logs:/logs",
                 "-e", f"OPENAI_API_KEY={env.get('OPENAI_API_KEY', '')}",
                 image, "bash", "-c", cmd],
                capture_output=True, text=True, timeout=timeout,
            )
            out = (r.stdout or "") + (r.stderr or "")
            m = re.search(r"---REWARD---\s*\n?\s*([0-9.]+)", out)
            reward = float(m.group(1)) if m else -1.0
            return reward, out
        except subprocess.TimeoutExpired:
            return -1.0, "TIMEOUT"
        except Exception as e:
            return -1.0, f"ERR {e}"


def smoke(workdir: Path, source: str | None = None, sample: int = 2) -> None:
    import tempfile

    keys = [source] if source else list(SOURCES)
    tag = "smoke_llm_judge"
    print("Building judge image (this also validates the Dockerfile)...")
    _build_image(JUDGE_DOCKERFILE, tag)
    print(f"  image {tag} ready")
    for key in keys:
        pq_path = workdir / "out" / f"{key}.parquet"
        import pandas as pd

        df = pd.read_parquet(pq_path)
        n = min(sample, len(df))
        print(f"\n=== SMOKE {key} ({n} of {len(df)} tasks) — empty /app -> reward ~0.0 ===")
        ok_low = 0
        for i in range(n):
            blob = df.iloc[i]["task_binary"]
            files = read_task(blob)
            with tempfile.TemporaryDirectory() as td:
                task_dir = Path(td) / "task"
                _materialize_task(files, task_dir)
                reward, out = _docker_run_verifier(tag, task_dir)
                low = reward <= 0.05
                ok_low += int(low)
                status = "OK" if low else "FAIL"
                print(f"  [{key}#{i}] empty_reward={reward}  {status}")
                if not low:
                    print("    --- output tail ---")
                    print(out[-800:])
        print(f"  summary: empty->low({0.0}) ok {ok_low}/{n}")


# --------------------------------------------------------------------------- #
# upload
# --------------------------------------------------------------------------- #
README_BODY = """\
# {dst}

LLM-judge-verified version of `{src}`.

Each task follows the `nemotron_gym` LLM-judge verifier contract:

* `instruction.md` — the original task instruction, with submission guidance
  appended directing the agent to write its answer to `/app/response.txt`.
* `tests/test_state.py` — a pytest-runnable LLM judge that reads
  `/app/response.txt` (legacy `/app/answer.txt` fallback), reads
  `tests/verifier_data.json` for the instruction + rubric, calls
  `litellm.completion` (default `openai/gpt-4o-mini`, temperature=0) with the
  rubric, parses `\\boxed{{<score>}}`, and writes the float score to
  `/logs/verifier/reward.txt`.
* `tests/test.sh` — defaults reward to 0, then runs
  `python3 -m pytest /tests/test_state.py`.
* `tests/verifier_data.json` — the task instruction + a per-dataset rubric.
* `environment/Dockerfile` — `ubuntu:24.04` + python3 + pip + openai + pytest
  + litellm.
* `task.toml` — `LLM_JUDGE_TASK_TOML`, so `OPENAI_API_KEY` / `JUDGE_MODEL`
  propagate into the verifier container via `[verifier].env` (`OPENAI_API_KEY`
  is required at trial time).

{notes}

**Rubric:** {rubric_list}

`{n_tasks}` tasks.
"""


def upload(workdir: Path) -> None:

    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    for key, cfg in SOURCES.items():
        pq_path = workdir / "out" / f"{key}.parquet"
        import pandas as pd

        n = len(pd.read_parquet(pq_path))
        readme = README_BODY.format(
            dst=cfg["dst_repo"],
            src=cfg["src_repo"],
            notes=cfg["notes"],
            rubric_list=" / ".join(r["id"] for r in cfg["rubric"]),
            n_tasks=n,
        )
        create_repo(cfg["dst_repo"], repo_type="dataset", token=token, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(pq_path),
            path_in_repo="tasks.parquet",
            repo_id=cfg["dst_repo"],
            repo_type="dataset",
            token=token,
        )
        api.upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=cfg["dst_repo"],
            repo_type="dataset",
            token=token,
        )
        print(f"[{key}] uploaded -> https://huggingface.co/datasets/{cfg['dst_repo']}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["process", "smoke", "upload"])
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--source", default=None, help="smoke: single source key")
    p.add_argument("--sample", type=int, default=2)
    args = p.parse_args()
    if args.cmd == "process":
        process(args.workdir)
    elif args.cmd == "smoke":
        smoke(args.workdir, args.source, args.sample)
    else:
        upload(args.workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
