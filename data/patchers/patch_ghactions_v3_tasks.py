#!/usr/bin/env python3
"""
v3 patcher for laion/exp_rpt_ghactions tasks.

Why v3
------
The v2 patcher only got 38% infra-ok and 0% solved (200/200 trials). Three
classes of failure caused that:

1. **Dockerfile patch incompatible with base image (124/200 infra-fail)**:
   v2 unconditionally appended ``apt-get`` and ``pip`` RUN lines, but the
   corpus uses 5 base images:
     - ``python:3.10-slim``  (3652 tasks) — apt+pip both work
     - ``node:20-slim``       (3592 tasks) — apt works, pip missing → fail
     - ``golang:1.21-alpine`` (2378 tasks) — apt is wrong (alpine uses apk) → fail
     - ``openjdk:17-slim``    ( 196 tasks) — DEPRECATED image, registry unresolvable → fail
     - ``rust:1.75-slim``     ( 112 tasks) — apt works, pip missing → fail
   Net: 41 alpine + 78 (node/openjdk/rust) + 5 openjdk-resolve = 124 fails.

2. **Misleading instruction body (most "infra-ok" 0/100)**:
   The original instructions describe complex CI/CD code projects ("write a
   Go benchmark", "implement a Node.js Helm-chart packager", ...) and the
   v2 preamble inverts that with a one-liner ``Place your solution at
   /app/.github/workflows/main.yml``. The agent followed the body, wrote
   code, and never created a workflow. Verifier ``find`` returned empty →
   ``No workflow YAML found under /app`` → reward 0.

3. **Verifier too strict (the remainder of "infra-ok" 0/100)**:
   - ``actionlint`` rejects any workflow that uses ``@v3`` action pins
     (now deprecated) — but the reference YAML itself uses ``@v3``, so the
     agent is being asked to reproduce a workflow we then reject.
   - ``compare_workflows.py`` requires every reference job name to appear
     verbatim in the agent's YAML, but the instruction doesn't tell the
     agent what the job names are.

v3 fixes
--------
A) **Unify base image** to ``python:3.10-slim`` for every task.
   Rationale: the only verifier dependency we need is ``yamllint`` + python
   for the structural comparator. The agent now only writes YAML — no
   language-specific runtimes are needed in the sandbox. This collapses 5
   Daytona snapshots → 1 (well under the 10/dataset cap).

B) **Rewrite the instruction** entirely. The body becomes a workflow-
   authoring task with concrete *hints from the reference*:
     - the workflow name
     - the trigger events (push / pull_request / schedule / etc.)
     - the job names and their runners
     - the *kinds* of steps in each job (action ``uses`` names — the
       semantic vocabulary, not the SHA pins)
   Without these hints the task is unguessable; with them it's a
   moderately-hard YAML synthesis problem.

C) **Loosen the verifier**: drop actionlint (it rejects the reference),
   keep yamllint, and switch the structural comparator from
   strict-job-name-match to fractional-coverage (>=70% of reference jobs
   present, >=50% of reference action ``uses`` referenced). The reward is
   still binary (1 = passes thresholds, 0 = fails) but the gate is now
   achievable.

CLI
---
    python -m data.patchers.patch_ghactions_v3_tasks \
        --root /tmp/ghactions_src \
        [--limit N] [--dry-run]

Idempotency: re-running is a no-op. Each output file is marker-gated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent

import yaml


# ---------------------------------------------------------------------------
# Idempotency markers
# ---------------------------------------------------------------------------

V3_DOCKERFILE_MARKER = "# --- laion v3 patch: ghactions unified base ---"
V3_TEST_SH_MARKER = "# --- laion v3 patch: ghactions verifier ---"
V3_INSTRUCTION_MARKER = "<!-- --- laion v3 patch: ghactions workflow-authoring task --- -->"
V3_COMPARE_PY_MARKER = "# --- laion v3 patch: ghactions comparator ---"


# ---------------------------------------------------------------------------
# Patch payloads
# ---------------------------------------------------------------------------

V3_DOCKERFILE = dedent(
    f"""\
    {V3_DOCKERFILE_MARKER}
    FROM python:3.10-slim
    WORKDIR /app
    RUN mkdir -p /output && chmod 777 /output
    RUN apt-get update && apt-get install -y --no-install-recommends \\
        ca-certificates curl bsdutils \\
     && rm -rf /var/lib/apt/lists/*
    RUN pip install --no-cache-dir 'yamllint==1.35.1' 'PyYAML==6.0.2'
    """
)


V3_TEST_SH_CONTENT = dedent(
    """\
    #!/usr/bin/env bash
    # --- laion v3 patch: ghactions verifier ---
    # Strategy: lint with yamllint (warnings ok), then run a permissive
    # structural comparator that scores fractional coverage of reference
    # jobs/actions. Reward is 1 if coverage >= thresholds, else 0.
    set +e
    mkdir -p /logs/verifier
    echo "0" > /logs/verifier/reward.txt

    # Find the agent's workflow YAML. Prefer canonical location, then
    # fall back to any YAML in /app.
    WF=""
    if [ -f /app/.github/workflows/main.yml ]; then
        WF=/app/.github/workflows/main.yml
    elif [ -f /app/.github/workflows/main.yaml ]; then
        WF=/app/.github/workflows/main.yaml
    else
        WF=$(find /app -type f \\( -name '*.yml' -o -name '*.yaml' \\) \\
              \\( -path '*/.github/workflows/*' -o -name 'workflow*' \\
                 -o -name 'ci*' -o -name 'main*' -o -name 'test*' \\) \\
              -print 2>/dev/null | head -1)
    fi
    if [ -z "$WF" ] || [ ! -f "$WF" ]; then
        echo "No workflow YAML found under /app"
        echo "0" > /logs/verifier/reward.txt
        exit 1
    fi
    echo "Found agent workflow: $WF"

    # 1) yamllint — warnings allowed, only hard syntax errors fail.
    yamllint -d "{extends: relaxed, rules: {line-length: disable, document-start: disable, truthy: disable, comments: disable, indentation: disable, empty-lines: disable}}" "$WF" 2>&1 \\
        | tee /logs/verifier/yamllint.txt
    yamllint_rc=${PIPESTATUS[0]}

    # 2) Permissive structural comparator (returns 0 = pass, 1 = fail).
    python /tests/compare_workflows.py /tests/reference_workflow.yml "$WF" 2>&1 \\
        | tee /logs/verifier/compare.txt
    compare_rc=${PIPESTATUS[0]}

    echo "v3 verifier: yamllint_rc=$yamllint_rc compare_rc=$compare_rc"

    # yamllint exit codes: 0 OK, 1 errors, 2 warnings. Allow 0 and 2.
    if { [ "$yamllint_rc" -eq 0 ] || [ "$yamllint_rc" -eq 2 ]; } && [ "$compare_rc" -eq 0 ]; then
        echo "1" > /logs/verifier/reward.txt
        exit 0
    else
        echo "0" > /logs/verifier/reward.txt
        exit 1
    fi
    """
)


V3_COMPARE_WORKFLOWS_PY = '''\
#!/usr/bin/env python3
"""
v3 permissive comparator for ghactions tasks.

Compares an agent workflow YAML against a reference. Passes (exit 0) iff:
  - both files parse as YAML mappings
  - agent has at least one ``jobs.*`` entry
  - agent has a recognizable ``on:`` trigger (or the PyYAML-True variant)
  - jobs coverage: # of ref jobs whose id appears in agent / # ref jobs >= 0.70
  - actions coverage: # of ref ``uses`` action-names referenced by agent /
    # ref ``uses`` action-names >= 0.50
  - any single-job-only reference passes if the agent has at least one job
    whose uses-set covers >= 0.50 of the reference's uses (id match optional)

Coverage thresholds are intentionally permissive: this is a reward signal,
not a static-analysis tool. The aim is to recognize "agent reproduced the
structural intent" without requiring exact name parity.

Exits 0 on pass, 1 on fail, 2 on usage error.
"""
# --- laion v3 patch: ghactions comparator ---
from __future__ import annotations
import sys
import yaml
from pathlib import Path


JOB_COVERAGE_THRESHOLD = 0.70
USES_COVERAGE_THRESHOLD = 0.50
SINGLE_JOB_USES_THRESHOLD = 0.50


def load_yaml(path):
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"YAML load failed for {path}: {e}")
        return None
    return data


def get_steps(job):
    if not isinstance(job, dict):
        return []
    return [s for s in (job.get("steps") or []) if isinstance(s, dict)]


def action_name(u):
    return u.split("@", 1)[0].strip() if isinstance(u, str) else None


def collect_uses(job):
    out = set()
    for step in get_steps(job):
        n = action_name(step.get("uses"))
        if n:
            out.add(n)
    return out


def has_on(d):
    if not isinstance(d, dict):
        return False
    if "on" in d:
        return True
    # PyYAML parses ``on:`` as boolean True at the top level.
    if True in d:
        return True
    return False


def main():
    if len(sys.argv) != 3:
        print("usage: compare_workflows.py <reference.yml> <agent.yml>")
        return 2

    ref = load_yaml(Path(sys.argv[1]))
    cand = load_yaml(Path(sys.argv[2]))
    if ref is None:
        print("FAIL: reference YAML did not parse (verifier bug)")
        return 2
    if cand is None:
        print("FAIL: agent YAML did not parse")
        return 1
    if not isinstance(cand, dict):
        print(f"FAIL: agent YAML is {type(cand).__name__}, expected mapping")
        return 1

    # Trigger block check
    if has_on(ref) and not has_on(cand):
        print("FAIL: missing top-level 'on:' trigger block")
        return 1

    ref_jobs = ref.get("jobs") if isinstance(ref, dict) else None
    cand_jobs = cand.get("jobs") if isinstance(cand, dict) else None
    if not isinstance(ref_jobs, dict) or not ref_jobs:
        # Reference has no jobs section — agent only needs an ``on:`` and a
        # non-empty ``jobs:`` mapping to pass.
        if not isinstance(cand_jobs, dict) or not cand_jobs:
            print("FAIL: agent missing 'jobs:' mapping")
            return 1
        print("OK: degenerate reference; agent has jobs and trigger")
        return 0
    if not isinstance(cand_jobs, dict) or not cand_jobs:
        print("FAIL: agent missing 'jobs:' mapping")
        return 1

    # --- Jobs coverage --------------------------------------------------
    ref_job_ids = set(ref_jobs.keys())
    cand_job_ids = set(cand_jobs.keys())
    matched_jobs = ref_job_ids & cand_job_ids
    jobs_cov = len(matched_jobs) / max(len(ref_job_ids), 1)

    # --- Uses coverage --------------------------------------------------
    ref_uses = set()
    for j in ref_jobs.values():
        ref_uses |= collect_uses(j)
    cand_uses = set()
    for j in cand_jobs.values():
        cand_uses |= collect_uses(j)
    if ref_uses:
        matched_uses = ref_uses & cand_uses
        uses_cov = len(matched_uses) / len(ref_uses)
    else:
        # Reference uses no actions — uses-check is vacuous.
        matched_uses = set()
        uses_cov = 1.0

    # --- Single-job fallback -------------------------------------------
    # If the reference has only one job (common), allow the agent to use any
    # job name as long as that job covers >= SINGLE_JOB_USES_THRESHOLD of
    # the reference's uses.
    single_job_pass = False
    if len(ref_job_ids) == 1 and ref_uses and cand_jobs:
        for cand_job in cand_jobs.values():
            cj_uses = collect_uses(cand_job)
            if cj_uses and len(cj_uses & ref_uses) / len(ref_uses) >= SINGLE_JOB_USES_THRESHOLD:
                single_job_pass = True
                break

    print(
        f"jobs: matched={sorted(matched_jobs)} cov={jobs_cov:.2f} "
        f"(ref={sorted(ref_job_ids)}, cand={sorted(cand_job_ids)})"
    )
    print(
        f"uses: matched={sorted(matched_uses)} cov={uses_cov:.2f} "
        f"(ref_n={len(ref_uses)} cand_n={len(cand_uses)})"
    )
    if single_job_pass:
        print("single-job fallback: agent has a job covering >= 50% of ref uses")

    jobs_ok = jobs_cov >= JOB_COVERAGE_THRESHOLD or single_job_pass
    uses_ok = uses_cov >= USES_COVERAGE_THRESHOLD or single_job_pass

    if jobs_ok and uses_ok:
        print("OK: thresholds met")
        return 0

    if not jobs_ok:
        print(
            f"FAIL: jobs coverage {jobs_cov:.2f} < {JOB_COVERAGE_THRESHOLD:.2f}"
        )
    if not uses_ok:
        print(
            f"FAIL: uses coverage {uses_cov:.2f} < {USES_COVERAGE_THRESHOLD:.2f}"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
'''


# Instruction template ------------------------------------------------------
#
# The instruction is built from the reference YAML on a per-task basis,
# so it lives as a function rather than a fixed string.

def _summarize_step(step):
    """Return a short human description of a workflow step.

    Falls back gracefully when the step lacks a ``name``, ``uses`` or ``run``
    field.
    """
    if not isinstance(step, dict):
        return None
    name = step.get("name")
    uses = step.get("uses")
    run = step.get("run")
    if name and uses:
        return f"`{name}` — uses `{action_name_safe(uses)}`"
    if name and run:
        first_run_line = str(run).strip().splitlines()[0] if str(run).strip() else ""
        snippet = first_run_line[:60] + ("..." if len(first_run_line) > 60 else "")
        return f"`{name}` — runs `{snippet}`" if snippet else f"`{name}`"
    if uses:
        return f"uses `{action_name_safe(uses)}`"
    if run:
        first_run_line = str(run).strip().splitlines()[0] if str(run).strip() else ""
        snippet = first_run_line[:60] + ("..." if len(first_run_line) > 60 else "")
        return f"runs `{snippet}`" if snippet else None
    if name:
        return f"`{name}`"
    return None


def action_name_safe(u):
    """Strip the SHA / tag from an action ``uses`` reference."""
    if isinstance(u, str):
        return u.split("@", 1)[0]
    return str(u)


def _normalize_triggers(on_block):
    """Turn the ``on:`` block into a short comma-separated string."""
    if on_block is True:
        return "push, pull_request (defaults)"
    if isinstance(on_block, str):
        return on_block
    if isinstance(on_block, list):
        return ", ".join(str(x) for x in on_block)
    if isinstance(on_block, dict):
        return ", ".join(sorted(on_block.keys()))
    return "(none specified)"


def _build_instruction(ref_yaml_text):
    """Build a workflow-authoring instruction from the reference YAML."""
    try:
        ref = yaml.safe_load(ref_yaml_text) or {}
    except Exception:
        # Fallback: generic instruction.
        ref = {}

    name = ref.get("name") if isinstance(ref, dict) else None
    on_block = ref.get("on") if isinstance(ref, dict) else None
    # PyYAML parses ``on:`` as True
    if on_block is None and isinstance(ref, dict) and True in ref:
        on_block = ref[True]
    triggers = _normalize_triggers(on_block)
    jobs = ref.get("jobs") if isinstance(ref, dict) else {}
    if not isinstance(jobs, dict):
        jobs = {}

    lines = []
    lines.append(V3_INSTRUCTION_MARKER)
    lines.append("# Task: write a GitHub Actions workflow")
    lines.append("")
    lines.append(
        "Write a GitHub Actions workflow file at "
        "`/app/.github/workflows/main.yml` matching the structural "
        "specification below. The verifier runs `yamllint` on your file "
        "and then compares its structure against a hidden reference "
        "workflow. You pass when the YAML parses cleanly and your workflow "
        "covers the reference's jobs and actions (thresholds: ≥ 70% of "
        "reference job names present, ≥ 50% of reference actions "
        "referenced; or, for single-job references, ≥ 50% of reference "
        "actions in any one of your jobs)."
    )
    lines.append("")
    lines.append("## Workflow spec")
    lines.append("")
    if name:
        lines.append(f"- **Workflow name**: `{name}`")
    lines.append(f"- **Triggers**: {triggers}")
    lines.append("")
    lines.append("### Required jobs")
    lines.append("")

    if jobs:
        for job_id, job in jobs.items():
            if not isinstance(job, dict):
                lines.append(f"- **`{job_id}`**")
                continue
            runs_on = job.get("runs-on", "ubuntu-latest")
            if isinstance(runs_on, list):
                runs_on_str = ", ".join(str(x) for x in runs_on)
            else:
                runs_on_str = str(runs_on)
            lines.append(f"- **`{job_id}`** (runs-on: `{runs_on_str}`)")
            steps = job.get("steps")
            if isinstance(steps, list):
                summaries = []
                for step in steps:
                    s = _summarize_step(step)
                    if s:
                        summaries.append(s)
                # Cap to first ~12 step summaries to keep the prompt scannable.
                for s in summaries[:12]:
                    lines.append(f"    - {s}")
                if len(summaries) > 12:
                    lines.append(f"    - ... ({len(summaries) - 12} more steps)")
    else:
        lines.append("- (reference workflow has no `jobs:` block; produce any valid workflow with an `on:` trigger and at least one job)")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Action SHA/tag pins are not required to match; only the action "
        "name before `@` is compared (e.g. `actions/checkout` matches any "
        "version)."
    )
    lines.append(
        "- You may include additional jobs or steps; the verifier only "
        "requires that the reference jobs/actions are *present*."
    )
    lines.append(
        "- `yamllint` is run in relaxed mode (line-length, indentation, "
        "comments, etc. are tolerated). Only hard syntax errors fail it."
    )
    lines.append(
        "- The sandbox base image is `python:3.10-slim` with `yamllint` "
        "and `PyYAML` preinstalled."
    )
    lines.append("")
    return "\n".join(lines)


# Reference-YAML filenames the source corpus uses (always actually YAML).
REFERENCE_YAML_CANDIDATES = (
    "reference_workflow.yml",  # v2 already renamed
    "test_solution.py",
    "test_solution.js",
    "solution_test.go",
    "TestSolution.java",
    "tests.rs",
)

_PY_PREAMBLE_LINES = (
    "import sys",
    "sys.path.insert(0, '/app')",
    "sys.path.insert(0, \"/app\")",
)


def _strip_py_preamble(text):
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped in _PY_PREAMBLE_LINES:
            i += 1
            continue
        break
    return "".join(lines[i:])


def _find_reference_yaml(tests_dir):
    for name in REFERENCE_YAML_CANDIDATES:
        p = tests_dir / name
        if p.is_file():
            return p
    return None


def _write_dockerfile(dockerfile, dry_run):
    """Replace Dockerfile with the v3 unified template (idempotent)."""
    if dockerfile.is_file():
        existing = dockerfile.read_text()
        if V3_DOCKERFILE_MARKER in existing and "FROM python:3.10-slim" in existing:
            return "already_patched"
    if not dry_run:
        dockerfile.parent.mkdir(parents=True, exist_ok=True)
        dockerfile.write_text(V3_DOCKERFILE)
    return "patched"


def _write_test_sh(test_sh, dry_run):
    if test_sh.is_file():
        existing = test_sh.read_text()
        if V3_TEST_SH_MARKER in existing:
            return "already_patched"
    if not dry_run:
        test_sh.write_text(V3_TEST_SH_CONTENT)
        test_sh.chmod(0o755)
    return "patched"


def _write_compare_py(tests_dir, dry_run):
    target = tests_dir / "compare_workflows.py"
    if target.is_file():
        existing = target.read_text()
        if V3_COMPARE_PY_MARKER in existing:
            return "already_patched"
    if not dry_run:
        target.write_text(V3_COMPARE_WORKFLOWS_PY)
        target.chmod(0o755)
    return "patched"


def _write_instruction(instruction_md, ref_yaml_text, dry_run):
    new_text = _build_instruction(ref_yaml_text)
    if instruction_md.is_file():
        existing = instruction_md.read_text()
        if V3_INSTRUCTION_MARKER in existing and existing.strip() == new_text.strip():
            return "already_patched"
    if not dry_run:
        instruction_md.write_text(new_text)
    return "patched"


def _ensure_reference_yaml(tests_dir, ref_path, cleaned_text, dry_run):
    """Make sure tests/reference_workflow.yml exists with the cleaned content."""
    target = tests_dir / "reference_workflow.yml"
    if target.is_file():
        # Already exists — make sure cleaned content is present (idempotent).
        if not dry_run and target.read_text() != cleaned_text:
            target.write_text(cleaned_text)
        if ref_path is not None and ref_path != target and ref_path.is_file():
            if not dry_run:
                ref_path.unlink()
        return "already_patched"
    if not dry_run:
        target.write_text(cleaned_text)
        if ref_path is not None and ref_path != target and ref_path.is_file():
            ref_path.unlink()
    return "patched"


def patch_task(task_dir, dry_run):
    tests_dir = task_dir / "tests"
    env_dir = task_dir / "environment"
    if not tests_dir.is_dir():
        return {"status": "error", "reason": "no tests/ dir"}

    # Find or use the existing reference YAML.
    ref_path = _find_reference_yaml(tests_dir)
    if ref_path is None:
        return {"status": "missing_reference",
                "reason": "no reference YAML file found"}

    try:
        raw = ref_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        return {"status": "error",
                "reason": f"read error: {exc.__class__.__name__}"}

    cleaned = _strip_py_preamble(raw) if ref_path.suffix == ".py" else raw
    try:
        loaded = yaml.safe_load(cleaned)
    except yaml.YAMLError as exc:
        return {"status": "dropped_invalid_yaml",
                "reason": f"yaml.safe_load failed: {exc.__class__.__name__}"}

    if loaded is None or not isinstance(loaded, dict):
        return {"status": "dropped_invalid_yaml",
                "reason": "YAML parsed to non-mapping / empty"}

    # All checks passed — apply patches.
    if not env_dir.is_dir() and not dry_run:
        env_dir.mkdir(parents=True, exist_ok=True)
    dockerfile_status = _write_dockerfile(env_dir / "Dockerfile", dry_run)
    ref_status = _ensure_reference_yaml(tests_dir, ref_path, cleaned, dry_run)
    test_sh_status = _write_test_sh(tests_dir / "test.sh", dry_run)
    compare_status = _write_compare_py(tests_dir, dry_run)
    instr_status = _write_instruction(task_dir / "instruction.md", cleaned, dry_run)

    return {
        "status": "ok",
        "reason": (
            f"dockerfile={dockerfile_status} ref={ref_status} "
            f"test_sh={test_sh_status} compare={compare_status} "
            f"instruction={instr_status}"
        ),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="v3 patch for DCAgent/exp_rpt_ghactions tasks.",
    )
    p.add_argument("--root", required=True,
                   help="Tasks dir (extracted parquet)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report actions only; write nothing")
    p.add_argument("--limit", type=int, default=0,
                   help="Patch at most N tasks (0 = all)")
    p.add_argument("--dropped-out", default=None,
                   help="Optional path to write dropped task names + reason")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    n_total = len(task_dirs)
    n_ok = 0
    n_dropped = 0
    n_error = 0
    n_missing = 0
    dropped = []

    for i, td in enumerate(task_dirs, 1):
        result = patch_task(td, dry_run=args.dry_run)
        status = result["status"]
        if status == "ok":
            n_ok += 1
        elif status == "dropped_invalid_yaml":
            n_dropped += 1
            dropped.append(f"{td.name}\tdropped_invalid_yaml\t{result['reason']}")
            if not args.dry_run:
                import shutil
                shutil.rmtree(td)
        elif status == "missing_reference":
            n_missing += 1
            dropped.append(f"{td.name}\tmissing_reference\t{result['reason']}")
            if not args.dry_run:
                import shutil
                shutil.rmtree(td)
        else:
            n_error += 1
            dropped.append(f"{td.name}\terror\t{result['reason']}")

        if i % 500 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] ok={n_ok} dropped_invalid_yaml={n_dropped} "
                f"missing_reference={n_missing} error={n_error}",
                flush=True,
            )

    print(
        f"\nDone. ok={n_ok}/{n_total}, dropped_invalid_yaml={n_dropped}, "
        f"missing_reference={n_missing}, error={n_error}, "
        f"dry_run={args.dry_run}"
    )

    if args.dropped_out and dropped:
        Path(args.dropped_out).write_text("\n".join(dropped) + "\n")
        print(f"Wrote dropped list to {args.dropped_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
