---
name: code-create-staged-plan
description: >-
  DESIGN a non-trivial codebase change (Harbor / MarinSkyRL / vLLM / OT-Agent / LLaMA-Factory) as a
  dependency-ordered STAGED PLAN before writing code — a feature port, a multi-step fix with parity
  requirements, a refactor, a kernel/perf change. Produces a parent plan doc + per-stage scope docs under
  notes/<codebase>/ (each stage = scope + GO/NO-GO validation gate + cost), with global invariants
  (flag-off byte-identical, parity gates), a borrow-map of code anchors (which drift), and safety
  considerations. Evidence/scoping breadcrumbs go in dated agent_logs/. Use when the user says "scope/plan
  this change", "stage it out", "design before coding", or a change is too big/risky for one shot. Pairs
  with code-execute-staged-plan (which runs the plan).
---

# code-create-staged-plan

Turn a substantial codebase change into a **dependency-ordered staged plan** that someone (you, a subagent,
or future-you) can execute incrementally with a clear gate at each step. This is the **design half**;
`code-execute-staged-plan` is the execution half. Plans for the same change/family share a `notes/<codebase>/`
home; scoping evidence lives in dated `agent_logs/`.

> Real examples to mirror: `notes/vllm/` (DCP GQA-LSE fix — `README.md` parent + `stage{0..6}_*_scope.md`),
> the SkyRL FSDP2-EP/router-replay port, the loop-reward B/C/D plan. Read one before writing a new plan.

## When to use
- A feature **port** (upstream → our fork), a **multi-step fix** (esp. with a parity/regression requirement), a **refactor**, a **kernel/perf** change, or any change too big or too risky to land in one commit.
- NOT for a one-line fix or a mechanical edit — just do those (and log if non-obvious).

## Where it lives
- **Parent plan + per-stage scope docs → `/Users/benjaminfeuer/Documents/notes/<codebase>/`** (`vllm/`, `skyrl/`, `harbor/`, `llama-factory/`, `ot-agent/`). One parent (`README.md` or `<change>_plan.md`) + one `stage<N>_<slug>_scope.md` per stage.
- **Scoping evidence / breadcrumbs → dated `/Users/benjaminfeuer/Documents/agent_logs/YYYY-MM-DD_<topic>.md`** (the repro that proved the bug, the upstream-check that ruled out "already fixed", measurements). The plan *cites* these logs.

## Parent plan doc — required sections
1. **Header:** date · **status** (`scoped — propose-only; no code yet` at creation) · target repo + **local path** + branch (the feature branch you'll cut, e.g. `feuer/<slug>`) · links to the evidence `agent_logs/`.
2. **Goal:** the precise, testable end state (e.g. "`dcp=N` rollout bit-identical to `dcp=1`: greedy token-ids identical + logprobs allclose atol 1e-2").
3. **Why / the mechanism:** the root cause or design rationale — enough that a fresh reader understands the change without re-deriving it.
4. **Stage map** (dependency-ordered table): `Stage | title | what | feature(s) | layer | cost (CPU / 1-GPU / N-GPU) | gate`. Order so each stage is independently testable and a later stage builds on an *already-gated* earlier one. Mark the **critical path**.
5. **Global invariants** (assert in EVERY stage): the **flag-off / default-off byte-identical** contract (a new feature is a no-op until its flag flips — mirror the EP/CP scaffold no-op tests); the **parity gate** (the load-bearing equivalence, e.g. G2 bit-identical); regression bounds (don't break MLA / the other arms); **minimal diff** (no gratuitous API/config churn).
6. **Borrow map** (don't reinvent): the exact files/functions/line-anchors you'll touch or copy from — **and a standing note that anchors DRIFT** (reconfirm at impl time; they're from a dated read).
7. **Safety / reward-hacking** (where relevant): policy-invariance for RL reward shaping, ground-truth anchors, "down-weight not zero", parse-real-signals-only.
8. **Validation discipline:** per-stage, what proves the gate (flag-off byte-identical first → behavior-on test → GPU smoke on the correct SIF/env). Name the measurement (paired McNemar + pass@k, `torch.equal`, allclose tol at the bf16 floor — don't loosen a tol silently).

## Per-stage scope doc (`stage<N>_<slug>_scope.md`) — required sections
- **Header:** date · status (`scoped GO` / `blocked` / …) · companion = the parent · "no fix yet" if scope-only.
- **Why this is the right next step** (cheapest / unblocks the rest).
- **Change-set:** exactly what files change (or "test-only; no `<repo>/` source touched this stage").
- **Validation gate (GO/NO-GO):** the concrete pass condition + cost. This is what `code-execute-staged-plan` checks before advancing.
- **Composes with / depends on:** the upstream stages it assumes are already green.

## Discipline
- **Local clone is ground truth** — the plan targets the local repo on a feature branch off `penfever/working` (vLLM: the fork on its branch); execution will commit→push→sync, never patch the cluster (see `supervisor-init`).
- **Default-off & reversible:** design new features so flag-off is byte-identical; that's the gating invariant that makes staged landing safe.
- **Cheapest-repro-first:** stage 0 is usually a unit/CPU harness that reproduces the target signal without a full multi-GPU run, so the fix loop is fast.
- At creation the plan is **propose-only** — no code, no edits, no rebuild. Hand off to `code-execute-staged-plan` to run it.
