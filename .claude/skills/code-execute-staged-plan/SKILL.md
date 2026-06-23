---
name: code-execute-staged-plan
description: >-
  EXECUTE a staged codebase plan (from code-create-staged-plan or an existing notes/<codebase>/ plan) one
  stage at a time, gate-by-gate, while keeping the local clone ground truth and a dated agent_logs/ progress
  log. For each stage: re-read the scope + reconfirm the (drifting) code anchors, make the edit in the LOCAL
  clone on the feature branch, run the validation gate (flag-off byte-identical FIRST, then behavior-on /
  parity / GPU smoke on the right SIF/env), commit+push and sync to the cluster (vLLM = build from source,
  never rsync), update the plan status, and only THEN advance. Log every stage/debug session in
  agent_logs/YYYY-MM-DD_<topic>.md so long runs don't lose context. Use when the user says "execute/run the
  plan", "do stage N", or "continue the <X> port/fix".
---

# code-execute-staged-plan

Run a staged plan incrementally: **one stage, one gate, one commit at a time**, never advancing past a red
gate. The plan lives in `notes/<codebase>/`; you keep a **dated progress log** in `agent_logs/`. This is the
execution half of `code-create-staged-plan`.

## Inputs
- The **plan**: `notes/<codebase>/{README.md | <change>_plan.md}` + the per-stage `stage<N>_<slug>_scope.md`. Read the parent (goal + invariants + stage map) and the current stage's scope doc.
- The **progress log**: a dated `agent_logs/YYYY-MM-DD_<topic>.md` — the running record for this change. If one exists for the change, append; else create it.

## Per-stage loop
1. **Re-read the stage scope + RECONFIRM anchors.** The borrow-map line/file anchors drift — grep/open the real files now; don't trust line numbers from the plan's authoring date.
2. **Edit the LOCAL clone on the feature branch.** `harbor` / `MarinSkyRL` / `vllm` / `OpenThoughts-Agent` under `~/Documents/`, feature branch off the canonical branch. Never hand-edit on a cluster, never patch-by-rsync (see `supervisor-init` — local is ground truth). Keep the diff minimal (global invariant G4).
3. **Run the validation gate, in order:**
   - **Flag-off byte-identical FIRST** — prove the change is a no-op when its flag is off (`torch.equal` / golden test / the EP-CP no-op test pattern). This is the gating invariant; assert it before any behavior-on test.
   - **Behavior-on / parity** — the stage's GO condition (e.g. `dcp=N`==`dcp=1` greedy bit-identical + logprobs allclose at the **bf16 floor** tol — don't loosen tol to pass).
   - **GPU smoke** where needed — via `sbatch` on the **correct SIF/env** (`.claude/ops/<cluster>/ENVIRONMENT_MAP.md`; torch is the version discriminator). Don't run a GPU parity test on the wrong runtime (the classic DCP false-NO-GO).
   - Lint/syntax via `mcp__ide__getDiagnostics`, not `py_compile`.
4. **Commit + push + sync.** One commit per stage/logical unit, descriptive message. The three Python repos: `git pull` on the cluster (editable, live). **vLLM: build from source on the cluster from the committed commit** (never rsync edits; some envs may run vanilla). Verify the cluster clone is at the pushed commit and **not dirty/divergent** — if it is, fold the drift into the local clone properly, then hard-reset the cluster.
5. **Record results:** in the stage scope doc + parent flip the status (`Stage N ✅ DONE — commit <sha>, gate <result>`); append to the `agent_logs/` log: what changed, the commit, the gate numbers, any blocker, and the **next step**.
6. **Advance only on GREEN.** A red/ambiguous gate stops the loop — diagnose (dispatch an investigative subagent if it's deep), log it, fix, re-gate. Don't paper over a failing parity gate.

## Progress-log discipline (`agent_logs/`)
- **One dated file per change/debug thread** (`YYYY-MM-DD_<topic>.md`); keep appending across stages and across sessions so a long debug never loses context (this is the whole point — the DCP/SkyRL/loop-reward threads each have such a log).
- Capture: the hypothesis, what was tried, **commit SHAs**, gate results (with numbers), what was ruled out, and the current blocker + next action. Future-you (or a fresh subagent) should be able to resume from the log alone.
- This log is ALSO the per-change failure/remediation record the cron sweep references for genuine FAILED jobs.

## Subagents (for parallel stages / deep investigations)
Dispatch with the rules baked in: edit local → commit/push → sync (never patch the cluster); secrets via env vars only (`source secrets.env`, reference by name); reconfirm anchors; run the gate and report numbers; STOP + report if a gate fails or scope is ambiguous rather than forcing a pass. **Verify the subagent's gate claim yourself** before flipping a stage to DONE.

## On completion
When the final stage is green: update the parent plan status to DONE (with the commit range + the headline gate result), make sure the change is committed+pushed on the canonical branch (merge the feature branch if that's the convention — e.g. SkyRL `feuer/<slug>` → `penfever/working`), confirm the clusters are synced/rebuilt, and write the closing `agent_logs/` entry. If it spawned follow-ups, note them (and `/schedule` only if there's a concrete dated obligation).
