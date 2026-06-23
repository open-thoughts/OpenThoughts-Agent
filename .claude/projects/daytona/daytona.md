# Daytona — dependency-specific operating notes

Daytona is the cloud sandbox backend Harbor uses for agentic RL rollouts and datagen trials. These are
the service-side constraints that shape how we launch and monitor jobs. Folded out of memory 2026-06-14.

---

## RL concurrency cap — ≤ 6 RUNNING RL jobs **per cluster**

Cap **concurrently RUNNING** RL jobs at **6 per cluster** (Jupiter, Leonardo, NYU Torch, … each have
their own headroom — **NOT** a cross-cluster aggregate). Each RL job spawns ~hundreds of concurrent
Daytona sandboxes per training step; the rate-limit horizon is about **per-source-cluster connection
count**, not total org load.

- **PENDING restart-chain jobs (`afterany` deps) don't count** — they're SLURM-queued, not creating sandboxes. Only RUNNING counts.
- Before launching an RL batch on a cluster, count `squeue --me -t RUNNING` RL jobs **on that cluster only**. If adding more would exceed 6, either chain behind existing runs (`--dependency afterany:<last>`) or defer.
- Eval jobs count as a *fraction* of an RL job — a few evals alongside 6 RL is OK.
- **Not** a SLURM cap — a Daytona service-side concern scoped per-cluster.
- History: raised 2026-04-23 after 6×16-node Qwen3-32B RL jobs; corrected 2026-05-21 (I had wrongly aggregated Jupiter+Perlmutter as one budget).

---

## Datagen per-job concurrency ceiling — ~100–128 (NOT aggregate)

A datagen job's effective trial concurrency (vLLM `Running:` reqs) tops out at **~100–128 PER JOB**,
regardless of requested `--trace-n-concurrent` / `max_num_seqs`. This is a **per-job ceiling, NOT
multi-job contention** — so running several datagen jobs in parallel is fine.

- **Don't set `max_num_seqs`/`--trace-n-concurrent` much above ~128 per job** — wasted (the job won't be fed beyond ~100–128 effective Running). seqs 128 already saturates; 256 buys nothing. Per-job pattern: req 32→Running~25, 64→~56, 128→~100, 256→~100.
- **Run datagen jobs in parallel freely** up to aggregate Daytona supply (≥~400 concurrent sustained on Jupiter); no need to serialize grids.
- Throughput tracks effective Running ~linearly (~7 gen tok/s/req MiniMax-M2.7, ~5 GLM-5.1-AWQ @32k). If Running caps at ~100, the lever is **prefix-caching** (shared prompts) or the real production target (131k, where KV finally binds), NOT more seqs. At 32k the engine is rarely KV-bound at Running ~100 (KV <40%).
- Evidence (Jupiter 2026-05-30): G9 requested 256 with one sibling and still capped ~100 — proves per-job, not aggregate starvation. The earlier "max 2 datagen jobs total" rule was an over-correction off a misread — **do NOT re-impose it.**

---

## Snapshot caps are HARD limits — never bypass the gate

The safety gates in `hpc/rl_launch_utils.py:prebuild_daytona_snapshots` are intentional hard limits:
- `max_new_snapshots: int = 10` — per-launch cap
- `max_org_snapshots: int = 60` — total org cap
- Both `raise ValueError(...)` when exceeded.

**NEVER** raise these defaults, pass a larger `max_new_snapshots`, or convert the `ValueError`s to a
warning-and-skip. Mass snapshot creation is expensive and starves other org users. (History: I bypassed
both — `max_new_snapshots` 10→10000 inline, then the org `ValueError` → print+return — both reverted in
`8afe8b70`, the structural_debug chain cancelled.)

**Two distinct over-cap cases — handle them differently:**
1. **Org total cap full** (org at quota, next dataset needs a new env) → **AUTONOMOUS: clean STALE/orphaned snapshots to make room**, then launch (it fits under the unchanged cap). Do NOT block/ask. Reclaim only orphaned `harbor__<hash>` envs from completed/cancelled work — snapshots are keyed by sandbox-ENVIRONMENT hash and **shared across every run using those tasks**, so never delete an env a running/queued job depends on (see [[reference-minimax-datagen-tracker]] — "do NOT reclaim per-dataset; leave the shared env snapshot").
2. **Per-launch cap** — a single dataset legitimately needing `> 10` UNIQUE envs (e.g. structural_debug = 5000 unique Dockerfiles) is a mis-designed/oversized dataset. Cleaning stale won't help (it wants 5000 NEW). **STOP + ask** the user: regenerate to share Dockerfiles / prune unique envs / explicitly approve a one-run cap change (named numeric value, documented in the commit message).

Org-cap-full ≠ dataset-needs-too-many-new — only the latter blocks. Either way: make room by deleting
stale, never by lifting the gate.

### How to clean stale snapshots (the procedure for case 1)

Tool: **`scripts/daytona/daytona_snapshot_manager.py`** (the *safe* reclaim tool — it ONLY audits and
deletes snapshots; do NOT confuse it with `cleanup_stale_sandboxes.py`, which targets running sandboxes).
Run from the otagent env; `source secrets.env` first (it reads the key from `--api-key-env`, default
`DAYTONA_DATA_API_KEY` — env var only, never inline).

```bash
# 1. AUDIT first (read-only default — no deletes). Shows used/headroom + which snapshots are STALE.
python scripts/daytona/daytona_snapshot_manager.py --api-key-env DAYTONA_DATA_API_KEY --stale-days 2
# 2. RECLAIM (dry-run unless --yes). Deletes only STALE harbor__* envs:
python scripts/daytona/daytona_snapshot_manager.py --api-key-env DAYTONA_DATA_API_KEY --delete-stale --yes --stale-days 2
```

- **Staleness = idle** (`last_used_at` older than `--stale-days`) **AND not protected/active.** Error-ish snapshots are deletable only once ALSO past the stale window. So this NEVER deletes an env a running/queued job depends on — it's the safe scoping the "leave shared envs" rule requires. `--stale-days 2` is the standard threshold (the RECLAIM EVENT used it to reclaim 13 envs idle 6–11d).
- **Exit codes:** `3` = `--delete-stale` requested but the confirm prompt was declined (use `--yes` to skip the prompt in autonomous runs).
- **Caveats (from the 2026-06-12 reclaim event):**
  - The manager's post-delete "N/60" re-read can lag (e.g. still shows `40/60` right after a reclaim) — that's a **Daytona list-API eventual-consistency artifact**, NOT a failed delete. Confirm real headroom by the next launch's prebuild line (`0 built` / registry-HIT = headroom freed).
  - "Failures" on `daytonaio/sandbox:*` base images + `daytona-gpu` are **non-org-snapshot images** (already gone) — harmless, ignore.
  - Some reclaimed `harbor__<hash>` envs are shared/re-buildable; a later run that needs one rebuilds it instantly as a registry hit (0 err) — reclaiming a stale-but-rebuildable env is safe.
- **This is autonomous at the cap** (case 1): never raise/bypass the cap, never block on the user; just reclaim stale and launch. Re-audit periodically as the sweep builds new-env snapshots. Tool added OT-Agent commit `a34e45db`.

**Daytona RL org snapshot quota = 40** (HARD, server-side) was observed during the TIS smoke (the r2egym
variant needed 1785 new snapshots → blocked; reused a 4-task subset whose snapshots already existed, 0
new). Snapshot-optimized dataset variants (converted to need FEW snapshots) launch fine — don't pre-flag
them as cap-blocked.
