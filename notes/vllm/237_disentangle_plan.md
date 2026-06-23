# #237 branch disentanglement — clean atomic-commit plan

- **Date:** 2026-06-19
- **Status:** disentangled — `feuer/237-clean` produced locally, propose-only (NO push, NO merge, NO force-push). Original `feuer/r3-rank-symmetry-norope` preserved untouched.
- **Target repo:** local vLLM fork — `/Users/benjaminfeuer/Documents/vllm` (GROUND TRUTH; clusters build from this fork from source).
- **Base / mainline:** `penfever/working` @ `5aa6fbfba` (== `origin/penfever/working`).
- **Messy branch being disentangled:** `feuer/r3-rank-symmetry-norope` @ `07e9fbca3`.
- **Clean branch produced:** `feuer/237-clean` (off `penfever/working`).
- **Evidence:** `vllm/agent_logs/2026-06-18_237_r3_rank_symmetry_fix.md` (the fix-work log; documents the rope regression and — in its later sections — the real #232 root cause).

---

## Goal

Replace the divergent `feuer/r3-rank-symmetry-norope` branch with a set of clean, atomic, independently-reviewable
commits off `penfever/working`, so each concern can be reviewed/landed ONE AT A TIME and we stop carrying a big
divergent branch.

## Key structural finding (reframes the whole task)

The merge-base of `feuer/r3-rank-symmetry-norope` against `penfever/working` is `5aa6fbfba` — which is a commit
**on the branch itself**. `penfever/working` is a strict **ancestor** of the messy branch. So:

- The entire `#232` chain (`690e4dce1` MQ chunk-size, `32a11afc8` transport split, `1b5a72e9a` FIX-A
  stream-sync drop) AND the rope fix (`28faed993`) AND the faulthandler instrument (`5aa6fbfba`) are
  **ALREADY ON MAINLINE** (`penfever/working`). They are NOT part of this branch's divergence — they need no
  disentangling here.
- The branch's TRUE divergence vs mainline is exactly **7 commits**: `git log --oneline penfever/working..feuer/r3-rank-symmetry-norope`.

This means bucket (E) (the #232 `690e4dce1` chunk-size fix) is moot for this exercise — it already landed
cleanly on mainline as its own commit. Verified: `git merge-base --is-ancestor 690e4dce1 penfever/working` → true.

## What the actual #237 failing signature is (verified, NOT the count)

Per the commit bodies and `agent_logs/2026-06-18_237_…`: the failing signature is **rank-ASYMMETRY in the number
of R3-capture GPU/stream ops issued across TP/CP ranks** in the per-step D2H epilogue. Pre-fix, the Stage-0
op-counter (job 923741) showed rank0 issuing `[sync_d2h_enter,main_copy,copystream_wait,pinned_copy,event_record]`
every step while rank1 (host_cache=None) issued only `[sync_d2h_enter]` — a 73/73-step mismatch. Under
`enforce_eager` (no CUDA graph to enforce identical op streams) the rank-0-only main-stream op skews rank 0's
collective launch timing and on long decode the group desyncs → 900s watchdog → EngineDead.

**The `count 607744` AllGather is BENIGN** (= 4×151936 padded vocab = the LM-head logits all-gather; rank-symmetric
in shape, fires thousands of times in healthy runs). Do not treat it as the deadlock.

---

## Classification table — every divergent commit → bucket → disposition

Divergent commits (newest first), `penfever/working..feuer/r3-rank-symmetry-norope`:

| Orig SHA | Subject | Bucket | Disposition in clean branch |
|---|---|---|---|
| `07e9fbca3` | Revert "re-run patch_rope_parameters after rope_scaling hf_override" | **D** rope revert | **KEPT** as its own commit (`revert: re-run patch_rope_parameters…`). It is a GENUINE regression fix — see the rope knot below. |
| `afe60f19a` | style(#237): ruff E501 line-length fixes | **F** pure style | **FOLDED** — the cosmetic fixes are absorbed into the rewritten fix/instrument/test commits (comments rewritten clean); no standalone style commit. |
| `36a23bf6a` | test(#237 Stage 3): dump generated token-ids for parity | **C** parity/test tooling | **FOLDED into the single test commit** (test harness at final state). |
| `3de064d3e` | fix(#237 Stage 1, FIX B): rank-symmetric R3 capture epilogue | **A** genuine #237 fix (+ woven-in **B** instrumentation) | **SPLIT** → the production fix (no instrumentation) = commit 1; the woven `_r3_op_*` hooks moved into the debug-instrument commit 2. |
| `269277979` | instrument(#237 Stage 0): self-dumping op counter + repro (GATE GREEN) | **B** debug instrumentation (+ **C** harness) | **SPLIT** → instrumentation hooks → commit 2 (env-gated); harness/repro/smoke files → commit 3 (test). |
| `e7025e03c` | instrument(#237 Stage 0): env-gated per-rank op-counter + repro | **B** debug instrumentation (+ **C** harness) | **SPLIT** → same as above; the env-gated counter is consolidated into commit 2, harness into commit 3. |
| `ba214a01e` | Revert "instrument(#232): faulthandler watchdog + per-step divergence log" | **(cancel pair)** / real fix | **KEPT** as its own commit. NOTE: this is NOT a no-op cancel pair here, because its counterpart `5aa6fbfba` is ON MAINLINE (not on the divergence). The faulthandler+steplog landed **default-ON** in `penfever/working`, so the revert is a genuine production-hygiene fix (see below). |

Mainline commits the task referenced as "nearby" but which are **already landed** (not part of divergence, no action):

| SHA | Subject | Bucket | Disposition |
|---|---|---|---|
| `690e4dce1` | fix(#232): size worker_response_mq chunk to cover R3 routed_experts at long ctx | **E** | Already on `penfever/working` as its own atomic commit. Independent & correct; nothing to do. |
| `32a11afc8` | fix(#232): publish R3 routed_experts off the per-step worker_response_mq deadline path | — | Already on mainline. |
| `1b5a72e9a` | fix(#232): drop rank-0-only R3 pre-forward stream-sync (FIX A) | — | Already on mainline. |
| `5aa6fbfba` | instrument(#232): faulthandler watchdog + per-step divergence log | **B** | Already on mainline, **default-ON** — reverted by commit 4 below. |
| `28faed993` | fix: re-run patch_rope_parameters after rope_scaling hf_override | **D** | Already on mainline — reverted by commit 5 below. |

### Cancel-pair note
`5aa6fbfba` (faulthandler) + `ba214a01e` (its revert) do **not** vanish as a clean cancel pair, because the
ADD is on mainline and only the REVERT is on the branch. Since the add is default-ON (regression in production),
the clean branch carries the revert as an explicit commit rather than dropping both.

---

## Clean atomic-commit sequence on `feuer/237-clean` (off `penfever/working`)

`git log --oneline penfever/working..feuer/237-clean` (oldest → newest):

| # | SHA | Commit | Bucket | Prod? | Per-commit validation |
|---|---|---|---|---|---|
| 1 | `34a6597b3` | `fix(#237): rank-symmetric R3 capture epilogue` | A | **PRODUCTION** | G-CAPTURE + G-PARITY (below). Code is byte-identical to FIX B minus the `_r3_op_*` hooks. |
| 2 | `eb6cb323f` | `instrument(#237): env-gated per-rank R3 op-issuance counter (debug-only)` | B | debug-only (inert; `VLLM_R3_OPCOUNT` default-OFF = byte-identical no-op) | Flag-off byte-identical (no-op-when-off); flag-on produces per-rank `oplog_rank{N}.json`. |
| 3 | `c53413cc3` | `test(#237): R3 rank-symmetry repro + op-log/parity gate harness` | C | test-only (no `vllm/` source) | The harness itself; runs the G-* gates. |
| 4 | `96775565c` | `revert(#232): drop default-on faulthandler watchdog + per-step divergence log` | B-hygiene | **PRODUCTION** | Worker init no longer arms the 300s faulthandler / per-step steplog; grep `VLLM_WEDGE232` absent. |
| 5 | `8be02331d` | `revert: re-run patch_rope_parameters after rope_scaling hf_override` | D | **PRODUCTION** | YaRN long-ctx model LOAD no longer crashes (`pow(): NoneType`); `applied_rope_override` absent from `config/model.py`. |

Production set to land = commits **1, 4, 5**. Commits **2, 3** are debug/test (land if desired for regression
localization, but never run in production: commit 2 is env-gated default-OFF; commit 3 is test-only).

Recommended landing order for review: **1 (the fix) → 5 (rope revert, unblocks YaRN load) → 4 (faulthandler revert,
prod hygiene) → 2 (debug instrument) → 3 (test harness)**. 1/4/5 are mutually independent and each independently
testable; 2 depends on 1 (hooks live in the fixed epilogue); 3 depends on 2 (reads the op-log).

### Equivalence check
`feuer/237-clean` is functionally equivalent to `feuer/r3-rank-symmetry-norope`: stripping comments, the only
residual diff is docstring wording in `finalize_pending_copy`. Executable code is byte-identical. The clean branch
just removes the commit+revert noise and splits production fix / debug instrument / test tooling into atomic units.

---

## Validation that proves the #237 fix (commit 1) without breaking generation parity

Harness: `tests/r3_rank_symmetry_repro.py` (small-MoE OLMoE-1B-7B, TP=2, `enforce_eager`, R3 ON, forced mixed
chunked-prefill+decode), launched via `tests/r3_237_smoke.sbatch`. Env knobs: `R3_MODEL`, `R3_TP` (=2),
`R3_OUT`, `R3_FLAG` (on|off), `R3_STEPS`, `R3_REPRO_MODE` (opcount|capture), `VLLM_R3_OPCOUNT=1`,
`VLLM_R3_OPCOUNT_DIR=<dir>`.

Three gates — **PASS** criteria:

1. **G-SYMMETRY (op-issuance, the deadlock gate)** — `R3_REPRO_MODE=opcount VLLM_R3_OPCOUNT=1`. Compare each TP
   worker's `oplog_rank{N}.json` collective-affecting subsequence (exclude the pure-host `scatter` token).
   - Pre-fix (mainline `5aa6fbfba`): RESULT=**MISMATCH**, rank0 ≠ rank1 (bug reproduced; was 73/73 mismatch, job 923741).
   - Post-fix (commit 1): RESULT=**IDENTICAL every step + no hang**. ← PASS.

2. **G-PARITY (no-op equivalence)** — run `R3_REPRO_MODE=capture` with `R3_FLAG=off` on stock (pre-fix) vs
   patched, dump `tokens.npz`. PASS = greedy token-ids **identical** (`np.array_equal`) → the fix is a no-op on the
   flag-off path (R3 disabled), proving no generation regression for non-R3 users.

3. **G-CAPTURE (R3-on decode parity)** — run `R3_REPRO_MODE=capture R3_FLAG=on` pre vs post fix; dump per-request
   routed_experts `.npz` + `tokens.npz`; compare with `tests/r3_compare_captures.py`. PASS = captured
   routed_experts arrays **`np.array_equal`** pre vs post AND the decode token trace identical → the rank-symmetric
   epilogue does not perturb what rank 0 captures (host snapshot stays on the main stream, anti-aliasing preserved).

GPU smoke (already run during the original work): R3-on long-ctx 16-node smoke on the rebaked SIF (job chain
923903…) — watch (a) no `pow(): NoneType` load crash, (b) no execute_model-timeout/EngineDead, (c) reach
global_step ≥ 1, (d) sane long-ctx token lengths.

---

## OPEN QUESTION + recommendation: the rope-revert knot (bucket D)

**What it is.** `07e9fbca3` reverts `28faed993` ("re-run `patch_rope_parameters` after `rope_scaling` hf_override").
The `-norope` branch is literally `feuer/r3-rank-symmetry` + this single revert (the two siblings differ ONLY in
`vllm/config/model.py`).

**Why it's there (resolved from `agent_logs/2026-06-18_237_…`, Step 1 + Context).** The rope fix is a **real
regression**, not masking. With a YaRN `rope_scaling` hf_override, re-running `patch_rope_parameters(config)` inside
`_apply_dict_overrides` produces a YaRN rope `base=None` → `TypeError: pow(): NoneType and Tensor` at MODEL LOAD
(job 923800, died at load — NOT the wedge, NOT FIX B). The known-good prior #232 chain (917349) ran clean WITHOUT
the rope fix. Mechanism: the v4 branch of `patch_rope_parameters` does `config.rope_parameters = rope_scaling`
(by-reference) then `standardize/validate`, which is not idempotent against an already-overridden config and
corrupts the YaRN params for the 40960→131072 long-ctx configs this work targets.

**Is the underlying rope fix needed for #237 / the long-ctx path?** No. #237's fix is purely the R3 capture
epilogue (commit 1). The rope re-patch is orthogonal and actively breaks the long-ctx YaRN LOAD path. The concern
the rope fix tried to address (a dict rope override applied after `get_config()` already patched, so a YaRN override
can be silently dropped) is real in principle, but the implementation is unsafe.

**Recommendation.** KEEP the revert in the production set (commit 5) — it unblocks YaRN long-ctx model load and is
required for any R3-on long-ctx run. Treat a *correct* re-application (deep-copy the config, guarded
standardize/validate, A/B'd against the YaRN 131k load) as a **separate, independently-tested future change**; it
must not ride along with — or block — the #237 fix.

---

## IMPORTANT scope caveat on what FIX B actually fixes (flag for reviewers)

The later sections of `agent_logs/2026-06-18_237_…` (the most recent #232 investigation) establish that the
standalone rank-asymmetry (FIX B) was **NOT** the production deadlock cause: three single-node repros (pure decode,
async mixed R3-off, async mixed R3-on at max pressure) all completed clean, exercising the count-607744 AllGather
thousands of times with no hang. The actual production #232 deadlock was pinned to a **watchdog-less custom-AR ↔
NCCL weight-sync interleave at global_step 0→1**, fixed by `disable_custom_all_reduce=True` (a SkyRL/config change
in OT-Agent `penfever/working` @ `305c2c9c`, NOT a vLLM source change).

→ Commit 1 (FIX B) is a **legitimate correctness improvement** — it eliminates a genuine TP/CP rank-asymmetry in the
R3 epilogue and is independently validated by the G-* gates above — but it should be reviewed/landed on its own
merit (op-stream symmetry under enforce_eager), NOT sold as "the #237/#232 production deadlock fix." The production
deadlock fix lived in SkyRL config.

---

## Could-not-cleanly-separate / flags

- **None blocking.** The split was clean: the FIX-B production logic and the `_r3_op_*` instrumentation were
  separable line-by-line (instrumentation is purely additive `self._r3_op_*` calls + helpers, env-gated). Verified
  the rebuilt instrumented file is byte-identical (code) to the original head capturer; the only intentional
  divergence is comment wording (dropped internal "FIX B"/"GI-1/3" plan jargon for upstream-clean comments).
- The `afe60f19a` ruff/style commit was folded (not preserved standalone) because its two cosmetic edits live
  inside lines that the clean fix/instrument commits rewrite anyway — the clean versions are already ruff-clean.
- IDE diagnostics tool (`mcp__ide__getDiagnostics`) was unavailable in this environment; syntax was verified via
  `ast.parse` on every touched file (a single-file parse, not the discouraged bash flake8/py_compile sweep).
