---
name: sft-cleanup-hf-only
description: >-
  Clean up a completed NON-AGENTIC / HF-only SFT model — HF upload WITHOUT Supabase DB registration. The
  counterpart to sft-job-cleanup (which uploads AND registers). Use when a completed SFT cell belongs to an
  HF-only series (config `enable_db_registration: false`, or a known HF-only series like the Delphi #6279
  54-grid): upload the weights to laion/ and STOP — do NOT run manual_db_push (DB-registering scaling-laws
  / sweep checkpoints pollutes the model registry; the artifacts are consumed by an eval grid, not the
  registry). Covers cell→hub_model_id resolution (incl. the Delphi launch_54_map.tsv), the Leonardo
  sbatch-tunnel upload, and the downstream eval hand-off. Use on discovering a completed Delphi/HF-only SFT
  cell during a sweep; for DB-registered models use sft-job-cleanup instead.
---

# sft-cleanup-hf-only

The **HF-upload-only** SFT cleanup: publish the model to HF and **stop — no DB registration**. This is the
deliberate path for non-agentic / scaling-laws SFT series (Delphi #6279 today) whose checkpoints are
consumed by an **eval grid**, not the Supabase model registry. For normal SFT runs that DO get registered,
use **`sft-job-cleanup`**.

> **Why no DB:** DB-registering N scaling-laws cells (e.g. the 54-cell Delphi grid) would pollute the model
> registry; the series is deliberately scoped HF-only (`enable_db_registration: false`). Documented
> per-series exception — see memory `project_delphi_sft_hf_only_no_db`. The cron sweep's generic
> "SFT → upload + register" routes HERE for HF-only cells.

## When to use
- A completed SFT cell whose config sets **`enable_db_registration: false`**, OR a known HF-only series (Delphi #6279 `delphi-*` cells).
- Discovered during a sweep, or "upload the completed Delphi cells (no DB)".
- If it's a normal registered SFT → use `sft-job-cleanup` (don't skip the DB step there).

## Procedure (per completed cell)
1. **Confirm it's HF-only + 100% complete.** Check the cell's config `enable_db_registration: false` (or that it's a Delphi `delphi-*` cell). Confirm `trainer_log.jsonl` `"percentage": 100.0` / final step == total — never upload a partial.
2. **Cancel the pending restart chain** for that cell (`squeue … | grep <job> | grep PENDING | awk … | xargs -r scancel`) so stale restarts don't fire mid-upload. Do NOT touch other cells' chains.
3. **Recognize the path:** 8B → root `model.safetensors` (direct upload); 32B/ZeRO-3 → consolidate first (rare for these grids). Drop intermediate `checkpoint-*` + `.cache`.
4. **Resolve `hub_model_id`:**
   - **Delphi:** name is `laion/delphi-<base>-<recipe>_lr1e5-sft`; the authoritative cell→hub_model_id↔jobid map is `/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-prepared-tok/launch_54_map.tsv`. Resolve from there (don't guess).
   - Otherwise read the rendered launch config's `hub_model_id`.
   - Qwen3.5: copy `preprocessor_config.json` from the base model into the checkpoint before upload.
4b. **Tokenizer sanity check (pre-upload, MANDATORY):** verify `tokenizer_config.json`'s `extra_special_tokens` is a **dict**, not a list (`python -c "import json;d=json.load(open('<ckpt>/tokenizer_config.json'));assert isinstance(d.get('extra_special_tokens',{}),dict)"`). If it's a list → set to `{}` and re-save before upload. (Root cause: transformers **5.x** SFT-save folds `additional_special_tokens` into `extra_special_tokens` as a list; the **4.57.6** RL/SkyRL loader `.keys()`-crashes on it in `get_tokenizer`. Eval env (evalchemy, 5.x) is unaffected, but coerce anyway for RL reuse. Bit swesmith cold-start 2026-06-14.)
5. **Upload to HF** (public default):
   - **Leonardo** (where the Delphi grid runs): the login node SIGKILLs long processes at ~100s → use the **sbatch compute-node + SSH-tunnel** upload (see `sft-launch-leonardo` §11 — the `start_proxy_tunnel.sh` SOCKS pattern; or, while the step-ca cert is expired, the detached login-node nohup fallback per `ops/leonardo/ops.md`). `hf upload`, NOT `upload-large-folder`.
   - **Jupiter:** login node has direct internet → `hf upload` in tmux.
   - `source secrets.env` for `HF_TOKEN` (env var only — never inline).
6. **STOP — do NOT run `manual_db_push.py`.** This is the defining difference from `sft-job-cleanup`. No DB row, no `--base-model` anchor (which would auto-create a base-model row).
7. **Clean** the cell's exp/checkpoint dir after the upload is verified (detached `rm` on GPFS; no `du`/`find`).
8. **Verify:** the `laion/<name>` repo exists with `model.safetensors` (>500MB) + tokenizer/config. Note the cell as done.

## Downstream chain — "move the chains" (3 legs, autonomous every sweep)
HF-only SFT cells exist to be **evaluated then recorded**. The standing autonomous chain (run it without asking, every sweep):
1. **SFT completes → HF upload** (this skill; no DB).
2. **upload completes → `eval-standard-launch`** on the series' eval grid for the newly-uploaded cell(s).
3. **eval completes → record scores in the experiment tracker** — for Delphi, append the cell's result to **`/Users/benjaminfeuer/Documents/experiments/active/delphi/rl-scaling-laws-6279/main_sft_evals/SCORES.md`** (the tracker relocated here from `notes/marin/...`; that older path is stale). Pull the score from the completed `delphi-eval/<RUN>/` output + the eval job's metrics.

Each leg also catches up backlog: any COMPLETED-SFT-not-uploaded, uploaded-not-evaled, or evaled-not-in-SCORES cell gets advanced. Idempotent — skip legs already done.

### Delphi eval traps (carry into the eval hand-off)
- **`HF_HUB_ENABLE_HF_TRANSFER` MUST be 0/unset** when pre-caching the `-sft` repos into `$HF_HUB_CACHE` for eval — `hf_transfer` isn't in the `evalchemy` env, so `=1` leaves an EMPTY `.incomplete` blob (silent no-op). Verify the snapshot has `model.safetensors` (>500MB) before submitting.
- The `-sft` repos ship the chat template as a separate **`chat_template.jinja`** file (NOT in `tokenizer_config.json`'s `chat_template` key) — the eval sbatch's `delphi_v0` override reads/replaces that file.
- **Per-cell TP must divide `num_attention_heads`:** 9e18 heads=9→TP=1, 2e19 heads=11→TP=1, 3e19 heads=12→TP=2 (the sbatch's hardcoded TP=2 is WRONG for 9e18/2e19 — pass the optional TP_OVERRIDE 4th positional arg).

> Full Delphi series data (the why, the map path, the score sheet) lives in memory
> `project_delphi_sft_hf_only_no_db`; Leonardo upload mechanics in `sft-launch-leonardo` §11; this skill is
> the reusable HF-only-cleanup procedure.
