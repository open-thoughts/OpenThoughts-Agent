# EVAL_GUIDE — cluster-diagnostic eval listener stack

Operational guide for firing `(model × dataset × scaffold)` evaluations
through `unified_eval_listener.py` and diagnosing failures with the
catalog in §7.

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Five firing categories](#2-five-firing-categories)
3. [Pinggy management](#3-pinggy-management)
4. [Pattern A/B/C/D — vLLM serving configs](#4-pattern-abcd--vllm-serving-configs)
5. [Yaml flip-restore workflow (Cat 4)](#5-yaml-flip-restore-workflow-cat-4)
6. [Monitoring + result extraction](#6-monitoring--result-extraction)
7. [Failure modes catalog](#7-failure-modes-catalog)
8. [Resume after walltime](#8-resume-after-walltime)
9. [Recovery scripts](#9-recovery-scripts)

---

## 1. Prerequisites

### Conda environments

The listener picks an env per fire via `--conda-env`. Configure each
mapping in your cluster config's `conda_envs:` block. Two envs are
load-bearing for evals:

| Env | When |
|---|---|
| `otagent-fix` | Default. transformers 4.x line. Used by terminus-2 reg eval, OOD presets, and swe-agent / openhands text-tools / mini-swe-agent / aider installed-agent fires. |
| `otagent2-fix` | transformers 5.x line. Used by axolotl-trained Qwen3 finetunes that crash to load on 4.x (`extra_special_tokens` list bug), Qwen3.5 family, and Pattern C scaffolds that need vLLM ≥ 0.17 for `--reasoning-parser-plugin` (Nemotron-Nano, Qwen3-Coder native). |

Recreate both via the slim yaml recipes in [`envs/`](envs/):

```bash
conda env create -f eval/envs/otagent-fix.yml      # primary
conda env create -f eval/envs/otagent2-fix.yml     # transformers 5.x line
```

Pinned versions (as of 2026-04-29):

| Package | otagent-fix | otagent2-fix |
|---|---|---|
| python | 3.12 | 3.12 |
| vllm | 0.13.0 | 0.17.1 |
| torch | 2.9.0+cu128 | 2.10.0+cu128 |
| transformers | 4.57.3 | 5.6.0.dev0 |
| flashinfer-python | 0.5.3 | 0.6.4 |
| huggingface-hub | 0.36.2 | 1.11.0 |
| daytona | 0.164.0 | 0.164.0 |
| harbor (editable) | from harbor-fix checkout | from harbor-fix checkout |

See [`envs/README.md`](envs/README.md) for the verification one-liner
and notes on when to retest after a version bump.

### harbor-fix — pin the patch stack

Harbor is installed *editable* from a checkout. The eval pipeline
depends on patches that aren't yet in upstream `main`; pin to:

```bash
git clone https://github.com/harbor-framework/harbor.git ~/harbor-fix
git -C ~/harbor-fix checkout penfever/temp-override
git -C ~/harbor-fix checkout 9980f967     # tip used in the last validated fire
pip install -e ~/harbor-fix               # inside each conda env
```

`9980f967` is the Daytona connection-pool fix (PR #1460); the rest of
`penfever/temp-override` carries:

- Loading-gate semaphore (cap = `n_concurrent * 2`)
- Treat "Bearer token invalid" as transient
- `assume_global_snapshot` flag for auto-snapshot pinning
- Multiplier-aware `max_timeout_sec` ordering
- Tmux dummy-session PTY + history-limit for swebench/tb2 task images
- swerex `dirs_exist_ok=True` (mode 12 in §7)
- ContextLengthExceeded → still run verifier
- Summarization-timeout default raised (mode 6 in §7)
- LiteLLM timeouts treated as env failures

If you fire Cat 3 (preferred-harness reproductions), also apply the
three uncommitted installed-agent patches documented in
[`envs/README.md`](envs/README.md) (aider provider routing, mini-swe-agent
api-key list, swe-agent venv PATH + set-u guard). Cat 1/2 fires don't
need them.

### Secrets

Source a dotenv before any fire:

```bash
source ~/.local/eval.env       # or wherever your cluster's secrets live
```

Required: `HF_TOKEN`, `SUPABASE_URL`, `SUPABASE_ANON_KEY` (and
`SUPABASE_KEY` aliased to it), `DAYTONA_API_KEY`. See
`hpc/dotenv/example.env` for the full list.

> **Two Daytona keys, two reliability tiers.** If you have a primary +
> secondary key, the secondary often shows higher auth-error rates
> under load. Route critical evals through the primary.

### Pinggy

Only Cat 3 (installed-agent reproductions) needs Pinggy — the agent
CLI runs inside a Daytona sandbox and can't reach the SLURM node's
localhost. Buy N pairs (one pair = one concurrent Cat 3 job) and load
them via `source ~/pinggy_pairs.env`. See §3 for management.

---

## 2. Five firing categories

All fires use the same launcher (`unified_eval_listener.py`); the
category determines preset, harbor config, scaffold flags, and
whether Pinggy is needed.

### Cat 1 — Reg eval (terminus-2 on v2 / swebench / tb2)

Default surface. Picks per-model serving config from
`baseline_model_configs_minimal.yaml`, runs against
`dcagent_eval_config.yaml`.

```bash
echo "Qwen/Qwen3-32B" > /tmp/list.txt
for preset in v2 swebench tb2; do
  python eval/unified_eval_listener.py \
    --cluster-config ~/.local/eval-cluster.yaml \
    --preset $preset \
    --priority-file /tmp/list.txt \
    --baseline-model-config eval/configs/baseline_model_configs_minimal.yaml \
    --conda-env otagent-fix \
    --enable-thinking --tp-size 2 --dp-size 2 --timeout-multiplier 2.0 \
    --slurm-partition <PARTITION> --slurm-time 24:00:00 \
    --max-jobs-submitted 32 --n-concurrent 32 \
    --no-disk-resume --auto-snapshot \
    --stagger-delay 2 --chain-batch-size 2 --once
done
```

### Cat 2 — OOD presets (aider / bfcl / medagentbench / gaia / financeagent / swebench_full)

OOD = "out-of-distribution" relative to terminus-2 training. Each
preset pins a different dataset + harbor config but runs the same
terminus-2 agent.

```bash
python eval/unified_eval_listener.py \
  --cluster-config ~/.local/eval-cluster.yaml \
  --preset bfcl \
  --priority-file /tmp/list.txt \
  --baseline-model-config eval/configs/baseline_model_configs_minimal.yaml \
  --conda-env otagent-fix \
  --enable-thinking --tp-size 2 --dp-size 2 --timeout-multiplier 2.0 \
  --slurm-partition <PARTITION> --slurm-time 24:00:00 \
  --max-jobs-submitted 32 --n-concurrent 32 \
  --no-disk-resume --auto-snapshot \
  --stagger-delay 2 --chain-batch-size 2 --once
```

Per-preset nuances:
- **financeagent**: SEC EDGAR caps at 10 req/s. Preset bumps
  `n_concurrent` to 16 (empirical p95 ~6.8 req/s aggregate, safe
  margin); don't override below.
- **swebench_full**: 500 trials, ~36–48h walltime. Bump
  `--slurm-time 48:00:00` and pre-download the dataset on a no-internet
  cluster.
- **aider / bfcl / medagentbench / gaia**: each pins
  `dcagent_eval_config_no_override.yaml` (force_build=true,
  n_concurrent_trials=4) to side-step the swe-agent `/workdir` bug
  on dev_set_v2/tb2-shaped datasets.

### Cat 3 — Preferred-harness reproduction

Paper-author scaffold + installed-agent CLI in a Daytona sandbox,
talking back to the served model via Pinggy. See
[`docs/PREFERRED_HARNESS_REPRODUCTION.md`](docs/PREFERRED_HARNESS_REPRODUCTION.md)
for the per-model recipes.

```bash
source ~/pinggy_pairs.env
python eval/unified_eval_listener.py \
  --cluster-config ~/.local/eval-cluster.yaml \
  --preset swebench \
  --priority-file /tmp/list.txt \
  --baseline-model-config eval/configs/baseline_model_configs_minimal.yaml \
  --conda-env otagent-fix \
  --config-yaml dcagent_eval_config_swe_agent.yaml \
  --agent-parser "" \
  --enable-thinking --tp-size 2 --dp-size 2 --timeout-multiplier 2.0 \
  --slurm-partition <PARTITION> --slurm-time 24:00:00 \
  --max-jobs-submitted 1 --n-concurrent 32 \
  --no-disk-resume --no-auto-snapshot \
  --pinggy-url "$PINGGY_URL_1" --pinggy-token "$PINGGY_TOKEN_1" \
  --stagger-delay 2 --chain-batch-size 2 --once --force-reeval
```

Swap `--config-yaml` per scaffold:

| Scaffold | `--config-yaml` |
|---|---|
| swe-agent regular / SERA-e2e | `dcagent_eval_config_swe_agent.yaml` |
| openhands native (text tools) | `dcagent_eval_config_openhands.yaml` |
| openhands Qwen3-Coder native | `dcagent_eval_config_openhands_qwen3_coder.yaml` |
| openhands Nemotron-Nano native | `dcagent_eval_config_openhands_toolcall.yaml` |
| mini-swe-agent | `dcagent_eval_config_mini_swe_agent.yaml` |
| aider (no thinking) | `dcagent_eval_config_aider_agent_nothink.yaml` |

For SERA-e2e, also export
`SWEAGENT_CONFIG=https://huggingface.co/datasets/DCAgent2/swe-agent-configs/resolve/main/sera_e2e.yaml`
before fire (the listener forwards it).

### Cat 4 — Yaml flip-restore

A model whose baseline-yaml entry uses Pattern B/C (parsers tuned for
its native installed-agent harness) gives **0% on terminus-2** because
the parser hijacks JSON the agent expects to consume directly. To
fire terminus-2 on such a model:

1. Edit its yaml entry — strip `tool_call_parser` and
   `reasoning_parser`. Keep tp/dp, swap_space, trust_remote_code,
   chat_template, extra_args. Add a header comment `# TEMP FLIP <date>`.
2. Fire the Cat 1 command.
3. Wait for batch completion (the listener bakes env vars at submit
   time, so restore is safe once all jobs are queued).
4. Restore the yaml. The repo's last-write rule is **the file ends in
   the most-common state** so future sessions see the right config.

See §5 for a worked example.

### Cat 5 — Per-model serving config

Lives entirely in `configs/baseline_model_configs_minimal.yaml`. No
fire change. Add an entry, choose Pattern A/B/C/D from §4, sanity-check
with a one-model dry-run, then fold into a Cat 1/2/3 batch.

---

## 3. Pinggy management

### What Pinggy is

A persistent reverse-tunnel service. Each pair (`url`, `token`) maps
to a stable `https://<...>.a.pinggy.link/v1` endpoint. The sbatch
opens an SSH reverse tunnel from the SLURM node so the served vLLM is
reachable from inside Daytona sandboxes (which can't reach SLURM
node localhost).

### Allocation

One pair per concurrent Cat 3 job. SSH reverse binding is exclusive —
the second tunnel to the same pair is rejected. Buy as many pairs as
you want concurrent installed-agent fires. Store as:

```bash
# ~/pinggy_pairs.env  (chmod 600)
export PINGGY_URL_1="abcd1234.a.pinggy.link"
export PINGGY_TOKEN_1="..."
export PINGGY_URL_2="..."
# ...etc
```

### Occupancy check (which pairs are in use right now)

```bash
for jid in $(squeue -u $USER --format='%.10i' --noheader); do
  url=$(grep -oE "[a-z0-9]+\.a\.pinggy\.link" eval/logs/data_${jid}.out 2>/dev/null | head -1)
  [ -n "$url" ] && echo "$jid -> $url"
done
```

### Cross-cluster collision check (before fire)

If pairs are shared with another cluster (e.g. you also use Jupiter
with the same account), the SSH binding races. The pair appears
healthy from your side but routes to the *other cluster's* model.
Detect by querying the served-model id and comparing root path:

```bash
URL=<pair-url>
curl -s --max-time 8 https://$URL/v1/models \
  | python3 -c "import sys, json; d=json.load(sys.stdin); m=d.get('data',[{}])[0]; print(f\"served={m.get('id')} root={m.get('root','')[:80]}\")"
```

If `root=` doesn't point inside *your* `paths.hf_cache`, the pair is
contaminated — cancel the job, fire on a different pair.

### Rules of thumb

- Cat 1/2 fires never need Pinggy (terminus-2 talks to vLLM directly
  on the SLURM node).
- Cat 3 fires always need Pinggy *unless* the scaffold ships a
  Daytona-internal endpoint (rare).
- A failed Cat 3 fire that left a hung tunnel: cancel the SLURM job;
  the SSH tunnel dies with it.

---

## 4. Pattern A/B/C/D — vLLM serving configs

The shape of `extra_args` + parsers in
`baseline_model_configs_minimal.yaml` falls into four patterns. Pick
based on (a) what scaffold the model was trained for, (b) whether the
model emits `<think>`, (c) tool-call format.

| Pattern | Scaffold | tool_call_parser | reasoning_parser | extra_args highlights |
|---|---|---|---|---|
| **A** | swe-agent SERA-e2e | `hermes` | _(none)_ | `--enforce-eager --disable-cascade-attn --seed 42`; no prefix-caching |
| **B** | swe-agent regular / openhands native | `hermes` | `qwen3` (when model emits `<think>`) | `--enable-prefix-caching` |
| **C** | openhands Qwen3-Coder / Nemotron-Nano native | `qwen3_coder` | `nano_v3` (Nemotron) | `--reasoning-parser-plugin ${DCFT}/eval/configs/nano_v3_reasoning_parser.py --chat-template ${DCFT}/eval/configs/nemotron_chat_template.jinja --override-generation-config '{"temperature":0.6,"top_p":0.95}' --enable-prefix-caching` |
| **D** | terminus-2 / openhands text-tools | _(none)_ | _(none)_ | `--enable-prefix-caching` (optional) |

### Why no parser for Pattern D

Terminus-2 and openhands text-tools both do their own JSON extraction
client-side. A vLLM-side `tool_call_parser` mangles the response —
silent 0%.

### Why `reasoning_parser: qwen3` only when needed

Adding it strips `<think>` blocks cross-turn. Models trained to keep
those (SA-SWE, some axolotl variants) regress hard. Default off; add
only when the scaffold needs the JSON-after-think shape.

### Pattern A footgun: `--enforce-eager` × 32B + retries

Eager mode adds ~5–14 min to vLLM startup. Default
`MAX_RETRIES=10` × 100 s = ~17 min — usually enough, but on a slow
node the harness can still give up before `Application startup
complete`. For Pattern A 32B fires, pass `--vllm-max-retries 30`
(~50 min headroom) on the listener.

---

## 5. Yaml flip-restore workflow (Cat 4)

Worked example: fire terminus-2 on `Qwen/Qwen3-32B`, which is
configured Pattern B in baseline yaml.

**Before** (Pattern B — installed-agent):

```yaml
"Qwen/Qwen3-32B":
  tensor_parallel_size: 2
  swap_space: 32
  trust_remote_code: true
  tool_call_parser: hermes
  reasoning_parser: qwen3
  extra_args: "--enable-prefix-caching"
```

**During fire** (Pattern D — terminus-2):

```yaml
"Qwen/Qwen3-32B":
  tensor_parallel_size: 2
  swap_space: 32
  trust_remote_code: true
  # TEMP FLIP <YYYY-MM-DD> — parsers stripped for terminus-2
  extra_args: "--enable-prefix-caching"
```

Fire the Cat 1 command, wait for `--max-jobs-submitted N` jobs to
queue, then restore.

**Restore rule**: the file's last-write state is what every future
fire — and every other contributor — sees. Restore *immediately* after
the batch is queued, not after the batch completes. The listener's
env-baking happens at submit time; the on-disk yaml doesn't matter
once jobs are queued.

---

## 6. Monitoring + result extraction

### Live progress

```bash
python eval/check_progress.py            # text summary
python eval/check_progress.py --live     # rich dashboard
```

### Per-trial result extraction

```python
import json, glob, os
from collections import Counter

run_dir = "jobs/<run_tag>"
results = glob.glob(os.path.join(run_dir, "*__*/result.json"))
total = len(results)
rewards, exceptions = [], Counter()
for r in results:
    d = json.load(open(r))
    rw = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if rw is not None:
        rewards.append(rw)
    exc = (d.get("exception_info") or {}).get("exception_type")
    if exc:
        exceptions[exc] += 1
errors = sum(exceptions.values())
scored = total - errors
correct = sum(1 for r in rewards if r == 1.0)
acc = correct / scored * 100 if scored else 0
print(f"trials={total} errors={errors} correct={correct} acc={acc:.1f}%")
for e, c in exceptions.most_common():
    print(f"  {e}: {c}")
```

### Trust threshold

**Make recipe decisions only when n ≥ 150 trials are scored.** Sub-150
reads can drift ±5–10 percentage points as more trials land — several
preferred-harness reads moved 5–8 pp in the pessimistic direction
between n=50 and n=150 in earlier work. If the dataset has fewer than
150 tasks total, run multiple seeds and aggregate.

### Health-check thresholds (running jobs)

| Symptom | Likely cause |
|---|---|
| No new `result.json` for 60+ min, job RUNNING | Harbor stall (all trials timing out simultaneously, or hung) |
| vLLM `Running: 0` for 10+ min | Agents not generating: env-build stall, auth degradation, drain |
| All trials done but job RUNNING | Zombie — Harbor process didn't exit. Cancel + manual upload |
| Repeated `Bearer token is invalid` in slurm log | Daytona auth degradation under load (see #5 in §7) |

---

## 7. Failure modes catalog

The cluster-diagnostic spine. Each entry: short symptom, root cause,
mitigation.

1. **vLLM health-check timeout on 32B cold start** — sbatch retried
   10× × 60s, model needs 12–15 min. *Refire.* If chronic, raise
   `--vllm-max-retries 20`.
2. **NCCL ALLREDUCE timeout during flashinfer autotune** — usually a
   bad node. *Clear* `~/.cache/vllm/torch_compile_cache`, refire on a
   different node.
3. **Torch-compile cache corrupted** — `Bytes object is corrupted,
   checksum does not match`. *Delete* the cache, refire.
4. **24h SLURM wall before upload** — long jobs with many
   `AgentTimeoutError` trials burn walltime before Harbor's upload
   step. *Use* `upload_overlong_jobs.py` (see §9).
5. **Bearer token invalid / Daytona load-shedding** — sustained sandbox
   creation > ~10/s triggers fake 401s. *Drop* concurrent fires;
   stagger with `--stagger-delay 2 --chain-batch-size 2`. The
   harbor-fix branch widens the connection pool.
6. **`SummarizationTimeoutError` dominates** — default 300s too short
   for 32B on long context. *Use* a harbor config with
   `summarization_timeout: 1800`.
7. **Daytona "Dockerfile cannot be empty"** — transient build flake at
   high concurrency. *Harbor* retries; usually clears. If >50% of
   trials hit it, cancel and back off.
8. **Cross-cluster Pinggy contamination** — pair shared with another
   cluster, SSH races. *Run* the §3 collision check before each Cat 3
   fire.
9. **vLLM port collision via DP workers** — DP=2 binds 3 contiguous
   ports; SLURM JIDs differing by ≤2 on the same node collide. *Cancel
   one, refire* (gets a new JID → new port range).
10. **Pattern A `--enforce-eager` × 32B startup race** — see §4.
    *Pass* `--vllm-max-retries 30`.
11. **HF Hub 429 (org-level rate limit)** — pre-download HEAD calls
    across many shards × many jobs blow the 12 k / 5 min quota.
    *Stagger* fires (≥60s apart); set `HF_HUB_OFFLINE=1` for
    fully-cached models.
12. **swe-agent `/workdir already exists`** — task containers on
    dev_set_v2 / tb2 pre-create `/workdir`; swe-agent's setup uses
    `copytree(exist_ok=False)`. *Use* openhands instead, or pin the
    swerex patch (`dirs_exist_ok=True`).
13. **Tool-call parser hijacks terminus-2 JSON** — `hermes` /
    `qwen3_coder` parser on a terminus-2 fire = silent 0%. *Pattern D
    has no parser*; flip baseline yaml (Cat 4) before fire.
14. **`reasoning_parser: qwen3` strips `<think>` cross-turn** — breaks
    models trained to keep them. *Add only when the scaffold needs the
    post-think JSON shape.*
15. **Engine-direct scaffolds aren't portable** — SkyRL's
    `OHCodeActAgent` bypasses the OpenAI API, custom chat templates
    aren't reproducible via Harbor. *Documented as structurally
    unreproducible (SA-SWE, OpenSWE).*
16. **Custom chat templates accumulate state at 32k** — a template
    that re-emits earlier `<think>` fills the context window;
    condenser fires, drops observations, agent gets stuck. *Don't add
    unless context budget can grow to 64k+.*
17. **v4-axolotl tokenizer crash** — the older transformers 4.x
    crashes on `extra_special_tokens: list`. *Use* `otagent2-fix`.
18. **HF 429 on fresh-upload batch** — cluster of jobs fails on the
    same safetensors shard before the model is fully on disk.
    *Refire* survivors individually.
19. **Pinggy pair user-collision** — another user holds the pair.
    *Refire* on a different pair.
20. **`--config-yaml <path>` path-doubling** — sbatch auto-prepends
    `${DCFT}/eval/configs/`. Passing an absolute path → `Harbor config
    not found`, crashes after vLLM warmup. *Pass bare filename.*
21. **Numeric model-id in DB after upload** — Harbor reads the served
    model name from `agent_info.model_info.name`, which for vLLM is a
    numeric serving id. *Pass* `--model-name laion/<real-name>` to
    `manual_db_eval_push.py`, or fix the model_id post-upload.
22. **Small-n accuracy reads overstate** — see §6 trust threshold.
    Decisions with n<150 are unreliable.
23. **Numeric subagent-cited paper IDs** — verify chat-template format
    against `tokenizer_config.json` directly, not against subagent
    summaries.
24. **Qwen3-Coder-tuned models with broken tool-call format** —
    axolotl SFT'd against hermes, deployed against openhands
    CodeActAgent → format mismatch. *Sample* a `trajectory.json` for
    `<parameter=...>` without surrounding `<function=...>` markers
    before declaring a recipe.
25. **Harness × model fit dominates accuracy** — same model on
    swe-agent regular vs openhands text-tools can swing 30 pp.
    *Confirm* the harness matches the SFT training before declaring
    "model is broken".
26. **Secondary Daytona key has worse capacity** — when two keys are
    available, the secondary often shows ~40% auth-error rate under
    load. *Route critical evals through the primary.*
27. **vLLM `auto-snapshot` cached image** — `--auto-snapshot` reuses a
    cached `action_execution_server` build that's broken for some
    openhands fires (silent 0%). *Drop* the flag for fresh builds, or
    rely on the preset's authoritative `auto_snapshot` field. Don't
    pass it on the CLI — let the preset decide.
28. **bfcl false-skip on listener restart** — silent skip leaves a
    Pending DB row with no result. *Refire* with `--force-reeval` only
    when silent-skipped (unconditional refire creates duplicates).
29. **OpenSWE-32B paper sampling crashes terminus-2** — `rep_penalty
    1.2` etc. in `override-generation-config` causes 75–77% STOE
    (stuck-on-exception) on OOD. *Drop* the entire override block for
    OOD fires.
30. **Stuck tail trial** — last 1–2 trials hang for hours past job-end
    threshold. *Cancel* + push partial results via
    `manual_db_eval_push.py`.

---

## 8. Resume after walltime

When a SLURM job hits its wallclock cap before harbor finishes the dataset,
the run dir is left at partial coverage with `finished_at=None`. Most
clusters can recover the missing trials without re-running the whole job
through the **resume tooling**:

- `eval/check_resume_needed.py` — inspector. Reads disk + squeue, classifies
  every run dir, prints a status table.
- `eval/resume_chunked.py` — orchestrator. Fires N-at-a-time resume listener
  invocations from a priority file.

Both reuse internals from `eval/unified_eval_listener.py` so classification
and infra-error definitions never drift.

> **Read the bug catalogue first.** Out-of-the-box `harbor jobs resume`
> re-runs the entire dataset due to a per-trial config mismatch (G12), and
> can leave the DB row stuck at Pending (G13). Both fixes are committed in
> this repo. The full cross-cluster catalogue + verification checklist
> lives at [`eval/docs/RESUME_HANDOFF.md`](docs/RESUME_HANDOFF.md). Apply
> the harbor-fix patches from `eval/envs/README.md` before relying on
> resume — without them you'll waste compute or hit 9-10h tail retry loops.

### Inspect first

```bash
python eval/check_resume_needed.py \
  --jobs-dir $EVAL_JOBS_DIR \
  --needs-resume-only
```

The output table classifies every run dir:

| Status | Meaning | Resume? |
|---|---|---|
| `INCOMPLETE` | `n_completed < n_total`, `finished_at=None` | Yes — standard walltime case |
| `DONE_WITH_ERRORS` | All trials done, infra errors > threshold | Yes |
| `PARTIAL` | `n_completed < n_total`, `finished_at` set, infra > threshold | Yes |
| `EARLY_KILL` | No `result.json`, `n_total=0` | Yes |
| `DONE` | All trials done, infra ≤ threshold | No (by default — see override) |
| `IN_FLIGHT` | Currently in squeue, or active map | No (wait) |
| `AT_RESUME_LIMIT` | `n_fires ≥ --max-total-fires` | No — hand to upload (§9) |
| `REJECTED` | `.no-resume` marker present | No (hidden by default) |

Stuck-at-full runs — full coverage on disk but no `finished_at`, classify as
`DONE` and get hidden by default. To resume them anyway, pass
`--resume-error-threshold -1` to the orchestrator. This promotes
`infra_errors=0` dirs from `DONE` to `DONE_WITH_ERRORS` so they become
eligible.

### Fire the resume

```bash
# Build a one-per-line priority file:
cat > /tmp/resume.txt <<EOF
<org>/<model_A>
<org>/<model_B>
EOF

python eval/resume_chunked.py \
  --priority-file /tmp/resume.txt \
  --preset tb2 --org eval \
  --jobs-dir $EVAL_JOBS_DIR \
  --tag-prefix r_$(date -u +%H%M%S) \
  --tp-size 2 --dp-size 2 --timeout-multiplier 16.0 \
  --conda-env otagent-fix \
  --chunk-size 6 --sleep-between 0
# Add --resume-error-threshold -1 for stuck-at-full DONE dirs.
```

One listener invocation = one sizing config (one preset, one TP/DP combo,
one conda env). Mixed sizes / presets → fire multiple invocations in
parallel, one per (preset × size) combo.

### Pre-walltime cancel

Daytona sandboxes run remotely. When SLURM SIGTERM hits at walltime, harbor
on the compute node dies but sandboxes keep writing back to the run dir on
shared FS for 30-60 minutes. Observed: 267-task dir went 267 → 352 trial
dirs after walltime; `n_completed` overshot `n_total`; the inspector
silently skips dirs with `n_completed > n_total`.

**Mitigation**: cancel before walltime hits.

```bash
# Cancel jobs at ≤15 min TIME_LEFT (~11h45m+ elapsed on a 12h cap):
squeue -u $USER --format='%.10i %.10M %.10L' --noheader \
  | awk '$3 ~ /^0:[0-3][0-9]:/ { print $1 }' | xargs -r scancel
```

Then wait ~60s and run the inspector. After a *walltime-killed* dir, wait
~60min instead for zombies to settle.

### Verify each resume

For every new JID, confirm:

```bash
# RUNNING within 30s
squeue -j <new_jid> --format='%.10i %.10T'

# G12 check — per-trial sed patch ran
grep "Patching api_base port in [0-9]+ per-trial" eval/logs/<log>_<new_jid>.out

# G12 check — zero "skipping" warnings
grep -c "not found in generated configs; skipping" eval/logs/<log>_<new_jid>.out
# expect: 0

# Harbor scheduled only the in-flight trials, not the whole dataset
grep -c "^Starting trial" eval/logs/<log>_<new_jid>.out
# expect: matches in_flight count from inspector pre-resume

# G13 check — DB row flipped to Started
python -c "from database.unified_db.utils import get_sandbox_job_by_name; \
  print(get_sandbox_job_by_name('<run_tag>')['job_status'])"
# expect: "Started"
```

If any check fails, see [`eval/docs/RESUME_HANDOFF.md`](docs/RESUME_HANDOFF.md)
§Bug G12 / §Bug G13 for the root cause and the corresponding source patches.

### Fire-cap

Default `--max-total-fires=2` (one original + one resume). Beyond that the
inspector marks the dir `AT_RESUME_LIMIT` — upload it as-is via §9 rather
than firing again. Daytona auth tokens have a 24h validity window in most
org configs, so a third fire usually starts failing in the auth-degradation
window anyway.

---

## 9. Recovery scripts

When resume isn't applicable (auto-upload failed, hit fire cap, or the run
just needs to be pushed as-is), use one of the upload helpers.

### When to use which

```
                    +----------------------------------+
                    | Has the run dir been resumed     |
                    | to satisfactory coverage         |
                    | (or is it AT_RESUME_LIMIT)?      |
                    +----------------+-----------------+
                                     |
                          yes        |       no
                  +------------------+----------------+
                  |                                   |
        Did SLURM hit TIMEOUT                  Run §8 resume first
        (24h walltime) before upload?
                  |
       yes        |       no
       +----------+---------+
       |                    |
upload_overlong       Did Harbor's auto-upload
_jobs.py              fail (numeric model_id,
                      missing result.json)?
                             |
                  yes        |       no
                  +----------+---------+
                  |                    |
        manual_db_eval_push.py    no action — verify
        (single job)              in Supabase
        batch_upload_eval.py
        (multiple jobs)
```

### `manual_db_eval_push.py` — single-job recovery

```bash
source ~/.local/eval.env
python scripts/database/manual_db_eval_push.py \
  --job-dir jobs/<run_tag> \
  --verbose
```

**Common flags**:
- `--forced-update` — overwrite existing DB rows (default: skip if row
  exists).
- `--skip-hf` — DB-only (skip the HuggingFace traces upload).
- `--hf-repo DCAgent3/<run_tag>-traces` — explicit override.

> **HF trace upload destination is `DCAgent3`.** `DCAgent` and `DCAgent2` are
> the previous orgs — both have hit their public-storage cap (HF returns 403
> on new uploads), so new traces go to `DCAgent3`. The two older orgs still
> host source benchmark datasets (e.g. `DCAgent2/terminal_bench_2`,
> `DCAgent/dev_set_v2`) — those are read-only and unchanged. If you see a
> sbatch fail at upload with `403 Forbidden` on a `DCAgent2/` path, the run
> was launched from a pre-migration sbatch checkout; re-upload via
> `batch_upload_eval.py` to land on `DCAgent3`.

**Watch out**: `agent_info.model_info.name` in `result.json` is the
vLLM-served numeric id (e.g. `1774950145766573`), not the HF model
name. The script may register a bogus model row. After upload, verify
the model name in Supabase and fix it if needed:

```bash
python -c "
import json
d = json.load(open('experiments/<run_tag>/configs/<run_tag>_eval_config.json'))
print(d['model_hf_name'])
"
```

### `upload_overlong_jobs.py` — batch TIMEOUT recovery

```bash
source ~/.local/eval.env
python scripts/database/upload_overlong_jobs.py --upload --force --parallel 8
```

Walks `$EVAL_JOBS_DIR`, picks runs whose latest `trial.log` mtime is
older than the threshold (= TIMEOUT-killed), pushes their partial
results to Supabase + HuggingFace as `status=overlong`.

### `batch_upload_eval.py` — multi-job upload

Cluster-agnostic batch helper for pushing many run dirs to Supabase + HF
in one invocation. Reads `HF_TOKEN` / `SUPABASE_*` from the environment;
no hardcoded paths — `--jobs-dir` defaults to `./jobs` and accepts any
location. Useful when an in-flight batch of evals all complete and you
want a single upload pass instead of N `manual_db_eval_push` calls.

```bash
source ~/.local/eval.env

# Upload explicit run dirs
python scripts/database/batch_upload_eval.py \
  $EVAL_JOBS_DIR/<run_tag_A> \
  $EVAL_JOBS_DIR/<run_tag_B>

# Auto-detect overlong (TIMEOUT-killed) runs under a jobs dir
python scripts/database/batch_upload_eval.py \
  --auto-detect-overlong --jobs-dir $EVAL_JOBS_DIR

# Parallel upload with N workers
python scripts/database/batch_upload_eval.py -p 8 $EVAL_JOBS_DIR/swebench_*

# Dry-run / DB-only / forced overwrite
python scripts/database/batch_upload_eval.py --dry-run $EVAL_JOBS_DIR/<run_tag>
python scripts/database/batch_upload_eval.py --skip-hf $EVAL_JOBS_DIR/<run_tag>
python scripts/database/batch_upload_eval.py --force $EVAL_JOBS_DIR/<run_tag>
```

Same `model_id` numeric-id pitfall applies as for `manual_db_eval_push.py`
— verify the model name in Supabase post-upload if the run used a vLLM
endpoint.

---

## See also

- `docs/PREFERRED_HARNESS_REPRODUCTION.md` — per-model recipe lookup
  with reproduction numbers.
- `docs/RESUME_HANDOFF.md` — cross-cluster bug catalogue (G10/G12/G13 +
  harbor #1617 + Bug #2) referenced from §8. Read before relying on
  resume on a new cluster.
- `clusters/example.yaml`, `hpc/dotenv/example.env` — populate these
  for your cluster before any fire.
- `python eval/unified_eval_listener.py --help` — full CLI surface.
