# Preferred-harness reproduction (PHR)

Per-model recipes for reproducing paper accuracy via the model
author's intended scaffold (swe-agent / openhands / mini-swe-agent /
aider) running the **installed-agent CLI inside a Daytona sandbox**,
talking back to the served model via Pinggy. This is Cat 3 of the
[`EVAL_GUIDE`](../EVAL_GUIDE.md).

Each entry below pins:

- **scaffold** + which `--config-yaml` to pass
- **conda env** + `tp-size` / `dp-size` / `timeout-multiplier`
- **baseline-yaml shape** (Pattern A/B/C/D — see EVAL_GUIDE §4)
- **observed accuracy** (n ≥ 150 unless noted) and the paper number
- known **scaffold-bound ceiling** (where reproduction isn't full)

> **Trust threshold**: numbers below are stable only at n ≥ 150.
> Earlier runs at n=50 drifted 5–10 pp pessimistic as more trials
> landed.

---

## §1 — Quick-fire reference

| Model | Scaffold | Env | tp/dp/tm | Pattern | Acc | Paper | Δ |
|---|---|---|---|---|---|---|---|
| `allenai/SERA-32B` | swe-agent SERA-e2e | otagent-fix | 2/2/16 | A | 48.2% | 49.5% | -1.3 |
| `allenai/SERA-8B` | swe-agent SERA-e2e | otagent-fix | 1/2/2 | A | 31.8% | 31.7% | +0.1 |
| `SWE-bench/SWE-agent-LM-32B` | swe-agent regular | otagent-fix | 2/2/16 | B | 42.4% | ~40% | +2.4 |
| `SWE-bench/SWE-agent-LM-7B` | swe-agent regular | otagent-fix | 1/2/2 | B | — | — | recipe known |
| `GAIR/daVinci-Dev-32B` | swe-agent regular | otagent-fix | 2/2/16 | B (no `reasoning_parser`) | 55.5% | — | matched |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | openhands Qwen3-Coder native | otagent2-fix | 2/2/16 | C | recipe set | — | not re-measured |
| `Skywork/Skywork-SWE-32B` | openhands text-tools | otagent-fix | 2/2/16 | D | 29.6% | 38.0% | -8.4 (drift) |
| `R2E-Gym/R2EGym-32B` | openhands text-tools | otagent-fix | 2/2/16 | D | 26.4% | 34.4% | -8.0 (drift) |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | openhands Nemotron native | otagent2-fix | 2/2/16 | C | 33.7% | 38.8% | -5.1 (ctx + drift) |
| `SWE-Lego/SWE-Lego-Qwen3-8B` | openhands text-tools | otagent-fix | 1/2/2 | D | 17.6% | 42.2% | -24.6 (32k ctx) |
| `SWE-Lego/SWE-Lego-Qwen3-32B` | openhands text-tools | otagent-fix | 2/2/16 | D | not fired | ~50s expected | — |
| `NovaSky-AI/SA-SWE-32B` | swe-agent regular | otagent-fix | 2/2/16 | best-of-bad | 24.7% | 39.4% | engine-direct, not portable |
| `GAIR/OpenSWE-32B` | (no config worked) | otagent-fix | 2/2/16 | — | 0% × 4 | 62.4% | structurally broken |

### Δ legend

- **drift**: scaffold ↔ training mismatch outside our control; the
  paper recipe isn't fully reproducible via Harbor + installed CLI.
- **32k ctx**: paper trained at 64k+; eval pinned at 32k can't
  surface the model's behavior on long-context tasks.
- **engine-direct, not portable**: paper used SkyRL's engine-direct
  agent (bypasses the OpenAI API); custom chat templates aren't
  reproducible via Harbor.
- **structurally broken**: no scaffold config we tried produces
  non-zero accuracy; flagged for the model owner.

---

## §2 — Per-model serving config

The yaml fragments below go into
[`../configs/baseline_model_configs_minimal.yaml`](../configs/baseline_model_configs_minimal.yaml).
Each is the *flipped state* — what to use for the listed scaffold.
For a Cat 4 flip back to terminus-2, drop the parser lines (see
EVAL_GUIDE §5).

### `allenai/SERA-32B` — Pattern A (SERA-e2e)

```yaml
"allenai/SERA-32B":
  conda_env: otagent-fix
  tensor_parallel_size: 2
  max_model_len: 32768
  swap_space: 32
  trust_remote_code: true
  tool_call_parser: hermes
  extra_args: "--enforce-eager --disable-cascade-attn --seed 42"
```

Fire with `SWEAGENT_CONFIG=https://huggingface.co/datasets/DCAgent2/swe-agent-configs/resolve/main/sera_e2e.yaml`
exported. Pass `--vllm-max-retries 30` to the listener (eager mode
needs the longer warmup window — see EVAL_GUIDE §4).

### `allenai/SERA-8B` — Pattern A

```yaml
"allenai/SERA-8B":
  conda_env: otagent-fix
  tensor_parallel_size: 1
  max_model_len: 32768
  swap_space: 32
  trust_remote_code: true
  tool_call_parser: hermes
  extra_args: "--enforce-eager --disable-cascade-attn --seed 42"
```

### `SWE-bench/SWE-agent-LM-32B` — Pattern B

```yaml
"SWE-bench/SWE-agent-LM-32B":
  conda_env: otagent-fix
  tensor_parallel_size: 2
  max_model_len: 32768
  swap_space: 32
  trust_remote_code: true
  tool_call_parser: hermes
  reasoning_parser: qwen3
  extra_args: "--enable-prefix-caching"
```

### `SWE-bench/SWE-agent-LM-7B` — Pattern B

```yaml
"SWE-bench/SWE-agent-LM-7B":
  conda_env: otagent-fix
  tensor_parallel_size: 1
  max_model_len: 32768
  swap_space: 32
  trust_remote_code: true
  tool_call_parser: hermes
  reasoning_parser: qwen3
  extra_args: "--enable-prefix-caching"
```

### `GAIR/daVinci-Dev-32B` — Pattern B (no `reasoning_parser`)

```yaml
"GAIR/daVinci-Dev-32B":
  conda_env: otagent-fix
  tensor_parallel_size: 2
  max_model_len: 32768
  swap_space: 32
  trust_remote_code: true
  tool_call_parser: hermes
  extra_args: "--enable-prefix-caching"
```

> **Note**: earlier drafts of this recipe carried
> `reasoning_parser: qwen3`. The validated 55.5% reproduction was
> served with **no** reasoning_parser — confirmed by reading the vLLM
> log of that fire (`reasoning_parser=none`). Don't add it here.

### `Qwen/Qwen3-Coder-30B-A3B-Instruct` — Pattern C

```yaml
"Qwen/Qwen3-Coder-30B-A3B-Instruct":
  conda_env: otagent2-fix
  tensor_parallel_size: 2
  max_model_len: 32768
  swap_space: 32
  trust_remote_code: true
  tool_call_parser: qwen3_coder
  extra_args: "--enable-prefix-caching"
```

### `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` — Pattern C

```yaml
"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16":
  conda_env: otagent2-fix
  tensor_parallel_size: 2
  max_model_len: 32768
  swap_space: 32
  trust_remote_code: true
  tool_call_parser: qwen3_coder
  reasoning_parser: nano_v3
  extra_args: '--reasoning-parser-plugin ${DCFT}/eval/configs/nano_v3_reasoning_parser.py --chat-template ${DCFT}/eval/configs/nemotron_chat_template.jinja --enable-prefix-caching --override-generation-config {"temperature":0.6,"top_p":0.95}'
```

`${DCFT}` is expanded by the listener (see EVAL_GUIDE §4 / Pattern C).

### `Skywork/Skywork-SWE-32B`, `R2E-Gym/R2EGym-32B`, `SWE-Lego/SWE-Lego-Qwen3-{8B,32B}` — Pattern D

These run through openhands text-tools (no native tool calls; the
agent extracts JSON itself). Pattern D yaml — no parsers:

```yaml
"<model>":
  conda_env: otagent-fix
  tensor_parallel_size: 2          # 1 for the 8B
  max_model_len: 32768
  swap_space: 32
  trust_remote_code: true
  extra_args: "--enable-prefix-caching"
```

### Models with structural reproduction issues

- `NovaSky-AI/SA-SWE-32B` — paper used SkyRL's engine-direct
  `OHCodeActAgent`, which bypasses the OpenAI API. Best-of-bad via
  swe-agent regular reaches 24.7% vs paper 39.4%.
- `GAIR/OpenSWE-32B` — terminus-2 fires hit 75–77% `STOE` (stuck on
  exception); `override-generation-config` from the paper's
  `rep_penalty 1.2` triggers it. Dropping the override gives 0% × 4
  configs. Flagged for the model owner.

---

## §3 — Firing checklist

Before each Cat 3 fire:

- [ ] Source `~/pinggy_pairs.env` and confirm at least one pair is
      free (EVAL_GUIDE §3 occupancy check).
- [ ] Cross-cluster collision check on the chosen pair (curl
      `/v1/models`, verify `root` is on this cluster).
- [ ] Source `~/.local/eval.env` for `HF_TOKEN`, `SUPABASE_*`,
      `DAYTONA_API_KEY`.
- [ ] If the model needs `otagent2-fix`, confirm the env exists in the
      cluster config's `conda_envs:` block.
- [ ] For Pattern A 32B fires, pass `--vllm-max-retries 30`.
- [ ] For SERA-e2e, export `SWEAGENT_CONFIG=...` before fire.
- [ ] Pass `--no-auto-snapshot` (or omit `--auto-snapshot`) — the
      preset's authoritative `auto_snapshot` field handles caching;
      CLI flips have caused silent 0%.

After fire:

- [ ] Wait n ≥ 150 trials before reading accuracy.
- [ ] Sample one `trajectory.json` to verify tool-call format hasn't
      regressed (broken `<function=...>` markers).
- [ ] Verify Supabase shows the correct HF model name (not a numeric
      vLLM-served id) — see EVAL_GUIDE §8.
