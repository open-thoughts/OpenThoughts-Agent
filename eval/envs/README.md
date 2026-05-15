# eval/envs/

Conda env recipes for the eval listener. Two envs ship because some
models can't be loaded by the same `transformers` line as the
default agent runtime — see [otagent-fix vs otagent2-fix](#which-env-when)
below.

## Files

- [`otagent-fix.yml`](otagent-fix.yml) — primary env (transformers 4.x).
- [`otagent2-fix.yml`](otagent2-fix.yml) — axolotl/Qwen3.5 + Pattern C
  reasoning-parser-plugin env (transformers 5.x, vLLM 0.17.x).

Both envs install [harbor](https://github.com/harbor-framework/harbor)
**editable** from the same checkout. The exact commit matters — see
the next section.

## Pinning harbor-fix

The eval pipeline depends on a curated stack of harbor patches that
isn't (yet) merged to upstream `main`. Pin your harbor checkout to a
state that includes them:

```bash
git clone https://github.com/harbor-framework/harbor.git ~/harbor-fix
git -C ~/harbor-fix checkout penfever/temp-override
git -C ~/harbor-fix checkout 9980f967    # Daytona connection-pool fix (PR #1460)
```

`9980f967` is the verified-working tip of `penfever/temp-override` as
of 2026-04-29. Newer commits on that branch are usually fine — this
is what was last fired against.

### Why this branch (not upstream main)

The patches the eval listener relies on, beyond what's in upstream
`main`:

| Area | Why it matters |
|---|---|
| Daytona connection pool 500 (#1460) | Stops "Bearer token invalid" load-shedding under high concurrency |
| Loading-gate semaphore | Prevents unbounded sandbox creation; cap = `n_concurrent * 2` |
| Treat "Bearer token invalid" as transient | Retry instead of fail-fast on transient Daytona auth |
| `assume_global_snapshot` flag | Lets the listener pin its `auto_snapshot` decision |
| max_timeout_sec multiplier ordering | Caps after multiplier so `--timeout-multiplier 16` doesn't blow past the agent/verifier max |
| Tmux dummy-session PTY + history-limit | Required for terminus-2 on swebench / tb2 task images |
| swerex `dirs_exist_ok` | swe-agent on dev_set_v2/tb2 — see EVAL_GUIDE §7 mode 12 |
| ContextLengthExceeded → run verifier | Don't drop a partial trial; let the verifier score what we have |
| Summarization timeout default raised | terminus-2 on 32B long-context — see EVAL_GUIDE §7 mode 6 |
| LiteLLM timeouts treated as env failures | Doesn't burn vLLM slots on transient client errors |

### Three uncommitted installed-agent patches

The maintainer's working checkout carries three small in-tree patches
on top of `9980f967` that the upstream branch hasn't accepted yet. If
your fire is Cat 3 (preferred-harness reproduction — see EVAL_GUIDE
§2), apply these locally:

1. **`src/harbor/agents/installed/aider.py`** — propagate
   `OPENAI_API_BASE` so aider/litellm can reach a local vLLM, and
   keep the `openai/` provider prefix when routing through the
   OpenAI-compatible vLLM endpoint. Also expose an `--edit-format`
   CLI flag.

2. **`src/harbor/agents/installed/mini_swe_agent.py`** — handle the
   `get_api_key_var_names_from_model_name` API change (returns a list
   of provider env-var names instead of a single string), and don't
   crash when the model is unknown.

3. **`src/harbor/agents/installed/swe_agent.py`** — prepend
   `/opt/sweagent-venv/bin` to PATH (the venv `install.sh` puts
   sweagent there), and guard the `set -u` trap when sourcing
   `/etc/profile.d/testbed-conda.sh` (it references
   `CONDA_DEFAULT_ENV` unbound, which under `set -u` terminates the
   whole shell — `|| true` can't rescue a sourced script's trap).

If you don't run installed-agent fires, you don't need these patches.

## Which env, when?

| Env | Use for |
|---|---|
| **`otagent-fix`** (transformers 4.x) | Default for everything. Terminus-2 reg eval, OOD presets, swe-agent / openhands text-tools / mini-swe-agent / aider installed-agent fires. Most baseline-yaml entries default to this env. |
| **`otagent2-fix`** (transformers 5.x) | Models that don't load on transformers 4.x: axolotl-trained Qwen3 finetunes (the `extra_special_tokens` list crash), Qwen3.5 family. Also required for Pattern C scaffolds that use `--reasoning-parser-plugin` (Nemotron-Nano, Qwen3-Coder native) — needs vLLM ≥ 0.17. |

Per-model env selection lives in
[`../configs/baseline_model_configs_minimal.yaml`](../configs/baseline_model_configs_minimal.yaml)
under each model's `conda_env:` field. The listener reads it and sets
`PYTHON_BIN` for the sbatch from your cluster config's
`conda_envs:` map.

## Verification (per env)

```bash
conda activate otagent-fix
python - <<'PY'
import vllm, torch, transformers, harbor
print("vllm",         vllm.__version__)
print("torch",        torch.__version__)
print("transformers", transformers.__version__)
print("harbor.__file__", harbor.__file__)   # should point inside ~/harbor-fix
PY
```

Expected (otagent-fix):

```
vllm         0.13.0
torch        2.9.0+cu128
transformers 4.57.3
harbor       <your-harbor-fix-checkout>/src/harbor/__init__.py
```

For otagent2-fix, expect `vllm 0.17.1`, `torch 2.10.0+cu128`,
`transformers 5.6.x`.

## When the recipe drifts

These yamls pin the *eval-shaping* deps. If you upgrade vLLM, torch,
or transformers, retest at least one model from each Pattern (A/B/C/D
in EVAL_GUIDE §4) before relying on the new pins for a batch fire —
small version bumps have changed agent behavior in non-obvious ways
(reasoning-parser plugin API, eos-token reading, NCCL flashinfer
autotune). Update this README when you do.
