# Sera evaluation artifacts

Eval-time configs and diagnostic notes that go alongside `baselines/sera/README.md`.

## SWE-agent scaffold configs (`sweagent_configs/`)

- **`e2e_upstream.yaml`** — verbatim snapshot of `https://github.com/allenai/SERA/blob/main/sera/configs/sweagent/e2e.yaml` (commit-of-record at the time of the 2026-04-25 sync). Use this as the reference; do NOT pass it directly because it inherits SWE-agent's default `agent.max_requeries: 3`, which is too low for our Sera-trained 8B (see diagnostic below).

- **`e2e_max_requeries_50.yaml`** — same config, with `agent.max_requeries: 50` injected to give the model 50 requery chances on `FormatError` / `_BlockedActionError` / `BashIncorrectSyntaxError` before the harness emits "Exit due to repeated format/blocklist/bash syntax errors". This is the version we pass via the `SWEAGENT_CONFIG` env var.

A copy of `e2e_max_requeries_50.yaml` is hosted at:
```
https://gist.githubusercontent.com/penfever/f327eabde26934630ee2aea1a59bb511/raw/sera_e2e_patched.yaml
```
The harbor `swe_agent.py` integration curls this URL into the daytona container at runtime when `SWEAGENT_CONFIG` is a URL. To inject it into a launcher-generated eval sbatch:
```bash
SB=experiments/<run_dir>/sbatch/<job_name>__eval.sbatch
sed -i '/^set -eo pipefail$/a export SWEAGENT_CONFIG="https://gist.githubusercontent.com/penfever/f327eabde26934630ee2aea1a59bb511/raw/sera_e2e_patched.yaml"' "$SB"
sbatch -D $SCRATCH/OpenThoughts-Agent "$SB"
```
(Submit from the OT-Agent root or pass `-D` — the in-sbatch DCFT-detection fallback resolves to `$PWD` and breaks if you submit from `$HOME`. See `feedback_perlmutter_sbatch_cwd.md`.)

## Diagnostic — `FormatError` storm at eval time (2026-04-25)

After Sera v7 (`Sera-4.6-Lite-T2-v4-1000-axolotl__Qwen3-8B-v7`) was uploaded and evaluated on swebench-verified-random-100-folders + SWE-agent harness, trials were exiting with **"Exit due to repeated format/blocklist/bash syntax errors"** even after we bumped `max_requeries` from 3 to 50. The patched ceiling IS in effect (debug log shows 50 requeries before exit), so the issue is structural.

### Root cause

The model emits Hermes `<tool_call>` blocks with **doubly-nested `name`/`arguments`**, e.g.:

```
<tool_call>
{"name": "str_replace_editor", "arguments": {"name": "view", "arguments": {"command": "view", "path": "/testbed/sympy/tensor/array/ndim_array.py"}}}}
</tool_call>
```

Two pathologies:
1. The outer tool name is correct (`str_replace_editor`), but its `arguments` wraps **another `{"name", "arguments"}` envelope** instead of the actual params.
2. Four closing braces vs. three opening braces → invalid JSON.

### Pipeline failure

1. vLLM's Hermes parser tries to parse the `<tool_call>` JSON → fails (invalid) → returns the response with empty `tool_calls`.
2. SWE-agent's `function_calling` parser sees no `tool_calls` → raises `FunctionCallingFormatError`.
3. Harbor requeries — but the bad pattern is now in the conversation history, so the model keeps reproducing it.
4. After 50 requeries (or 3 with the upstream config), exit-format triggers.

### What we ruled out

- **Not a training-data bug.** Scanned all 34,793 `<tool_call>` blocks in `laion/Sera-4.6-Lite-T2-v4-1000` and **0 are doubly-nested**. All 16,815 `str_replace_editor` calls use the canonical `{command, path, new_str?, old_str?, file_text?, view_range?, insert_line?}` arg shape.
- **Not a `transform_traj_hermes` bug.** Our port at `scripts_dataset_build/subset_sera_v4.py::_tool_call_to_action` faithfully passes through `tc["function"]["arguments"]` — and the upstream OpenAI `tool_calls` we read are already in the right shape.
- **Not the `max_requeries` ceiling.** Bumping to 50 still exits because the model never produces a parseable JSON envelope on requery.

### Hypothesis

Sera v7 (1000 rows × 12 epochs) is **under-trained on the schema distinction** between the outer envelope (`{"name": tool, "arguments": {...}}`) and the inner sub-command schema for `str_replace_editor` (`{command: "view"|"create"|..., path: ..., ...}`). The model conflates the two schemas under long-context pressure, producing a hybrid where the outer envelope's `arguments` slot gets filled with another envelope.

### Next steps to evaluate

1. **Parser shim** (highest leverage near-term, no retrain): subclass SWE-agent's `function_calling` parser to detect the `{"name": X, "arguments": {"name": Y, "arguments": Z}}` pattern and recursively extract `Z` as the actual args (with `X` as the tool name). Implement at the harbor level so it applies cluster-wide for any future Sera SFT eval.
2. **More data** (mid-leverage, requires retrain): Sera v4 sizes 3160 / 10000 / 31600 likely outgrow the schema-confusion mode the way 100k clearly does (Nemotron-Corpus 100k 32B already produces clean output).
3. **Schema-disambiguating data** (mid-leverage, requires data work): include a small augment set where assistant turns explicitly distinguish the envelope vs the str_replace_editor.command schema (e.g. by dataset-level rendering hints or a cleaner system prompt).

(1) is the unblocking path for evaluating ALL existing Sera-v\* weights; (2)+(3) are what would actually fix the model.
