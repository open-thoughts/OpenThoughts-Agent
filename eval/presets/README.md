# Eval presets

Shared catalog of eval-run defaults. One YAML file per preset, named
`<preset>.yaml` (the stem is the preset name used on the CLI, e.g. `swebench`).

The OpenThoughts-Agent consumer is the **SLURM orchestrator** —
`eval/unified_eval_listener.py` (`--preset`). Iris agentic evals live in
Marin's standalone `experiments/agentic_evals` package (Marin PR #7246), which
ships its own preset catalog so it can be installed independently of this repo.

## Format

Each file is a flat mapping. `load_presets()` returns
`{stem: parsed_yaml_mapping}` in sorted-key order, with field types preserved
(bools stay bools, ints stay ints, `datasets` stays a list).

| Field | Type | Meaning |
|---|---|---|
| `datasets` | list[str] | HF dataset ids for the SLURM listener to evaluate. |
| `log_suffix` | str | Suffix for the listener's log file. (SLURM-only) |
| `n_concurrent` | int | Harbor `--n-concurrent`. |
| `error_threshold` | int | Max invalid errors before abort. (SLURM-only) |
| `vllm_max_retries` | int | vLLM startup retries. (SLURM/serve-only) |
| `agent_kwargs` | list[str] | Generic extra harbor agent-kwargs as `key=value` strings, each forwarded as `--agent-kwarg key=value`. **Thinking is NOT set here** — it is **per-model authoritative**, sourced from the shared model-config registry (`eval/configs/model_configs.yaml`, the default path), so a preset can never force thinking on a non-thinking model. Affects results. |
| `agent_parser` | str | Harbor agent-kwarg `parser=<value>` (e.g. `xml`). Affects results. |
| `auto_snapshot` | bool | Pre-build Daytona snapshots. (SLURM-only) |
| `config_yaml` | str | Listener eval config YAML. (SLURM-only) |
| `slurm_time` | str | SLURM time limit. (SLURM-only) |
| `agent_envs` | str | Comma-separated `KEY=VALUE` envs forwarded into the sandbox. (SLURM-only) |

## Adding a preset

Drop a new `<name>.yaml` here with the fields above. No code change is needed;
the SLURM listener picks it up automatically (and `--preset` choices update).
