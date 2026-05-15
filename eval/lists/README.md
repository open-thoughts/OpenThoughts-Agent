# eval/lists/

Priority files consumed by `unified_eval_listener.py --priority-file`.
One HuggingFace model id per line; blank lines and `# ...` comments are
ignored.

```
# example list — one model per line
Qwen/Qwen3-32B
Qwen/Qwen3-Coder-30B-A3B-Instruct
allenai/SERA-32B
```

The listener filters this list against:
1. The baseline model config (`baseline_model_configs_minimal.yaml`)
   — models without an entry use defaults.
2. Already-evaluated jobs in Supabase (skipped unless `--force-reeval`).
3. `--require-priority-list` — when set, only models in the file fire,
   even if other models would otherwise be eligible.

## Generating lists

The DB-aware helper picks models with no eval row yet for a given
benchmark family:

```bash
python scripts/database/query_unevaled_models.py \
  --benchmark dev_set_v2 --size 8 \
  -o eval/lists/<your-list>.txt
```

Requires `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`.
