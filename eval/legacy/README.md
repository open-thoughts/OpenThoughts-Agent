# Legacy Eval Scripts

These are frozen v4/v6-MBZ eval scripts kept for backward compatibility.

- `unified_eval_harbor_v4.sbatch` — v4 sbatch (MBZ-specific, hardcoded paths)
- `unified_eval_harbor_v6_mbz.sbatch` — old v6 sbatch (MBZ-specific, before cluster-agnostic rewrite)
- `unified_eval_listener_v4.py` — v4 listener (MBZ-specific)

**Do not modify these files.** Use the cluster-agnostic versions in `eval/` instead:
- `eval/unified_eval_harbor.sbatch`
- `eval/unified_eval_listener.py`
