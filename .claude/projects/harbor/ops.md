## Harbor Job File Organization

Harbor eval jobs use a **single unified directory** per eval run at `$EVAL_JOBS_DIR/<run_tag>/`.
Run tags follow the format `eval-<SAFE_MODEL>_<SAFE_REPO>` (model first, `eval-` prefix).

A `trace_jobs/<run_tag>` symlink in the working dir points to the unified run dir so Harbor writes there directly.

**Contents of `$EVAL_JOBS_DIR/<run_tag>/`**:
- `<task_name>__<trial_id>/agent/trajectory.json` — full agent conversation trace
- `<task_name>__<trial_id>/exception.txt` — error traceback if the trial failed
- `<task_name>__<trial_id>/verifier/` — verifier output and reward
- `result.json` — aggregate results, exception stats, metrics
- `config.json` — Harbor run configuration
- `meta.env` — model, dataset, SLURM job ID, DB job ID
- `vllm.log` — vLLM server log
- `upload.log` — DB/HF upload log
- `slurm.log` — symlink to the SLURM output log

To debug DaytonaErrors or other trial failures, read `exception.txt` in the trial directory:
```bash
cat $EVAL_JOBS_DIR/<run_tag>/<task>__<id>/exception.txt
```

**Config mismatch on auto-resume**: If Harbor fails with `FileExistsError: Job directory ... already exists and cannot be resumed with a different config`, the run dir has a `config.json` from a previous run with different settings. To fix, delete only the specific stale run dir **after confirming no useful trials exist**:
```bash
# Check if the dir has any completed trials before deleting
ls $EVAL_JOBS_DIR/<run_tag>/*/result.json 2>/dev/null | wc -l
# If zero, safe to delete
rm -rf $EVAL_JOBS_DIR/<run_tag>
```

## Cluster clones + sync (canonical upstream)

**Canonical upstream = `marin-community/harbor`** (see `harbor.md`). The editable harbor install on each cluster lives in these clones — `git pull` here is how a laptop push goes live:
- **Jupiter:** `/e/scratch/jureap59/feuer1/harbor`
- **Leonardo:** `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/harbor`

Both clones' `origin` MUST be `https://github.com/marin-community/harbor.git`. **Gotcha (fixed 2026-06-17):** both were found pointing at the stale fork `laude-institute/harbor` (frozen at an old commit), so `git pull` reported "Already up to date" while silently missing every new push. Repointed via `git remote set-url origin https://github.com/marin-community/harbor.git`. Deploy pattern (clone is editable → live for new processes after pull; **running jobs keep the old code until they restart**):
```bash
cd <clone> && git remote get-url origin   # MUST be marin-community/harbor
git fetch origin penfever/working && git pull --ff-only origin penfever/working
git log -1 --oneline                       # confirm the expected HEAD
```
