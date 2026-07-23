---
name: eval-agentic-launch-iris
description: Launch, monitor, and manually clean up Harbor agentic evals on Marin Iris TPU or CoreWeave GPU through Marin's standalone agentic-evals package.
---

# eval-agentic-launch-iris

Iris agentic eval orchestration lives in Marin's standalone
`experiments/agentic_evals` package (Marin PR #7246), not OpenThoughts-Agent.
For Iris datagen/tracegen, use `datagen-launch-iris` instead.

## Preconditions

1. Start from a Marin checkout containing `experiments/agentic_evals` and install
   the package with the `iris` and `serve` extras.
2. Use the MAIN Daytona credential (`DAYTONA_API_KEY`) for evals. Eval Harbor
   configs use `environment.force_build: true`; do not pre-build snapshots.
3. Read `.claude/ops/iris/ops.md` before submitting. Never stop a RUNNING job
   without current-thread authorization.

## Standard Iris eval

```bash
cd /Users/benjaminfeuer/Documents/marin/experiments/agentic_evals
python -m agentic_evals.launch \
  --preset <name> \
  --harbor_config <harbor-yaml> \
  --model <hf-model-id> \
  --dataset_path <tasks-or-hf-id> \
  --tpu v6e-4 \
  --job-name "eval-<model>-<benchmark>-$(date +%Y%m%d-%H%M%S)" \
  --secrets-env "$DC_AGENT_SECRET_ENV" \
  --no-wait
```

`--preset` supplies benchmark defaults; explicit CLI settings take precedence.
Use `python -m agentic_evals.launch --help` for the package's authoritative
surface and durable-output options.

## Separately served CoreWeave model

Serving and evaluation are separate jobs. Use the package's external endpoint
mode (for the established Grug profile, `--external-profile grug`): it submits
the parent-delegated `marin-serve` job, uses Marin's `iris endpoints wait-and-mint`
primitive to obtain a parent-scoped capability URL, and starts the durable eval.
Never mint from the CoreWeave peer: peer-signed URLs are not Daytona-reachable.

The evaluator duration must not exceed the minted-token TTL. A serving endpoint
with no model requests expires through its configured idle timeout; health checks
must not reset that timer.

## Monitor and cleanup

Use the generic Iris Harbor analyzer from this repo:

```bash
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python \
  /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/iris/analyze_iris_harbor_job.py \
  /benjaminfeuer/<job> --output /tmp/<job>_history.md --resync
```

For a partial run, relaunch `agentic_evals.launch` with the same job name after
checking its persisted Harbor artifacts. Preserve artifacts before cleanup.
