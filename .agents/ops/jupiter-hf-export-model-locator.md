# Debugging log for Jupiter HF export model references

Preserve an independently resolvable model reference when an offline RL launch prefetches a Hugging Face model.

## Initial status

`construct_rl_sbatch_script` downloads a Hub model into the shared Hugging Face cache, then replaces
`trainer.policy.model.path` with the task-local snapshot path. Explicit HF export requests reject that path because no
durable `model.source_uri` and `model.source_identity` accompany it.

## Hypothesis 1

Prefetching only needs to populate the shared cache. Passing the original Hub repository ID to SkyRL lets offline
`from_pretrained` resolve the cached snapshot and gives the later export job an independently resolvable model reference.

## Changes to make

Extract the RL model-prefetch boundary, cover Hub IDs and explicit local paths, then keep the Hub ID after a successful
prefetch.

## Results

Before the fix, the regression returned `/cache/hub/models--Qwen--Model/snapshots/abc123` instead of `Qwen/Model`.
The explicit local-path case remained unchanged.

The launcher now uses prefetch only to populate the shared cache. The Hub repository ID remains in the generated SkyRL
configuration and can be recorded in `HFExportRequest` without `model_source_uri` metadata.

The complete `tests/hpc` suite passes: 317 tests.
