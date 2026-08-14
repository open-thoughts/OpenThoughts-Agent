# Debugging log for explicit Hugging Face export

Prevent offline compute jobs from enabling Hugging Face checkpoint export unless a repository ID is explicitly configured.

## Initial status

`build_skyrl_hydra_args` replaces an absent `hf_hub_repo_id` with `laion/{job_name}`. On Jupiter, this installs the periodic export callback even though task-local model paths cannot be materialized by a later export job. The resulting `ModelLocatorError` terminates training at the first export interval.

## Hypothesis 1

The unconditional job-name-derived repository ID is the only launcher behavior that turns an otherwise disabled export into an enabled export.

## Changes to make

Add regression coverage at the Hydra configuration boundary for both an omitted repository ID and an explicitly configured repository ID.

## Results

The new regression failed before the fix: the generated Hydra arguments contained `trainer.hf_hub_repo_id=laion/tasktrove-arm`. This confirms that the launcher-derived repository ID enabled the callback.

Removed the derivation. A repository ID is now forwarded only when the caller explicitly supplies one; an absent ID remains absent. Documentation now describes checkpoint export as opt-in.

The focused configuration test passed after the fix. The full HPC unit suite passed with 315 tests. The first repo-wide collection picked up an incompatible local Marin checkout and failed while importing `rigging.telemetry`; forcing the tests to use the locked Iris package produced 554 passing tests and 2 skips.

## Future work

- [ ] Consider startup validation for explicitly requested exports whose model source cannot be materialized.
