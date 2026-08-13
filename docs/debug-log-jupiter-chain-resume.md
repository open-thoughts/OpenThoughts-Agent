# Debugging log for Jupiter chain resume

Make a from-scratch Jupiter dependency chain consume checkpoints banked by earlier links.

## Initial status

AUTO resume discovery ran once during submission. With no checkpoint present, the launcher serialized
`trainer.resume_mode=none` into the JSON shared by every Slurm link. Later links therefore restarted at step 0
even after an earlier link wrote `latest_ckpt_global_step.txt`.

## Hypothesis 1

The durable resume intent is lost when `RLPathManager.resolve()` converts AUTO into the current concrete mode.
Preserving that intent in `RLJobConfig` would let each link repeat the same validated discovery after Slurm starts
it and before Ray or the trainer starts.

## Changes to make

Add a resume policy to the resolved path contract and serialized job config. Mark implicit AUTO launches for
link-start discovery; keep explicit `none`, `latest`, `from_path`, and forced-fresh launches fixed. At link start,
replace only the concrete resume mode and path in the Hydra argument list.

## Results

The regression runs the same serialized job twice. With an empty checkpoint directory the first run keeps
`resume_mode=none`. After the test banks `global_step_6`, the second run resolves `resume_mode=latest` and the
exact checkpoint path. The checkpoint-path and chain-guard suite passes (32 tests).

## Future work

- [ ] Confirm the first replacement campaign link logs `Resuming from global_step: 6` or later.
