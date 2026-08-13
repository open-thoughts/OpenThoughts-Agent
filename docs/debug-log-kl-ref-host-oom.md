# Debugging log for KL reference host OOM

Prevent KL-enabled 30B TaskTrove arms from exhausting host memory at policy optimizer and checkpoint-load peaks.

## Initial status

All KL arms placed a 16-rank, CPU-offloaded reference model on the four policy nodes. Those nodes used 100–170
GiB more memory than the no-KL arm and workers were killed during the first optimizer step or checkpoint load.

## Hypothesis 1

`colocate_all: false` was assumed to separate the policy and reference models, but the MarinSkyRL base setting
`colocate_policy_ref: true` still puts both actor groups in one placement group.

## Changes to make

Inspect the resolved launcher geometry and MarinSkyRL actor placement.

## Results

Confirmed. The campaign reserves six GH200 nodes: four policy nodes and two rollout nodes. The inherited
`colocate_policy_ref: true` assigns each policy actor 0.75 GPU and each reference actor 0.25 GPU in the same
four-node placement group. Reserving a separate reference group would increase each arm from six to ten nodes.

## Hypothesis 2

Keeping reference parameters on GPU removes the measured reference-model host allocation while retaining the
existing six-node topology. FSDP still shards the reference over all 16 ranks, while the policy remains CPU
offloaded and can use the recovered host-memory headroom for Adam first-touch, grad-norm temporaries, and
checkpoint restore.

## Changes to make

Disable CPU offload for the 30B Jupiter reference model, declare policy/reference colocation explicitly, and
add a regression test for that campaign invariant.

## Results

The regression and adjacent RL path/chain tests pass (32 tests). The regression loads the shipped Jupiter
configuration and verifies that policy/reference colocation cannot silently reintroduce reference CPU offload.
Ruff passes over the complete `tests/` tree. The complete local pytest suite cannot collect three Iris watcher
modules because the local Marin checkout lacks `rigging.telemetry`; GitHub CI installs the locked environment.

## Future work

- [ ] Record the policy-node host-memory watermark on the next KL run through its first optimizer step.
