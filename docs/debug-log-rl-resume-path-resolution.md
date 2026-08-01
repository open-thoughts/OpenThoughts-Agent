# RL resume path resolution

## Failure contract

RL launcher artifacts may be collision-renamed from `<run>` to `<run>_N`. Checkpoint, export, and trial paths
were independently derived from that mutable launcher path, while the auto-resume guard inspected only the
unrenamed path. A checkpoint written under a fork could therefore be omitted from the next launch and training
would begin at step zero.

## Hypothesis

A single RL path manager can resolve the durable state root before Hydra arguments are built. Resume validation
and subsequent checkpoint, export, trial upload, and cleanup paths can then consume the same resolved object.
YAML path values are inputs to that decision, while explicit Hydra overrides retain final precedence.

## Test design

The path contract covers checkpoint discovery across canonical and numbered fork roots, highest-step selection,
strict validation of explicit `resume_mode=latest`, malformed marker rejection, ambiguous equal-step siblings,
canonical placement for a new run whose launcher artifacts were collision-renamed, and rejection when an explicit
resume source differs from the checkpoint write directory.

## Decision rule

An automatically selected checkpoint makes its containing run root the owner of checkpoints, exports, and
trials for the resumed launch. It uses `resume_mode=latest` on that fixed checkpoint directory so chained
restarts observe checkpoints written after submission. With no checkpoint, durable state uses the canonical root even if configs and
logs were collision-renamed. Explicit resume settings are validated only at their stated path and never fall
back to a sibling. Built-in launcher configs leave `resume_mode` unset so first launches use automatic
discovery; a user-authored `resume_mode=latest` is strict and requires a usable checkpoint at the resolved path.
