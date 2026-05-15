# SERAlike — SWE-Smith tasks with SERA-style vague prompts

Rewrites every row of [`DCAgent/swesmith-sandboxes-with_tests-25k`](https://huggingface.co/datasets/DCAgent/swesmith-sandboxes-with_tests-25k) so that each task's `instruction.md` is replaced with a vague SERA-style "patch or abstain" prompt, while leaving the docker image, the buggy commit checkout, and the test verifier untouched.

## Why

[SERA's SVG method](https://arxiv.org/abs/2601.20789) trains agents on synthetic data where:
1. The teacher is told there's a vague bug related to a function (sampled from one of 51 bug categories).
2. The teacher invents some plausible code change and submits.
3. The change is taken as ground truth — what the model "fixed" was the bug (retroactively).

SERA's actual published training corpus uses SWE-Smith's 121-codebase set as the base — they just reuse SWE-Smith's docker images for the environment and synthesize the rest. Our SERAlike pipeline is a leaner read of the same idea:

- Reuse SWE-Smith's task tarballs **including the existing pytest verifier** (a strict improvement over SVG's line-level recall ground truth).
- Replace the upstream specific bug report (which would be too easy for the agent) with one of SERA's 51 vague prompts.
- Add an **explicit abstain branch** in the prompt (option to refuse before generating, cleaner than SERA's post-hoc self-evaluation).

The result is a Harbor-format task dataset suitable for downstream datagen via `hpc.launch --job_type datagen` against any teacher (e.g. GLM-4.7, Claude, GLM-4.5-Air).

## Pipeline

```
DCAgent/swesmith-sandboxes-with_tests-25k   (upstream — real SWE-Smith bugs + tests)
        │
        ▼  rewrite.py: deterministic SHA-256(path) → 1 of 51 SERA bug categories
        │              gunzip+untar → swap instruction.md → repack
        ▼
DCAgent/SERAlike-swesmith-25k               (this dataset — vague prompts, real verifier)
        │
        ▼  hpc.launch --job_type datagen + terminus-2 + GLM-4.7 (future)
        ▼
Trained Qwen3 SFT data
```

## Files

- `bug_prompts.py` — verbatim port of SERA's 51 `ROLLOUT_ONE_PROMPTS`, with `{{start_fn}}` references rewritten to "any function in this codebase" (we don't currently emit a per-task function pointer; this is option a from the design discussion). Also defines `render_instruction()` which builds the full `instruction.md` body including the abstain branch.
- `rewrite.py` — streaming row-by-row rewriter. Streams the upstream dataset, swaps `instruction.md` inside each gzipped tarball, writes parquet shards under `./out/`.
- `upload.py` — pushes `./out/*.parquet` plus a generated README to `DCAgent/SERAlike-swesmith-25k`.

## Usage

```bash
# 1. Smoke test on 50 rows
python -m data.seralike.rewrite --limit 50

# 2. Inspect a sample
python -c "
import io, gzip, tarfile, pyarrow.parquet as pq
t = pq.read_table('data/seralike/out/shard-00000.parquet')
row = t.to_pylist()[0]
gz = gzip.GzipFile(fileobj=io.BytesIO(row['task_binary']))
tf = tarfile.open(fileobj=gz)
print(tf.extractfile('instruction.md').read().decode())
"

# 3. Full run (25k rows)
python -m data.seralike.rewrite

# 4. Upload to HF
python -m data.seralike.upload --repo DCAgent/SERAlike-swesmith-25k
```

## Notes

- **Function pointer**: SERA prompts originally reference a specific `{{start_fn}}` to anchor the agent. We don't emit a per-task function pointer in v0. If quality is lacking, a future v1 can pre-compute per-repo function manifests (one image-pull per unique base image, ~120 of them) and inject a random fn into the prompt.
- **Abstain semantics at datagen time**: the prompt asks the agent to output `<abstain/>` if no real change is found. Most harnesses (SWE-agent, terminus-2) already have an `exit_forfeit` / abstain action; if the model emits `<abstain/>` as text the harness can be adapted to recognize it. Trace filtering should drop traces where the agent abstained or where the existing tests fail after the patch.
- **Bug-label distribution**: roughly uniform across the 51 categories (deterministic SHA-256 keying). See `metadata.jsonl` for per-row labels.
- **Reproducibility**: bug labels are derived from `sha256(path)`. Re-running on the same source dataset yields byte-identical output.
