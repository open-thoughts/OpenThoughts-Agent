# Debugging log for offline trace upload

Stop Jupiter links from promising a Hugging Face trace upload while Hub access is disabled.

## Initial status

The 30B Jupiter recipe enabled post-run trace uploads and exported `HF_HUB_OFFLINE=1`. Every link retained its
local traces but the uploader failed while creating the Hugging Face dataset repository.

## Hypothesis 1

The conflict is fully knowable at launch validation: both settings live in the same parsed YAML. Rejecting the
combination before submission prevents a delayed failure after hours of training.

## Changes to make

Validate `terminal_bench.trace_upload.enabled` against the host and Apptainer Hub-offline variables before any
dataset or model staging. Disable trace upload in the affected Jupiter recipe and retain `trials_dir` for
archive-based collection.

## Results

The focused trace-upload and durable-path suite passes (16 tests). Ruff passes over the complete `tests/` tree.
The validator rejects host or Apptainer `HF_HUB_OFFLINE` values before launch and permits the same environment
when trace upload is disabled.

## Future work

- [ ] Add an archive uploader when the campaign needs automatic publication from a networked login node.
