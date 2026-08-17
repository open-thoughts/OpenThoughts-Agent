# Jupiter HF export model-reference invariant

RL model prefetching populates the shared Hugging Face cache but must preserve the configured model reference. Replacing
a Hub repository ID with a task-local cache snapshot makes a later `HFExportRequest` impossible to replay independently:
the export worker cannot resolve that snapshot without separate `model.source_uri` and `model.source_identity` metadata.

`prefetch_rl_model` therefore returns its input reference after a successful prefetch. Hub references remain Hub
references, and explicit local model paths remain unchanged. Regression coverage verifies both cases. The complete
`tests/hpc` suite passes: 317 tests.
