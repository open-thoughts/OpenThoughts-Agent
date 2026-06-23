# AJudge — How-To / Onboarding

Project notes for **AJudge** (the certified-judgment library — a *separate* project from OpenThoughts-Agent
ML-ops). Folded out of cross-project memory 2026-06-14.

## Workflow
- **Iterative loop:** after making changes, always run `pytest` **and** update `README.md` before
  considering a step done.
- Tests live in `tests/`:
  - `test_imports.py` — 7 import/instantiation tests.
  - `test_registries.py` — 8 registry-validation tests.
- The **pre-commit hook** validates registries via `pytest tests/test_registries.py`.
- No `uv` on this machine — run tests with `.venv/bin/python -m pytest`.

## Key architecture decisions
- **Context is a HuggingFace `Dataset`** (not raw dicts).
- **`CertifiedJudgment`** bundles all 3 certs into one addressable blob (keyed by `public_hash`),
  optionally published to the HF Hub.
- **Rules** never mutate (but *can* redact). **Screens** always mutate (and must report). **Guidelines**
  are post-hoc aggregate checks.
- Build backend is **`hatchling`** (not `uv_build`).
