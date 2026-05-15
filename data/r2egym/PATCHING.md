# R2E-Gym task flattening

The raw R2E-Gym parquet (`data/r2egym/generate.py`) produces tasks whose
`environment/Dockerfile` is `FROM namanjain12/<repo>_final:<commit_sha>` — a
unique prebuilt image per task. This defeats Daytona's `auto_snapshot` cache
(1 snapshot per task, 1785 rebuilds for the codex-solved subset).

The patcher at `../patchers/patch_r2egym_tasks.py` flattens tasks down to
**3 unique snapshots** (one per Python version) by swapping the prebuilt
image for a generic `python:X.Y-bookworm` base and moving all per-task
content into `setup_files/` (uploaded by Harbor at runtime, excluded from
the environment-snapshot hash).

## Snapshot reduction (verified)

| Dataset | Tasks | Raw snapshots | Flattened snapshots |
|---|---|---|---|
| `penfever/r2egym_gpt5_codex_solved_tasks` | 1,785 | 1,785 | **3** |
| `R2E-Gym/R2E-Gym-V1` (via `--from-upstream`) | 8,101 | 8,101 | **3** |

Snapshots split into 5 buckets to maximize oracle yield without exploding
snapshot count. Three "generic" buckets keyed on Python version, plus two
specialized ones for repos whose source `pip install -e .` is unreliable on
old commits:

| Snapshot | Repos | Why |
|---|---|---|
| `python:3.9-bookworm` (generic) | tornado, scrapy, pillow, pyramid, datalad, coveragepy | Most repos; their fix-commit source builds cleanly with modern pip |
| `python:3.11-bookworm` (generic) | rare upstream tasks | Some upstream commits used 3.11 |
| `python:3.12-bookworm` (generic) | rare upstream tasks | Some upstream commits used 3.12 |
| `python:3.9-bookworm` (scientific) | numpy, pandas, orange3, matplotlib, sympy | Heavy scientific-stack pre-installed as binary wheels (numpy<2, scipy, pandas, scikit-learn, sympy, matplotlib, hypothesis, nose, etc.). Tasks **skip `pip install -e .`** and pytest runs from `/tmp` so the source tree at `/testbed` doesn't shadow the pre-installed package. Trade-off: oracle verifies "pre-installed wheel satisfies expected outcomes" rather than "this exact commit's source compiles and passes" — for 2017–2022 fixes that are now upstream, this is reliable. |
| `python:3.6-slim-buster` (aiohttp) | aiohttp | Pre-3.7 aiohttp source has `yield from asyncio.async(...)` which is a `SyntaxError` on Python ≥ 3.7 (`async` became reserved). Python 3.6 parses it. pytest-json-ctrf is install-best-effort here (needs Python 3.8+); test.sh detects and falls back to plain pytest. |

Verify any time:
```bash
python -m scripts.harbor.count_snapshots_from_tasks \
    --local-dataset /path/to/patched --verbose
```

## How the flattening works

| File | Before | After |
|---|---|---|
| `environment/Dockerfile` | `FROM namanjain12/<repo>_final:<sha>` | `FROM python:X.Y-bookworm` + apt toolchain + pytest/chardet |
| `environment/workspace/metadata.json` | per-task | **removed** (moved to `setup_files/test_info.json`) |
| `setup_files/test_info.json` | — | `{github_repo, base_commit, python_version, test_file_names, expected_output_json}` |
| `setup_files/r2e_tests/*.py` | — | Test source from upstream V1 `execution_result_content.test_file_codes` |
| `instruction.md` | wordy r2egym prompt (may contain `[ISSUE]...[/ISSUE]`) | Same body, with setup preamble prepended (git clone + checkout + `pip install -e .` + stage r2e_tests); `[ISSUE]` tags stripped |
| `tests/test.sh` | 200-line inline shell w/ internet-dependent `uv pip install` | `timeout` + `python -m pytest /testbed/r2e_tests/` + pytest-based grader → `/logs/verifier/ctrf.json` + `reward.txt` |
| `tests/test_state.py` | — (previously inlined into test.sh) | Standalone pytest grader: parse log, compare to `expected_output_json` |
| `solution/solve.sh` | — (missing, caused 1785/1785 oracle failures) | Oracle that copies `patched_files/*` onto `/testbed/` and processes `deleted_files.txt`; derived from `parsed_commit_content.file_diffs` |
| `solution/patched_files/*` | — | Full post-fix contents of every modified or new file from the fix commit |
| `solution/deleted_files.txt` | — | One path per line for files the fix commit removed (rare — 2/8101 in V1) |

## Repo → GitHub map (13 repos in R2E-Gym-V1)

Defined in `REPO_TO_GITHUB` inside `patch_r2egym_tasks.py`:

| r2egym `repo_name` | github path | Tasks in codex-solved subset |
|---|---|---|
| aiohttp | aio-libs/aiohttp | 129 |
| coveragepy | nedbat/coveragepy | 45 |
| datalad | datalad/datalad | 76 |
| matplotlib | matplotlib/matplotlib | 0 |
| moto | getmoto/moto | 0 |
| numpy | numpy/numpy | 428 |
| orange3 | biolab/orange3 | 174 |
| pandas | pandas-dev/pandas | 287 |
| pillow | python-pillow/Pillow | 268 |
| pyramid | Pylons/pyramid | 108 |
| scrapy | scrapy/scrapy | 139 |
| sympy | sympy/sympy | 0 |
| tornado | tornadoweb/tornado | 131 |

## Python version distribution (from `execution_result_content`, floored to 3.9)

| Raw upstream python | Tasks | Normalized to |
|---|---|---|
| 3.5 / 3.6 / 3.7 / 3.8 / 3.9 | majority | 3.9 |
| 3.11 | 2 | 3.11 |
| 3.12 | 2 | 3.12 |

Old Python versions (3.5-3.8) are floored to 3.9 because `python:3.X-bookworm`
only exists for X ≥ 3.9. Same approach as `patch_swe_rebench_tasks.py`.

## Usage

Two entry points to the patcher — pick based on which dataset you want.

### Full V1 (all 8,101 tasks) — `--from-upstream`

No raw extraction needed; the patcher reads V1 parquets directly:

```bash
python data/patchers/patch_r2egym_tasks.py --from-upstream \
    --output-dir /path/to/tasks_flat_v1
```

Outputs task dirs `r2egym-v1-00000` … `r2egym-v1-08100`. First run
rebuilds the index cache (~1 min); subsequent runs reuse it.

### Codex-solved subset (1,785 tasks)

Two-step: extract, then patch:

```bash
# 1. Extract raw tasks from the filtered parquet
python -m scripts.datagen.extract_tasks_from_parquet \
    --parquet penfever/r2egym_gpt5_codex_solved_tasks \
    --output_dir /path/to/tasks_raw \
    --on_exist overwrite

# 2. Flatten
python data/patchers/patch_r2egym_tasks.py /path/to/tasks_raw \
    --output-dir /path/to/tasks_flat
```

### Verify snapshot count (both paths)

```bash
python -m scripts.harbor.count_snapshots_from_tasks \
    --local-dataset /path/to/tasks_flat --verbose
```

Both paths produce the **same 3 snapshot hashes**, so if you run them on
the same cluster they'll share Daytona snapshots.

## Caveats

- **Install best-effort**: the preamble runs `pip install -e . || pip install .`
  on the cloned repo. Upstream R2E-Gym pins every transitive dep via a uv
  lockfile baked into the docker image; we trade that pinning for snapshot
  sharing. Expect some per-task install failures on old commits where
  unpinned deps break with modern pip/setuptools. Symptom: the pytest run
  errors on import, so `test_state.py` reports "missing tests" and writes
  `reward=0`. If the failure rate is material, the next iteration would pull
  `execution_result_content.setup_res_stdout` (which contains a `pip freeze`-
  style dump) and emit per-task `setup_files/requirements.txt`.
- **Python-version mismatch**: flooring everything to 3.9 is fine for most
  repos but may fail for tasks that rely on Python-3.7-specific behaviors
  (e.g. old `asyncio` APIs). Numpy / pandas in particular span Python 3.7
  upstream — monitor task success rates before committing to this floor.
- **Filtered subset only**: the codex-solved subset has 10 of 13 repos
  (no matplotlib/moto/sympy). If you re-run on the full
  `R2E-Gym/R2E-Gym-V1`, all 13 repos are in `REPO_TO_GITHUB`.
- **Upstream cache is 372 MB** — it holds `test_file_codes` for all 8,101 V1
  tasks. Not checked in; regenerated on first patcher run.
