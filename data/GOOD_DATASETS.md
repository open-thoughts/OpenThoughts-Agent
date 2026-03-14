# Good Datasets Summary

**Date**: 2026-02-19
**Eval model**: openai/gpt-5-mini-2025-08-07 (unless noted)
**Sample size**: 25 tasks per dataset
**Total good datasets**: 38

## All Good Datasets (sorted by pass rate)

| # | Dataset | HuggingFace Repo | Pass Rate (mini) | Tasks | Source | Languages |
|---|---------|------------------|-------------------|-------|--------|-----------|
| 1 | stack-php | DCAgent/exp_rpt_stack-php-v2 | 100% | 500 | v2-fixed | PHP |
| 2 | stack-junit | DCAgent/exp_rpt_stack-junit | 84% | original | original | Java |
| 3 | crosscodeeval-csharp | DCAgent/exp_rpt_crosscodeeval-csharp | 68% | original | original | C# |
| 4 | r2egym-easy | DCAgent/exp-rdb-r2egym-easy | 65% | original | original | Python |
| 5 | r2egym-trivial | DCAgent/exp-rdb-r2egym-trivial | 60% | original | original | Python |
| 6 | bugsinpy-mf | DCAgent/exp_rpt_bugsinpy-mf | 8% (mini) / 0% gpt-5 | 500 | v5-rewrite | Python |
| 7 | r2egym-medium | DCAgent/exp-rdb-r2egym-medium | 56% | original | original | Python |
| 8 | r2egym-hard | DCAgent/exp-rdb-r2egym-hard | 56% | original | original | Python |
| 9 | bigcodebench | DCAgent/exp_rpt_bigcodebench-v3 | 54% (mini) / 40% nano | 500 | v3-fat-dockerfile | Python |
| 10 | codeelo | DCAgent/exp_rpt_codeelo-v2 | 52% | 500 | v2-fixed | Multi |
| 11 | crosscodeeval-python | DCAgent/exp_rpt_crosscodeeval-python-v2 | 48% | 500 | v2-fixed | Python |
| 12 | crosscodeeval-java | DCAgent/exp_rpt_crosscodeeval-java | 48% | original | original | Java |
| 13 | crosscodeeval-typescript | DCAgent/exp_rpt_crosscodeeval-typescript | 48% | original | original | TypeScript |
| 14 | e2egit | DCAgent/exp_rpt_e2egit-v2 | 48% | 500 | v2-fixed | Python |
| 15 | bugsinpy | DCAgent/exp_rpt_bugsinpy-v4 | 8% (mini) / 8% gpt-5 | 500 | v4-no-copy-dockerfile | Python |
| 16 | pymethods2test | DCAgent/exp_rpt_pymethods2test-v3 | 40% | 500 | v3-fixed | Python |
| 17 | stack-pytest | DCAgent/exp_rpt_stack-pytest-v2 | 40% | 500 | v2-fixed | Python |
| 18 | unitsyn-python | DCAgent/exp_rpt_unitsyn-python-v3 | 36% | 500 | v3-fixed | Python |
| 19 | stack-bash-withtests-gpt5mini | DCAgent/exp_rpt_stack-bash-withtests-gpt5mini | 32% | original | original | Bash |
| 19b | nemotron-bash | DCAgent/exp_rpt_nemotron-bash-v2 | 33% (nano) | 10 | v2-timeout-wrapper | Bash |
| 19c | nemotron-pytest | DCAgent/exp_rpt_nemotron-pytest-gpt5mini-v2 | 30% (nano) / 40% gpt-5 | 10 | v2-pythonpath-hints | Python |
| 20 | methods2test | DCAgent/exp_rpt_methods2test-v2 | 28% | 500 | v2-fixed | Java |
| 21 | r2egym-very_hard | DCAgent/exp-rdb-r2egym-very_hard | 28% | original | original | Python |
| 22 | stack-cpp | DCAgent/exp_rpt_stack-cpp | 24% | original | original | C++ |
| 23 | stack-bash-withtests | DCAgent/exp_rpt_stack-bash-withtests | 24% | original | original | Bash |
| 24 | codereval-python | DCAgent/exp_rpt_codereval-python-v2 | 24% | 230 | v2-fixed | Python |
| 25 | stack-bash | DCAgent/exp_rpt_stack-bash | 20% | original | original | Bash |
| 26 | stack-dockerfile | DCAgent/exp_rpt_stack-dockerfile-v2 | 16% | 497 | v2-fixed | Dockerfile |
| 27 | stack-selfdoc | DCAgent/exp_rpt_stack-selfdoc-v2 | 16% | 500 | v2-fixed | Python |
| 28 | defects4j | DCAgent/exp_rpt_defects4j-v3 | 4% / 12% gpt-5 / 8% gpt-5 2x | 464 | real bugs (rufimelo/defects4j) | Java |
| 29 | manybugs | DCAgent/exp_rpt_manybugs-v2 | 7% / **54% gpt-5 2x** | v2-minimal-dockerfile | real bugs (repairbenchmarks) | C |
| 30 | stack-csharp | DCAgent/exp_rpt_stack-csharp | 12% | original | original | C# |
| 31 | stack-ruby | DCAgent/exp_rpt_stack-ruby | 12% | original | original | Ruby |
| 32 | stack-rust | DCAgent/exp_rpt_stack-rust | 12% | original | original | Rust |
| 33 | softwareheritage | DCAgent/exp_rpt_softwareheritage-v2 | 12% | 500 | v2-fixed | Python |
| 34 | stack-pytest-gpt5mini | DCAgent/exp_rpt_stack-pytest-gpt5mini | 8% | original | original | Python |
| 35 | exercism-python | DCAgent/exp_rpt_exercism-python | 24% (mini) / 44% gpt-5 | original | original | Python |
| 36 | taco | DCAgent/exp_rpt_taco | 24% (mini) | original | competitive programming | Multi |
| 37 | codenet-python | DCAgent/exp_rpt_codenet-python | 24% (mini) | original | algorithmic problems | Python |
| 38 | stack-jest | DCAgent/exp_rpt_stack-jest-v2 | 8% | 500 | v2-fixed | JavaScript |

## Recent Fixes (2026-02-19)

### Newly working datasets (previously 0%)

| Dataset | Old Rate | New Rate (mini) | gpt-5 Rate | Fix Applied |
|---------|----------|-----------------|------------|-------------|
| bugsinpy-mf | 0% | **8%** | 0% gpt-5 | Full rewrite (v5): multi-fault bug combos from BugsInPy with self-contained tests + Dockerfile COPY fix. |
| bugsinpy | 0% | **8%** | 8% gpt-5 | Full rewrite (v7): real BugsInPy bugs + buggy starter code + import validation + Dockerfile COPY fix. |
| pymethods2test | 0% | **40%** | 20% | `from solution import *` fix + test.sh fallback to auto-find solution.py |
| unitsyn-python | 0% | **36%** | 8% | `from solution import *` fix + test.sh fallback to auto-find solution.py |
| defects4j | 0% | 4% (mini) | 12% gpt-5 / 8% gpt-5 2x | Rewrite using real rufimelo/defects4j bugs (467 bugs). Genuinely hard Java bugs. |
| manybugs | 0% | 7% (mini) | **54% gpt-5 2x** | Rewrite using real ManyBugs scenario tarballs. Needs gpt-5 + 2x timeout (30 min). |

### Key fixes explained

1. **solution.py fallback** (pymethods2test, unitsyn): Tests import `from solution import *` but agents create arbitrarily-named .py files. test.sh now auto-copies any .py to solution.py before running tests.

2. **Real bug data** (defects4j, manybugs, bugsinpy): Previously used synthetic LLM-generated bugs. Rewrote to download and use real bugs from actual benchmark datasets. LLM only generates instruction text or test cases, not the bugs themselves.

3. **Instruction clarity**: Changed prompts from "place in /app/" to "write in `/app/solution.py`" so agents know the exact filename expected.

## Dockerfile / Infrastructure Fixes (2026-03-01)

Fixed 5 previously-broken datasets that had 100% infrastructure failure (SandboxBuildFailedError, OOM, format errors). All now have 0% build errors.

### Fixed datasets

| Dataset | Old Issue | Fix Applied | New HF Repo | Nano Rate | gpt-5 Rate |
|---------|-----------|-------------|-------------|-----------|------------|
| bigcodebench | Unique Dockerfile per task (per-task `get_dockerfile_for_libs()`) | Single fat Dockerfile pre-installing ~90 packages | `exp_rpt_bigcodebench-v3` | **40%** | — |
| nemotron-bash | Verifier scripts hang (infinite loops, long sleeps) | `timeout 120` wrapper + filtering dangerous patterns (`while true`, `sleep >10s`, background forks) | `exp_rpt_nemotron-bash-v2` | **33%** | — |
| bugsinpy | `COPY solution.py` in Dockerfile made each task unique | Removed COPY, use runtime fallback in test.sh | `exp_rpt_bugsinpy-v4` | 0% | 0% |
| manybugs | Consolidated Dockerfile OOM killed (exit 137) | Minimal `FROM ubuntu:22.04` — test.sh only does diff-based validation, no compilation needed | `exp_rpt_manybugs-v2` | 0% | 0% (timeout) |
| nemotron-pytest | Invalid task.toml + missing PYTHONPATH + no import hints | `textwrap.dedent()` + `PYTHONPATH=/app` in test.sh + auto-generated import hints | `exp_rpt_nemotron-pytest-gpt5mini-v2` | **30%** | **40%** |

### Code changes

1. **`generate_bugsinpy_tasks.py` + `generate_bugsinpy_mf_tasks.py`**: Removed `COPY solution.py` line from Dockerfile generation. Now uses `dockerfile = get_dockerfile("python")` (shared image). The buggy starter code is still written to `environment/solution.py` for the test.sh runtime fallback.

2. **`generate_bigcodebench_tasks.py`**: Replaced `get_dockerfile_for_libs(libs)` with `BIGCODEBENCH_FAT_DOCKERFILE` constant that pre-installs all BigCodeBench dependencies (~90 packages from requirements-eval.txt).

3. **`generate_bash_tasks.py` (nemotron)**: Added `timeout 120` wrapper around verifier execution in `adapt_verifier_for_harbor()`. Added hang-detection filtering in `is_verifier_script()`: rejects `while true`, `sleep > 10s`, and background process forks.

4. **`generate_pytest_tasks_gpt5mini.py` (nemotron) + `commons.py`**: Fixed `create_standard_task_toml()` and `create_llm_verifier_task_toml()` to use `textwrap.dedent()`, removing leading whitespace from TOML output. Added instruction validation: skip tasks with empty or short (< 50 chars) instructions. Added `PYTHONPATH=/app` to test.sh so pytest can find agent-created modules. Added `_build_import_hint()` that parses test imports and appends a "File structure required by the tests" block to each instruction showing exact file paths and exports needed.

5. **`generate_manybugs_tasks.py`**: Replaced 7 per-project Dockerfiles + `DEFAULT_DOCKERFILE` with single minimal `MANYBUGS_DOCKERFILE` (`FROM ubuntu:22.04` + `mkdir`). The test.sh uses diff-based validation (comparing agent's fix against known-good fix), so no build tools are needed.

### Notes
- bugsinpy and manybugs have 0% pass rate even with gpt-5; these are genuinely hard tasks (real-world bug repair). Tests run cleanly — agent just can't solve them in 1 episode.
- nemotron-pytest went from 0% to **30% nano / 40% gpt-5** after adding `PYTHONPATH=/app` to test.sh and auto-generating import hints showing exact file paths.

## Previously working v2 fixes (12 datasets)

| Dataset | Old Rate | New Rate | Fix Applied |
|---------|----------|----------|-------------|
| stack-php | 0% | 100% | PHPUnit PHAR download instead of Composer |
| bigcodebench | 0% | 54% | Python stdlib filter + libs parsing via ast.literal_eval |
| codeelo | 0% | 52% | Removed openjdk from Dockerfile, use train split |
| crosscodeeval-python | 28% | 48% | Fresh regen with fixed generator |
| e2egit | 0% | 48% | Removed Xvfb/selenium, LLM converts E2E to pytest unit tests |
| stack-pytest | 0% | 40% | Strict import filtering + PYTHONPATH fix |
| methods2test | 0% | 28% | Java base image fix (eclipse-temurin:17-jdk-jammy) |
| codereval-python | 0% | 24% | Fixed placeholder tests, real test extraction from CoderEval |
| stack-dockerfile | 0% | 16% | LLM Dockerfile rewrite + input filtering |
| stack-selfdoc | 0% | 16% | Python3 validation + dependency detection |
| softwareheritage | 0% | 12% | Fresh regen with fixed generator |
| stack-jest | 0% | 8% | Added bsdutils + better test file filtering |

## Breakdown by Language

| Language | Datasets | Count |
|----------|----------|-------|
| Python | bigcodebench, crosscodeeval-python, e2egit, stack-pytest, codereval-python, stack-selfdoc, softwareheritage, stack-pytest-gpt5mini, pymethods2test, unitsyn, bugsinpy, bugsinpy-mf, r2egym-* (5) | 17 |
| Java | stack-junit, crosscodeeval-java, methods2test, defects4j | 4 |
| Bash | stack-bash, stack-bash-withtests, stack-bash-withtests-gpt5mini | 3 |
| C# | crosscodeeval-csharp, stack-csharp | 2 |
| C | manybugs | 1 |
| C++ | stack-cpp | 1 |
| JavaScript | stack-jest | 1 |
| TypeScript | crosscodeeval-typescript | 1 |
| PHP | stack-php | 1 |
| Ruby | stack-ruby | 1 |
| Rust | stack-rust | 1 |
| Dockerfile | stack-dockerfile | 1 |
| Multi | codeelo (competitive programming) | 1 |

## Datasets NOT included (and why)

| Dataset | Pass Rate | Reason Excluded |
|---------|-----------|-----------------|
| ghactions | 0% | Tasks too hard, agent timeouts |
| bugswarm | 0% | Only 2 tasks generated |
| swebench | -- | Skipped (too complex for this pipeline) |
| stack-go | 0% | Tasks too hard even after infra fix |
| r2egym-impossible | 0% | Expected to be impossible |

## Notes

- Pass rates measured on 25-task samples with gpt-5-mini (upgraded from gpt-5-nano)
- Datasets marked "v3-rewrite" had generators completely rewritten to use real data
- Datasets marked "v3-fixed" had test infrastructure fixes (solution.py fallback)
- **manybugs** is solvable with gpt-5 + 2x timeout (54%), but only 7% with gpt-5-mini at 1x. 6/24 still timeout even at 30 min.
- **defects4j** is genuinely hard: 4% mini, 12% gpt-5, 8% gpt-5 2x. Java bugs remain stubbornly difficult regardless of model/timeout.
- codereval-python has 230 tasks (entire source dataset), all others have ~500
- "original" datasets are on HF under their base repo names
- "v2-fixed" datasets are on HF with `-v2` suffix appended to repo name
- "v3-fixed/rewrite" datasets are on HF with `-v3` suffix appended to repo name
- Machine-readable manifest: `data/all_good_datasets.json`
