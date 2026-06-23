# Local (this Mac) — ops

The control machine: where launches, uploads, Supabase queries, code edits, and analysis run from. Clusters
do the GPU work; the Mac is the orchestrator (commit→push→`git pull` on the cluster — see
`.claude/projects/marinskyrl/marinskyrl.md` sync discipline). Captured 2026-06-14; re-probe specs if acting
on this months later.

## System
- **macOS 26.5.1** (build 25F80), **Apple M4 Max** (`Mac16,5`), **arm64**.
- **14 CPU cores** (10 performance + 4 efficiency), **36 GB** unified RAM.
- **No local NVIDIA GPU** — all CUDA/training/vLLM-serving work happens on the clusters. Local GPU paths (MPS) are only for tiny smoke tests; treat the Mac as CPU-only for ML purposes.

## Disk
- Single 926 GB internal volume. **~188 GB free** (Data volume 80% full, ~707 GB used) as of capture.
- Implication: the four code checkouts + HF caches + `agent_logs`/`notes` are the bulk of local usage. **Do NOT pull large model weights / trace datasets to the Mac** — stage those on cluster scratch. Keep local to code + notes + logs.

## Homebrew
- **Homebrew 5.1.14**, prefix `/opt/homebrew` (arm64; `brew` at `/opt/homebrew/bin/brew`).
- **Leaf formulae:** `agg asciinema cmake codex coreutils gh git-lfs go helm helm@3 k3d kubelogin kubernetes-cli kustomize llama.cpp marp-cli netcat pandoc pango poppler rpm rustup socat step tectonic wget`.
- **Casks:** `docker-desktop`, `gcloud-cli`, fonts (`font-lato`, `font-raleway`).
- Load-bearing tools: `gh` (GitHub PRs/issues), `git-lfs` (HF repos), `step` (CINECA/Leonardo step-ca SSH certs), `gcloud-cli` (Iris/TPU + GCS), `kubernetes-cli`/`helm`/`k3d` (k8s cloud path), `tectonic` (LaTeX), `coreutils` (GNU `g*` tools), `wget`, `pandoc`.

## Conda
- **conda 25.5.1**, base at **`/Users/benjaminfeuer/miniconda3`** (base: Python 3.13.5, torch 2.8.0 — don't train in base).
- **`otagent` is the canonical env for all OT-Agent / Harbor / SkyRL work** (Python 3.12.12, torch 2.9.0). Per repo `CLAUDE.md`, **use the full interpreter path — symlinks don't work in the sandbox:**
  ```
  /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python
  ```
- Other envs (`conda env list`): `harbor` (3.13.9), `llama-factory` (3.12.12), `marin` (3.13.13), `oumi` (3.12.11), `ajudge`, `marvis`, `abb`, `openreview`, `sweagent`, `tokviz-rt`. `otagent` is the default unless a task is specific to one of these (e.g. `ajudge` for the AJudge project — see `.claude/projects/ajudge/`).
- Syntax/lint check: use the IDE MCP `mcp__ide__getDiagnostics`, not `python -m py_compile`/`flake8` (per repo CLAUDE.md — bash output capture is unreliable).

## Local codebases (`/Users/benjaminfeuer/Documents/`)
All editable-installed into the relevant conda env; **edit here, commit, push, then `git pull` on the cluster** (never patch on the cluster). Branch SoT per repo:
- **`OpenThoughts-Agent/`** — the launcher/orchestration repo. Branch `penfever/working` (origin `open-thoughts/OpenThoughts-Agent`). See `.claude/projects/ot-agent/`.
- **`harbor/`** — agent framework. Branch `penfever/working`, remote `marin` = `marin-community/harbor`. See `.claude/projects/harbor/`.
- **`MarinSkyRL/`** — RL framework. Branch `penfever/working` (= `marin-community/MarinSkyRL`). See `.claude/projects/marinskyrl/`.
- **`vllm/`** — inference-engine fork (`mlfoundations/vllm`, our own upstream). Currently on `feuer/dcp-gqa-lse-fix`. Local clone is ground truth; it's compiled, so it's **built from source on each cluster (per-arch) from the committed fork** (or baked into a SIF from a committed commit) — **never** rsync'd working-tree edits or hand-patched. Every cluster keeps an env with our fork built for it; some envs may run vanilla vLLM. See `.claude/projects/vllm/`.

## Other local paths
- **`agent_logs/`** — dated investigation/post-mortem logs, `YYYY-MM-DD_<topic>.md` (~109 files). Write a dated entry here when diagnosing a genuine FAILED job (per the cron-sweep skill).
- **`experiments/`** — per-experiment working state: one subdir per experiment/series, each with its own tracker(s). See `.claude/ops/experiments/ops.md`.
- **`notes/`** — the private knowledge base (~17 subdirs: `RL/`, `marin/`, `harbor/`, `llama-factory/`, `vllm/`, `ot-agent/`, `jsc/`, `nvidia/`, `scaling_laws_papers/`, … + many single-file notes). Canonical source-of-truth for several things the `.claude/` docs mirror (e.g. the MiniMax datagen tracker, the pinggy bank). Per-cluster notes: `notes/leonardo.md`, `notes/jsc/`, `notes/perlmutter` (in `ot-agent/`).
- **`secrets.env`** → **`/Users/benjaminfeuer/Documents/secrets.env`** (NOT `~/secrets.env` — that's the *cluster* convention). 48 lines. Load so a subprocess inherits the vars:
  ```bash
  set -a; source /Users/benjaminfeuer/Documents/secrets.env; set +a
  ```
- **`.claude/secret.md`** (untracked, gitignored) — privileged values pulled out of the committable skills/ops docs (pinggy URL+token bank, Leonardo HedgeDoc URL). Referenced by name from those docs.

### secrets.env — key NAMES only (values never leave the file)
Provides credentials for: HuggingFace (`HF_TOKEN`), **Daytona** (`DAYTONA_API_KEY`, `DAYTONA_B_KEY`,
`DAYTONA_DATA_API_KEY`, `DAYTONA_RL_API_KEY` — datagen uses `DAYTONA_DATA_API_KEY`), **Supabase**
(`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_ANON_KEY`), W&B (`WANDB_API_KEY`), OpenAI
(`OPENAI_API_KEY` — see the LLM-judge datagen note re: the OUMI key), **LAION S3** (`LAION_ENDPOINT`,
`LAION_BUCKET_NAME`, `LAION_ACCESS_KEY`, `LAION_SECRET_KEY`), **Marin GCS HMAC** (`MARIN_HMAC_ACCESS_ID`,
`MARIN_HMAC_SECRET`, `MARIN_PREFIX`), AWS (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), Docker
(`DOCKER_USER_ID`, `DOCKER_TOKEN`), **Pinggy** (`PINGGY_API_KEY`, `PINGGY_IDENTITY_FILE`), Modal
(`MODAL_PROFILE`), Together (`TOGETHER_API_KEY`), Vast (`VAST_API_KEY`), OpenReview
(`OPENREVIEW_USERNAME`, `OPENREVIEW_PASSWORD`), and `SSH_KEY`. **Reference these by name only** in any
committable doc — never paste a value (the `.claude/secret.md` convention).

## Cluster SSH aliases (`~/.ssh/config`)
`Jupiter`, `Leonardo`, `perlmutter`, `torch` (NYU), `TACCVista`, `ALCFPolaris`, `OLCFFrontier`,
`TUDCapella`, `EmpireAI_Alpha1`, `OumiLambdaSLURM`, `OracleCloud`, plus NYU `hegde-lambda-{1,2}`. Active
cron scope is **Jupiter + Leonardo** (Perlmutter dropped). Per-cluster access/paths/preamble live in
`.claude/ops/<cluster>/`. Leonardo uses a `step ssh certificate` cert that expires ~12h (refresh from here;
see `.claude/ops/leonardo/ops.md`).
