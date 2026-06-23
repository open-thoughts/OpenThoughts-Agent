# JSC Jupiter Access

**SSH**: `ssh Jupiter` (alias in `~/.ssh/config`). User `feuer1`, group `jureap59`. If the alias isn't
configured, IPv4 is required: `ssh -i ~/.ssh/id_ed25519_jsc feuer1@login01.jupiter.fz-juelich.de -4`.

**Cluster**: GH200 96GB GPUs (aarch64), 4/node, 48 nodes, SLURM. **No internet on compute nodes** (proxy
via SSH tunnel on compute; **login nodes have direct internet / HF Hub access**) — pre-download
datasets/models on the login node before submitting jobs.

**Non-interactive SSH**: `$DCFT_ACTIVATE_ENV` does NOT work over non-interactive SSH — use full paths, e.g.
```bash
ssh Jupiter '/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python ...'
```

**Tmux**: sessions persist across disconnects — `tmux ls`; `tmux attach -t 2` (main work session).

**Pre-launch preamble** (run before launching any job — pulls latest code; `GIT_TERMINAL_PROMPT=0`
prevents interactive-auth hangs):
```bash
source ~/.bashrc; source ~/secrets.env; \
cd /e/scratch/jureap59/feuer1/harbor && git stash && git pull; \
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent/SkyRL && git stash && git pull; \
conda activate otagent; \
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent && GIT_TERMINAL_PROMPT=0 git pull && \
git submodule update --init --remote sft/llamafactory; \
source hpc/dotenv/jupiter.env
```

**Key paths**:
- Code (`$DCFT`): `/e/scratch/jureap59/feuer1/OpenThoughts-Agent` — experiments in `experiments/`, eval logs
  in `eval/jupiter/logs/`, dotenv `hpc/dotenv/jupiter.env`.
- Harbor: `/e/scratch/jureap59/feuer1/harbor`
- Conda env: `/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/`
- **Personal data root (`$DCFT_DATA`) — USE THIS**: `/e/data1/datasets/playground/ot-baf`
  → HF cache (`$HF_HUB_CACHE`/`$HF_HOME`) `…/ot-baf/hf_hub`, checkpoints (`$CHECKPOINTS_DIR`)
  `…/ot-baf/checkpoints/`, wheels `…/ot-baf/wheels/`.
- Eval job files: `/e/data1/datasets/playground/ot/eval_jobs/`
- **Legacy shared data — avoid for new WRITES**: `/e/data1/datasets/playground/ot` (owned by `nezhurina1`;
  its xet/datasets cache subdirs were created by other users with `0755`, causing `Permission denied` on HF
  Xet uploads + dataset lock files). Read-only references to existing `/ot` artifacts are fine.

**Job management** (SLURM): `sqme` (your queued/running jobs), `squeue -u feuer1` (detailed), `scancel <job_id>`.

**Rsync files to local** (from Mac):
```bash
rsync -avz --progress -e "ssh -i ~/.ssh/id_ed25519_jsc -4" \
  feuer1@login01.jupiter.fz-juelich.de:/remote/path /local/path
```

> Which **runtime / conda env / SIF** to use for which workstream (RL venv vs the MoE/0.20.2rc0 SIFs vs
> `otagent`/`sft-qwen35`) lives in **`ENVIRONMENT_MAP.md`** in this directory — not duplicated here.

---

# Filesystem & GPFS hygiene

- **Never run `find` or `du` on Jupiter GPFS** (`/e/scratch`, `/e/data1`) — `stat`-walks are very slow and stall the SSH session for minutes. To locate logs/dirs use canonical paths + depth-1 `ls -td <dir>/*JOBID*`, `ls | wc -l`, or `squeue -j JOBID -o '%Z'` (the `%Z` workdir). Same caution on Perlmutter/Leonardo parallel FS.
- **Cleanup subagents must bake these rules into their prompt** (subagents don't inherit memory): never `du`/`find` to size or locate; **detach** the long `rm -rf` (`nohup … &` or a tmux session logging `RM_DONE <dir> exit=$?`) and let the agent EXIT — do NOT poll a multi-hundred-thousand-file GPFS delete to completion (it's idempotent/resumable). History: a `du -sh` on a `trace_jobs/` tree hung 75min; two synchronous `rm -rf` blocked cleanup agents 2.5h — and all the important checklist steps had already completed.

## Inode quota (the bind that bites) — EDQUOT can masquerade as sig53
`/e/scratch/jureap59` has a **project-shared inode quota** (~8.0M soft / 8.8M hard, shared across all jureap59 members, 2–4h lag). Datagen jobs create thousands of trial subdirs → we hit inodes long before the data limit.
- Inspect: `jutil project dataquota -p jureap59 | grep exa_scratch` (project), `df -i /e/scratch` (live), `du -s --inodes <subdir>` (mine — but avoid on huge trees).
- **EDQUOT presents as a sig53 sbatch failure** (9–13s, exit `0:53`, empty `logs/` dir): the kernel can't create the `.out` → SIGRTMIN+19. Diagnostic separation — if the **launcher python** errors at `paths.sbatch.mkdir()` with `OSError: [Errno 122] Disk quota exceeded` → EDQUOT; if the launcher succeeds but Slurm reports sig53 with NO `.out` at all → lean true trap (see below).
- **Over-soft writes only work during the GPFS grace period** (~7 days). Once grace expires, every new-inode op fails as if over-hard even while "under hard" — verify with `touch <existing-dir>/probe_$$`. Don't revert `OT_AGENT_RAY_LOG_DIR`/scratch-dodge patches on "under hard" alone.
- **Freeing inodes (order):** `rm -rf ~/.cache/uv` (biggest disposable target — uv extract cache, rebuilds itself; freed 131k and cleared EDQUOT once), then `~/.cache/{pip,wandb,torch,curator,flashinfer}`, then old experiment dirs. Always re-survey your own usage before accepting an "it's other members'/out-of-hands" diagnosis.
- Last-resort dodge: `--experiments_dir /e/data1/datasets/playground/ot-baf/experiments` + `OT_AGENT_RAY_LOG_DIR=/e/data1/...` to avoid `/e/scratch` entirely (needs the `ray_utils.py:520` patch, commit `122cae2d`).

## Inode allocations — counts + maxima per Jupiter allocation (CHECK EACH SWEEP) {#inode-allocations}
Inodes (file/dir COUNT), not bytes, are the binding constraint on Jupiter — datagen/eval create thousands of tiny task/trial files. **Check via `jutil project dataquota -p <project>` (the `inode-usage / inode-soft-limit / inode-hard-limit` columns) + `df -i /e/data1 /e/scratch`.** Per-allocation limits (soft → hard; over-soft only works during the GPFS ~7-day grace, then fails as if over-hard):

| Allocation (path) | Project | inode soft | inode hard | typical use |
|---|---|---|---|---|
| `/e/data1/datasets` (`exa_data1`) | **datasets** (SHARED, `hagemeier2:datasets`) | **110M** | **121M** | our `…/playground/ot-baf` lives here |
| `/e/scratch/jureap59` (`exa_scratch`) | jureap59 | 8.0M | 8.8M | RL/datagen scratch (the EDQUOT-sig53 area above) |
| `/e/project1/jureap59` (`exa_project1`) | jureap59 | 4.0M | 4.4M | — |
| `/e/scratch/laionize` | laionize | 8.0M | 8.8M | — |
| `/e/project1/laionize` (`exa_project1`) | laionize | 4.0M | 4.4M | — |
| `/p/project1/{jureap59,laionize}` | — | 3.0M / 6.0M | 3.3M / 6.6M | — |

**`/e/data1/datasets/playground/ot-baf` is the chronic offender.** The `datasets` project is **SHARED across all its members** and has run **OVER the 110M soft limit (~118M used, ~98% of the 121M hard)** — when it hits hard, *everyone's* writes fail. Our footprint is dominated by per-experiment **`trace_jobs/` + `tasks/` subtrees** (each = thousands of tiny trial/task dirs). **The standing rule: a cleanup is NOT done until the artifact dir is actually `rm`'d — uploading to HF then leaving the trace/task tree on disk is the #1 inode leak** (subagents habitually skip the delete). After any RL/SFT/datagen/eval cell is archived to HF, its `trace_jobs/`/`tasks/`/`exports`-already-pushed subtrees MUST be deleted (detached `rm`, per the GPFS-delete discipline above), and inode reclaim verified with `df -i`/`jutil`.

## sbatch signal-53 trap (true cluster-side variant)
Distinct from EDQUOT: `sbatch` returns a JID, RUNS 9–18s, then FAILS `0:53` `Reason=RaisedSignal:53(Real-time_signal_19)`, **no log file at all** (script's first line never runs). `srun` from the same shell works fine; already-running sbatches keep running — affects NEW submissions only. Per-user/per-account, not per-node. Ruled out: reservation, account, node count, cpus-per-task, mail dirs, `--export=NONE`, WorkDir, `--exclude`, even raw `--wrap='echo hello'`. **Probe:** `sbatch --reservation=reformo --account=reformo --partition=booster --time=00:02:00 --nodes=1 --gres=gpu:4 --wrap='echo hello'` — if it FAILs 0:53 the trap is active → fall back to `srun` for one-offs, or use `python -m hpc.launch` (its submission path has been observed healthy). Untried next steps: fresh login shell, different login node, CPU-only sbatch, JSC support (slurmstepd/spank-side). First seen 2026-04-29.

## Ray bootstrap transients (NOT code/config bugs)
A fresh RL launch can die during Ray bring-up; these are transient infra, recovered by the `afterany` restart chain (don't manually resubmit — risks the ≤6 RUNNING-RL cap):
- **Cold-start DNS race:** head exits `code 255` (before writing its `ray_head_<node>.log`) OR driver reports `Ray cluster did not reach desired resources within 600 seconds`. Cause: Ray's `get_node_ip_address()` probes external DNS (8.8.8.8) for the local IP; compute nodes have no internet → ~49s timeout → late GCS → workers never all register in the 600s window. Amplified by two multi-node clusters bootstrapping in the same minute → **stagger launches**. Durable (unvalidated) fix: explicit `--node-ip-address` on head+worker `ray start` / skip the DNS probe / raise `wait_for_cluster`.
- **SLURM node-prolog wedge:** job sits `RUNNING Reason=Prolog` for HOURS, `.batch` never launches → **NO `.out` at all** (empty `_N/logs/`), GPUs idle. Signature = RUNNING but `*.out` absent/empty after ~2–3min → **scancel + resubmit FAST** (new allocation draws different nodes); don't let it hold the allocation.
- **login01 fork-saturation → FALSE empty squeue:** the `Jupiter` alias is `login01`, which periodically fork-saturates (`fork: Resource temporarily unavailable`, ssh exit 254/127) → `ssh Jupiter "squeue"` returns EMPTY = a false "drained". **Re-check via login02/03/04** (`ssh -i ~/.ssh/id_ed25519_jsc -o AddressFamily=inet feuer1@login02.jupiter.fz-juelich.de "<cmd>"`). Keep ssh commands SIMPLE (single inline string) — nested loops / `$VAR="ssh…"` indirection exit 127 under this shell.

## Compiled DP>1 illegal-memory-access = MNNVL fused allreduce
MiniMax-M2.7-AWQ / GLM-4.7-AWQ compiled (cudagraphs ON) at **DP>1** crash with `CUDA driver error: an illegal memory access` in `profile_cudagraph_memory` during startup capture (DP=1 is fine). Cause: vLLM's `fuse_allreduce_rms` pass swaps in flashinfer's `trtllm_mnnvl_allreduce_fusion` (Multi-Node NVLink) kernel, but Jupiter's cross-node transport is **InfiniBand** → writes to a non-existent NVLink peer. **Fix (one flag):** `--compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'`. The MoE all-to-all is a red herring. Diagnostic that cracked it: `CUDA_LAUNCH_BLOCKING=1` (propagate to the cross-node Ray DP actor via `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=CUDA_LAUNCH_BLOCKING`) → synchronous traceback names the kernel.

# Debugging tooling (SIF / Ray-actor / multi-node hangs)
Hard-won during the #232 long-ctx RL TP-rank-desync wedge (2026-06-18). What works vs doesn't inside the Apptainer SIF + Ray-actor vLLM workers:

- **`ptrace` is BLOCKED in the SIF — py-spy AND gdb both fail** with "Operation not permitted" *even on a self-spawned child* (`/proc/sys/kernel/yama/ptrace_scope=2` + `CapEff=0`, no `SYS_PTRACE` cap). Do NOT burn time trying `py-spy dump`/`gdb -p` on a wedged worker; they cannot attach. (Untried: an `apptainer exec --add-caps CAP_SYS_PTRACE` re-enter, or a `--cap-add` at submit — may or may not be permitted.)
- **In-process stack capture is the substitute that WORKS (no ptrace):** `faulthandler`. Arm `faulthandler.dump_traceback_later(SECS, repeat=True, file=<per-rank file>)` at worker init (SECS below the NCCL+vLLM watchdogs, e.g. 240-300s; healthy steps are ~20-40s so it only fires on a real hang), and/or install a `SIGUSR1 → faulthandler.dump_traceback` handler so you can `kill -SIGUSR1 <pid>` a live wedge. This is the ONLY way to get the **lagging rank's** Python stack in a TP desync (see below).
- **`/proc/<pid>/environ` is readable without ptrace** — use it to VERIFY an env var actually reached a worker process (we caught `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` NOT propagating to the Ray-actor engine workers this way).
- **Env vars don't reach Ray-actor-spawned vLLM workers via `APPTAINERENV_` alone.** The engine EngineCore/mp workers are Ray actors; they inherit the driver's env only through the Ray `runtime_env.env_vars` passthrough (SkyRL `ray_wrapped_inference_engine._build_inference_engine_runtime_env`), with TP child actors via `placement_group_capture_child_tasks=True`. Setting `APPTAINERENV_FOO` gets it to the driver, NOT the collective-running workers — always `/proc`-verify on a worker pid.
- **NCCL flight recorder (FR) — useful but has THREE gotchas:** (1) var is `TORCH_FR_BUFFER_SIZE` now (deprecated `TORCH_NCCL_TRACE_BUFFER_SIZE` auto-maps); enable `TORCH_NCCL_DUMP_ON_TIMEOUT=1` + a writable `TORCH_NCCL_DEBUG_INFO_TEMP_FILE=<dir>/rank` (torch appends rank). (2) vLLM's `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900` **preempts NCCL's default 1800s watchdog** → NCCL never reaches its dump path. To get an FR dump, drop `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` BELOW 900 (e.g. 600). (3) **FR only dumps a rank that is IN a collective** — in a TP desync where the lagging rank *never issues* the collective the others block on, that rank has nothing in-flight → no dump on the rank you most need. FR catches the *blocked* ranks (0/1), not the *diverged* one (rank2). For the diverged rank, use faulthandler.
- **NCCL `COLL` trace lines as a stack substitute:** with `NCCL_DEBUG=INFO` / FR-on, each rank logs `AllReduce/AllGather: opCount … count … comm …`. Aligning the per-rank op streams by `opCount` pinpoints WHICH collective desyncs (the lagging rank stops emitting at op N while peers advance to op N+1). This cracked the #232 desync localization when py-spy + FR both failed. (Caveat: these verbose lines contain the substring "opCount dead" — a hex/marker, NOT an error — which falsely trips naive `grep dead`/EngineDead monitors; match real tokens only: `EngineDeadError`, `execute_model timed out`, `Watchdog`.)
- **Two independent watchdogs:** vLLM's `execute_model` RPC timeout (`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, multiproc/Ray executor) vs NCCL's collective heartbeat (`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`). A multi-node hang trips whichever is shorter; tune their relative values to control which fires first + whether you get a dump. A 900s `execute_model` timeout = a genuinely *wedged* step (normal steps are ms–seconds), not slow compute.
