# JSC Jupiter Access

> **⛔ HARD DOWNTIME — MDC maintenance Jun 23 → Jul 12, 2026** ("Critical Incident" / data-center acceptance testing). ALL login nodes (login01–04) print "This node is in maintenance" and **close the connection before any command runs** — fully inaccessible, not queue-gated. **Sweeps skip Jupiter** until access is re-verified (~Jul 12; re-check, may shift). Stranded while down: EP/R3 debug chain (953566 EPDIAG2 → `sel_rows`) + 7 flawed-summ campaign resumes (953480/82/88/91, 953500/03/11) — all preserve valid trials, re-resume on return (do NOT migrate mid-flight). R3-OFF is the working RL fallback meanwhile.

**SSH**: `ssh Jupiter` (alias in `~/.ssh/config`). User `feuer1`, group `jureap59`. Fallback (IPv4 required): `ssh -i ~/.ssh/id_ed25519_jsc feuer1@login01.jupiter.fz-juelich.de -4`.

**Cluster**: GH200 96GB GPUs (aarch64), 4/node, 48 nodes, SLURM. **No internet on compute nodes** (proxy via SSH tunnel); **login nodes have direct internet / HF Hub** → pre-download datasets/models on a login node before submitting.

**Non-interactive SSH**: `$DCFT_ACTIVATE_ENV` does NOT work — use full paths:
```bash
ssh Jupiter '/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python ...'
```

**Tmux** (persists across disconnects): `tmux ls`; `tmux attach -t 2` (main work session).

**Pre-launch preamble** (before any job — pulls latest code; `GIT_TERMINAL_PROMPT=0` blocks interactive-auth hangs):
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
- Code (`$DCFT`): `/e/scratch/jureap59/feuer1/OpenThoughts-Agent` — **tracks `origin/main`** (switched from `penfever/working` 2026-08-29, owner instruction; the preamble's `git pull` keeps it current). If `origin/main` falls behind `origin/penfever/working`, check that these Jupiter launcher fixes are present before launching RL: rollout-section rendering (`569b0d14`), `++` terminal-bench overrides (`274c9a6c`), wandb-offline→console downgrade (`778df5c8`), skyrl-gym container PYTHONPATH (`3e7b3540`) — missing any of them kills RL restart links (6h RPC watchdog / hydra composition / `wandb.init` hang / `skyrl_gym.verification` import). Experiments in `experiments/`, eval logs in `eval/logs/` (per `eval/clusters/jupiter.yaml` `eval_logs_dir`), dotenv `hpc/dotenv/jupiter.env`.
- Harbor: `/e/scratch/jureap59/feuer1/harbor`
- Conda env: `/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/`
- **Personal data root (`$DCFT_DATA`) — USE THIS**: `/e/data1/datasets/playground/ot-baf`
  → HF cache (`$HF_HUB_CACHE`/`$HF_HOME`) `…/ot-baf/hf_hub`, checkpoints (`$CHECKPOINTS_DIR`) `…/ot-baf/checkpoints/`, wheels `…/ot-baf/wheels/`.
- Eval job files: `/e/data1/datasets/playground/ot/eval_jobs/`
- **Legacy shared data — avoid for new WRITES**: `/e/data1/datasets/playground/ot` (owned by `nezhurina1`; subdirs `0755` from other users → `Permission denied` on HF Xet uploads + dataset locks). Read-only references to existing `/ot` artifacts are fine.

**Job management** (SLURM): `sqme` (queued/running), `squeue -u feuer1` (detailed), `scancel <job_id>`.

**Rsync to local** (from Mac):
```bash
rsync -avz --progress -e "ssh -i ~/.ssh/id_ed25519_jsc -4" \
  feuer1@login01.jupiter.fz-juelich.de:/remote/path /local/path
```

> Which **runtime / conda env / SIF** for which workstream (RL venv vs MoE/0.20.2rc0 SIFs vs `otagent`/`sft-qwen35`) → **`ENVIRONMENT_MAP.md`** in this directory.

---

# Filesystem & GPFS hygiene

- **Never `find` or `du` on Jupiter GPFS** (`/e/scratch`, `/e/data1`) — `stat`-walks stall the SSH session for minutes. Locate logs/dirs via canonical paths + depth-1 `ls -td <dir>/*JOBID*`, `ls | wc -l`, or `squeue -j JOBID -o '%Z'` (the `%Z` workdir). Same caution on Perlmutter/Leonardo parallel FS.
- **Cleanup subagents must bake these rules into their prompt** (no inherited memory): never `du`/`find` to size or locate; **detach** long `rm -rf` (`nohup … &` or a tmux session logging `RM_DONE <dir> exit=$?`) and EXIT — do NOT poll a multi-hundred-thousand-file GPFS delete (idempotent/resumable).

## Image-backed RL trial artifacts

RL configs can set `container.artifact_store.enabled: true`. The launcher creates a sparse ext4 image under the durable inner run root, and each chain link mounts it below node-local `/tmp/otagent-artifact-stores` on the Ray head before Apptainer and Ray start. The live mount cannot reside below GPFS: Jupiter's `fusermount` refuses that filesystem even when `fuse2fs` reports success. The SkyRL entrypoint and Harbor coordinators are pinned to the mount node. Check `artifact_authority.json` for the active Slurm job, node, mount, and trial paths.

Use `rw,fakeroot`, not `allow_other`, for the writer mount. A new ext4 root is owned by UID 0, so `fakeroot` is required for the submitting user to create `trace_jobs`. Jupiter does not enable `user_allow_other` in `/etc/fuse.conf`, and every process that needs the mount already runs as the submitting user.

Never mount `artifact_store.img` from a login node while its Slurm link is running. The image has one read-write authority, enforced across hosts by `artifact_store.img.mount-lease` and on one host by `artifact_store.img.lock`; a second mount can observe inconsistent ext4 metadata. Live inspection must run through the recorded node and mount:

```bash
srun --mpi=none --overlap --jobid <job_id> -w <node> -N1 -n1 ls <mount>/trace_jobs
```

After the link exits, repository readers mount the image read-only and release it automatically. For an ad-hoc inspection, use the same helper instead of a loop mount:

```bash
python -m hpc.artifact_store <run>/artifact_store.img
```

Every link requests `SIGTERM` 180 seconds before walltime. The batch shell forwards it to MarinSkyRL, waits for cooperative shutdown, then syncs and unmounts the image. A successor obtains the same writer lock, runs `e2fsck -p`, and remounts before resume.

## Inode quota (the binding constraint) — EDQUOT can masquerade as sig53
`/e/scratch/jureap59` has a **project-shared inode quota** (~8.0M soft / 8.8M hard, shared across all jureap59 members, 2–4h lag). Datagen jobs create thousands of trial subdirs → inodes bind long before bytes.
- **Inspect:** `jutil project dataquota -p jureap59 | grep exa_scratch` (project), `df -i /e/scratch` (live), `du -s --inodes <subdir>` (mine — avoid on huge trees).
- **EDQUOT presents as a sig53 sbatch failure** (9–13s, exit `0:53`, empty `logs/`): kernel can't create the `.out` → SIGRTMIN+19. Separation: launcher failing at `paths.sbatch.mkdir()` with `OSError: [Errno 122] Disk quota exceeded` → EDQUOT; launcher succeeds but Slurm reports sig53 with NO `.out` → lean true trap (below).
- **Over-soft writes only work during the GPFS grace period** (~7 days). Once grace expires, every new-inode op fails as if over-hard even while "under hard" — verify with `touch <existing-dir>/probe_$$`. Don't revert `OT_AGENT_RAY_LOG_DIR`/scratch-dodge patches on "under hard" alone.
- **Freeing inodes (order):** `rm -rf ~/.cache/uv` (biggest disposable — uv extract cache, rebuilds itself), then `~/.cache/{pip,wandb,torch,curator,flashinfer}`, then old experiment dirs.
- **Last-resort dodge:** `--experiments_dir /e/data1/datasets/playground/ot-baf/experiments` + `OT_AGENT_RAY_LOG_DIR=/e/data1/...` to avoid `/e/scratch` entirely (needs the `ray_utils.py:520` patch).

## Inode allocations — per Jupiter allocation (CHECK EACH SWEEP) {#inode-allocations}
**Check via `jutil project dataquota -p <project>` (the `inode-usage / inode-soft-limit / inode-hard-limit` columns) + `df -i /e/data1 /e/scratch`:**

| Allocation (path) | Project | inode soft | inode hard | typical use |
|---|---|---|---|---|
| `/e/data1/datasets` (`exa_data1`) | **datasets** (SHARED, `hagemeier2:datasets`) | **110M** | **121M** | our `…/playground/ot-baf` lives here |
| `/e/scratch/jureap59` (`exa_scratch`) | jureap59 | 8.0M | 8.8M | RL/datagen scratch (EDQUOT-sig53 area above) |
| `/e/project1/jureap59` (`exa_project1`) | jureap59 | 4.0M | 4.4M | — |
| `/e/scratch/laionize` | laionize | 8.0M | 8.8M | — |
| `/e/project1/laionize` (`exa_project1`) | laionize | 4.0M | 4.4M | — |
| `/p/project1/{jureap59,laionize}` | — | 3.0M / 6.0M | 3.3M / 6.6M | — |

**`/e/data1/datasets/playground/ot-baf` is the chronic offender.** The `datasets` project is **SHARED across all members** and has run **OVER the 110M soft limit (~118M used, ~98% of the 121M hard)** — at hard, *everyone's* writes fail; our footprint is dominated by per-experiment **`trace_jobs/` + `tasks/` subtrees**. **Standing rule: a cleanup is NOT done until the artifact dir is `rm`'d — uploading to HF then leaving the trace/task tree on disk is the #1 inode leak.** After any RL/SFT/datagen/eval cell is archived to HF, its `trace_jobs/`/`tasks/`/`exports`-already-pushed subtrees MUST be deleted (detached `rm`); verify reclaim with `df -i`/`jutil`.

## sbatch signal-53 trap (true cluster-side variant)
Distinct from EDQUOT: `sbatch` returns a JID, RUNS 9–18s, then FAILS `0:53` `Reason=RaisedSignal:53(Real-time_signal_19)`, **no log file at all** (script's first line never runs). `srun` from the same shell works; already-running sbatches keep running — affects NEW submissions only. Per-user/per-account, not per-node. Ruled out: reservation, account, node count, cpus-per-task, mail dirs, `--export=NONE`, WorkDir, `--exclude`, raw `--wrap='echo hello'`. **Probe:**
```
sbatch --account=reformo --partition=booster --time=00:02:00 --nodes=1 --gres=gpu:4 --wrap='echo hello'
```
FAIL 0:53 → trap active → fall back to `srun` for one-offs, or use `python -m hpc.launch` (its submission path has been observed healthy). Untried: fresh login shell, different login node, CPU-only sbatch, JSC support.

## Ray bootstrap transients (NOT code/config bugs)
A fresh RL launch can die during Ray bring-up; these are transient infra, recovered by the `afterany` restart chain (don't manually resubmit — risks the ≤6 RUNNING-RL cap):
- **Cold-start DNS race:** head exits `code 255` (before writing `ray_head_<node>.log`) OR driver reports `Ray cluster did not reach desired resources within 600 seconds`. Cause: Ray's `get_node_ip_address()` probes external DNS (8.8.8.8); no compute internet → ~49s timeout → late GCS → workers miss the 600s window. Amplified when two multi-node clusters bootstrap in the same minute → **stagger launches**. (Unvalidated fix: explicit `--node-ip-address` on head+worker `ray start` / skip the DNS probe / raise `wait_for_cluster`.)
- **SLURM node-prolog wedge:** job sits `RUNNING Reason=Prolog` for HOURS, `.batch` never launches → **NO `.out` at all** (empty `_N/logs/`), GPUs idle. Signature = RUNNING but `*.out` absent/empty after ~2–3min → **scancel + resubmit FAST** (new allocation draws different nodes).
- **login01 fork-saturation → FALSE empty squeue:** the `Jupiter` alias is `login01`, which periodically fork-saturates (`fork: Resource temporarily unavailable`, ssh exit 254/127) → `ssh Jupiter "squeue"` returns EMPTY = a false "drained". **Re-check via login02/03/04** (`ssh -i ~/.ssh/id_ed25519_jsc -o AddressFamily=inet feuer1@login02.jupiter.fz-juelich.de "<cmd>"`). Keep ssh commands SIMPLE (single inline string) — nested loops / `$VAR="ssh…"` indirection exit 127 under this shell.

## Compiled DP>1 illegal-memory-access = MNNVL fused allreduce
MiniMax-M2.7-AWQ / GLM-4.7-AWQ compiled (cudagraphs ON) at **DP>1** crash with `CUDA driver error: an illegal memory access` in `profile_cudagraph_memory` during startup capture (DP=1 fine). Cause: vLLM's `fuse_allreduce_rms` pass swaps in flashinfer's `trtllm_mnnvl_allreduce_fusion` (Multi-Node NVLink) kernel, but Jupiter's cross-node transport is **InfiniBand** → writes to a non-existent NVLink peer. **Fix (one flag):** `--compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'`. (MoE all-to-all is a red herring.) Diagnostic that cracked it: `CUDA_LAUNCH_BLOCKING=1` (propagate to the cross-node Ray DP actor via `VLLM_RAY_EXTRA_ENV_VARIABLES_TO_COPY=CUDA_LAUNCH_BLOCKING`) → synchronous traceback names the kernel.

---

# Debugging tooling (SIF / Ray-actor / multi-node hangs)

What works vs doesn't inside the Apptainer SIF + Ray-actor vLLM workers (hard-won during the #232 long-ctx RL TP-rank-desync wedge):

- **`ptrace` is BLOCKED in the SIF** — py-spy/gdb fail ("Operation not permitted"; `yama/ptrace_scope=2`, no `SYS_PTRACE`). Don't try them on a wedged worker. (Untried: `apptainer exec --add-caps CAP_SYS_PTRACE`.)
- **In-process stack capture WORKS (no ptrace):** `faulthandler.dump_traceback_later(SECS, repeat=True, file=<per-rank file>)` at worker init (SECS below the watchdogs, e.g. 240–300s; healthy steps ~20–40s → only fires on a real hang), and/or a `SIGUSR1 → faulthandler.dump_traceback` handler to `kill -SIGUSR1 <pid>` a live wedge. The ONLY way to get the **lagging rank's** stack in a TP desync.
- **`/proc/<pid>/environ` is readable without ptrace** — VERIFY an env var reached a worker (caught `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` NOT propagating to Ray-actor engine workers this way).
- **Env vars don't reach Ray-actor vLLM workers via `APPTAINERENV_` alone.** EngineCore/mp workers are Ray actors; they inherit the driver's env only via Ray `runtime_env.env_vars` passthrough (SkyRL `ray_wrapped_inference_engine._build_inference_engine_runtime_env`), TP child actors via `placement_group_capture_child_tasks=True`. `APPTAINERENV_FOO` reaches the driver, NOT the workers — `/proc`-verify on a worker pid.
- **NCCL flight recorder (FR) — three gotchas:** (1) var is `TORCH_FR_BUFFER_SIZE` now (deprecated `TORCH_NCCL_TRACE_BUFFER_SIZE` auto-maps); enable `TORCH_NCCL_DUMP_ON_TIMEOUT=1` + writable `TORCH_NCCL_DEBUG_INFO_TEMP_FILE=<dir>/rank` (torch appends rank). (2) `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900` **preempts NCCL's default 1800s watchdog** → NCCL never dumps. To get an FR dump, drop `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` BELOW 900 (e.g. 600). (3) **FR only dumps a rank IN a collective** — in a TP desync where the lagging rank *never issues* the blocking collective, it has nothing in-flight → no dump on the rank you most need. FR catches the *blocked* ranks (0/1), not the *diverged* one (rank2); use faulthandler for that.
- **NCCL `COLL` trace lines as a stack substitute:** with `NCCL_DEBUG=INFO`/FR-on, each rank logs `AllReduce/AllGather: opCount … count … comm …`. Aligning per-rank streams by `opCount` pinpoints WHICH collective desyncs (lagging rank stops at op N while peers advance to N+1); cracked #232 when py-spy+FR failed. ⚠️ These lines contain the substring "opCount dead" — a hex/marker, NOT an error — falsely tripping naive `grep dead`/EngineDead monitors; match real tokens only: `EngineDeadError`, `execute_model timed out`, `Watchdog`.
- **`scancel` DESTROYS the per-rank Ray logs — copy them off BEFORE killing a wedged job.** Ray syncs each node's session dir to the shared experiment dir (`<exp>/ray_logs/ray_<jobid>_workers/<node>/…`) **at teardown, not live**, and a cancel kills the node before it flushes. Measured across 12 wedged TaskTrove arms: all **6/6** that ended `TIMEOUT` (full walltime) retained worker trees; **5 of 6** `CANCELLED` retained **none**, the sixth kept 1 node of 5. Corollary: a wedged job left to burn its walltime is *more* forensically useful than one killed early. Even so, every TIMEOUT job was missing ≥1 node — **the teardown sync is never complete**, so never assume a full rank set. To preserve evidence before a kill: `srun --jobid=<jid> --overlap -w <node> cp -r <ray_session_dir> <shared_path>` per node, or `rsync` from each `$TMPDIR`/local session root while the allocation is still alive.
- **`NCCL_DEBUG` is NOT on by default** — an arm launched without it has **zero** `opCount` lines, so the COLL-trace alignment above is unavailable retrospectively. Confirmed the hard way: 12 wedged arms carried only `NCCL_COLLNET_ENABLE=0`, and the planned `opCount` alignment could not be run on any of them. If a wedge class is under investigation, set `NCCL_DEBUG=INFO` on the *next* launch — you cannot recover it from logs already written.
- **Head-node `python-core-worker-*.log` is ~1.2 GB/job of pure noise** (Ray `Event stats` / `IO Service Stats` timer dumps, no training or collective content). Exclude it when pulling logs off Jupiter — it is ~90% of the bytes. Keep the same files on *worker* nodes, where they belong to the policy ranks. Also note `ray_<jobid>_workers/` lists **non-head nodes only** (the head's logs live in `ray_<jobid>/`), so an empty node dir there means missing data, not "that was the head".
- **Two independent watchdogs:** vLLM's `execute_model` RPC timeout (`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, multiproc/Ray executor) vs NCCL's collective heartbeat (`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`). A multi-node hang trips whichever is shorter; tune their relative values to control which fires first + whether you get a dump. 900s `execute_model` timeout = a genuinely *wedged* step (normal steps are ms–seconds).

---

# SFT (Jupiter particulars for the `sft-launch` skill)

Cluster-agnostic flow + backend/Delphi decisions → **`.agents/skills/sft-launch`**; this is the Jupiter-specific layer. GH200, 4 GPUs/node, aarch64. Conda env **`otagent`** (dense Qwen3 / Llama-3-tokenizer), **`sft-qwen35`** (Qwen3.5 hybrid arch, transformers ≥5.3), **`sft-axolotl`** (`--sft_backend axolotl`).

## Preamble (run FIRST, every session — pulls latest code + submodules)
```bash
source ~/.bashrc; source ~/secrets.env; cd /e/scratch/jureap59/feuer1/harbor; git stash; git pull; \
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent/SkyRL; git stash; git pull; conda activate otagent; \
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent; git pull; \
git submodule update --init --remote sft/llamafactory; git submodule update --init sft/axolotl; source hpc/dotenv/jupiter.env;
```
- `git submodule update … sft/llamafactory` is essential (SFT won't run on a stale submodule). `sft/axolotl` is **pinned** — update WITHOUT `--remote` to honor the pin; only needed for `--sft_backend axolotl`.
- **🚧 SUBMIT FROM THE REPO DIR WITH `DCFT` SET.** `universal_sft.sbatch` resolves `WORKDIR` from `DCFT_PRIVATE → DCFT → $PWD`; submitting from `$HOME`/scratch with `DCFT` unset trips the WORKDIR guard → immediate `exit 1` (`FATAL: WORKDIR=… is not the OpenThoughts-Agent repo root`). The preamble sets `DCFT`; re-run it if you see that FATAL.

## Wall / QOS / account
- **`--time_limit` max `11:59:00`** — booster QOS caps wall at 12h. For longer, chain-restart with `--max_restarts N` (auto-resumes from latest checkpoint).
- **Account: leave default `reformo`** — do NOT pass `--account jureap59` (its booster QOS is **suspended** → `Reason=InvalidQOS`, never schedules). `hpc.py` hardwires `reformo` for Jupiter SFT; SFT shares that allocation with RL/datagen, so to schedule faster **free a reformo slot** (don't switch accounts).

## Checkpoints → `/e/data1/datasets/playground/ot/checkpoints`
Prefer **`--output_dir /e/data1/datasets/playground/ot/checkpoints/<job_name>`**. `hpc/dotenv/jupiter.env` defaults `CHECKPOINTS_DIR=$DCFT_DATA/checkpoints` (= `…/ot-baf/checkpoints`) and clobbers any `export CHECKPOINTS_DIR=…` in the preamble → the explicit flag is the reliable way to land in canonical `ot/checkpoints`. Dry-run and confirm the rendered `output_dir`.

## 8B vs 32B (recognition + cleanup shape → skill §6)
- **8B** (also Qwen3.5-9B): `--hub_model_id laion/<name>` sets the repo; `push_to_hub` defaults **False** on Jupiter (compute has no direct internet), so upload is a **login-node `hf upload`** cleanup step + `manual_db_push.py` to register (the `--upload_to_database` flag is eval-only). Full: CLAUDE.md "8B SFT Job Cleanup Checklist".
- **32B** (`…_32b*.yaml`): ZeRO-3 writes sharded `global_stepN/`, NO root safetensors → **launch WITHOUT `--hub_model_id`**, then: (1) `--job_type consolidate --consolidate_input $CKPT/<job> --consolidate_output_repo laion/<name> --consolidate_workdir <wd>/<name> --time_limit 02:00:00 --num_nodes 1`; (2) **manually `hf upload` from `final_repo/`** (the consolidate auto-push has hit `BrokenPipeError` on big uploads). Full: CLAUDE.md "32B SFT Job Cleanup Checklist". Recognition: `ls $CKPT/<job>/` shows `global_stepN/`+`zero_to_fp32.py`, no root safetensors.

## Qwen3.5 hybrid — `sft-qwen35` env
Qwen3.5 (9B/27B) GatedDeltaNet+Attention arch isn't in transformers 4.x → run the preamble but `conda activate sft-qwen35`. After training, copy `preprocessor_config.json` from the base into the ckpt (LF doesn't emit it; vLLM needs it). 9B → root safetensors (SKIP consolidate); 27B → 32B consolidate flow. The launcher handles env activation on Jupiter.

## Operating notes
- **Axolotl SFT (Sera/CoderForge baselines): default `--nodes=4` (16 GH200) + `zero3_bf16.json`** even for small 1-epoch jobs (~4× per-step speedup vs ~50min/1-node; dodges the 1-node/zero1 step-5 OOM at 8B/32k). NOT `zero1.json`. Scale to 2 nodes only if the 4-node queue is congested. (Standalone-sbatch Sera/CoderForge path, distinct from the `--sft_backend axolotl` launcher path — see `.agents/projects/axolotl/axolotl.md`.)
- **HPO sweeps: always pass `--job_name <short>`** so SLURM name + exp dir + checkpoint dir match (else the derived long dataset-string name breaks resume-from-checkpoint).
- **Scaling-ablation workflow:** one SSH session → preamble → queue all sizes back-to-back (record every JID) → ScheduleWakeup 600s + 1800s for early health (PENDING-past-30min, arrow-cache race, ENOSPC, NCCL, OOM — only caught by tailing `.out`) → switch to a 2h CronCreate once RUNNING-advancing/PENDING-with-ETA → fire the per-size post-training flow as EACH lands (32B consolidate `--time_limit 06:00:00` NOT 24h → `manual_db_push`; 8B skip consolidate). Verify the Supabase row, then `rm -rf` exp + consolidate dirs.
