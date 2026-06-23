## CINECA Leonardo Access

**SSH**: Uses ControlMaster multiplexing + step-ca certificate auth:
```bash
ssh Leonardo    # Complete 2FA once; socket persists 8h
```
> **Host keys ROTATE (benign — NOT an anomaly).** The round-robin login nodes rotate host keys, so a fresh
> connection (esp. `-o ControlPath=none` / `-o ControlMaster=no`) can hit `REMOTE HOST IDENTIFICATION HAS
> CHANGED` / a `known_hosts` mismatch. This is expected, not a compromise — just use the standard `ssh Leonardo`
> (ControlMaster socket), which works. Do NOT flag it as a failure or block on a `known_hosts` refresh.

**Pre-launch preamble** (run before launching any new job):
```bash
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh && \
conda activate otagent && \
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent && GIT_TERMINAL_PROMPT=0 git pull && \
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/harbor && GIT_TERMINAL_PROMPT=0 git pull && \
source hpc/dotenv/leonardo.env && source ~/secrets.env && \
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent
```

**Key paths**:
- Code: `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent`
- Harbor: `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/harbor`
- MarinSkyRL: `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/MarinSkyRL` (the old `/code/SkyRL` clone was removed 2026-06-08)
- evalchemy (standard / pass@k evals): `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/evalchemy-marin` — the **single canonical** clone (remote `origin` = `marin-community/evalchemy`, branch `main`; editable-installed in the `evalchemy-marin` conda env; see ENVIRONMENT_MAP §2e). The redundant `/code/evalchemy` (stale `mlfoundations` clone) and its `/code/evalchemy-resume-test` linked worktree were removed 2026-06-18.
- Data/HF cache: `/leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub`
- Experiments: `/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments`

> **⚠ WRITE-PATH MANDATE (quota-bind — obey in every launcher/sbatch/subagent).** Two filesystems,
> two purposes:
> - **`$WORK` (`/leonardo_work/AIFAC_5C0_290/bfeuer00`, ~1.4 PB GPFS, persistent)** — **ALL persistent/large
>   writes go here:** RL/SFT **checkpoints**, `trainer.export_path`, HF cache (`$WORK/data/hub`), envs,
>   experiment outputs. The dotenv already sets `CHECKPOINTS_DIR=$WORK/experiments` — **use it; never
>   hardcode a checkpoint/export path onto scratch.**
> - **`$SCRATCH_FAST` (`/leonardo_scratch/fast/AIFAC_5C0_290/bfeuer00`, 1 TB Lustre, auto-purged)** — **ONLY
>   ephemeral caches/tmp** (`VLLM_CONFIG_ROOT`/`TRITON_CACHE_DIR`/`FLASHINFER_WORKSPACE_BASE`). It is **1 TB,
>   shared, chronically OVER quota** — a checkpoint/export written here fails with `OSError: [Errno 122] Disk
>   quota exceeded` (NOT an OOM). **History (2026-06-19):** all 12 Delphi #6279 RL cells crashed at their
>   first ckpt-write because `sbatch_delphi_math_rl.sh` hardcoded `$SF/rl_ckpts`; fix = redirect ckpts to
>   `$WORK`. **Any launch subagent MUST verify its sbatch's checkpoint/export paths resolve to `$WORK`
>   (`$CHECKPOINTS_DIR`), not `$SF`/`$SCRATCH_FAST`, BEFORE submitting.**

**Correct upstream per codebase** (SoT = the Mac clones under `/Users/benjaminfeuer/Documents/`; aligned on Leonardo 2026-06-05). All on branch `penfever/working`:
- **OpenThoughts-Agent** → remote `origin` = `open-thoughts/OpenThoughts-Agent`.
- **harbor** → remote `marin` = `marin-community/harbor` (NOT laude-institute / `penfever/temp-override` / `otagent-latest` — those were stale on Leonardo).
- **MarinSkyRL** → `marin-community/MarinSkyRL` `penfever/working` (see `.claude/projects/marinskyrl/marinskyrl.md`; `penfever/SkyRL` is obsolete).
- Sync discipline: commit on the Mac → push → `git pull` on the cluster (Leonardo can't push). The preamble also needs `git submodule update --init --remote sft/llamafactory` for SFT runs.
- Leonardo is back in use as of 2026-06-05; cron covers Jupiter + Leonardo (Perlmutter dropped).

**Cluster details**: A100 64GB GPUs, 4/node, 3456 nodes, SLURM scheduler. No internet on compute nodes (use proxychains/SSH tunnel). User: `bfeuer00`. Account: `AIFAC_5C0_290`.

**Important**: Compilers come from conda (GCC 15.2, CUDA 13.2) — do NOT load system modules (`module load gcc cuda`), they are too old.

**Max wall time**: 24 hours (`--time 23:59:00`). The boost_usr_prod partition has a 1-day limit.

## Leonardo HF Upload — Use sbatch, NOT the Login Node

Leonardo's login nodes SIGKILL any long-running user process after ~100 seconds,
regardless of how it's detached. We've verified this kills:
- `nohup hf upload ... &` / `nohup huggingface-cli ... &` (~80s)
- `tmux new-session -d -s ... "hf upload ..."` (~2 min)
- `systemd-run --user --unit=... hf upload ...` (also SIGKILLed at ~100s)

(Tested with both the legacy `huggingface-cli` and the current `hf` CLI;
the killer is process-agnostic, not command-specific.)

The login node DOES have direct internet (no proxychains needed there) — the
problem is purely the process killer, not network.

**The reliable path is an sbatch job on a compute node with an SSH tunnel back
to the login node.** Compute nodes have no direct internet, but the existing
`eval/leonardo/start_proxy_tunnel.sh` opens a SOCKS5 forward from the compute
node to login05 and prints a `proxychains4 -q -f <config>` command prefix
that wraps any HF-bound command.

### Pre-flight (from your local Mac)

The intra-cluster SSH cert expires every ~12h. Refresh if stale:
```bash
step ssh certificate 'bfeuer00' --provisioner cineca-hpc \
  ~/.ssh/leonardo_daytona --no-password --insecure
ssh-keygen -R login.leonardo.cineca.it && \
rsync -avz -e 'ssh -i ~/.ssh/leonardo_daytona -o IdentitiesOnly=yes -o StrictHostKeyChecking=no' \
  ~/.ssh/leonardo_daytona ~/.ssh/leonardo_daytona.pub ~/.ssh/leonardo_daytona-cert.pub \
  bfeuer00@login.leonardo.cineca.it:~/.ssh/
```

Verify with `ssh-keygen -L -f ~/.ssh/leonardo_daytona-cert.pub | grep Valid`. Key on Leonardo:
`/leonardo/home/userexternal/bfeuer00/.ssh/leonardo_daytona`. The `start_proxy_tunnel.sh` script opens a
SOCKS5 forward compute→login05 and returns the `proxychains4 -q -f <config>` prefix. (The auth-required
HedgeDoc setup-instructions URL is in `.claude/secret.md` — untracked. The `leonardo_daytona` step-ca cert
is for the Daytona/SOCKS proxy; the intra-cluster `ssh Leonardo` key is a separate credential.)

### Cert expired → refresh it; do NOT route around the sbatch path
The sbatch-tunnel path depends on the `~/.ssh/leonardo_daytona` step-ca cert, which **expires ~12h**. When
it's expired the SOCKS tunnel gets `Permission denied (publickey)` and the `upload_*.sbatch` job dies in
~19s — **when a Leonardo upload sbatch fails fast on a publickey error, suspect the expired cert first** and
refresh it via the pre-flight above (needs interactive CINECA SSO 2FA in a browser; can't be done headless).
**The sbatch-tunnel remains the upload path** — do NOT fall back to a detached login-node `nohup` upload:
Leonardo's login killer takes down `nohup`/`disown`/tmux at ~100s (see `ops/all/hf_tmux.md`), so a
login-node upload of anything non-trivial will be SIGKILLed and leave a partial. Refresh the cert and use
the sbatch job.

### sbatch template for HF upload

```bash
cat > /leonardo_work/AIFAC_5C0_290/bfeuer00/upload_<job_name>.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=hf_upload_<short>
#SBATCH --output=<workdir>/upload_logs/upload_sbatch.log
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_5C0_290
#SBATCH --gres=gpu:1
#SBATCH --qos=boost_qos_dbg

set -e
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh
conda activate otagent
source ~/secrets.env

WD=<workdir>
DCFT=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent

unset LD_PRELOAD
export PATH="/leonardo_work/AIFAC_5C0_290/bfeuer00/proxychains/bin:${PATH}"
CMD_PREFIX=$(bash "$DCFT/eval/leonardo/start_proxy_tunnel.sh")

cd $WD/final_repo   # or $CHECKPOINTS_DIR/<job_name> for 8B path
$CMD_PREFIX /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/envs/otagent/bin/hf upload \
    <hub_model_id> . . \
    --repo-type=model
EOF
cd /leonardo_work/AIFAC_5C0_290/bfeuer00
sbatch upload_<job_name>.sbatch
```

Then `squeue -j <jobid>` and `tail -f <workdir>/upload_logs/upload_sbatch.log`.

### Numbers / sizing

- 131GB consolidated 32B → ~4 min wall through the tunnel (sbatch + tunnel pattern)
- 30 min wall fits `boost_qos_dbg` (debug QOS); longer jobs need a different QOS
- `hf upload` is sequential (no `--num-workers` knob) — slower than the legacy
  `huggingface-cli upload-large-folder` looked on paper, but the latter is now
  a deprecation stub AND deadlocks against HF Hub LFS rate limits in practice.
  See "HF Uploads + Long-Running Login-Node Commands" near the RL Cleanup
  section for the full story.
- Resume is automatic — `.cache/huggingface/` persists state; if the job is
  requeued, `hf upload` picks up where it left off

### Why this matters for the SFT checklists

Both the 8B (step 2) and 32B (step 3) cleanup checklists invoke `hf upload`
(in a `tmux` session). On Jupiter / Perlmutter / NYU Torch that works
straight from the login node (login has direct internet, no kill policy).
On Leonardo the same one-liner WILL die at ~100 s and leave a partial
upload — use the sbatch template above instead.

## Agentic Harbor eval IS feasible on Leonardo compute (via the SOCKS5 proxy) — 2026-06-20

**Agentic terminus-2 eval runs end-to-end on Leonardo *compute* nodes** (no native internet) through the
existing `eval/leonardo/start_proxy_tunnel.sh` SOCKS5 forward (compute→login05 dynamic SSH `-D` +
proxychains4). Verified 2026-06-20 with a real smoke (job 47436018, Qwen3-1.7B × tb2 subset → Supabase
`sandbox_jobs` row `d9eef9e5-525b-476a-b4bd-f8937bf1588b` `Finished`, 9 trials, 0 errors, traces on HF,
benchmark auto-registered).

**Why eval works on Leonardo but agentic RL does NOT:** for terminus-2 the LLM call is made by the
**orchestrator process on the compute node** to the locally-served vLLM (`api_base=http://$MASTER_ADDR:8000`);
the Daytona sandbox only runs shell commands and never reaches the model. So the only outbound traffic is
orchestrator→Daytona-API + →HF, both carried by the proxy — **no pinggy / served-model tunnel needed.** The
older "Daytona infeasible on Leonardo" statement is **eval-vs-RL specific** (RL's heavier cross-node/Daytona
coupling), not absolute.

**Single cross-cluster entrypoint:** `eval/jupiter/unified_eval_listener.py` now takes
`--cluster {jupiter,leonardo}` (a `CLUSTER_DEFAULTS` table picking the per-cluster sbatch + log-dir;
`EVAL_LISTENER_CLUSTER` / `EVAL_LISTENER_SBATCH` / `EVAL_LISTENER_LOG_DIR` env vars still win). **Default is
`jupiter`, byte-for-behavior identical to the pre-flag invocation — the live Jupiter campaign is unaffected.**
To launch a Leonardo agentic-eval campaign (in tmux on Leonardo, after the preamble):
```bash
python eval/jupiter/unified_eval_listener.py --cluster leonardo --preset <tb2|swebench|v2|...> \
  --require-priority-list --priority-file <list> --pre-download --once --verbose
```

**Load-bearing gotcha:** `eval/leonardo/unified_eval_harbor.sbatch` MUST pass `--jobs-dir "$EVAL_JOBS_DIR"`
to harbor (fixed 2026-06-20, commit `b56f9c1c`) — without it harbor writes trials to the wrong dir and the
Supabase auto-register silently no-ops. Eval auto-registration to Supabase is BY DESIGN (the
`enable_db_registration:false` guardrail applies to RL/SFT *training* YAMLs, not evals). Depends on the
`~/.ssh/leonardo_daytona` step-ca cert being fresh (~12h; refresh per the pre-flight above) — a stale cert
makes the proxy fail `Permission denied (publickey)`.

## vLLM 0.20.2rc0 cross-cluster twin — build decision (2026-06-16)

Leonardo's analogue of Jupiter's prod `skyrl_megatron_vllm0202rc0_r3.sif` (vLLM fork
`penfever/working @ 5d7319dd1`: 0.20.2rc0 + R3 capture + DCP GQA-LSE fp32 fix). Built as a
**writable singularity sandbox dir** on `$WORK` (NOT a `.sif` — `mksquashfs` OOMs / `lustre.lov`
xattr errors on the Lustre login node). Full runtime detail + recipe paths: `ENVIRONMENT_MAP.md` §2d.

**The CUDA question (decided):** Leonardo A100 nodes load kernel driver `535.274.02` (native CUDA
≤12.2); `singularity --nv` binds that host driver and a container can't replace it. The *toolkit*,
however, can be CUDA-13 via **forward compatibility** — `cuda-compat-13` (bundled in NGC cu13 images
at `/usr/local/cuda/compat`, with `LD_LIBRARY_PATH=/usr/local/cuda/compat` ahead of the `--nv` bind)
provides a newer userspace `libcuda` that runs a cu13 toolkit on the older datacenter driver (A100 =
datacenter, supported). This must be verified empirically (forward-compat has a per-version
minimum-driver floor; confirm 535 is within cu13's floor by running a cu13 kernel on an A100).

**Decision (user-approved):**
1. **Prefer** the true **CUDA-13 / torch-2.9** twin (NGC 25.09 + `cuda-compat-13`, arch 8.0) for real
   cross-cluster parity — gated on the forward-compat test passing on the 535 driver.
2. **Fallback (sanctioned, "if we have no other option"):** the **torch-2.8 / NGC-25.06 / CUDA-12.9.1**
   twin (recipe already written under `sif_build/recipes/`) — differs from Jupiter only in the
   torch/CUDA floor; same fork commit, R3, DCP fix, SkyRL/Megatron/TE, arch 8.0.
3. A CINECA driver upgrade (→≥580.65 for native CUDA-13) is **NOT** required; the fallback is acceptable.

**✅ FORWARD-COMPAT GATE PASSED → taking path (1), the true cu13/torch-2.9 twin (2026-06-16).** Built
the cu13 base as a writable sandbox (`$WORK/containers/pytorch_2509_sbx`, NGC `pytorch:25.09-py3`, 19 G)
and ran a real CUDA-13 fp32 matmul on an A100 under the 535.274.02 driver: `torch 2.9.0a0+…nv25.09`,
`cuda 13.0`, cap `(8,0)`, real result; `/proc/self/maps` confirms torch loaded the bundled
`cuda-compat-13` `libcuda.so.580.82.07`, not a host 535 libcuda. So the 535 branch is within cu13's
forward-compat floor — the fallback is **not** needed. A packed `.sif` pull FATAL'd at
`while creating squashfs: create command failed: signal: killed` (login mksquashfs OOM/kill +
`lustre.lov` xattr) → use `singularity build --sandbox` with `TMPDIR`/`SINGULARITY_TMPDIR` forced onto
GPFS WORK (default `TMPDIR=/scratch_local` is Lustre → xattr storm). The cu13 build recipe is
`sif_build/recipes/{README_vllm0202rc0_r3_leonardo_cu13.md, build_vllm0202rc0_r3_leonardo_cu13.sbatch}`;
the torch-2.8 recipe is retained as the documented fallback. Run convention:
`SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat/lib.real singularity exec --nv …`.

## ptrace LOCKED DOWN cluster-wide (`ptrace_scope=2`) — software profilers/debuggers FAIL (CVE-2026-46333)
**Since 2026-05-15** (CVE-2026-46333, a ptrace root-priv-escalation flaw), CINECA raised the kernel
`kernel.yama.ptrace_scope` from `0` → **`2`** (only admin-privileged processes may ptrace). This is a
**cluster-wide kernel setting** (broader than the SIF-only ptrace block on Jupiter — `ops/jupiter/ops.md`
§Debugging tooling). **Consequence: any ptrace/software-sampling tool fails** — `py-spy dump`/`py-spy record`,
`gdb -p`, and **software-sampling profilers**. Do NOT burn time trying them on a wedged Leonardo process.
- **General rule:** use **hardware-based sampling (`perf`-backed)**, not software (ptrace) sampling. Check
  whether a tool's collection method is configurable to hw before running.
- **Intel VTune** (`vtune -collect <analysis_type>`) — verify hw-sampling support per analysis with
  `vtune -help collect <analysis_type>`:
  - **NOT affected (work as-is):** `performance-snapshot`, `uarch-exploration`, `hpc-performance`, `io`,
    `system-overview`.
  - **Affected (need the hw-sampling knob):** `hotspots`, `threading`, `memory-consumption`.
  - **Tested-working mitigations:**
    - `vtune -collect hotspots  -r vtune_hotspots  -knob sampling-mode=hw`
    - `vtune -collect threading -r vtune_threading -knob sampling-and-waits=hw`
  - ⚠ **`memory-consumption` has NO hw-sampling mode → unusable under `ptrace_scope=2`.**
- **For our use (diagnosing wedged RL/inference):** the Jupiter playbook already applies — ptrace is out,
  so rely on **NCCL trace + per-rank `opCount` alignment** + last-log-line-per-rank + `/proc/<pid>/{stack,environ}`
  (readable without ptrace) to localize a hang. faulthandler/`SIGUSR1`-stack-dump is in-process (no ptrace)
  and still works if instrumented at launch.