#!/bin/bash
#SBATCH --job-name=delphi_math_rl
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Delphi RL scaling-laws (marin #6279) — MATH-500 GRPO. STARTER TEMPLATE.
# 1 node x 4 A100-64GB. singularity exec --nv the marinskyrl writable SANDBOX +
# the external uv venv, running run_delphi_math_rl.sh with the GOLD hparams
# (main_rl_evals/RL_CONVENTION.md §3), LOCAL pre-staged model + dataset, fully offline.
#
# Mechanism is identical to hpc/skyrl_yaml/leonardo/sbatch_math_grid.sh (sandbox dir,
# uv venv, gcc-for-Triton, writable HOME). Differs only in: per-CELL MODEL_PATH /
# RUN_NAME / STAGE passthrough, ckpt dir keyed by RUN_NAME, and the `normal` QOS /
# 24h wall (a 100-step MATH-500 run does not fit boost_qos_dbg's 30 min).
#
# USAGE (key=val args, last-wins; --job-name is MANDATORY so squeue/logs are model-specific):
#   sbatch --job-name=rl_<RUN_NAME> sbatch_delphi_math_rl.sh \
#     MODEL_PATH=laion/delphi-1e21-p33m67-9p25b-lr0_67-9cf8da-wc386k_lr1e5-sft \
#     DATASET=rlvr_math   RUN_NAME=delphi-1e21-p33m67-wc386k_sft-rl-rlvr_math  STAGE=sft
#   DATASET ∈ {rlvr_math, dapo_math, math500, rlvr_ifeval}; selects $WORK/data/rl/$DATASET +
#   its verifier env (aime for the 3 math sets; ifeval for rlvr_ifeval — NOT yet wired, will fail).
#   # FLAGGED fallback cell (lr1e-5 collapsed entropy on a weak ckpt): append a hydra override:
#   #   ... STAGE=sft trainer.policy.optimizer_config.lr=3.0e-6
set -euxo pipefail

# Pull MODEL_PATH=/RUN_NAME=/STAGE= off the front of "$@" (env-style); the REST
# (anything with a dot, e.g. trainer.x=y) passes through to hydra untouched.
HYDRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    MODEL_PATH=*) export MODEL_PATH="${arg#MODEL_PATH=}" ;;
    RUN_NAME=*)   export RUN_NAME="${arg#RUN_NAME=}" ;;
    STAGE=*)      export STAGE="${arg#STAGE=}" ;;
    DATASET=*)    export DATASET="${arg#DATASET=}" ;;
    *)            HYDRA_ARGS+=("$arg") ;;
  esac
done

WORK=/leonardo_work/AIFAC_5C0_290/bfeuer00
SF=/leonardo_scratch/fast/AIFAC_5C0_290/bfeuer00
SANDBOX=$SF/marinskyrl_sandbox
VENV=$SF/marin_venv
MARIN=$WORK/code/MarinSkyRL/skyrl-train
# Point CFG at wherever these starter scripts are staged on Leonardo (e.g. the
# OT-Agent leonardo dir, or a copy of main_rl_evals/). run_delphi_math_rl.sh must
# sit next to this file.
CFG=${CFG:-$WORK/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo}

# Dataset → DATA_DIR + verifier env (RL_CONVENTION §2.2 / §4). DATA_DIR built by rl_dataset_prep.py (§5).
export DATASET=${DATASET:-math500}                           # rlvr_math | dapo_math | math500 | rlvr_ifeval
export DATA_DIR=${DATA_DIR:-$WORK/data/rl/$DATASET}
case "$DATASET" in
  rlvr_math|dapo_math|math500) export ENV_CLASS=aime ;;      # boxed/answer-match (Minerva)
  rlvr_ifeval)
    export ENV_CLASS=ifeval                                  # TODO: NOT yet in MarinSkyRL (§2.2)
    echo "WARNING: DATASET=rlvr_ifeval needs a skyrl_gym/envs/ifeval verifier that does not exist yet — this run WILL fail until it is wired (RL_CONVENTION.md §2.2)." ;;
  *) echo "ERROR: unknown DATASET '$DATASET' (rlvr_math|dapo_math|math500|rlvr_ifeval)"; exit 2 ;;
esac
: "${MODEL_PATH:?pass MODEL_PATH=<starting checkpoint HF id>}"
export MODEL_PATH
export RUN_NAME=${RUN_NAME:-delphi_${DATASET}_rl}
export STAGE=${STAGE:-sft}                                    # sft → delphi_v0 template; base → no template
export NUM_GPUS=${NUM_GPUS:-4}

# Fresh per-cell ckpt dir keyed by RUN_NAME — never share across cells (Wave-2 lesson).
# NOTE (2026-06-19 infra fix): write ckpts to $WORK, not $SF. The scratch-fast quota
# (leonardo_scratch_fast-22147958) is over 1T (was 370% / 3.7T, blown by the 3.4T stale
# grid_ckpts HPO sweep) → the first save_steps write (ckpt_interval=20, ~1-1.7h in) hit
# `OSError [Errno 122] Disk quota exceeded` and FAILED all 12 cells. $WORK has 88T free.
# Location-only change; NO hparam/science change.
export CKPT_DIR=$WORK/rl_ckpts/$RUN_NAME
rm -rf "$CKPT_DIR"

# Offline / cache env (compute nodes have no internet).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=$WORK/data/hub
export HF_HUB_CACHE=$WORK/data/hub
export WANDB_MODE=offline
export VLLM_CACHE_ROOT=$SF/vllm_cache
export TRITON_CACHE_DIR=$SF/vllm_cache/triton
export FLASHINFER_WORKSPACE_BASE=$SF/vllm_cache/flashinfer
export CONTAINER_HOME=$SF/canary_home          # SkyRL defaults paths to $HOME; container /home is RO
mkdir -p "$CKPT_DIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$CONTAINER_HOME"

# gcc 14.3.0 for Triton JIT (ray base image ships none) + silence ray telemetry.
export CONDA_BIN=$WORK/miniforge3/envs/otagent/bin
export RAY_USAGE_STATS_ENABLED=0
export VENV_PY=$VENV/bin/python

# RAY socket-path guard (2026-06-20 infra fix). Ray builds an AF_UNIX socket at
# $RAY_TMPDIR/.../session_<ts>_<pid>/sockets/plasma_store, which must stay <107 bytes.
# The leonardo.env dotenv exports RAY_TMPDIR=$WORK/jobtmp/$ID/ray (a LONG $WORK path);
# singularity does NOT --cleanenv, so that long var LEAKS into the container and Ray's
# socket path blew past 107 → `OSError: AF_UNIX path length cannot exceed 107 bytes`,
# which FAILED all 4 D3/D4 requeue cells (47470111-114) at ~3 min. (The 3 D1 cells that
# happened to be submitted from a shell without RAY_TMPDIR set survived on Ray's default.)
# FIX: pin a SHORT node-local RAY_TMPDIR (/tmp/...; ~79 bytes incl. the session+socket
# suffix) and pass it EXPLICITLY into the --env block so it overrides any leaked value.
export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID:-$$}
mkdir -p "$RAY_TMPDIR" || true

# ---------------------------------------------------------------------------
# LOAD-BEARING (RL_CONVENTION.md §2.3): delphi_v0 chat-template override.
# Ported verbatim from delphi_eval.sbatch §"delphi chat-template override". The SFT
# repos SHIP A PLAIN 656-char llama3 chat_template (no always-on system header, no
# <|start_think|> protocol) — the delphi ReasoningTemplate the models were SFT'd with
# did NOT persist into the repo. Serving RL rollout with the shipped template is a
# train/rollout/eval mismatch. FIX: for STAGE=sft (or rl), override the CACHED
# tokenizer's chat_template to delphi_v0.jinja2 BEFORE the singularity exec, so the
# in-container vLLM generator + the FSDP policy both read the same delphi_v0 the
# held-out eval uses. enable_thinking stays UNSET (model reasons in-channel).
# Idempotent (leaves a .plainbak; re-applies the same template). Runs on the HOST via
# the otagent conda python (has transformers + huggingface_hub); HF_HUB_OFFLINE=1 so
# snapshot_download returns the cached path (no network) — repo MUST be pre-staged (§5).
# STAGE=base/midtrained have NO chat template → skipped here; see the RL-zero caveat below.
DELPHI_TEMPLATE=${DELPHI_TEMPLATE:-$WORK/code/OpenThoughts-Agent/chat_templates/delphi_v0.jinja2}
if [ "$STAGE" = "sft" ] || [ "$STAGE" = "rl" ]; then
  echo ">>> applying delphi_v0 chat template to cached tokenizer for ${MODEL_PATH}"
  "$CONDA_BIN/python" - "$MODEL_PATH" "$DELPHI_TEMPLATE" <<'PYEOF'
import json, os, sys
from huggingface_hub import snapshot_download
repo, tmpl_path = sys.argv[1], sys.argv[2]
with open(tmpl_path) as f:
    delphi = f.read()
# resolve cached snapshot dir (HF_HUB_OFFLINE=1 -> cached path, no network)
snap = snapshot_download(repo)
tc_path = os.path.join(snap, "tokenizer_config.json")
jinja_path = os.path.join(snap, "chat_template.jinja")
# back up originals (copy resolved content, not the cache symlink) once
for p in (tc_path, jinja_path):
    if os.path.exists(p):
        bak = p + ".plainbak"
        if not os.path.exists(bak):
            with open(p, "rb") as s, open(bak, "wb") as d:
                d.write(s.read())
# write delphi_v0 as a REAL chat_template.jinja (break the cache symlink; keep the blob intact)
if os.path.islink(jinja_path):
    os.remove(jinja_path)
with open(jinja_path, "w") as f:
    f.write(delphi)
# and as the chat_template key in tokenizer_config.json (break symlink)
with open(tc_path) as f:
    tc = json.load(f)
tc["chat_template"] = delphi
if os.path.islink(tc_path):
    os.remove(tc_path)
with open(tc_path, "w") as f:
    json.dump(tc, f, ensure_ascii=False, indent=2)
# verify the loaded tokenizer now renders delphi_v0
from transformers import AutoTokenizer
ct = AutoTokenizer.from_pretrained(snap).chat_template
assert ct and ct.strip() == delphi.strip(), "chat_template override did NOT take (len=%s)" % (len(ct) if ct else None)
assert "<|start_think|>" in ct, "delphi_v0 think protocol missing after override"
print(">>> delphi_v0 applied + verified: chat_template len=%d (snapshot=%s)" % (len(ct), snap))
PYEOF
else
  # RL-zero (STAGE=base): the base/midtrained checkpoints have NO chat template, so the
  # math_dataset.py chat-formatted prompts cannot be rendered by the generator as-is.
  # DECIDE per the RL-zero rows (queue 3/4) how to format prompts (e.g. apply delphi_v0
  # anyway for contract-parity, or use a raw-completion prompt) and wire it here. Until
  # then, do NOT run an sft-grade RL-zero cell through this template-less path silently.
  echo ">>> STAGE=$STAGE: NO chat-template override (base/midtrained has none). RL-zero prompt formatting is unresolved — see RL_CONVENTION.md §2.3 / §4."
fi

nvidia-smi || true

singularity exec --nv \
  --no-home \
  --bind /leonardo_work:/leonardo_work,/leonardo_scratch:/leonardo_scratch \
  --pwd "$MARIN" \
  --env HOME=$CONTAINER_HOME,PATH=$CONDA_BIN:/usr/local/bin:/usr/bin:/bin,CC=$CONDA_BIN/gcc,CXX=$CONDA_BIN/g++ \
  --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_HOME=$HF_HOME,HF_HUB_CACHE=$HF_HUB_CACHE,WANDB_MODE=offline,VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT,TRITON_CACHE_DIR=$TRITON_CACHE_DIR,FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE,RAY_USAGE_STATS_ENABLED=0,RAY_TMPDIR=$RAY_TMPDIR,TMPDIR=$RAY_TMPDIR,DATA_DIR=$DATA_DIR,MODEL_PATH=$MODEL_PATH,NUM_GPUS=$NUM_GPUS,CKPT_DIR=$CKPT_DIR,RUN_NAME=$RUN_NAME,ENV_CLASS=$ENV_CLASS,LOGGER=console,VENV_PY=$VENV_PY \
  "$SANDBOX" bash "$CFG/run_delphi_math_rl.sh" "${HYDRA_ARGS[@]}" \
  trainer.resume_mode=null trainer.ckpt_path="$CKPT_DIR" trainer.export_path="$CKPT_DIR/exports"

echo "DELPHI_RL_EXIT=$?"
