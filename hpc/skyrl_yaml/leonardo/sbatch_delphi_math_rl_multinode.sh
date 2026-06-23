#!/bin/bash
#SBATCH --job-name=delphi_math_rl_mn
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Delphi RL scaling-laws (marin #6279) — MATH-RLVR GRPO, MULTI-NODE + THINKING-ON SMOKE.
# The foundation for a frontier-shaped RL HPO: prove (a) multi-node Ray bringup across the
# SLURM nodes, (b) policy + inference engines placed across all nodes, (c) a few training
# steps with reward logged, (d) NON-EMPTY think blocks in the rollout, (e) non-degenerate
# reward, (f) bounded entropy.
#
# Combines:
#   * the proven Leonardo MULTI-NODE Ray bring-up of sbatch_math_grid_multinode.sh
#     (ib0-pinned head/worker ray start inside the sandbox, GPU-count wait, trainer on head),
#   * the delphi-specific env of sbatch_delphi_math_rl.sh (account, delphi_v0 chat-template
#     override, short RAY_TMPDIR socket-path fix, offline caches), and
#   * run_delphi_math_rl_think.sh (thinking-on, validated-best hparams, smoke budget).
#
# DEFAULT = the 2-NODE / 8xA100 thinking-on smoke. Override --nodes=N for more nodes;
# POLICY_NUM_NODES is auto-derived from the SLURM node count so the FSDP mesh + engines
# span every node.
#
# USAGE (key=val args first; last-wins. --job-name MANDATORY so squeue/logs are run-specific):
#   sbatch --nodes=2 --job-name=rl_<RUN_NAME> sbatch_delphi_math_rl_multinode.sh \
#     MODEL_PATH=laion/delphi-...-sft \
#     DATASET=rlvr_math  RUN_NAME=delphi-...-rl-rlvr_math-think-mn2-smoke  STAGE=sft
#   DATASET in {rlvr_math, dapo_math, math500} -> env_class=aime. rlvr_ifeval -> ifeval.
#   THINK=false  -> A/B the thinking-off control (same code path, empty think prefix primed).
#   Any trailing token with a dot (trainer.x=y) passes through to hydra untouched.
set -euxo pipefail

# Pull env-style key=vals off the front of "$@"; the REST (dotted) -> hydra.
HYDRA_ARGS=()
for arg in "$@"; do
  case "$arg" in
    MODEL_PATH=*) export MODEL_PATH="${arg#MODEL_PATH=}" ;;
    RUN_NAME=*)   export RUN_NAME="${arg#RUN_NAME=}" ;;
    STAGE=*)      export STAGE="${arg#STAGE=}" ;;
    DATASET=*)    export DATASET="${arg#DATASET=}" ;;
    THINK=*)      export THINK="${arg#THINK=}" ;;
    THINK_MODE=*) export THINK_MODE="${arg#THINK_MODE=}" ;;
    DELPHI_TEMPLATE=*) export DELPHI_TEMPLATE="${arg#DELPHI_TEMPLATE=}" ;;
    *)            HYDRA_ARGS+=("$arg") ;;
  esac
done

WORK=/leonardo_work/AIFAC_5C0_290/bfeuer00
SF=/leonardo_scratch/fast/AIFAC_5C0_290/bfeuer00
SANDBOX=$SF/marinskyrl_sandbox
VENV=$SF/marin_venv
MARIN=$WORK/code/MarinSkyRL/skyrl-train
CFG=${CFG:-$WORK/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo}

# Dataset -> DATA_DIR + verifier env (built by rl_dataset_prep.py).
export DATASET=${DATASET:-rlvr_math}                          # rlvr_math | dapo_math | math500 | rlvr_ifeval
export DATA_DIR=${DATA_DIR:-$WORK/data/rl/$DATASET}
case "$DATASET" in
  rlvr_math|dapo_math|math500) export ENV_CLASS=aime ;;       # boxed/answer-match (Minerva)
  rlvr_ifeval)
    export ENV_CLASS=ifeval
    echo "WARNING: DATASET=rlvr_ifeval needs the skyrl_gym/envs/ifeval verifier (RL_CONVENTION §2.2)." ;;
  *) echo "ERROR: unknown DATASET '$DATASET'"; exit 2 ;;
esac
: "${MODEL_PATH:?pass MODEL_PATH=<starting checkpoint HF id>}"
export MODEL_PATH
export RUN_NAME=${RUN_NAME:-delphi_${DATASET}_rl_think_mn}
export STAGE=${STAGE:-sft}                                    # sft -> delphi_v0 template; base -> no template
export NUM_GPUS=${NUM_GPUS:-4}                                # GPUs PER NODE (Leonardo A100x4)
export THINK=${THINK:-true}                                   # thinking ON by default
export THINK_MODE=${THINK_MODE:-kwarg}                        # kwarg (smoke: batched=false+kwarg) | forced (batched=true + prefill template)
# In forced mode, bake the PREFILL template (delphi_v0_think) into the cached tokenizer so BOTH
# the vLLM engine and the policy render the forced '<|start_think|>' (consistent). kwarg mode
# keeps delphi_v0 (model-choice). Override DELPHI_TEMPLATE to pin a specific template.
if [ "$THINK_MODE" = "forced" ]; then
  export DELPHI_TEMPLATE=${DELPHI_TEMPLATE:-$WORK/code/OpenThoughts-Agent/chat_templates/delphi_v0_think.jinja2}
fi

# RESUME_MODE controls checkpoint resumption across an afterany restart CHAIN.
#   null (default)  -> one-shot smoke semantics: WIPE the per-run ckpt dir, no resume.
#   latest          -> chain semantics: PRESERVE the ckpt dir + resume from the latest ckpt
#                      (passed through to hydra as trainer.resume_mode=latest). The head run
#                      of a chain finds an empty/fresh dir and starts at step 0; restarts resume.
export RESUME_MODE=${RESUME_MODE:-null}

# Per-run ckpt dir on $WORK (scratch-fast quota is tight; ckpt OFF in the smoke anyway).
export CKPT_DIR=$WORK/rl_ckpts/$RUN_NAME
if [ "$RESUME_MODE" = "null" ]; then
  rm -rf "$CKPT_DIR"   # smoke / non-resumable: always start clean
fi

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
export CONDA_BIN=$WORK/miniforge3/envs/otagent/bin
export RAY_USAGE_STATS_ENABLED=0
export VENV_PY=$VENV/bin/python
mkdir -p "$CKPT_DIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$CONTAINER_HOME"

# RAY socket-path guard: Ray's AF_UNIX plasma_store socket must stay <107 bytes. Pin a SHORT
# node-local temp-dir and pass it EXPLICITLY into the --env so a leaked long RAY_TMPDIR is overridden.
export RAY_TMP=/tmp/ray_${SLURM_JOB_ID:-$$}
mkdir -p "$RAY_TMP" || true

# Cross-node fabric: pin to the routable InfiniBand NIC (ib0). The eno* mgmt NICs are NOT
# routable between Leonardo compute nodes -> without this the Ray GCS + NCCL pick the wrong
# interface and the run hangs at cluster join / first collective.
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export RAY_PORT=6479

# ---------------------------------------------------------------------------
# LOAD-BEARING: delphi_v0 chat-template override (sft/rl only). The SFT repos ship a plain
# 656-char llama3 template with NO <|start_think|> protocol; serving rollout with it is a
# train/rollout/eval mismatch (and there is NO think channel at all). Override the CACHED
# tokenizer's chat_template to delphi_v0.jinja2 on the HOST (otagent conda python; HF_HUB_OFFLINE
# -> cached path, no network) so the in-container vLLM generator AND the FSDP policy both render
# the delphi_v0 think protocol. The enable_thinking=true KWARG (passed by run_*_think.sh via
# generator.chat_template_kwargs) then prevents the empty-think prefix from being primed.
DELPHI_TEMPLATE=${DELPHI_TEMPLATE:-$WORK/code/OpenThoughts-Agent/chat_templates/delphi_v0.jinja2}
if [ "$STAGE" = "sft" ] || [ "$STAGE" = "rl" ]; then
  echo ">>> applying delphi_v0 chat template to cached tokenizer for ${MODEL_PATH}"
  "$CONDA_BIN/python" - "$MODEL_PATH" "$DELPHI_TEMPLATE" <<'PYEOF'
import json, os, sys
from huggingface_hub import snapshot_download
repo, tmpl_path = sys.argv[1], sys.argv[2]
with open(tmpl_path) as f:
    delphi = f.read()
snap = snapshot_download(repo)
tc_path = os.path.join(snap, "tokenizer_config.json")
jinja_path = os.path.join(snap, "chat_template.jinja")
for p in (tc_path, jinja_path):
    if os.path.exists(p):
        bak = p + ".plainbak"
        if not os.path.exists(bak):
            with open(p, "rb") as s, open(bak, "wb") as d:
                d.write(s.read())
if os.path.islink(jinja_path):
    os.remove(jinja_path)
with open(jinja_path, "w") as f:
    f.write(delphi)
with open(tc_path) as f:
    tc = json.load(f)
tc["chat_template"] = delphi
if os.path.islink(tc_path):
    os.remove(tc_path)
with open(tc_path, "w") as f:
    json.dump(tc, f, ensure_ascii=False, indent=2)
from transformers import AutoTokenizer
ct = AutoTokenizer.from_pretrained(snap).chat_template
assert ct and ct.strip() == delphi.strip(), "chat_template override did NOT take (len=%s)" % (len(ct) if ct else None)
assert "<|start_think|>" in ct, "delphi_v0 think protocol missing after override"
print(">>> delphi_v0 applied + verified: chat_template len=%d (snapshot=%s)" % (len(ct), snap))
PYEOF
else
  echo ">>> STAGE=$STAGE: NO chat-template override (base/midtrained has none). RL-zero prompt formatting unresolved."
fi

# ---- Discover head node + its IB ip (multi-node Ray) ----
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
NNODES=${#NODES[@]}
HEAD_NODE=${NODES[0]}
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --overlap \
    bash -c "ip -o -4 addr show ib0 | awk '{print \$4}' | cut -d/ -f1")
HEAD_IP=$(echo "$HEAD_IP" | tr -d '[:space:]')
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
# FSDP policy/ref/critic mesh spans every node.
export POLICY_NUM_NODES=$NNODES
echo "MN_BRINGUP: NNODES=$NNODES HEAD_NODE=$HEAD_NODE HEAD_IP=$HEAD_IP RAY_ADDRESS=$RAY_ADDRESS NODES=${NODES[*]} POLICY_NUM_NODES=$POLICY_NUM_NODES"

nvidia-smi || true

# Common singularity env string (shared by head / worker / trainer execs).
SING_ENV="HOME=$CONTAINER_HOME,PATH=$CONDA_BIN:/usr/local/bin:/usr/bin:/bin,CC=$CONDA_BIN/gcc,CXX=$CONDA_BIN/g++"
SING_ENV="$SING_ENV,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_HOME=$HF_HOME,HF_HUB_CACHE=$HF_HUB_CACHE"
SING_ENV="$SING_ENV,WANDB_MODE=offline,VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT,TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
SING_ENV="$SING_ENV,FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE,RAY_USAGE_STATS_ENABLED=0"
SING_ENV="$SING_ENV,NCCL_SOCKET_IFNAME=ib0,GLOO_SOCKET_IFNAME=ib0,RAY_ADDRESS=$RAY_ADDRESS"
SING_ENV="$SING_ENV,RAY_TMPDIR=$RAY_TMP,TMPDIR=$RAY_TMP"
SING_ENV="$SING_ENV,DATA_DIR=$DATA_DIR,MODEL_PATH=$MODEL_PATH,NUM_GPUS=$NUM_GPUS,CKPT_DIR=$CKPT_DIR"
SING_ENV="$SING_ENV,RUN_NAME=$RUN_NAME,ENV_CLASS=$ENV_CLASS,LOGGER=console,VENV_PY=$VENV_PY"
SING_ENV="$SING_ENV,POLICY_NUM_NODES=$POLICY_NUM_NODES,THINK=$THINK,THINK_MODE=$THINK_MODE"
# Put the skyrl-train repo on PYTHONPATH so EVERY Ray actor (incl. remote-node FSDP policy
# workers) imports skyrl_train.* from source — NOT via the setuptools editable-finder, which
# omits skyrl_train.dataset on the actors and crashes worker init at 16-node scale
# (ImportError: cannot import name 'PromptDataset' ... unknown location). Mirrors the Jupiter
# multi-node launch (PYTHONPATH=.../skyrl-train). Additive: venv deps stay on site-packages.
SING_ENV="$SING_ENV,PYTHONPATH=$MARIN"

SING_BIND="/leonardo_work:/leonardo_work,/leonardo_scratch:/leonardo_scratch"

# ---- Start Ray HEAD on node 0 (inside container, blocking, backgrounded) ----
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --overlap \
    --output="${SLURM_JOB_NAME}_${SLURM_JOB_ID}_rayhead.out" \
    singularity exec --nv --no-home \
        --bind "$SING_BIND" --pwd "$MARIN" \
        --env "$SING_ENV" \
        "$SANDBOX" \
        "$VENV_PY" -m ray.scripts.scripts start --head \
            --node-ip-address="$HEAD_IP" --port="$RAY_PORT" \
            --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$NUM_GPUS" \
            --temp-dir="$RAY_TMP" --disable-usage-stats --block &
RAY_HEAD_SRUN_PID=$!

# Wait for the head GCS to come up.
wait_for_ray() {
    for _ in $(seq 1 60); do
        if srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --overlap \
            singularity exec --nv --no-home --bind "$SING_BIND" --pwd "$MARIN" \
              --env "$SING_ENV" "$SANDBOX" \
              "$VENV_PY" -m ray.scripts.scripts status --address "$RAY_ADDRESS" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    echo "MN_BRINGUP: FATAL ray head at $RAY_ADDRESS not ready after 300s" >&2
    return 1
}
wait_for_ray || { echo "RAY_HEAD_FAIL"; scancel "$SLURM_JOB_ID"; exit 1; }
echo "MN_BRINGUP: ray head READY at $RAY_ADDRESS"

# ---- Start Ray WORKERS on nodes 1..N-1 ----
WORKER_PIDS=()
for ((i=1; i<NNODES; i++)); do
    WN=${NODES[$i]}
    echo "MN_BRINGUP: starting worker $i on $WN -> $RAY_ADDRESS"
    srun --nodes=1 --ntasks=1 -w "$WN" --overlap \
        --output="${SLURM_JOB_NAME}_${SLURM_JOB_ID}_rayw${i}.out" \
        singularity exec --nv --no-home \
            --bind "$SING_BIND" --pwd "$MARIN" \
            --env "$SING_ENV" \
            "$SANDBOX" \
            "$VENV_PY" -m ray.scripts.scripts start --address "$RAY_ADDRESS" \
                --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$NUM_GPUS" \
                --disable-usage-stats --block &
    WORKER_PIDS+=($!)
    sleep 5
done

# Wait until the cluster reports all expected GPUs (NNODES * NUM_GPUS).
EXPECT_GPUS=$((NNODES * NUM_GPUS))
NGPU=0
for _ in $(seq 1 60); do
    NGPU=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --overlap \
        singularity exec --nv --no-home --bind "$SING_BIND" --pwd "$MARIN" \
          --env "$SING_ENV" "$SANDBOX" \
          "$VENV_PY" -c "import ray; ray.init(address='$RAY_ADDRESS'); print(int(ray.cluster_resources().get('GPU',0)))" 2>/dev/null | tail -1)
    echo "MN_BRINGUP: cluster GPUs=$NGPU / expected=$EXPECT_GPUS"
    [ "$NGPU" = "$EXPECT_GPUS" ] && break
    sleep 5
done
[ "$NGPU" = "$EXPECT_GPUS" ] || { echo "RAY_WORKER_JOIN_FAIL: got $NGPU expected $EXPECT_GPUS"; }

# ---- Run the trainer on the HEAD (joins the existing Ray cluster via RAY_ADDRESS) ----
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" --overlap \
    --output="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
    singularity exec --nv --no-home \
        --bind "$SING_BIND" --pwd "$MARIN" \
        --env "$SING_ENV" \
        "$SANDBOX" bash "$CFG/run_delphi_math_rl_think.sh" "${HYDRA_ARGS[@]}" \
        trainer.resume_mode=$RESUME_MODE trainer.ckpt_path="$CKPT_DIR" trainer.export_path="$CKPT_DIR/exports"
TRAIN_RC=$?
echo "DELPHI_RL_MN_EXIT=$TRAIN_RC"

# ---- Teardown: stop Ray everywhere, kill backgrounded bring-up sruns ----
for ((i=0; i<NNODES; i++)); do
    srun --nodes=1 --ntasks=1 -w "${NODES[$i]}" --overlap \
        singularity exec --nv --no-home --bind "$SING_BIND" --pwd "$MARIN" \
          --env "$SING_ENV" "$SANDBOX" \
          "$VENV_PY" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
done
kill "$RAY_HEAD_SRUN_PID" "${WORKER_PIDS[@]}" 2>/dev/null || true
rm -rf "$RAY_TMP" || true
exit $TRAIN_RC
