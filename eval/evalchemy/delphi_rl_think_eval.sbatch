#!/bin/bash
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --job-name=delphi-think-eval
#SBATCH --output=/leonardo_work/AIFAC_5C0_290/%u/experiments/delphi-eval/%x-%j.log
#
# PREREQUISITE (multi-operator): the --output directory above must EXIST BEFORE SUBMIT.
# SLURM opens the log file before the job script runs and does NOT create intermediate
# directories -- a missing dir fails the job instantly with "Unable to open file", which
# looks like a launch bug rather than a missing mkdir. It already exists for the original
# operator; a NEW operator runs this once:
#   mkdir -p /leonardo_work/AIFAC_5C0_290/$USER/experiments/delphi-eval
#
# Held-out math eval for the 4 FORCED-THINKING RL grid cells (consolidated local HF dirs).
# IDENTICAL harness to eval/evalchemy/evalchemy_eval.sbatch (same tasks, seeds, gen_kwargs,
# gsm8k-via-lm_eval, AIME24 10-seed, MATH500 via eval.eval) EXCEPT:
#   * MODEL is a LOCAL DIR (no snapshot_download / no auto-TP probe; TP + max_model_len pinned).
#   * chat-template = delphi_v0_THINK.jinja2 (forced-prefill <|start_think|>) baked into the dir's
#     tokenizer BEFORE eval — matches how THINK_MODE=forced cells were trained (delphi_v0 would
#     inject an empty think block on these forced models -> artificially zero).
#   * STAGE=rl -> --apply_chat_template ; IFEval skipped (math-only cells).
# Pins: TP=2 (30 attention heads -> 30%2==0, 30%4 fails), max_model_len=4096, max_gen_toks=3584
#       (Delphi malformed-rope 4k clamp), temp 0.7 top_p 1.0.
#
# Usage: sbatch --job-name=delphi-think-eval-<RUN> delphi_rl_think_eval.sbatch <LOCAL_MODEL_DIR> <RUN_NAME>
set -euo pipefail

# ----- $USER-scoped paths (multi-operator) -----------------------------------------------------
# Resolves to the original bfeuer00 strings byte-identically when USER=bfeuer00. Read-only shared
# runtime (conda, evalchemy-marin clone) falls back to the bfeuer00 copy; WRITE paths are per-user.
_OTA_WORK="/leonardo_work/AIFAC_5C0_290/${USER:-$(id -un)}"
_OTA_SHARED="/leonardo_work/AIFAC_5C0_290/bfeuer00"
_pick() { [ -e "$1" ] && echo "$1" || echo "$2"; }
# Conda: pick the root that actually CONTAINS the env. Testing only for a miniforge3 dir is not
# enough -- an operator can have a partial install (e.g. otagent only) and would then fail with
# "EnvironmentNameNotFound: evalchemy-marin" instead of falling back to the shared root.
_pick_conda() { [ -d "${_OTA_WORK}/miniforge3/envs/$1" ] && echo "${_OTA_WORK}/miniforge3" || echo "${_OTA_SHARED}/miniforge3"; }
_OTA_CONDA_ROOT="${OTAGENT_CONDA_ROOT:-$(_pick_conda evalchemy-marin)}"
_OTA_REPO="${OTAGENT_REPO_DIR:-${_OTA_WORK}/code/OpenThoughts-Agent}"


# Clean any propagated python env that can shadow conda's base-python stdlib
# (intermittent "ImportError: cannot import name 'JSONEncoder' from 'json'" on some compute nodes).
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP 2>/dev/null || true
source "${_OTA_CONDA_ROOT}/etc/profile.d/conda.sh"
# Activate with a small retry — the JSONEncoder import flake is transient (GPFS/conda).
for try in 1 2 3 4 5; do
  conda activate evalchemy-marin 2>/dev/null || true
  if python -c "from json import JSONEncoder; import sys; assert 'evalchemy-marin' in sys.executable" 2>/dev/null; then
    echo ">>> conda evalchemy-marin active (try $try): $(which python)"; break
  fi
  echo ">>> conda activate flake on try $try; retrying in 10s"; conda deactivate 2>/dev/null || true; sleep 10
  source "${_OTA_CONDA_ROOT}/etc/profile.d/conda.sh"
done
python -c "from json import JSONEncoder; import sys; assert 'evalchemy-marin' in sys.executable" \
  || { echo "FATAL: could not activate evalchemy-marin cleanly after retries"; exit 17; }
cd "${EVALCHEMY_DIR:-$(_pick "${_OTA_WORK}/code/evalchemy-marin" "${_OTA_SHARED}/code/evalchemy-marin")}"

source ~/secrets.env
export HF_HOME="${HF_HOME:-${_OTA_WORK}/data/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${_OTA_WORK}/data/hub}"
export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

CACHE_ROOT="${CACHE_ROOT:-${_OTA_WORK}/experiments/delphi-eval/.cache}"
mkdir -p "$CACHE_ROOT"/{flashinfer,triton,inductor,vllm,xdg,hf_local}
export HOME="$CACHE_ROOT/hf_local"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
mkdir -p "$HOME/.cache/flashinfer"

MODEL_DIR="${1:?usage: sbatch ... <LOCAL_MODEL_DIR> <RUN_NAME>}"
RUN_NAME="${2:?usage: sbatch ... <LOCAL_MODEL_DIR> <RUN_NAME>}"
THINK_TEMPLATE="${THINK_TEMPLATE:-${_OTA_REPO}/chat_templates/delphi_v0_think.jinja2}"
OUT_ROOT="${OUT_ROOT:-${_OTA_WORK}/experiments/delphi-eval}"
OUT="${OUT_ROOT}/${RUN_NAME}"
TP_SIZE=2
MAX_MODEL_LEN=4096
MAX_GEN_TOKS=3584
CHAT_TEMPLATE_FLAG="--apply_chat_template"

mkdir -p "${OUT}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="delphi-think-eval-${RUN_NAME}" 2>/dev/null || true
fi
echo ">>> EVAL_JOBMAP jobid=${SLURM_JOB_ID:-NA} run=${RUN_NAME} repo=${MODEL_DIR} stage=rl template=delphi_v0_think"
{ echo "JOBID=${SLURM_JOB_ID:-NA}"; echo "RUN_NAME=${RUN_NAME}"; echo "MODEL_REPO=${MODEL_DIR}"; echo "STAGE=rl"; echo "TEMPLATE=delphi_v0_think"; } > "${OUT}/meta.env"

# ----- bake delphi_v0_think template into the local model dir (idempotent; keeps .plainbak) -----
echo ">>> baking delphi_v0_think template into ${MODEL_DIR}"
python - "$MODEL_DIR" "$THINK_TEMPLATE" <<'PYEOF'
import json, os, sys
mdir, tmpl_path = sys.argv[1], sys.argv[2]
with open(tmpl_path) as f:
    tmpl = f.read()
tc_path = os.path.join(mdir, "tokenizer_config.json")
jinja_path = os.path.join(mdir, "chat_template.jinja")
for p in (tc_path, jinja_path):
    if os.path.exists(p):
        bak = p + ".plainbak"
        if not os.path.exists(bak):
            with open(p, "rb") as s, open(bak, "wb") as d:
                d.write(s.read())
with open(jinja_path, "w") as f:
    f.write(tmpl)
with open(tc_path) as f:
    tc = json.load(f)
tc["chat_template"] = tmpl
with open(tc_path, "w") as f:
    json.dump(tc, f, ensure_ascii=False, indent=2)
from transformers import AutoTokenizer
ct = AutoTokenizer.from_pretrained(mdir).chat_template
assert ct and ct.strip() == tmpl.strip(), "template override did NOT take"
# confirm the forced-prefill is present
assert "<|start_think|>" in ct, "delphi_v0_think missing start_think prefill"
print(">>> delphi_v0_think baked + verified: chat_template len=%d" % len(ct))
PYEOF

echo "RUN=$RUN_NAME MODEL_DIR=$MODEL_DIR TP=$TP_SIZE MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_GEN_TOKS=$MAX_GEN_TOKS OUT=$OUT"

[ -d "${OUT}/seed42" ] && { echo "already done: ${OUT}/seed42 exists, skipping"; exit 0; }

# ----- MATH500 : seed 42 (evalchemy chat_benchmark) -----
python -m eval.eval \
  --model vllm \
  --model_args "pretrained=${MODEL_DIR},tensor_parallel_size=${TP_SIZE},dtype=bfloat16,max_model_len=${MAX_MODEL_LEN},seed=42" \
  --tasks MATH500 \
  ${CHAT_TEMPLATE_FLAG} \
  --batch_size auto \
  --max_tokens ${MAX_GEN_TOKS} \
  --gen_kwargs "temperature=0.7,top_p=1.0,max_gen_toks=${MAX_GEN_TOKS}" \
  --verbosity INFO \
  --output_path "${OUT}/seed42"

# ----- gsm8k : plain lm_eval (single vLLM engine) -> strict + flexible -----
lm_eval --model vllm \
  --model_args "pretrained=${MODEL_DIR},tensor_parallel_size=${TP_SIZE},dtype=bfloat16,max_model_len=${MAX_MODEL_LEN},gpu_memory_utilization=0.9,seed=42" \
  --tasks gsm8k \
  ${CHAT_TEMPLATE_FLAG} \
  --batch_size auto \
  --gen_kwargs "temperature=0.7,top_p=1.0,max_gen_toks=${MAX_GEN_TOKS}" \
  --seed 42 \
  --output_path "${OUT}/seed42"

# ----- AIME24 : 10 seeds -----
for S in 42 43 44 45 46 47 48 49 50 51; do
  python -m eval.eval \
    --model vllm \
    --model_args "pretrained=${MODEL_DIR},tensor_parallel_size=${TP_SIZE},dtype=bfloat16,max_model_len=${MAX_MODEL_LEN},seed=${S}" \
    --tasks AIME24 \
    ${CHAT_TEMPLATE_FLAG} \
    --batch_size auto \
    --max_tokens ${MAX_GEN_TOKS} \
    --gen_kwargs "temperature=0.7,top_p=1.0,max_gen_toks=${MAX_GEN_TOKS}" \
    --verbosity INFO \
    --output_path "${OUT}/seed${S}"
done

echo "DONE: ${RUN_NAME} (stage=rl, delphi_v0_think) -> ${OUT}"
