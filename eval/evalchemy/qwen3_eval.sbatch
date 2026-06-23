#!/bin/bash
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:00
#SBATCH --job-name=qwen3-baseline
#SBATCH --output=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/qwen3-baseline/%x-%j.log
#
# Qwen3 dense POST-TRAINED baseline — ONE downstream-eval job (MATH500 + gsm8k + AIME24).
# SAME fixed protocol as delphi_eval.sbatch (EVAL_CONVENTION.md §2.2-§2.5, §3.3), with the Qwen3
# adaptations documented in baseline_evals/grid.md. This is a FORK of delphi_eval.sbatch; the deltas
# vs the Delphi sbatch (all by design, for the public Qwen3 dense instruct ladder) are:
#   (1) Chat template: use each model's OWN built-in ChatML template. We pass --apply_chat_template
#       but DO NOT apply the delphi_v0.jinja2 override (that override is Delphi-specific and would
#       corrupt Qwen3's ChatML). The entire delphi-template-override block is REMOVED here.
#   (2) Context / generation budget: Qwen3 have real long context (max_position_embeddings=40960,
#       clean rope_scaling=None) — NOT the malformed-rope 4k Delphi clamp. We use MAX_MODEL_LEN=32768
#       and MAX_GEN_TOKS=30720, IDENTICAL across all 6 Qwen3 models for within-grid comparability.
#   (3) Thinking mode: Qwen3 supports enable_thinking; we use the model DEFAULT (thinking ON) — the
#       intended post-trained / strong-math setting. We do NOT pass enable_thinking=false anywhere.
#   (4) TP: vLLM requires num_attention_heads % TP == 0. All 6 Qwen3 head counts (16,16,32,32,40,64)
#       are divisible by 4 -> TP=4 uniformly. Override per model with $4 if ever needed.
#   (5) No HF push / NO DB. These are public Qwen models we don't own. Results stay local.
#
# Usage:
#   sbatch --job-name="qwen3-baseline-<RUN>" qwen3_eval.sbatch <MODEL_REPO> <RUN> [TP_SIZE]
#     $1 MODEL_REPO : HF repo, e.g. Qwen/Qwen3-8B
#     $2 RUN        : output label, e.g. qwen3-8b
#     $3 TP_SIZE    : tensor-parallel size (default 4; must divide num_attention_heads)
#
# *** MODEL-SPECIFIC NAMING IS MANDATORY — always pass --job-name="qwen3-baseline-<RUN>". ***

set -euo pipefail

# ----- env: conda env `evalchemy-marin` (lm-eval v0.4.12 standard; cutover 2026-06-15 from
#        `evalchemy`=0.4.9.1 after MATH500 validation 9.6% vs 10.2%, grader byte-identical) -----
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh
conda activate evalchemy-marin
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/evalchemy-marin

source ~/secrets.env                                            # HF_TOKEN etc. (BEFORE HOME override)
export HF_HOME=/leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub
export HF_HUB_CACHE=/leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub
export HF_HUB_OFFLINE=1            # repos pre-cached on login node; no compute-node internet
export HF_ALLOW_CODE_EVAL=1       # required for gsm8k execution grading (evals.py:64)
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1  # only matters for any pre-download; harmless here

# ----- writable caches (Leonardo: $HOME is READ-ONLY from compute AND login nodes) (§3.0 amendment) -----
CACHE_ROOT=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/.cache
mkdir -p "$CACHE_ROOT"/{flashinfer,triton,inductor,vllm,xdg,hf_local}
export HOME="$CACHE_ROOT/hf_local"            # flashinfer derives FLASHINFER_WORKSPACE_DIR from $HOME/.cache
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
mkdir -p "$HOME/.cache/flashinfer"

# ----- args -----
MODEL_REPO="${1:?usage: sbatch --job-name=qwen3-baseline-<RUN> qwen3_eval.sbatch <MODEL_REPO> <RUN> [TP_SIZE]}"
RUN_NAME="${2:?usage: sbatch --job-name=qwen3-baseline-<RUN> qwen3_eval.sbatch <MODEL_REPO> <RUN> [TP_SIZE]}"
TP_SIZE="${3:-4}"
# vLLM sampler-warmup seq cap. Default 256 (vLLM default) is fine for ≤14B, but the 32B at
# max_model_len=32768 OOMs warming up 256 dummy sampler requests (job 46868341) — pass a smaller
# value (e.g. 32) for big models. 4th positional arg keeps the default for the small cells unchanged.
MAX_NUM_SEQS="${4:-256}"
OUT="/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/qwen3-baseline/${RUN_NAME}"

# ----- model-specific naming + jobid->model map -----
mkdir -p "${OUT}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="qwen3-baseline-${RUN_NAME}" 2>/dev/null || true
fi
echo ">>> EVAL_JOBMAP jobid=${SLURM_JOB_ID:-NA} run=${RUN_NAME} repo=${MODEL_REPO} tp=${TP_SIZE}"
{ echo "JOBID=${SLURM_JOB_ID:-NA}"; echo "RUN_NAME=${RUN_NAME}"; echo "MODEL_REPO=${MODEL_REPO}"; echo "TP_SIZE=${TP_SIZE}"; } > "${OUT}/meta.env"

# ----- chat-template handling (Qwen3 ADAPTATION #1) -----
# Qwen3 instruct models ship their OWN correct ChatML chat_template. Pass --apply_chat_template so the
# harness renders prompts with the model's built-in template. DO NOT override it with delphi_v0.jinja2
# (the Delphi-specific override block from delphi_eval.sbatch is intentionally ABSENT here — applying
# it would replace Qwen3's ChatML with a Llama-3 think-protocol template and corrupt every prompt).
# Thinking mode (ADAPTATION #3): we leave enable_thinking UNSET -> Qwen3 default = thinking ON.
CHAT_TEMPLATE_FLAG="--apply_chat_template"

# ----- context length (Qwen3 ADAPTATION #2) -----
# Qwen3 dense have max_position_embeddings=40960 with clean rope_scaling=None (NOT the malformed-rope
# 4k clamp the Delphi checkpoints carry). Clamping to 3584 would cripple these hybrid reasoning models
# on MATH500/AIME24. Use a generous budget, IDENTICAL across all 6 for within-grid comparability.
MAX_MODEL_LEN=32768
MAX_GEN_TOKS=30720
echo "RUN=$RUN_NAME  MODEL_REPO=$MODEL_REPO  OUT=$OUT  TP_SIZE=$TP_SIZE  MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_GEN_TOKS=$MAX_GEN_TOKS  CHAT_TEMPLATE_FLAG='${CHAT_TEMPLATE_FLAG}'"

# ----- idempotent skip (§3.4): seed42 dir means MATH500+gsm8k already ran for this cell -----
[ -d "${OUT}/seed42" ] && { echo "already done: ${OUT}/seed42 exists, skipping"; exit 0; }

# ----- MATH500 : 1 seed (42) (§2.4). evalchemy chat_benchmark -> --max_tokens + --verbosity INFO required (§3.3 a,b) -----
python -m eval.eval \
  --model vllm \
  --model_args "pretrained=${MODEL_REPO},tensor_parallel_size=${TP_SIZE},dtype=bfloat16,max_model_len=${MAX_MODEL_LEN},max_num_seqs=${MAX_NUM_SEQS},seed=42" \
  --tasks MATH500 \
  ${CHAT_TEMPLATE_FLAG} \
  --batch_size auto \
  --max_tokens ${MAX_GEN_TOKS} \
  --gen_kwargs "temperature=0.7,top_p=1.0,max_gen_toks=${MAX_GEN_TOKS}" \
  --verbosity INFO \
  --output_path "${OUT}/seed42"

# ----- gsm8k : 1 seed (42) via the PLAIN lm-eval CLI (single engine, avoids the double-engine OOM) (§3.3 c) -----
lm_eval --model vllm \
  --model_args "pretrained=${MODEL_REPO},tensor_parallel_size=${TP_SIZE},dtype=bfloat16,max_model_len=${MAX_MODEL_LEN},gpu_memory_utilization=0.9,max_num_seqs=${MAX_NUM_SEQS},seed=42" \
  --tasks gsm8k \
  ${CHAT_TEMPLATE_FLAG} \
  --batch_size auto \
  --gen_kwargs "temperature=0.7,top_p=1.0,max_gen_toks=${MAX_GEN_TOKS}" \
  --seed 42 \
  --output_path "${OUT}/seed42"

# ----- AIME24 : averaged over 10 seeds (small set, high variance) (§2.4) -----
for S in 42 43 44 45 46 47 48 49 50 51; do
  python -m eval.eval \
    --model vllm \
    --model_args "pretrained=${MODEL_REPO},tensor_parallel_size=${TP_SIZE},dtype=bfloat16,max_model_len=${MAX_MODEL_LEN},max_num_seqs=${MAX_NUM_SEQS},seed=${S}" \
    --tasks AIME24 \
    ${CHAT_TEMPLATE_FLAG} \
    --batch_size auto \
    --max_tokens ${MAX_GEN_TOKS} \
    --gen_kwargs "temperature=0.7,top_p=1.0,max_gen_toks=${MAX_GEN_TOKS}" \
    --verbosity INFO \
    --output_path "${OUT}/seed${S}"
done

echo "DONE: ${RUN_NAME} (Qwen3 baseline) -> ${OUT}"
