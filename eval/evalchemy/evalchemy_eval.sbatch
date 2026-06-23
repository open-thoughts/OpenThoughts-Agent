#!/bin/bash
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --job-name=evalchemy-eval
#SBATCH --output=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/%x-%j.log
#
# ==============================================================================
# evalchemy_eval.sbatch — GENERIC evalchemy downstream-eval runner (MATH500 + gsm8k + AIME24)
#
# Runs the fixed reasoning-eval protocol (EVAL_CONVENTION §2.2-§2.5, §3.3) on ONE served model
# under the `evalchemy-marin` conda env (lm-eval). Parameterized by (MODEL_REPO, RUN_NAME, STAGE).
# This is the generic successor to the per-TP delphi_eval_tp{1,3,4}.sbatch one-line copies:
# **tensor-parallel size is now computed at runtime** from the cached model's num_attention_heads
# (see "auto-TP" below), so odd-head models (e.g. 9 -> TP3, 11 -> TP1, 14 -> TP2) just work.
#
# Generic vs. model-specific:
#   - The generic body runs MATH500 (chat_benchmark) + gsm8k (lm-eval native) + AIME24 (10 seeds).
#   - The ONLY model-family-specific bit is the optional chat-template OVERRIDE, gated behind the
#     CHAT_TEMPLATE_OVERRIDE env var. Delphi SFT/RL repos need the delphi_v0.jinja2 override (their
#     shipped template is a train/eval mismatch); Qwen3 and most models do NOT (their own template is
#     correct). For delphi, use the thin wrapper `delphi_eval.sbatch` which sets the override + 4k clamp.
#
# Usage:
#   sbatch --job-name="<PREFIX>-<RUN_NAME>" evalchemy_eval.sbatch <MODEL_REPO> <RUN_NAME> [STAGE]
#     $1 MODEL_REPO : HF repo, e.g. Qwen/Qwen3-8B  or  laion/delphi-9e19-...-magpie_lr1e5
#     $2 RUN_NAME   : output label (== the cell label); score row maps 1:1 to this
#     $3 STAGE      : sft | rl | base   (default: sft; chat-template ON for sft/rl, OFF for base)
#
# Tunable env vars (all optional; the delphi wrapper sets the delphi defaults):
#   OUT_ROOT               output root dir          (default: .../experiments/delphi-eval)
#   JOB_PREFIX             squeue/job-name prefix    (default: evalchemy-eval)
#   MAX_MODEL_LEN          vLLM max_model_len        (default: model-native via vLLM; unset here -> see below)
#   MAX_GEN_TOKS           generation budget         (default: 30720)
#   CHAT_TEMPLATE_OVERRIDE path to a .jinja2 to force onto the cached tokenizer (sft/rl only; default: unset)
#   GPUS_PER_NODE          GPUs available for TP     (default: 4, A100 node)
#
# *** MODEL-SPECIFIC NAMING IS MANDATORY: always pass --job-name="<PREFIX>-<RUN_NAME>" at submit. ***
# ==============================================================================

set -euo pipefail

# ----- env: conda env `evalchemy-marin` (lm-eval; the env every recent SCORES.md row used) -----
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh
conda activate evalchemy-marin
# Must cd into the evalchemy-MARIN clone (matches the env's editable install). evalchemy-marin's
# eval/eval.py imports parse_eval_args/setup_parser from its eval/lm_eval_compat.py shim — unlike
# the now-removed `code/evalchemy` clone, whose eval/eval.py did `from lm_eval.__main__ import
# parse_eval_args, setup_parser` (both dropped in lm-eval 0.4.12 -> ImportError at startup).
# (`code/evalchemy` + its `evalchemy-resume-test` worktree were deleted 2026-06-18; this is the sole clone.)
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/evalchemy-marin

source ~/secrets.env                                            # HF_TOKEN etc. (BEFORE HOME override)
export HF_HOME=/leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub
export HF_HUB_CACHE=/leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub
export HF_HUB_OFFLINE=1            # repos pre-cached on login node; no compute-node internet
export HF_ALLOW_CODE_EVAL=1       # required for gsm8k execution grading
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1  # only matters for any pre-download; harmless here
# IFEval's loose-accuracy checks use nltk sentence tokenization (punkt/punkt_tab). On a read-only,
# offline compute node lm_eval's download_nltk_resources() would try (and fail) to fetch them, so
# point NLTK_DATA at the pre-downloaded copy under $WORK (populated once on a login node:
#   python -c 'import nltk; d="$WORK/data/nltk_data"; nltk.download("punkt_tab",download_dir=d); nltk.download("punkt",download_dir=d)'
# ). Harmless for the math tasks (they don't import nltk).
export NLTK_DATA="${NLTK_DATA:-/leonardo_work/AIFAC_5C0_290/bfeuer00/data/nltk_data}"

# ----- writable caches (Leonardo: $HOME is READ-ONLY from compute AND login nodes) -----
# vLLM's flashinfer/triton/inductor JIT caches default under $HOME/.cache and hard-crash every worker
# on a read-only HOME. Redirect HOME + every cache to the writable work filesystem.
CACHE_ROOT=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/.cache
mkdir -p "$CACHE_ROOT"/{flashinfer,triton,inductor,vllm,xdg,hf_local}
export HOME="$CACHE_ROOT/hf_local"            # flashinfer derives FLASHINFER_WORKSPACE_DIR from $HOME/.cache
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
mkdir -p "$HOME/.cache/flashinfer"

# ----- args + tunables -----
MODEL_REPO="${1:?usage: sbatch --job-name=<PREFIX>-<RUN_NAME> evalchemy_eval.sbatch <MODEL_REPO> <RUN_NAME> [STAGE]}"
RUN_NAME="${2:?usage: sbatch --job-name=<PREFIX>-<RUN_NAME> evalchemy_eval.sbatch <MODEL_REPO> <RUN_NAME> [STAGE]}"
STAGE="${3:-sft}"
OUT_ROOT="${OUT_ROOT:-/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval}"
JOB_PREFIX="${JOB_PREFIX:-evalchemy-eval}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-30720}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CHAT_TEMPLATE_OVERRIDE="${CHAT_TEMPLATE_OVERRIDE:-}"
OUT="${OUT_ROOT}/${RUN_NAME}"

# ----- model-specific naming + jobid->model map (so eval scores map 1:1 to the model) -----
mkdir -p "${OUT}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  scontrol update JobId="${SLURM_JOB_ID}" JobName="${JOB_PREFIX}-${RUN_NAME}" 2>/dev/null || true
fi
echo ">>> EVAL_JOBMAP jobid=${SLURM_JOB_ID:-NA} run=${RUN_NAME} repo=${MODEL_REPO} stage=${STAGE}"
{ echo "JOBID=${SLURM_JOB_ID:-NA}"; echo "RUN_NAME=${RUN_NAME}"; echo "MODEL_REPO=${MODEL_REPO}"; echo "STAGE=${STAGE}"; } > "${OUT}/meta.env"

# ----- chat-template handling (§2.5) -----
# sft/rl checkpoints render with a chat template (--apply_chat_template); base = raw completion.
case "$STAGE" in
  base)        CHAT_TEMPLATE_FLAG="" ;;
  sft|rl)      CHAT_TEMPLATE_FLAG="--apply_chat_template" ;;
  *) echo "ERROR: STAGE must be one of: sft | rl | base (got '$STAGE')"; exit 2 ;;
esac

# ----- optional chat-template OVERRIDE (model-family-specific; e.g. delphi_v0.jinja2) -----
# Some checkpoints ship a template that mismatches how they were trained. If CHAT_TEMPLATE_OVERRIDE is
# set (a path to a .jinja2) AND stage is sft/rl, force it onto the cached tokenizer BEFORE any eval.
# Idempotent; keeps a .plainbak of the originals. base stage is skipped (no template / raw completion).
if [ -n "${CHAT_TEMPLATE_OVERRIDE}" ] && { [ "$STAGE" = "sft" ] || [ "$STAGE" = "rl" ]; }; then
  echo ">>> applying chat-template override '${CHAT_TEMPLATE_OVERRIDE}' to cached tokenizer for ${MODEL_REPO}"
  python - "$MODEL_REPO" "$CHAT_TEMPLATE_OVERRIDE" <<'PYEOF'
import json, os, sys
from huggingface_hub import snapshot_download
repo, tmpl_path = sys.argv[1], sys.argv[2]
with open(tmpl_path) as f:
    tmpl = f.read()
# resolve cached snapshot dir (HF_HUB_OFFLINE=1 -> returns the cached path, no network)
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
# write override as a REAL chat_template.jinja (break the cache symlink; keep the blob intact)
if os.path.islink(jinja_path):
    os.remove(jinja_path)
with open(jinja_path, "w") as f:
    f.write(tmpl)
# and as the chat_template key in tokenizer_config.json (break symlink)
with open(tc_path) as f:
    tc = json.load(f)
tc["chat_template"] = tmpl
if os.path.islink(tc_path):
    os.remove(tc_path)
with open(tc_path, "w") as f:
    json.dump(tc, f, ensure_ascii=False, indent=2)
# verify the loaded tokenizer now renders the override
from transformers import AutoTokenizer
ct = AutoTokenizer.from_pretrained(snap).chat_template
assert ct and ct.strip() == tmpl.strip(), "chat_template override did NOT take (len=%s)" % (len(ct) if ct else None)
print(">>> chat-template override applied + verified: chat_template len=%d (snapshot=%s)" % (len(ct), snap))
PYEOF
fi

# ----- context length -----
# MAX_MODEL_LEN may be pre-set by a wrapper (e.g. delphi's malformed-rope 4k clamp). If unset, derive
# it from the cached model config's max_position_embeddings (capped at a sane 32768), and pin
# MAX_GEN_TOKS to fit. This keeps the generic runner safe for both 4k-clamp and long-context models.
if [ -z "${MAX_MODEL_LEN:-}" ]; then
  MAX_MODEL_LEN=$(python - "$MODEL_REPO" <<'PYEOF'
import json, os, sys, glob
from huggingface_hub import snapshot_download
repo = sys.argv[1]
snap = snapshot_download(repo)
cfgs = glob.glob(os.path.join(snap, "config.json"))
if not cfgs:
    sys.stderr.write("ERROR: no config.json under cached snapshot %s\n" % snap); sys.exit(3)
with open(cfgs[0]) as f:
    cfg = json.load(f)
mpe = cfg.get("max_position_embeddings", 32768)
print(min(int(mpe), 32768))
PYEOF
)
  # leave headroom for the prompt
  if [ "$MAX_GEN_TOKS" -gt "$((MAX_MODEL_LEN - 512))" ]; then
    MAX_GEN_TOKS=$((MAX_MODEL_LEN - 512))
  fi
fi

# ----- auto-TP: compute tensor_parallel_size from the cached model's num_attention_heads -----
# vLLM requires num_attention_heads % tensor_parallel_size == 0. We pick the LARGEST d in {4,3,2,1}
# with (num_heads % d == 0) and d <= GPUS_PER_NODE (maximizing parallelism). This replaces the old
# hardcoded TP_SIZE=2 (fatal for odd-head models) and makes the per-TP variant copies obsolete.
# Verified: 8->4, 9->3, 11->1, 12->4, 14->2, 16->4, 18->3, 20->4, 30->3.
TP_SIZE=$(python - "$MODEL_REPO" "$GPUS_PER_NODE" <<'PYEOF'
import json, os, sys, glob
from huggingface_hub import snapshot_download
repo, gpus = sys.argv[1], int(sys.argv[2])
# locate the cached snapshot config.json (HF_HUB_OFFLINE=1 -> returns cached path, no network)
snap = snapshot_download(repo)
cfg_path = os.path.join(snap, "config.json")
if not os.path.exists(cfg_path):
    hits = glob.glob(os.path.join(snap, "**", "config.json"), recursive=True)
    cfg_path = hits[0] if hits else None
if not cfg_path or not os.path.exists(cfg_path):
    sys.stderr.write("FATAL auto-TP: no config.json found under cached snapshot %s for %s\n" % (snap, repo))
    sys.exit(3)
with open(cfg_path) as f:
    cfg = json.load(f)
# text-config nesting (some multimodal/MoE configs wrap text params)
heads = cfg.get("num_attention_heads")
if heads is None and isinstance(cfg.get("text_config"), dict):
    heads = cfg["text_config"].get("num_attention_heads")
if heads is None:
    sys.stderr.write("FATAL auto-TP: num_attention_heads missing in %s\n" % cfg_path)
    sys.exit(4)
heads = int(heads)
tp = next((d for d in (4, 3, 2, 1) if d <= gpus and heads % d == 0), 1)
sys.stderr.write(">>> auto-TP: num_attention_heads=%d gpus=%d -> TP=%d (config=%s)\n" % (heads, gpus, tp, cfg_path))
print(tp)
PYEOF
)
if [ -z "${TP_SIZE}" ]; then
  echo "FATAL: auto-TP computation failed (empty TP_SIZE) for ${MODEL_REPO}"; exit 3
fi

echo "STAGE=$STAGE  CHAT_TEMPLATE_FLAG='${CHAT_TEMPLATE_FLAG}'  MODEL_REPO=$MODEL_REPO  OUT=$OUT  TP_SIZE=$TP_SIZE  MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_GEN_TOKS=$MAX_GEN_TOKS"

# ----- idempotent skip (§3.4): seed42 dir means MATH500+gsm8k already ran for this cell -----
[ -d "${OUT}/seed42" ] && { echo "already done: ${OUT}/seed42 exists, skipping"; exit 0; }

# ----- MATH500 : 1 seed (42) (§2.4) -----
# evalchemy chat_benchmark -> --max_tokens drives gen length; --verbosity INFO is REQUIRED
# (eval.py does getattr(logging, args.verbosity); default None crashes after vLLM init).
python -m eval.eval \
  --model vllm \
  --model_args "pretrained=${MODEL_REPO},tensor_parallel_size=${TP_SIZE},dtype=bfloat16,max_model_len=${MAX_MODEL_LEN},seed=42" \
  --tasks MATH500 \
  --log_samples \
  ${CHAT_TEMPLATE_FLAG} \
  --batch_size auto \
  --max_tokens ${MAX_GEN_TOKS} \
  --gen_kwargs "temperature=0.7,top_p=1.0,max_gen_toks=${MAX_GEN_TOKS}" \
  --verbosity INFO \
  --output_path "${OUT}/seed42"

# gsm8k : run via the PLAIN lm-eval CLI (single vLLM engine; evalchemy eval.eval double-builds the
# engine for native tasks and the 2nd OOMs). gsm8k metric = exact_match on the final number.
lm_eval --model vllm \
  --model_args "pretrained=${MODEL_REPO},tensor_parallel_size=${TP_SIZE},dtype=bfloat16,max_model_len=${MAX_MODEL_LEN},gpu_memory_utilization=0.9,seed=42" \
  --tasks gsm8k \
  --log_samples \
  ${CHAT_TEMPLATE_FLAG} \
  --batch_size auto \
  --gen_kwargs "temperature=0.7,top_p=1.0,max_gen_toks=${MAX_GEN_TOKS}" \
  --seed 42 \
  --output_path "${OUT}/seed42"

# ----- IFEval : instruction-following pass-rate (lm-eval-native; single seed) -----
# Primary held-out metric for the D2 / RLVR-IFeval RL cells (their training reward is the open-instruct
# constraint verifier on allenai/RLVR-IFeval; this scores generalization on the disjoint held-out
# google/IFEval 541-prompt benchmark — NOT train-on-test). IFEval is an lm-eval-NATIVE generate_until
# task, so it runs through the PLAIN `lm_eval` CLI (one vLLM engine), exactly like gsm8k — running it
# under evalchemy eval.eval would double-build the engine and OOM. lm_eval ships the `ifeval` task
# (dataset_path=google/IFEval) at v0.4.12; it reports prompt/inst-level strict+loose accuracy. The
# headline "pass-rate" = prompt_level_strict_acc (fraction of prompts whose EVERY constraint is met).
# Single seed (42) is sufficient (deterministic constraint grading over 541 prompts; matches IFEval
# leaderboard practice). Dataset + nltk punkt/punkt_tab must be pre-cached on a login node (see NLTK_DATA
# above; `python -c 'from datasets import load_dataset; load_dataset("google/IFEval", split="train")'`).
# IFEVAL_ENABLE=0 to skip (e.g. a math-only cell).
if [ "${IFEVAL_ENABLE:-1}" = "1" ]; then
  lm_eval --model vllm \
    --model_args "pretrained=${MODEL_REPO},tensor_parallel_size=${TP_SIZE},dtype=bfloat16,max_model_len=${MAX_MODEL_LEN},gpu_memory_utilization=0.9,seed=42" \
    --tasks ifeval \
    --log_samples \
    ${CHAT_TEMPLATE_FLAG} \
    --batch_size auto \
    --gen_kwargs "temperature=0.7,top_p=1.0,max_gen_toks=${MAX_GEN_TOKS}" \
    --seed 42 \
    --output_path "${OUT}/seed42"
fi

# ----- AIME24 : averaged over 10 seeds (small set, high variance) (§2.4) -----
for S in 42 43 44 45 46 47 48 49 50 51; do
  python -m eval.eval \
    --model vllm \
    --model_args "pretrained=${MODEL_REPO},tensor_parallel_size=${TP_SIZE},dtype=bfloat16,max_model_len=${MAX_MODEL_LEN},seed=${S}" \
    --tasks AIME24 \
    --log_samples \
    ${CHAT_TEMPLATE_FLAG} \
    --batch_size auto \
    --max_tokens ${MAX_GEN_TOKS} \
    --gen_kwargs "temperature=0.7,top_p=1.0,max_gen_toks=${MAX_GEN_TOKS}" \
    --verbosity INFO \
    --output_path "${OUT}/seed${S}"
done

echo "DONE: ${RUN_NAME} (stage=${STAGE}) -> ${OUT}"
