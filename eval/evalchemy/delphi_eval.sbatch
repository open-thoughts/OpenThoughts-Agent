#!/bin/bash
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --job-name=delphi-eval
#SBATCH --output=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/%x-%j.log
#
# ==============================================================================
# delphi_eval.sbatch — THIN delphi wrapper around evalchemy_eval.sbatch (#6279).
#
# Sets the Delphi-specific defaults and execs the generic runner in the same SLURM allocation:
#   - CHAT_TEMPLATE_OVERRIDE = chat_templates/delphi_v0.jinja2  (the SFT/RL repos ship a PLAIN
#     Llama-3 template — a train/eval mismatch; delphi_v0 restores the <|start_think|> protocol).
#     Applied only for sft/rl stages by the generic runner; base = raw completion (no override).
#   - MAX_MODEL_LEN=4096 / MAX_GEN_TOKS=3584  (the Delphi checkpoints carry max_position_embeddings=4096
#     with a malformed llama3 rope_scaling block; vLLM derives max_model_len=4096 and rejects 32768).
#   - JOB_PREFIX=delphi-eval  (squeue + %x-%j.log encode the model).
# tensor_parallel_size is computed automatically by the generic runner (auto-TP) — no per-TP copies.
#
# Usage (identical CLI to the old standalone delphi_eval.sbatch):
#   sbatch --job-name="delphi-eval-<RUN_NAME>" delphi_eval.sbatch <MODEL_REPO> <RUN_NAME> [STAGE]
#     $1 MODEL_REPO : HF repo, e.g. laion/delphi-9e19-p33m67-coldstart-tulu3_lr2e5
#     $2 RUN_NAME   : output label (== the cold-start cell label)
#     $3 STAGE      : sft | rl | base   (default: sft)
# ==============================================================================

set -euo pipefail

# Resolve this wrapper's directory so it finds evalchemy_eval.sbatch next to it (repo-relative).
# NOTE: under SLURM the submitted script is copied to /var/spool/slurmd/jobNNN/ and run from there,
# so BASH_SOURCE[0]/dirname resolves to the spool dir (where evalchemy_eval.sbatch does NOT exist) —
# that produced a 4-sec ExitCode-127 "No such file or directory". Pin to the repo clone instead,
# falling back to the BASH_SOURCE dir only for interactive/non-SLURM invocation.
REPO_EVAL_DIR="/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/eval/evalchemy"
if [[ -f "${REPO_EVAL_DIR}/evalchemy_eval.sbatch" ]]; then
  SELF_DIR="${REPO_EVAL_DIR}"
else
  SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Delphi-specific defaults (overridable from the environment).
export CHAT_TEMPLATE_OVERRIDE="${CHAT_TEMPLATE_OVERRIDE:-/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/chat_templates/delphi_v0.jinja2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export MAX_GEN_TOKS="${MAX_GEN_TOKS:-3584}"
export JOB_PREFIX="${JOB_PREFIX:-delphi-eval}"

# Exec the generic runner in this same allocation (its #SBATCH lines are ignored on exec).
exec bash "${SELF_DIR}/evalchemy_eval.sbatch" "$@"
