#!/bin/bash
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --job-name=delphi-eval
#SBATCH --output=/leonardo_work/AIFAC_5C0_290/%u/experiments/delphi-eval/%x-%j.log
#
# PREREQUISITE (multi-operator): the --output directory above must EXIST BEFORE SUBMIT.
# SLURM opens the log file before the job script runs and does NOT create intermediate
# directories -- a missing dir fails the job instantly with "Unable to open file", which
# looks like a launch bug rather than a missing mkdir. It already exists for the original
# operator; a NEW operator runs this once:
#   mkdir -p /leonardo_work/AIFAC_5C0_290/$USER/experiments/delphi-eval
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
# $USER-scoped so a second operator runs THEIR checkout, not another user's. Resolves to the
# original bfeuer00 paths byte-identically when USER=bfeuer00.
#
# Resolution is by PROBE, not by assumption: try each candidate and take the first that actually
# contains the marker file. A bare "$WORK/$USER/code/OpenThoughts-Agent" default only works for
# operators who cloned at exactly that path; anyone else fell through to the BASH_SOURCE branch,
# which under SLURM is the /var/spool/slurmd/jobNNN/ copy — i.e. it re-created the very
# ExitCode-127 "No such file or directory" this block was written to prevent. It also silently
# mis-pointed CHAT_TEMPLATE_OVERRIDE below (and THINK_TEMPLATE in delphi_rl_think_eval.sbatch).
_OTA_WORK="/leonardo_work/AIFAC_5C0_290/${USER:-$(id -un)}"
_OTA_SHARED="/leonardo_work/AIFAC_5C0_290/bfeuer00"
_OTA_MARKER="eval/evalchemy/evalchemy_eval.sbatch"
# Walk up from a directory looking for the marker: SLURM_SUBMIT_DIR is wherever the operator ran
# sbatch, which is often a subdirectory of the repo rather than its root.
_ota_up() {
  local d="${1:-}"
  [ -n "$d" ] || return 1
  while [ -n "$d" ] && [ "$d" != "/" ]; do
    [ -f "$d/${_OTA_MARKER}" ] && { printf '%s' "$d"; return 0; }
    d="$(dirname "$d")"
  done
  return 1
}
_ota_first_repo() {
  local c
  for c in "$@"; do
    [ -n "$c" ] && [ -f "$c/${_OTA_MARKER}" ] && { printf '%s' "$c"; return 0; }
  done
  return 1
}
_OTA_REPO="$(_ota_first_repo \
  "${OTAGENT_REPO_DIR:-}" \
  "${_OTA_WORK}/code/OpenThoughts-Agent" \
  "$(_ota_up "${SLURM_SUBMIT_DIR:-}" || true)" \
  "${_OTA_SHARED}/code/OpenThoughts-Agent" \
  "$(_ota_up "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)" || true)" \
  || true)"
if [ -z "${_OTA_REPO}" ]; then
  echo "ERROR: cannot locate an OpenThoughts-Agent checkout containing ${_OTA_MARKER}." >&2
  echo "  tried, in order:" >&2
  echo "    OTAGENT_REPO_DIR   = ${OTAGENT_REPO_DIR:-<unset>}" >&2
  echo "    \$USER checkout     = ${_OTA_WORK}/code/OpenThoughts-Agent" >&2
  echo "    SLURM_SUBMIT_DIR   = ${SLURM_SUBMIT_DIR:-<unset>} (and parents)" >&2
  echo "    shared checkout    = ${_OTA_SHARED}/code/OpenThoughts-Agent" >&2
  echo "    script dir         = $(dirname "${BASH_SOURCE[0]}") (and parents)" >&2
  echo "  Set OTAGENT_REPO_DIR=/path/to/OpenThoughts-Agent and resubmit." >&2
  exit 1
fi
SELF_DIR="${_OTA_REPO}/eval/evalchemy"

# Delphi-specific defaults (overridable from the environment).
export CHAT_TEMPLATE_OVERRIDE="${CHAT_TEMPLATE_OVERRIDE:-${_OTA_REPO}/chat_templates/delphi_v0.jinja2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export MAX_GEN_TOKS="${MAX_GEN_TOKS:-3584}"
export JOB_PREFIX="${JOB_PREFIX:-delphi-eval}"

# Exec the generic runner in this same allocation (its #SBATCH lines are ignored on exec).
exec bash "${SELF_DIR}/evalchemy_eval.sbatch" "$@"
