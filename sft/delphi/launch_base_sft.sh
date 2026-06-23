#!/bin/bash
# Delphi #6279 — launch the 18-cell BASE-model SFT grid (priority-queue rows 2&4)
# on Leonardo. 9 bases × 2 recipes {magpie (math-strong), wc386k (math-weak)}.
#
# Config: sft/lf_configs/delphi/4k_sft_ckpt2k_dsfs.yaml (lr 1e-5, save_steps 2000,
#   data_shared_file_system: true, template delphi, cutoff 4096, 1 epoch, push_to_hub false).
# model_path -> the per-base prepared-tok dir (matches the canonical launch_54 midtrained grid).
# 90/10 instruction+delphi_warmup mix via the sft/delphi registry. HF-only, NO DB.
#
# CONCURRENCY = 8 (held via afterany gating): cells 1..8 gate only on the prep job;
# cells 9..18 each gate on prep + the (i-8)-th cell, so <=8 of THIS grid's SFT jobs
# are ever eligible (un-gated) at once.
#
# Usage:  bash launch_base_sft.sh <PREP_JOBID>
#   (PREP_JOBID = the sft/delphi/prep_base_tok.sbatch job; all SFT cells gate afterany on it.)
set -eo pipefail

PREP_JID="${1:?usage: launch_base_sft.sh <PREP_JOBID>}"
DCFT=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent
PREPROOT=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-prepared-tok-base
CFG=sft/lf_configs/delphi/4k_sft_ckpt2k_dsfs.yaml
LOG=$PREPROOT/launch_base_sft.log
MAP=$PREPROOT/launch_base_sft_map.tsv
cd "$DCFT"
mkdir -p "$PREPROOT"
: > "$MAP"
echo "DRIVER START $(date) PREP_JID=$PREP_JID" | tee -a "$LOG"

# base-id  nodes
BASES=(
  "delphi-3e18-447Mparams-1.2Btokens 1"
  "delphi-9e18-550Mparams-2.9Btokens 1"
  "delphi-2e19-837Mparams-3.6Btokens 1"
  "delphi-3e19-998Mparams-5Btokens 1"
  "delphi-9e19-1.4Bparams-10.6Btokens 2"
  "delphi-2e20-1.9Bparams-14.8Btokens 2"
  "delphi-3e20-2.5Bparams-18.6Btokens 2"
  "delphi-1e21-3.4Bparams-46.3Btokens 4"
  "delphi-1e22-9.7Bparams-160Btokens 4"
)
RECIPES=( "magpie magpie" "wc386k wildchat_386k" )

# Build the ordered cell list (base-major, recipe-minor) -> 18 cells.
declare -a CELL_BASE CELL_NODES CELL_RLABEL CELL_INSTR
for entry in "${BASES[@]}"; do
  read -r BID NODES <<<"$entry"
  for r in "${RECIPES[@]}"; do
    read -r RLABEL INSTR <<<"$r"
    CELL_BASE+=("$BID"); CELL_NODES+=("$NODES"); CELL_RLABEL+=("$RLABEL"); CELL_INSTR+=("$INSTR")
  done
done
N=${#CELL_BASE[@]}   # 18

declare -a CELL_JID
for i in $(seq 0 $((N-1))); do
  BID=${CELL_BASE[$i]}; NODES=${CELL_NODES[$i]}; RLABEL=${CELL_RLABEL[$i]}; INSTR=${CELL_INSTR[$i]}
  MODELP="$PREPROOT/$BID-prepared-tok"
  HUBID="laion/$BID-base-${RLABEL}_lr1e5-sft"

  # dependency: always behind prep; cells 9..18 also behind the (i-8)-th cell (8-cap).
  DEP="afterany:$PREP_JID"
  if [ "$i" -ge 8 ]; then
    DEP="$DEP:${CELL_JID[$((i-8))]}"
  fi

  echo "===== LAUNCH [$((i+1))/$N] $BID $RLABEL nodes=$NODES dep=$DEP -> $HUBID  $(date)" | tee -a "$LOG"

  OUT=$(DISABLE_VERSION_CHECK=1 python -m hpc.launch \
    --train_config_path "$CFG" \
    --time_limit 23:59:00 \
    --num_nodes "$NODES" --gpus_per_node 4 \
    --model_path "$MODELP" \
    --dataset_dir sft/delphi \
    --dataset "$INSTR,delphi_warmup" \
    --mix_strategy interleave_under --interleave_probs 0.9,0.1 \
    --hub_model_id "$HUBID" \
    --internet_node \
    --max_restarts 2 \
    --dependency "$DEP" 2>&1)
  echo "$OUT" | tee -a "$LOG"

  JID=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+|Job [0-9]+ submitted' | grep -oE '[0-9]+' | tail -1)
  CELL_JID+=("$JID")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$JID" "$BID" "$RLABEL" "$NODES" "$DEP" "$HUBID" | tee -a "$MAP"
done
echo "DRIVER DONE $(date) — $N cells" | tee -a "$LOG"
echo "MAP: $MAP"
