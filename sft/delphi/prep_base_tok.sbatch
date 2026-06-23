#!/bin/bash
# Delphi #6279 — prepare-tok for the 9 BASE models (rows 2&4 base-SFT grid).
# Runs sft/delphi/prepare_delphi_tokenizer.py over each of the 9 collapsed
# marin-community/delphi-* bases, writing prepared-tok dirs into a sibling
# `delphi-prepared-tok-base/` dir (the midtrained `delphi-prepared-tok/` holds
# the 27 midtrained run-ids — keep separate to avoid collision).
# Matches the methodology of the canonical 54-cell midtrained launch_54 grid,
# whose train_configs read model_name_or_path=…/delphi-prepared-tok/<id>-prepared-tok.
#SBATCH --job-name=delphi-base-tokprep
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-prepared-tok-base/tokprep_%j.log
set -eo pipefail
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh
conda activate otagent
source /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/hpc/dotenv/leonardo.env
source ~/secrets.env
export HF_HUB_OFFLINE=1   # bases pre-cached; no compute-node internet
DCFT=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent
OUTROOT=/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-prepared-tok-base
mkdir -p "$OUTROOT"
cd "$DCFT"

BASES=(
  delphi-3e18-447Mparams-1.2Btokens
  delphi-9e18-550Mparams-2.9Btokens
  delphi-2e19-837Mparams-3.6Btokens
  delphi-3e19-998Mparams-5Btokens
  delphi-9e19-1.4Bparams-10.6Btokens
  delphi-2e20-1.9Bparams-14.8Btokens
  delphi-3e20-2.5Bparams-18.6Btokens
  delphi-1e21-3.4Bparams-46.3Btokens
  delphi-1e22-9.7Bparams-160Btokens
)
echo "START $(date) — ${#BASES[@]} bases"
for b in "${BASES[@]}"; do
  OUT="$OUTROOT/$b-prepared-tok"
  if [ -f "$OUT/model.safetensors" ] || ls "$OUT"/model-*.safetensors >/dev/null 2>&1; then
    echo "SKIP $b (already prepared at $OUT)"; continue
  fi
  echo "===== PREP $b -> $OUT  $(date)"
  python sft/delphi/prepare_delphi_tokenizer.py \
    --model "marin-community/$b" \
    --output "$OUT"
  echo "RC=$? base=$b"
done
echo "DONE $(date)"
