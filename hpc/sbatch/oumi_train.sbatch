#!/bin/bash -x
#SBATCH --nodes={num_nodes}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --mem={mem_per_node}
#SBATCH --time={time_limit}
#SBATCH --job-name={job_name}
#SBATCH --output={experiments_dir}/logs/%x_%j.out

set -e

# Environment activation (expected to be defined in dotenv/oumi.env)
if [ -n "$DCFT_ACTIVATE_ENV" ]; then
    eval "$DCFT_ACTIVATE_ENV"
fi

SECRET_FILE="${DC_AGENT_SECRET_ENV:-${KEYS:-}}"
if [[ -n "${SECRET_FILE}" ]]; then
    if [[ -f "${SECRET_FILE}" ]]; then
        echo "Sourcing secrets from ${SECRET_FILE}"
        set -a
        # shellcheck disable=SC1090
        source "${SECRET_FILE}"
        set +a
    else
        echo "Warning: secrets file ${SECRET_FILE} not found; database registration may fail." >&2
    fi
else
    echo "Warning: DC_AGENT_SECRET_ENV is not set; database registration may fail." >&2
fi

for _supabase_var in SUPABASE_URL SUPABASE_ANON_KEY SUPABASE_SERVICE_ROLE_KEY; do
    if [[ -n "${!_supabase_var:-}" ]]; then
        export "${_supabase_var}=${!_supabase_var}"
    else
        echo "Warning: ${_supabase_var} is not set; Supabase registration may fail." >&2
    fi
done

# Basic NCCL networking configuration (adjust if needed)
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=120
export OMP_NUM_THREADS=1

# Rendezvous details
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=20043
export NUM_NODES=$SLURM_JOB_NUM_NODES
export NUM_GPUS_PER_NODE={gpus_per_node}
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))

# Paths and config
CONFIG={train_config_path_out}
OUTPUT_DIR={experiments_dir}
echo -e "CONFIG: $CONFIG\nOUTPUT_DIR: $OUTPUT_DIR"
TMP_DIR=$OUTPUT_DIR/tmp
mkdir -p "$OUTPUT_DIR" "$TMP_DIR"

# Accelerate configuration
{accelerate_config_block}

# Main command
LAUNCHER="python -u -m accelerate.commands.launch \
    --rdzv_conf \"rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT\" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --machine_rank \\${SLURM_PROCID} \
    --role \\$(hostname -s): --tee 3 \
    "

CMD="$LAUNCHER sft/llamafactory/src/train.py $CONFIG"

SRUN_ARGS="
    --nodes=$NUM_NODES \
    --gpus-per-node=$NUM_GPUS_PER_NODE \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --label \
    --jobid $SLURM_JOBID"

# Run
cd "$DCFT"
srun $SRUN_ARGS bash -c "$CMD"

# Cleanup
rm -f "$ACCELERATE_CONFIG_FILE"
