#!/bin/bash
# Setup script for vLLM experiments
# This script generates datasets, configs, and sbatch scripts

set -e

echo "=========================================="
echo "vLLM Experiment Setup"
echo "=========================================="
echo ""

# Parse arguments
PHASE="${1:-all}"
DATASET="${2:-medium_balanced.jsonl}"
REQUEST_RATE="${3:-10.0}"

echo "Configuration:"
echo "  Phase: $PHASE"
echo "  Dataset: $DATASET"
echo "  Request Rate: $REQUEST_RATE req/s"
echo ""

# Create directories
echo "Creating directories..."
mkdir -p configs datasets results logs
echo "  ✓ Directories created"
echo ""

# Generate datasets
echo "Generating benchmark datasets..."
python dataset_loader.py --create-suite
echo "  ✓ Datasets created"
echo ""

# Generate configs
echo "Generating experiment configurations..."
if [ "$PHASE" = "quick" ]; then
    python config_generator.py --quick-test
    echo "  ✓ Quick test config created"
elif [ "$PHASE" = "all" ]; then
    python config_generator.py --phase all
    echo "  ✓ All phase configs created"
else
    python config_generator.py --phase "$PHASE"
    echo "  ✓ Phase '$PHASE' configs created"
fi
echo ""

# Generate sbatch scripts
echo "Generating SLURM batch scripts..."
if [ "$PHASE" = "quick" ]; then
    python generate_sbatch.py \
        --single-config configs/quick_test.json \
        --dataset "datasets/quick_test.jsonl" \
        --request-rate "$REQUEST_RATE"
else
    python generate_sbatch.py \
        --config-dir configs \
        --dataset "datasets/$DATASET" \
        --request-rate "$REQUEST_RATE"
fi
echo "  ✓ Batch scripts created"
echo ""

# Count configs
NUM_CONFIGS=$(find configs -name "*.json" -type f | wc -l)
NUM_SCRIPTS=$(ls run_*_head.sbatch 2>/dev/null | wc -l)

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Configs generated: $NUM_CONFIGS"
echo "  Batch scripts: $NUM_SCRIPTS"
echo ""
echo "Next steps:"
echo "  1. Review configs: ls -lh configs/"
echo "  2. Check datasets: ls -lh datasets/"
echo "  3. Launch experiments: ./launch_all_experiments.sh"
echo "  4. Monitor jobs: squeue -u \$USER"
echo "  5. Check logs: tail -f logs/vllm_*.out"
echo ""
echo "After completion:"
echo "  python aggregate_results.py --enrich-configs"
echo ""
