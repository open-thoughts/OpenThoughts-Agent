#!/bin/bash
# Test script to verify all components work

set -e

echo "=========================================="
echo "Testing vLLM Experiment Framework"
echo "=========================================="
echo ""

# Test 1: Config generator
echo "Test 1: Config generator..."
python config_generator.py --quick-test --output-dir configs
if [ -f "configs/quick_test.json" ]; then
    echo "  ✓ Config generation works"
else
    echo "  ✗ Config generation failed"
    exit 1
fi

# Test 2: Dataset loader
echo ""
echo "Test 2: Dataset loader..."
python dataset_loader.py --num-prompts 5 --output-dir datasets
if [ -f "datasets/custom_5.jsonl" ]; then
    echo "  ✓ Dataset generation works"
else
    echo "  ✗ Dataset generation failed"
    exit 1
fi

# Test 3: Sbatch generator
echo ""
echo "Test 3: SLURM script generator..."
python generate_sbatch.py \
    --single-config configs/quick_test.json \
    --dataset datasets/custom_5.jsonl \
    --request-rate 5.0 \
    --output-dir .

if ls run_*_head.sbatch 1> /dev/null 2>&1; then
    echo "  ✓ Sbatch generation works"
else
    echo "  ✗ Sbatch generation failed"
    exit 1
fi

# Test 4: File structure
echo ""
echo "Test 4: Directory structure..."
for dir in configs datasets results logs; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/ exists"
    else
        echo "  ✗ $dir/ missing"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "All tests passed! ✓"
echo "=========================================="
echo ""
echo "Framework is ready to use."
echo ""
echo "Next steps:"
echo "  1. Update model path in config_generator.py"
echo "  2. Run: ./setup_experiments.sh quick"
echo "  3. Launch: ./launch_all_experiments.sh"
echo ""
