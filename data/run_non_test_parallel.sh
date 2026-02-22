#!/bin/bash
# Run all non-test-generating experiments in parallel.
# Dataset is pre-cached so no HF API rate limiting issues.
set -u

PROJECT_ROOT="/scratch/10000/eguha3/dc-agent"
cd "$PROJECT_ROOT"

# Source secrets
if [ -f "$PROJECT_ROOT/rl/hpc/dotenv/secret.env" ]; then
    source "$PROJECT_ROOT/rl/hpc/dotenv/secret.env"
fi
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/scripts/harbor:$PYTHONPATH"

LOG_DIR="$PROJECT_ROOT/data/nontest_logs"
mkdir -p "$LOG_DIR"

SCRIPTS=(
    "data/exp_8_1_style_transfer/generate_github_issue.py"
    "data/exp_8_1_style_transfer/generate_slack_message.py"
    "data/exp_8_1_style_transfer/generate_stackoverflow_question.py"
    "data/exp_8_1_style_transfer/generate_code_review_comment.py"
    "data/exp_8_1_style_transfer/generate_error_report.py"
    "data/exp_8_2_spec_modality/generate_nl_prose.py"
    "data/exp_8_2_spec_modality/generate_pseudocode.py"
    "data/exp_8_2_spec_modality/generate_io_examples.py"
    "data/exp_8_2_spec_modality/generate_type_signatures.py"
    "data/exp_8_2_spec_modality/generate_failing_test_description.py"
    "data/exp_8_3_granularity/generate_vague.py"
    "data/exp_8_3_granularity/generate_bullets.py"
    "data/exp_8_3_granularity/generate_detailed.py"
    "data/exp_8_5_context_padding/generate_mild.py"
    "data/exp_8_5_context_padding/generate_moderate.py"
    "data/exp_8_5_context_padding/generate_heavy.py"
    "data/exp_8_6_error_state/generate_subtle.py"
    "data/exp_8_6_error_state/generate_structural.py"
    "data/exp_8_10_knowledge_gradient/generate_expert.py"
    "data/exp_8_10_knowledge_gradient/generate_intermediate.py"
    "data/exp_8_10_knowledge_gradient/generate_novice.py"
    "data/exp_1_1_difficulty_mixing/generate_goldilocks.py"
    "data/exp_1_1_difficulty_mixing/generate_70_20_10.py"
    "data/exp_1_1_difficulty_mixing/generate_50_30_20.py"
    "data/exp_1_1_difficulty_mixing/generate_equal_bands.py"
    "data/exp_1_2_instruction_specificity/generate_minimal.py"
    "data/exp_1_2_instruction_specificity/generate_rich.py"
    "data/exp_1_3_test_quality/generate_top50.py"
    "data/exp_1_3_test_quality/generate_top25.py"
    "data/exp_1_4_deduplication/generate_deduplicated.py"
    "data/exp_2_1_dense_rewards/generate_proportional.py"
    "data/exp_2_1_dense_rewards/generate_threshold_80.py"
    "data/exp_4_3_skill_distillation/generate_curated_1k.py"
    "data/exp_5_1_consistency_verification/generate_filtered_solvable.py"
    "data/exp_7_1_dockerfile_curriculum/generate_d1_full.py"
    "data/exp_7_1_dockerfile_curriculum/generate_d3_some.py"
    "data/exp_7_1_dockerfile_curriculum/generate_d5_bare.py"
    "data/exp_7_2_info_dropout/generate_random_30.py"
    "data/exp_7_2_info_dropout/generate_type_specific.py"
    "data/exp_7_5_staged_rewards/generate_staged.py"
    "data/exp_7_6_temporal_reward/generate_speed_bonus.py"
)

echo "Launching ${#SCRIPTS[@]} non-test experiments in parallel..."
echo "Logs at: $LOG_DIR"

PIDS=()
for script in "${SCRIPTS[@]}"; do
    name=$(basename "$script" .py)
    echo "  Starting: $script"
    python "$PROJECT_ROOT/$script" > "$LOG_DIR/${name}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#SCRIPTS[@]} scripts launched. Waiting for completion..."

FAILED=0
SUCCEEDED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    script=${SCRIPTS[$i]}
    if wait $pid; then
        SUCCEEDED=$((SUCCEEDED + 1))
        echo "  DONE: $script"
    else
        FAILED=$((FAILED + 1))
        echo "  FAIL: $script (see log: $LOG_DIR/$(basename "$script" .py).log)"
    fi
done

echo ""
echo "============================================="
echo "Results: $SUCCEEDED succeeded, $FAILED failed out of ${#SCRIPTS[@]} total"
echo "============================================="
