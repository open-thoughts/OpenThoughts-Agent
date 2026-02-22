#!/bin/bash
# Run all non-test-generating experiments to produce 10k task datasets.
# These experiments keep original tests — only instruction/dockerfile/reward changes.
# Each script uploads to its own HF repo.
set -e
cd "$(dirname "$0")/.."

# Source environment
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source secrets if available
if [ -f "$PROJECT_ROOT/rl/hpc/dotenv/secret.env" ]; then
    source "$PROJECT_ROOT/rl/hpc/dotenv/secret.env"
elif [ -f /scratch/10000/eguha3/old-dc-agent/secret.env ]; then
    source /scratch/10000/eguha3/old-dc-agent/secret.env
fi

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/scripts/harbor:$PYTHONPATH"

echo "============================================="
echo "Running non-test-generating experiments (10k)"
echo "============================================="

# Theme 8 (instruction-only rewrites)
SCRIPTS_8=(
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
)

# Theme 1
SCRIPTS_1=(
    "data/exp_1_1_difficulty_mixing/generate_goldilocks.py"
    "data/exp_1_1_difficulty_mixing/generate_70_20_10.py"
    "data/exp_1_1_difficulty_mixing/generate_50_30_20.py"
    "data/exp_1_1_difficulty_mixing/generate_equal_bands.py"
    "data/exp_1_2_instruction_specificity/generate_minimal.py"
    "data/exp_1_2_instruction_specificity/generate_rich.py"
    "data/exp_1_3_test_quality/generate_top50.py"
    "data/exp_1_3_test_quality/generate_top25.py"
    "data/exp_1_4_deduplication/generate_deduplicated.py"
)

# Theme 2
SCRIPTS_2=(
    "data/exp_2_1_dense_rewards/generate_proportional.py"
    "data/exp_2_1_dense_rewards/generate_threshold_80.py"
)

# Theme 4
SCRIPTS_4=(
    "data/exp_4_3_skill_distillation/generate_curated_1k.py"
)

# Theme 5 (non-test only)
SCRIPTS_5=(
    "data/exp_5_1_consistency_verification/generate_filtered_solvable.py"
)

# Theme 7 (non-test only)
SCRIPTS_7=(
    "data/exp_7_1_dockerfile_curriculum/generate_d1_full.py"
    "data/exp_7_1_dockerfile_curriculum/generate_d3_some.py"
    "data/exp_7_1_dockerfile_curriculum/generate_d5_bare.py"
    "data/exp_7_2_info_dropout/generate_random_30.py"
    "data/exp_7_2_info_dropout/generate_type_specific.py"
    "data/exp_7_5_staged_rewards/generate_staged.py"
    "data/exp_7_6_temporal_reward/generate_speed_bonus.py"
)

ALL_SCRIPTS=("${SCRIPTS_8[@]}" "${SCRIPTS_1[@]}" "${SCRIPTS_2[@]}" "${SCRIPTS_4[@]}" "${SCRIPTS_5[@]}" "${SCRIPTS_7[@]}")

echo "Total scripts to run: ${#ALL_SCRIPTS[@]}"
echo ""

for script in "${ALL_SCRIPTS[@]}"; do
    echo "---------------------------------------------"
    echo "Running: $script"
    echo "---------------------------------------------"
    python "$script" 2>&1 | tail -5
    echo ""
done

echo "============================================="
echo "All non-test experiments complete!"
echo "============================================="
