#!/bin/bash
# Launch remaining experiments with gpt-5-mini in parallel
set -u
source /scratch/10000/eguha3/old-dc-agent/secret.env
cd /scratch/10000/eguha3/dc-agent

MODEL="openai/gpt-5-mini-2025-08-07"
CONFIG="eval/tacc/dcagent_eval_config.yaml"
PID=$$
echo "PID=$PID"

launch() {
    local name="$1"
    local task_dir="$2"
    local job_name="mini_${name}_${PID}"
    echo "Launching: ${job_name}"
    harbor jobs start \
        -p "$task_dir" \
        --n-concurrent 8 \
        --agent terminus-2 \
        --model "$MODEL" \
        --env daytona \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$job_name" \
        --config "$CONFIG" &
}

# Theme 8 remaining
launch "8.10_expert" "/scratch/10000/eguha3/data_cache/exp-8-10-ex_ztii28ub"
launch "8.10_intermediate" "/scratch/10000/eguha3/data_cache/exp-8-10-in_r6iikszg"
launch "8.10_novice" "/scratch/10000/eguha3/data_cache/exp-8-10-no_2inug0fi"
launch "8.1_code_review" "/scratch/10000/eguha3/data_cache/exp-8-1-cr_4o88k5a7"
launch "8.1_error_report" "/scratch/10000/eguha3/data_cache/exp-8-1-er_rync2dc2"
launch "8.1_github_issue" "/scratch/10000/eguha3/data_cache/exp-8-1-gh__ixxh63q"
launch "8.1_slack_message" "/scratch/10000/eguha3/data_cache/exp-8-1-sl_ywyqsujf"
launch "8.1_stackoverflow" "/scratch/10000/eguha3/data_cache/exp-8-1-so_nwpd5yu2"
launch "8.2_failing_test" "/scratch/10000/eguha3/data_cache/exp-8-2-ft_2q3zfxfc"
launch "8.2_io_examples" "/scratch/10000/eguha3/data_cache/exp-8-2-io_ue_2blrg"
launch "8.2_nl_prose" "/scratch/10000/eguha3/data_cache/exp-8-2-nl_6a5n0yy_"
launch "8.2_pseudocode" "/scratch/10000/eguha3/data_cache/exp-8-2-ps_1jqp3xad"
launch "8.2_type_signatures" "/scratch/10000/eguha3/data_cache/exp-8-2-ts_cgjg_2n9"
launch "8.3_bullets" "/scratch/10000/eguha3/data_cache/exp-8-3-bl_r0pg_vz9"
launch "8.3_detailed" "/scratch/10000/eguha3/data_cache/exp-8-3-dt_w9h1bhql"
launch "8.3_vague" "/scratch/10000/eguha3/data_cache/exp-8-3-vg_r_n1tss9"
launch "8.5_heavy" "/scratch/10000/eguha3/data_cache/exp-8-5-hv_1x0e8rvb"
launch "8.5_mild" "/scratch/10000/eguha3/data_cache/exp-8-5-mi_t5lcxdt6"
launch "8.5_moderate" "/scratch/10000/eguha3/data_cache/exp-8-5-mo_v7xwd7r8"

# Other themes remaining
launch "baseline_source" "/scratch/10000/eguha3/data_cache/baseline_h8w46h4o"
launch "1.1_goldilocks" "/scratch/10000/eguha3/data_cache/exp-1-1-goldilocks_eth979_8"
launch "1.1_70_20_10" "/scratch/10000/eguha3/data_cache/exp-1-1-70-20-10_3nhq_egq"
launch "1.2_minimal" "/scratch/10000/eguha3/data_cache/exp-1-2-minimal_24dzgwa8"
launch "1.2_rich" "/scratch/10000/eguha3/data_cache/exp-1-2-rich_gh4_v1iv"
launch "1.4_deduplicated" "/scratch/10000/eguha3/data_cache/exp-1-4-dedup_2uagiowg"
launch "4.3_curated_1k" "/scratch/10000/eguha3/data_cache/exp-4-3-curated-1k_4git9_w5"
launch "5.1_solvable" "/scratch/10000/eguha3/data_cache/exp-5-1-solvable_yxxq3c2a"
launch "7.1_d1_full" "/scratch/10000/eguha3/data_cache/exp-7-1-d1-full_w9tjv6sb"
launch "7.1_d5_bare" "/scratch/10000/eguha3/data_cache/exp-7-1-d5-bare_vc9gdpm3"
launch "7.2_random_30" "/scratch/10000/eguha3/data_cache/exp-7-2-dropout30_iycegoc7"
launch "7.6_speed_bonus" "/scratch/10000/eguha3/data_cache/exp-7-6-speed_1tfj3asn"

echo "All launched"
wait
echo "Done"
