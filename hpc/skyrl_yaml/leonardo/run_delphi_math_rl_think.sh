#!/bin/bash
# Delphi RL scaling-laws (marin #6279) — MATH-RLVR GRPO with THINKING ENABLED.
# THINKING-ON variant of run_delphi_math_rl.sh. Mirrors that script's VALIDATED-BEST
# hparams (lr=1e-5, max_grad_norm=0.5, entropy OFF, no KL, grpo) but turns ON real
# (non-empty) chain-of-thought in the rollout. Foundation for a frontier-shaped RL HPO.
#
# Runs via skyrl_train.entrypoints.main_base inside the marinskyrl sandbox
# (singularity), fully offline, no Harbor/Daytona. Invoked by
# sbatch_delphi_math_rl_multinode.sh (multi-node Ray) — also works single-node.
#
# Reward/contract: env_class=aime = the Minerva/boxed `Answer: \boxed{}` verifier on
# data built by hpc/skyrl_yaml/leonardo/rl_dataset_prep.py. NOTE the verifier
# (skyrl_gym/envs/aime/utils.compute_score) truncates solution_str to its LAST 300
# chars before the `Answer:\s*([^\n<]+)` regex extract — so it reads the FINAL answer
# AFTER any think block, exactly what we want with thinking ON.
#
# ===========================  HOW THINKING IS ENABLED  =======================
# The delphi_v0 chat template (chat_templates/delphi_v0.jinja2) only INJECTS an EMPTY
# `<|start_think|>\n\n<|end_think|>` prefix into the generation prompt WHEN
# `enable_thinking is defined AND is false` (template lines 72-74). With enable_thinking
# UNSET (the prior STAGE=sft default) OR set TRUE, NO empty block is primed and the model
# is free to emit its own `<|start_think|>...real reasoning...<|end_think|>`.
#
# Two coupled changes vs run_delphi_math_rl.sh make thinking REAL + non-empty:
#   1. generator.batched=false  — the SkyRLGymGenerator NON-batched path is the only one
#      that threads generator.chat_template_kwargs into tokenizer.apply_chat_template
#      (skyrl_gym_generator.py:186). batched=true HARD-RAISES when chat_template_kwargs is
#      set (skyrl_gym_generator.py:111-113: "not compatible with batched=True"), because
#      batched lets the vLLM engine template internally and the kwargs are dropped.
#   2. generator.chat_template_kwargs.enable_thinking=true — passed into apply_chat_template
#      so the empty-block branch is NOT taken; the model generates its own think region.
# (Belt-and-suspenders: the sbatch ALSO re-applies delphi_v0 to the cached tokenizer, so
#  the FSDP policy + the vLLM generator both render the same think protocol.)
# =============================================================================
#
# Env (from sbatch): DATA_DIR MODEL_PATH NUM_GPUS CKPT_DIR RUN_NAME [LOGGER VENV_PY]
#                    POLICY_NUM_NODES (multi-node placement; default 1)
set -x

: "${DATA_DIR:?set DATA_DIR (=\$WORK/data/rl/rlvr_math, built by rl_dataset_prep.py)}"
: "${MODEL_PATH:?set MODEL_PATH (the starting checkpoint: SFT repo, midtrained, or base)}"
: "${NUM_GPUS:=4}"                     # GPUs PER NODE (Leonardo A100x4)
: "${CKPT_DIR:?set CKPT_DIR (fresh per-cell)}"
: "${RUN_NAME:=delphi_math_rl_think}"
: "${LOGGER:=console}"                 # offline -> console (WANDB_MODE=offline)
: "${INFERENCE_BACKEND:=vllm}"
: "${VENV_PY:=python}"

# --- VALIDATED-BEST hparams (carried verbatim from run_delphi_math_rl.sh; sweep cell
#     gc05_lr1e5_e0 47447531: pass@8 0.75, +reward, entropy bounded). DO NOT re-tune here.
: "${LR:=1.0e-5}"                      # VALIDATED-BEST
: "${MAX_GRAD_NORM:=0.5}"              # cheap brake on runaway entropy
: "${USE_ENTROPY_LOSS:=false}"         # entropy regularization OFF (the +0.01 bonus drove the explosion)
: "${ENTROPY_COEF:=0.0}"               # only applies if USE_ENTROPY_LOSS=true
: "${N_SAMPLES:=8}"                    # GRPO group size

# --- SMOKE budget: short, ckpt OFF, no HF upload, modest batch (fits 8xA100, fast).
: "${MAX_GEN_LEN:=3584}"               # 4k ceiling - 512 prompt; think block + boxed answer must fit
: "${MAX_PROMPT_LEN:=512}"
: "${TEMPERATURE:=0.7}"
: "${TOP_P:=1.0}"
: "${TBS:=64}"                         # 64 prompts x n8 = 512 episodes/step; dp=8 -> 64 ep/GPU
: "${EPOCHS:=20}"
: "${MAX_STEPS:=10}"                   # SMOKE: a few steps to prove multi-node + thinking
: "${HF_SAVE_INTERVAL:=20}"            # HF-format export every N steps. main_base already registers
                                       # HFModelSaveCallback via DefaultCallbackHandler when >0 (-1=off was the
                                       # silent default → the grid's empty exports/). 20: fast Leonardo steps,
                                       # ~20GB per 9.7B export → keeps periodic models without flooding disk.
: "${MICRO_FWD:=4}"
: "${MICRO_TRAIN:=2}"
: "${ENV_CLASS:=aime}"                 # boxed/answer-match (Minerva)

# THINKING toggle (default ON for this script). Set THINK=false to A/B against thinking-off.
: "${THINK:=true}"

# Multi-node placement. POLICY_NUM_NODES drives the FSDP policy/ref mesh across nodes.
# Default 1 = single-node (back-compatible). 2 = the 2-node smoke.
: "${POLICY_NUM_NODES:=1}"

# 4 engines per node x TP1 (colocated). Total engines = NUM_GPUS/TP * POLICY_NUM_NODES,
# i.e. one inference engine per GPU, spread across all nodes by SkyRL's placement.
TP=1
ENGINES=$(( NUM_GPUS / TP * POLICY_NUM_NODES ))

# THINK_MODE — how real (non-empty) thinking is driven into the rollout:
#   'kwarg'  (default; the validated smoke path): generator.batched=false +
#            chat_template_kwargs.enable_thinking — relies on the MODEL to populate the
#            think region. Smoke 47525016 showed THIS sft ckpt emits an EMPTY block here.
#   'forced' (the bake-in / fast path): generator.batched=true (FAST) and NO kwarg
#            (batched refuses chat_template_kwargs). Thinking is FORCED by the template
#            itself — pass DELPHI_TEMPLATE=.../delphi_v0_think.jinja2 to the sbatch so the
#            cached tokenizer (used by BOTH the vLLM engine AND the policy → consistent)
#            PREFILLS '<|start_think|>\n', so the response MUST begin inside the think region.
: "${THINK_MODE:=kwarg}"
if [ "$THINK_MODE" = "forced" ]; then
  THINK_ARGS=( generator.batched=true )
else
  THINK_ARGS=( generator.batched=false "+generator.chat_template_kwargs.enable_thinking=$THINK" )
fi

"$VENV_PY" -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=$POLICY_NUM_NODES \
  trainer.placement.ref_num_nodes=$POLICY_NUM_NODES \
  trainer.placement.critic_num_nodes=$POLICY_NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP \
  trainer.epochs=$EPOCHS \
  trainer.max_steps=$MAX_STEPS \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=false \
  trainer.eval_interval=0 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TBS \
  trainer.policy_mini_batch_size=$TBS \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FWD \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN \
  trainer.ckpt_interval=999999 \
  trainer.hf_save_interval=$HF_SAVE_INTERVAL \
  trainer.max_prompt_length=$MAX_PROMPT_LEN \
  generator.sampling_params.max_generate_length=$MAX_GEN_LEN \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.max_num_batched_tokens=8192 \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.max_grad_norm=$MAX_GRAD_NORM \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_entropy_loss=$USE_ENTROPY_LOSS \
  trainer.algorithm.entropy_loss_coef=$ENTROPY_COEF \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  "${THINK_ARGS[@]}" \
  generator.enforce_eager=false \
  environment.env_class=$ENV_CLASS \
  environment.skyrl_gym.aime.max_gen_length=$MAX_GEN_LEN \
  generator.n_samples_per_prompt=$N_SAMPLES \
  generator.gpu_memory_utilization=0.85 \
  generator.vllm_stats_interval=1 \
  trainer.logger="$LOGGER" \
  trainer.project_name="delphi_rl_scaling_6279" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_DIR" \
  trainer.export_path="$CKPT_DIR/exports" \
  "$@"
