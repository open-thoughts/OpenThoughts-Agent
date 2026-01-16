#!/usr/bin/env python3
"""Generate ablation experiment configuration files."""

import yaml
from pathlib import Path

CONFIGS_DIR = Path(__file__).parent / "configs"
CONFIGS_DIR.mkdir(exist_ok=True)

# Output dataset prefix
DATASET_PREFIX = "exp_tas_"


def get_base_config():
    return {
        "input_repo_id": "mlfoundations-dev/stackexchange-tezos-sandboxes",
        "agent": {
            "name": "terminus-2",
            "model_name": "hosted_vllm/glm",
            "kwargs": {
                "temperature": 1.0,
                "max_episodes": 128,
                "enable_summarize": True,
                "proactive_summarization_threshold": 8192,
                "interleaved_thinking": False,
                "parser_name": "json",
                "tmux_pane_width": 160,
                "tmux_pane_height": 40,
                "trajectory_config": {
                    "raw_content": True,
                    "linear_history": True,
                },
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                    },
                },
            },
        },
        "job": {
            "n_concurrent": 64,
            "n_attempts": 1,
            "max_retries": 0,
            "timeout_multiplier": 2.0,
            "disable_verification": True,
        },
    }


def write_config(name: str, description: str, config: dict):
    config["experiment_name"] = name
    config["description"] = description
    config["output_repo_suffix"] = f"{DATASET_PREFIX}{name}"

    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Created: {path}")


# =============================================================================
# BASELINE
# =============================================================================
config = get_base_config()
write_config("baseline", "Baseline: temp=1.0, max_episodes=128", config)

# =============================================================================
# TEMPERATURE (2x increments)
# =============================================================================
for temp in [0.25, 0.5, 2.0, 4.0]:
    config = get_base_config()
    config["agent"]["kwargs"]["temperature"] = temp
    write_config(f"temp_{temp}", f"Temperature = {temp}", config)

# =============================================================================
# TOP-P (nucleus sampling)
# =============================================================================
for top_p in [0.8, 0.9, 0.95]:
    config = get_base_config()
    config["agent"]["kwargs"]["extra_completion_params"] = {"top_p": top_p}
    write_config(f"top_p_{top_p}", f"Top-p nucleus sampling = {top_p}", config)

# =============================================================================
# TOP-K (2x increments)
# =============================================================================
for top_k in [16, 32, 64, 128]:
    config = get_base_config()
    config["agent"]["kwargs"]["extra_body"]["top_k"] = top_k
    write_config(f"top_k_{top_k}", f"Top-k sampling = {top_k}", config)

# =============================================================================
# MIN-P (vLLM sampling)
# =============================================================================
for min_p in [0.01, 0.05, 0.1]:
    config = get_base_config()
    config["agent"]["kwargs"]["extra_body"]["min_p"] = min_p
    write_config(f"min_p_{min_p}", f"Min-p sampling = {min_p}", config)

# =============================================================================
# REPETITION PENALTY
# =============================================================================
for rep_pen in [1.05, 1.1, 1.2]:
    config = get_base_config()
    config["agent"]["kwargs"]["extra_body"]["repetition_penalty"] = rep_pen
    write_config(f"repetition_penalty_{rep_pen}", f"Repetition penalty = {rep_pen}", config)

# =============================================================================
# PRESENCE PENALTY
# =============================================================================
for pres_pen in [0.25, 0.5, 1.0]:
    config = get_base_config()
    config["agent"]["kwargs"]["extra_completion_params"] = {"presence_penalty": pres_pen}
    write_config(f"presence_penalty_{pres_pen}", f"Presence penalty = {pres_pen}", config)

# =============================================================================
# FREQUENCY PENALTY
# =============================================================================
for freq_pen in [0.25, 0.5, 1.0]:
    config = get_base_config()
    config["agent"]["kwargs"]["extra_completion_params"] = {"frequency_penalty": freq_pen}
    write_config(f"frequency_penalty_{freq_pen}", f"Frequency penalty = {freq_pen}", config)

# =============================================================================
# MAX TOKENS (2x increments)
# =============================================================================
for max_tok in [1024, 2048, 4096, 8192]:
    config = get_base_config()
    config["agent"]["kwargs"]["extra_completion_params"] = {"max_tokens": max_tok}
    write_config(f"max_tokens_{max_tok}", f"Max tokens = {max_tok}", config)

# =============================================================================
# MAX EPISODES (2x increments)
# =============================================================================
for max_ep in [32, 64, 256, 512]:
    config = get_base_config()
    config["agent"]["kwargs"]["max_episodes"] = max_ep
    write_config(f"max_episodes_{max_ep}", f"Max episodes = {max_ep}", config)

# =============================================================================
# PROACTIVE SUMMARIZATION THRESHOLD (2x increments)
# =============================================================================
for threshold in [2048, 4096, 16384, 32768]:
    config = get_base_config()
    config["agent"]["kwargs"]["proactive_summarization_threshold"] = threshold
    write_config(
        f"summarize_threshold_{threshold}",
        f"Proactive summarization threshold = {threshold} tokens",
        config,
    )

# =============================================================================
# PARSER NAME
# =============================================================================
config = get_base_config()
config["agent"]["kwargs"]["parser_name"] = "xml"
write_config("parser_xml", "Parser = xml (instead of json)", config)

# =============================================================================
# TMUX PANE SIZE
# =============================================================================
config = get_base_config()
config["agent"]["kwargs"]["tmux_pane_width"] = 240
config["agent"]["kwargs"]["tmux_pane_height"] = 60
write_config("tmux_large", "Large terminal: 240x60", config)

# =============================================================================
# BINARY TOGGLES
# =============================================================================
config = get_base_config()
config["agent"]["kwargs"]["interleaved_thinking"] = True
write_config("interleaved_thinking_on", "Interleaved thinking enabled", config)

config = get_base_config()
config["agent"]["kwargs"]["trajectory_config"]["raw_content"] = False
write_config("raw_content_off", "Raw content disabled (parsed tool_calls)", config)

config = get_base_config()
config["agent"]["kwargs"]["trajectory_config"]["linear_history"] = False
write_config("linear_history_off", "Linear history disabled (single trajectory file)", config)

config = get_base_config()
config["agent"]["kwargs"]["enable_summarize"] = False
write_config("summarize_off", "Context summarization disabled", config)

# =============================================================================
# COMBINED CONFIGS
# =============================================================================
config = get_base_config()
config["agent"]["kwargs"]["temperature"] = 2.0
config["agent"]["kwargs"]["extra_completion_params"] = {"top_p": 0.95}
config["agent"]["kwargs"]["extra_body"]["top_k"] = 64
write_config(
    "high_diversity",
    "High diversity: temp=2.0, top_p=0.95, top_k=64",
    config,
)

config = get_base_config()
config["agent"]["kwargs"]["temperature"] = 0.25
config["agent"]["kwargs"]["extra_completion_params"] = {"top_p": 0.8}
write_config(
    "low_diversity",
    "Low diversity: temp=0.25, top_p=0.8",
    config,
)

config = get_base_config()
config["agent"]["kwargs"]["interleaved_thinking"] = True
config["agent"]["kwargs"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] = True
write_config(
    "full_thinking",
    "Full thinking: interleaved_thinking=true, enable_thinking=true",
    config,
)

config = get_base_config()
config["agent"]["kwargs"]["trajectory_config"]["raw_content"] = False
config["agent"]["kwargs"]["trajectory_config"]["linear_history"] = False
write_config(
    "trajectory_minimal",
    "Minimal trajectory: raw_content=false, linear_history=false",
    config,
)

print(f"\nGenerated {len(list(CONFIGS_DIR.glob('*.yaml')))} config files")
