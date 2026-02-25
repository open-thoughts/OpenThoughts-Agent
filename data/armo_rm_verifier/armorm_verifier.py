import os
from pathlib import Path
from data.commons import create_standard_task_toml

'''
Templates and logic for the ArmoRM verifier.
'''

VERIFIER_TEMPLATE = '''
import torch
import os
import sys
import json
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def parse_trajectory_authentic_multiturn(trajectory_path: Path):
    """
    Parses ATIF trajectory JSON into a multi-turn message list.
    Preserves authentic content while mapping sources to roles.
    Ignores the final observation.
    """
    with open(trajectory_path, "r") as f:
        data = json.load(f)
    
    steps = data.get("steps", [])
    messages = []
    num_steps = len(steps)
    
    for i, step in enumerate(steps):
        source = step.get("source")
        message = step.get("message", "")
        
        if source == "user":
            messages.append({"role": "user", "content": message})
            
        elif source == "agent":
            # Build the Assistant's turn (Thought + Tool Calls)
            content = message
            tool_calls = step.get("tool_calls", [])
            for call in tool_calls:
                func = call.get("function_name", "unknown")
                args = json.dumps(call.get("arguments", {}))
                content += f"\\nAction: {func}({args})"
            
            messages.append({"role": "assistant", "content": content})
            
            if i < num_steps - 1:
                observation = step.get("observation", {}).get("results", [])
                obs_content = "\\n".join([res.get("content", "") for res in observation if res.get("content")])
                
                if obs_content:
                    messages.append({"role": "user", "content": f"Observation:\\n{obs_content}"})
                
    return messages

def run_verifier():
    print("--- Starting ArmoRM Verification (Raw Score Mode) ---")
    
    # 1. Load context from Harbor logs
    traj_path = Path("/logs/agent/trajectory.json")
    reward_file = Path("/logs/verifier/reward.txt")
    
    if not traj_path.exists():
        print(f"Error: {traj_path} not found.")
        reward_file.parent.mkdir(parents=True, exist_ok=True)
        reward_file.write_text("0")
        return

    messages = parse_trajectory_authentic_multiturn(traj_path)
    
    if not messages:
        print("Error: No valid steps found in trajectory.")
        reward_file.parent.mkdir(parents=True, exist_ok=True)
        reward_file.write_text("0")
        return

    # Ensure the last message is from the assistant
    if messages[-1]["role"] != "assistant":
        messages.pop()

    print(f"Parsed {len(messages)} turns from trajectory.")

    # 2. Load Model and Tokenizer
    model_id = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    print(f"Loading Reward Model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # 3. Process and Score
    input_ids = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to(model.device)

    with torch.no_grad():
        output = model(input_ids)
        raw_reward = output.score.float().item()

    # 4. Output Reward
    reward_file.parent.mkdir(parents=True, exist_ok=True)
    reward_file.write_text(f"{raw_reward:.6f}")
    
    print(f"Verification Complete.")
    print(f"Final Reward: {raw_reward:.6f}")

if __name__ == "__main__":
    try:
        run_verifier()
    except Exception as e:
        print(f"Verifier Crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''

TEST_SH_TEMPLATE = '''#!/bin/bash
# System-level setup
apt-get update
apt-get install -y curl jq
apt-get install -y python3-pip python3-full

# Install ML dependencies using system-package bypass
# Specifically pinning transformers==4.41.2 for ArmoRM compatibility
pip install --break-system-packages torch transformers==4.41.2 accelerate

# Run the ArmoRM Judge script
python3 -u /tests/test_state.py
'''

RESOURCES_TEMPLATE = '''
    [resources]
    cpus = 8
    memory = "24G"
    storage = "10G"
'''

def inject_armorm_verifier(dataset_dir: str):
    """Adds the ArmoRM verifier files to each task directory."""
    tasks_root = Path(dataset_dir)
    print(f"Injecting ArmoRM verifier into tasks at: {tasks_root}")
    
    # Get the baseline TOML and update the timeout for ArmoRM loading
    base_toml = create_standard_task_toml()
    # Increase verifier timeout from 720 to 1200 seconds
    updated_toml = base_toml.replace("timeout_sec = 720.0", "timeout_sec = 1200.0")
    
    # Append ArmoRM-specific hardware requirements
    armorm_task_toml = updated_toml.strip() + "\n" + RESOURCES_TEMPLATE + "\n"

    for task_dir in tasks_root.iterdir():
        if not task_dir.is_dir(): continue
            
        # Write the customized task.toml
        with open(task_dir / "task.toml", "w") as f:
            f.write(armorm_task_toml)

        # Setup tests directory
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        # Write the python logic to test_state.py
        with open(tests_dir / "test_state.py", "w") as f:
            f.write(VERIFIER_TEMPLATE)
            
        # Write the bash entrypoint
        test_sh_path = tests_dir / "test.sh"
        with open(test_sh_path, "w") as f:
            f.write(TEST_SH_TEMPLATE)
        os.chmod(test_sh_path, 0o755)
        
    print("Verifier injection complete.")
