import os
from pathlib import Path
from data.commons import create_standard_task_toml

'''
Templates and logic for the ArmoRM Reward Verifier (Response File Mode).
This version requires the agent to write its final answer to 'response.txt'.
Instruction is parsed from the trajectory log.
'''

VERIFIER_TEMPLATE = '''
import torch
import os
import sys
import json
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def get_instruction_from_trajectory(trajectory_path: Path):
    """Extracts the core instruction from the first step of the trajectory."""
    with open(trajectory_path, "r") as f:
        data = json.load(f)
    
    steps = data.get("steps", [])
    if not steps:
        return ""
        
    for i, step in enumerate(steps):
        if step.get("source") == "user" and i == 0:
            message = step.get("message", "")
            
            # Extract content between "Task Description:" and "Current terminal state:"
            if "Task Description:" in message:
                content = message.split("Task Description:", 1)[1]
            else:
                content = message
                
            if "Current terminal state:" in content:
                content = content.split("Current terminal state:", 1)[0]
                
            return content.strip()
    return ""

def run_verifier():
    print("--- Starting ArmoRM Verification (Response File Mode) ---")
    
    # 1. Load context
    traj_path = Path("/logs/agent/trajectory.json")
    response_path = Path("response.txt")
    reward_file = Path("/logs/verifier/reward.txt")
    
    if not traj_path.exists():
        print(f"Error: {traj_path} not found.")
        reward_file.parent.mkdir(parents=True, exist_ok=True)
        reward_file.write_text("0.0")
        return

    if not response_path.exists():
        print("Error: response.txt not found. Agent failed to provide the required deliverable.")
        reward_file.parent.mkdir(parents=True, exist_ok=True)
        reward_file.write_text("0.0")
        return

    # 2. Extract Instruction and Answer
    user_prompt = get_instruction_from_trajectory(traj_path)
    agent_answer = response_path.read_text().strip()

    if not user_prompt:
        print("Error: Could not parse instruction from trajectory.")
        reward_file.parent.mkdir(parents=True, exist_ok=True)
        reward_file.write_text("0.0")
        return

    if not agent_answer:
        print("Error: response.txt is empty.")
        reward_file.parent.mkdir(parents=True, exist_ok=True)
        reward_file.write_text("0.0")
        return

    # 3. Format for ArmoRM (Two-Turn)
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": agent_answer}
    ]
    
    print(f"DEBUG: Messages sent to model: {json.dumps(messages, indent=2)}")

    # 4. Load Model and Tokenizer
    model_id = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    print(f"Loading Reward Model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        device_map="auto",
        trust_remote_code=True,
    )

    # 5. Process and Score
    input_ids = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to(model.device)

    with torch.no_grad():
        output = model(input_ids)
        raw_reward = output.score.float().item()

    # 6. Output Reward
    reward_file.parent.mkdir(parents=True, exist_ok=True)
    reward_file.write_text(f"{raw_reward:.6f}")
    
    print(f"Verification Complete.")
    print(f"Final Reward (Raw): {raw_reward:.6f}")

if __name__ == "__main__":
    try:
        run_verifier()
    except Exception as e:
        print(f"Verifier Crashed: {e}")
        import traceback
        traceback.print_exc()
        Path("/logs/verifier/reward.txt").write_text("0.0")
        sys.exit(1)
'''

TEST_SH_TEMPLATE = '''#!/bin/bash
# Verifier Entrypoint (Response File Mode)

# 1. System-level setup
apt-get update
apt-get install -y curl jq
apt-get install -y python3-pip python3-full

# 2. Install ML dependencies
# Specifically pinning transformers==4.41.2 for ArmoRM compatibility
pip install --break-system-packages torch transformers==4.41.2 accelerate

# 3. Run the ArmoRM Judge script
python3 -u /tests/test_state.py
'''

RESOURCES_TEMPLATE = '''
    [environment]
    cpus = 8
    memory_mb = 24576
    storage_mb = 10240
'''

def inject_armorm_response_verifier(dataset_dir: str):
    """Adds the ArmoRM Response verifier files and requirements to tasks."""
    tasks_root = Path(dataset_dir)
    print(f"Injecting ArmoRM Response verifier into tasks at: {tasks_root}")
    
    base_toml = create_standard_task_toml()
    # Increase verifier timeout from 720 to 1200 seconds for 8B model
    updated_toml = base_toml.replace("timeout_sec = 720.0", "timeout_sec = 1200.0")
    armorm_task_toml = updated_toml.strip() + "\n" + RESOURCES_TEMPLATE + "\n"

    for task_dir in tasks_root.iterdir():
        if not task_dir.is_dir(): continue
            
        # 1. Update task.toml
        with open(task_dir / "task.toml", "w") as f:
            f.write(armorm_task_toml)

        # 2. Setup tests directory
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        with open(tests_dir / "test_state.py", "w") as f:
            f.write(VERIFIER_TEMPLATE)
            
        test_sh_path = tests_dir / "test.sh"
        with open(test_sh_path, "w") as f:
            f.write(TEST_SH_TEMPLATE)
        os.chmod(test_sh_path, 0o755)

        # 3. Update instruction.md with the deliverable requirement
        instr_path = task_dir / "instruction.md"
        if instr_path.exists():
            instruction = instr_path.read_text()
            
            deliverable_requirement = (
                "After you have completed your analysis and formulated your answer, "
                "you MUST write your final, comprehensive response into a file named "
                "'response.txt' in the current directory."
            )

            if "response.txt" not in instruction:
                instr_path.write_text(instruction + deliverable_requirement)
        
    print("ArmoRM Response-based verifier injection complete.")
