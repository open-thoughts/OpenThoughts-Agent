"""
Logic for authoring and injecting functional Python verifiers for Self-Instruct tasks.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Tuple

AUTHORING_PROMPT_TEMPLATE = """
You are authoring a verification harness for a task completion evaluation. The harness consists of two files:

- `test.sh`: bash entrypoint — installs the Python packages imported by `test_state.py` (if `test_state.py` only uses stdlib modules like `re`, `os`, `pathlib`, no pip installs are needed), then runs `python3 -u /tests/test_state.py`.
- `test_state.py`: Python script — verifies the functional outcome of the task and writes a reward score.

The harness must verify whether the agent successfully completed the "Task Instruction" below by checking the state of the sandbox (files created, content of files, permissions, etc.).

================ CORE PRINCIPLES ================
1. Verify Outcomes, Not Just Code: Focus on the side effects. If the task asks to create a file, check if that file exists and has the correct content.
2. Deliverables: The instruction often lists specific "Deliverables". Your verifier MUST check for these files at their specified paths.
3. Exhaustive Constraint Matching & Comprehensive Testing: Identify every technical constraint (e.g., 'recursive', 'case-insensitive', 'human-readable'). Your Python script MUST independently verify each requirement by creating specific test scenarios (e.g., nested folders for recursion, varied cases for case-insensitivity). The final score must be 1.0 only if **all** checks pass.
4. Descriptive Logging: Print clear "CHECK PASSED" or "CHECK FAILED" messages for every specific requirement you verify.
5. Robustness: Handle whitespace, case sensitivity (if appropriate), and varied formatting in the agent's output. 
6. The Contract: The Python script MUST write a scalar score (1.0 for success, 0.0 for failure) to the file `/logs/verifier/reward.txt`.
7. Environment: The `test.sh` bash script must install any dependencies required by your `test_state.py` Python script.
8. Path Handling: The agent will be working in a sandbox. Focus your checks on the file names and relative paths specifically mentioned in the "Task Instruction." Do not assume absolute paths (like /app) unless they are explicitly stated in the instruction.

================ REQUIRED OUTPUT FORMAT ================
Emit ONLY the two XML blocks below — no prose before, between, or after them.

<bash_script>
test.sh content here
</bash_script>

<python_script>
test_state.py content here
</python_script>

================ REFERENCE EXAMPLE ================

--- EXAMPLE INSTRUCTION ---
# Replace default with custom in /tmp .log files

Goal: Find all files in the /tmp directory with a .log extension and replace the word "default" with "custom" in each file, editing them in place.

Steps:

1. Create a script named fix_tmp_logs.sh that will:
   - Enumerate all .log files under /tmp (for example using find /tmp -type f -name "*.log" -print0).
   - For every file found, apply an in-place substitution that replaces the word default with custom, matching whole words only.
   - Print a short summary of how many files were updated.

2. Example implementation outline (you may adapt the exact syntax):
   - #!/usr/bin/env bash
     set -euo pipefail
     count=0
     while IFS= read -r -d '' file; do
       perl -0777 -pe 's/\\bdefault\\b/custom/g' -i "$file" || exit 1
       ((count++))
     done < <(find /tmp -type f -name "*.log" -print0)
     echo "$count file(s) updated."

3. Run the script and verify:
   - bash fix_tmp_logs.sh
   - Verify there are no remaining 'default' occurrences in /tmp/*.log:
     grep -R --word-regexp -n 'default' /tmp/*.log 2>/dev/null || true

Deliverables:
- A single executable script named fix_tmp_logs.sh that performs the task when run. Include executable permissions (chmod +x fix_tmp_logs.sh) as part of instructions. Optionally provide a short log of the changes.

--- EXAMPLE VERIFIER ---
<bash_script>
#!/bin/bash
# 1. Standard setup
apt-get update > /dev/null 2>&1
apt-get install -y python3 > /dev/null 2>&1
# 2. Run the functional judge
python3 -u /tests/test_state.py
</bash_script>

<python_script>
import os
import subprocess
from pathlib import Path

def verify():
    print("--- Starting Functional Verification ---")
    reward_file = Path("/logs/verifier/reward.txt")
    reward_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Locate the agent's deliverable
    script_name = "fix_tmp_logs.sh"
    # Robust search for the script (check both current dir and /app)
    candidate_paths = [Path(script_name), Path("/app") / script_name]
    script_path = next((p for p in candidate_paths if p.exists()), None)
    
    if not script_path:
        print(f"FAIL: {{script_name}} not found.")
        reward_file.write_text("0.0")
        return

    # 2. Setup a test case (Prepare dummy data)
    test_log = Path("/tmp/test_verifier_dummy.log")
    test_log.write_text("This is a default value. Also, another default here.")
    print(f"Created test log at {{test_log}}")

    # 3. Execute the agent's work
    try:
        os.chmod(script_path, 0o755)
        result = subprocess.run(["bash", str(script_path)], capture_output=True, text=True, timeout=30)
        print(f"STDOUT: {{result.stdout}}")
    except Exception as e:
        print(f"ERROR: Script failed to run: {{e}}")
        reward_file.write_text("0.0")
        return

    # 4. Validate the outcome (Side effects)
    content = test_log.read_text()
    if "custom" in content and "default" not in content:
        print("PASS: Substitution successful.")
        reward_file.write_text("1.0")
    else:
        print("FAIL: Substitution failed or 'default' remains.")
        reward_file.write_text("0.0")

    if test_log.exists(): test_log.unlink()

if __name__ == "__main__":
    verify()
</python_script>

---
### Task Instruction to Process:
{instruction}
"""

def author_verifier_harness(instruction: str, task_name: str, model_name: str = "gpt-5-nano") -> Tuple[str, str]:
    """Calls the LLM to author both test.sh and test_state.py."""
    try:
        from openai import OpenAI
        from openai.types.chat import ChatCompletionUserMessageParam
        client = OpenAI()
        
        prompt = AUTHORING_PROMPT_TEMPLATE.format(instruction=instruction)
        
        messages: List[ChatCompletionUserMessageParam] = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1.0
        )
        
        content = response.choices[0].message.content
        
        # Extract scripts
        bash_match = re.search(r"<bash_script>(.*?)</bash_script>", content, re.DOTALL)
        if not bash_match:
            print(f"ERROR [{task_name}]: Missing <bash_script>")
        bash_script = bash_match.group(1).strip() if bash_match else "# Error"
        
        python_match = re.search(r"<python_script>(.*?)</python_script>", content, re.DOTALL)
        if not python_match:
            print(f"ERROR [{task_name}]: Missing <python_script>")
        python_script = python_match.group(1).strip() if python_match else "# Error"
        
        return bash_script, python_script
        
    except Exception as e:
        print(f"Error calling LLM for verifier authoring on {task_name}: {e}")
        return "#!/bin/bash\nexit 1", f"# Error: {str(e)}"

def inject_self_instruct_verifier(dataset_dir: str, questions: List[str], model_name: str = "gpt-5-nano", max_workers: int = 30):
    """Orchestrates the authoring and injection of verifiers into tasks."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks_root = Path(dataset_dir)
    task_dirs = sorted([d for d in tasks_root.iterdir() if d.is_dir()], key=lambda x: x.name)

    print(f"Authoring functional verifiers for {len(task_dirs)} tasks using {model_name}...")

    def process_task(task_dir, instruction):
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        if (tests_dir / "test_state.py").exists():
            print(f"[{task_dir.name}] Skipping - verifier already exists.")
            return

        bash_code, python_code = author_verifier_harness(instruction, task_dir.name, model_name)

        with open(tests_dir / "test_state.py", "w") as f:
            f.write(python_code)
        with open(tests_dir / "test.sh", "w") as f:
            f.write(bash_code)
        os.chmod(tests_dir / "test.sh", 0o755)
        print(f"[{task_dir.name}] Verifier generated.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_task, td, instr) for td, instr in zip(task_dirs, questions)]
        for f in as_completed(futures):
            f.result()
