"""
Logic for authoring and injecting structural Python verifiers for InferredBugs.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Tuple

AUTHORING_PROMPT_TEMPLATE = """
You are authoring a verification harness for a bug fix task. The harness consists of two files:

- `test.sh`: bash entrypoint — installs ONLY the Python packages actually imported by `test_state.py` (if `test_state.py` only uses stdlib modules like `re`, `os`, `pathlib`, no pip installs are needed), then runs `python3 -u /tests/test_state.py`. Nothing else.
- `test_state.py`: Python script — reads the agent's submitted file, checks whether the bug is fixed, and writes a reward score

The harness must verify whether the bug described in the "Bug Report" below has been correctly fixed, by statically inspecting the source code (no compilation or execution).

================ CORE PRINCIPLES ================
1. Analyze Structure, Not Runtime: The environment lacks the .NET SDK. Use Python's `re` module or string parsing to inspect the source code files directly.
2. Target File: The bug report specifies the target file path. Read this file from the candidate paths (relative and `/app/` prefixed).
3. Analyze the Bug First: Read the "Before (Buggy File)" section in the bug report. Identify the exact buggy pattern and what class of fixes would address it. Document this in a comment in your verifier code.
4. The Contract: The Python script MUST write a scalar score (1.0 for success, 0.0 for failure) to the file `/logs/verifier/reward.txt`.
5. `test.sh` contains ONLY: environment setup (apt-get, pip install) and the single command `python3 -u /tests/test_state.py`. Nothing else — no Python code, no heredocs, no file creation, no cat commands.
6. `test_state.py` contains ALL verification logic: reading the target file, extracting the method body, checking the fix, and writing the reward. The target file is written by the agent being evaluated — do NOT create or modify it in either script.

================ VERIFICATION QUALITY RULES ================
These rules are CRITICAL. Violating them produces a useless verifier.

7. Scope checks to the specific buggy method/function named in the bug report. Do NOT check the entire file — patterns that already exist elsewhere in the file will cause false positives on unfixed code.

8. Design a check that would return 0.0 on the ORIGINAL buggy code and 1.0 on any correct fix. You decide the best strategy — presence of a fix construct, absence of a bug pattern, or a combination — whichever is most reliable for this specific bug type and language. Before finalizing, mentally run your check against the original buggy code shown in the bug report: if it would return 1.0, revise it.

9. Accept multiple valid fix patterns. There is rarely only one correct fix. Your check should pass for any reasonable fix, not just one specific implementation.

10. Use a brace-counting parser to extract the exact method body, not a regex that stops at the first closing brace. The reference example shows a simple brace-counter approach.

================ REQUIRED OUTPUT FORMAT ================
Emit ONLY the two XML blocks below — no prose before, between, or after them.

<bash_script>
test.sh content here
</bash_script>

<python_script>
test_state.py content here
</python_script>

================ REFERENCE EXAMPLE ================

<bash_script>
#!/bin/bash
# Ensure standard setup
apt-get update > /dev/null 2>&1
apt-get install -y python3-pip > /dev/null 2>&1
# Run the judge with unbuffered output for real-time logs
python3 -u /tests/test_state.py
</bash_script>

<python_script>
import re
import os
import sys
import traceback
from pathlib import Path

RELATIVE_TARGET = "src/Storage/Database.cs"
BUGGY_METHOD = "void SaveData("

def extract_method_body(code, method_signature):
    # Brace-counting extractor -- handles nested braces correctly.
    idx = code.find(method_signature)
    if idx == -1:
        return None
    brace_start = code.find('{{', idx)
    if brace_start == -1:
        return None
    depth = 0
    for i in range(brace_start, len(code)):
        if code[i] == '{{':
            depth += 1
        elif code[i] == '}}':
            depth -= 1
            if depth == 0:
                return code[brace_start:i+1]
    return None

def verify():
    reward_file = Path("/logs/verifier/reward.txt")
    reward_file.parent.mkdir(parents=True, exist_ok=True)

    candidate_paths = [Path(RELATIVE_TARGET), Path("/app") / RELATIVE_TARGET]
    target_path = next((p for p in candidate_paths if p.exists()), None)
    if not target_path:
        print(f"ERROR: target file not found. Searched: {{[str(p) for p in candidate_paths]}}")
        reward_file.write_text("0.0")
        print("VERIFIER: FAIL")
        return

    code = target_path.read_text()
    body = extract_method_body(code, BUGGY_METHOD)
    if body is None:
        print("ERROR: buggy method not found in file.")
        reward_file.write_text("0.0")
        print("VERIFIER: FAIL")
        return

    # Check for presence of fix construct (using or finally+Dispose)
    has_using = bool(re.search(r'\busing\s*\(', body))
    has_finally_dispose = bool(re.search(r'finally', body)) and bool(re.search(r'\.Dispose\(', body))
    fixed = has_using or has_finally_dispose

    reward_file.write_text("1.0" if fixed else "0.0")
    print("VERIFIER: PASS" if fixed else "VERIFIER: FAIL")

if __name__ == "__main__":
    try:
        verify()
    except Exception:
        traceback.print_exc()
        Path("/logs/verifier/reward.txt").write_text("0.0")
        sys.exit(1)
</python_script>

---
### Bug Report to Process:
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
        )
        
        content = response.choices[0].message.content
        
        # Extract Bash script
        bash_match = re.search(r"<bash_script>(.*?)</bash_script>", content, re.DOTALL)
        if not bash_match:
            print(f"ERROR [{task_name}]: Model response is missing <bash_script> tags.")
        bash_script = bash_match.group(1).strip() if bash_match else "# Error: No bash script authored"
        
        # Extract Python script
        python_match = re.search(r"<python_script>(.*?)</python_script>", content, re.DOTALL)
        if not python_match:
            print(f"ERROR [{task_name}]: Model response is missing <python_script> tags.")
        python_script = python_match.group(1).strip() if python_match else "# Error: No python script authored"
        
        return bash_script, python_script
        
    except Exception as e:
        print(f"Error calling LLM for verifier authoring on {task_name}: {e}")
        return f"#!/bin/bash\\necho 'Error: {e}'\\nexit 1", f"# Error: {str(e)}"

def inject_inferredbugs_verifier(dataset_dir: str, questions: List[str], model_name: str = "gpt-5-nano", max_workers: int = 30):
    """Orchestrates the authoring and injection of verifiers into tasks."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks_root = Path(dataset_dir)
    task_dirs = sorted([d for d in tasks_root.iterdir() if d.is_dir()], key=lambda x: x.name)

    print(f"Authoring verifier harnesses for {len(task_dirs)} tasks using {model_name} (workers={max_workers})...")
    """Orchestrates the authoring and injection of verifiers into tasks."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks_root = Path(dataset_dir)
    task_dirs = sorted([d for d in tasks_root.iterdir() if d.is_dir()], key=lambda x: x.name)

    print(f"Authoring verifier harnesses for {len(task_dirs)} tasks using {model_name} (workers={max_workers})...")

    def process_task(task_dir, instruction):
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        test_sh_path = tests_dir / "test.sh"
        test_py_path = tests_dir / "test_state.py"

        if test_sh_path.exists() and test_py_path.exists():
            print(f"  - Verifier already exists for {task_dir.name}. Skipping.")
            return

        bash_code, python_code = author_verifier_harness(instruction, task_dir.name, model_name)

        if python_code.startswith("# Error:") or bash_code.startswith("#!/bin/bash\necho 'Error:"):
            print(f"  - Skipping {task_dir.name}: LLM authoring failed.")
            return

        with open(test_py_path, "w", encoding="utf-8") as f:
            f.write(python_code)
        with open(test_sh_path, "w", encoding="utf-8") as f:
            f.write(bash_code)
        with open(test_sh_path, "w") as f:
            f.write(bash_code)
        os.chmod(test_sh_path, 0o755)
        print(f"  - Harness generated for {task_dir.name}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_task, td, instr) for td, instr in zip(task_dirs, questions)]
        for f in as_completed(futures):
            f.result()
