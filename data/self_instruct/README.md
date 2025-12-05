# Self-Instruct Synthetic Data Generation

A pipeline for generating synthetic terminal-based tasks using LLMs. This system transforms seed text from pretraining corpora into structured, verifiable tasks for training AI agents.

## How It Works

### 1. **Seed Text → Task Idea**
- Take a text snippet from a pretraining corpus (e.g., Stack Overflow, documentation, tutorials)
- Use an LLM to generate a task concept with a name, description, and difficulty level
- Example: A text about "database backups" becomes a task to "write a bash script that automates PostgreSQL backups"

### 2. **Task Idea → Generated Files**
- Spin up an isolated Docker container
- Use an LLM agent (via `mini-swe-agent`) to generate:
  - `instruction.md` - Detailed task description
  - `solve.sh` - Reference solution script
  - `test.sh` - Verification tests
- The LLM is guided by few-shot examples from the `few-shots/` directory

### 3. **Verification**
- Run the generated tests inside the container
- Parse test results to confirm the task is valid and solvable
- Only keep tasks that pass verification

### 4. **Package as Sandbox**
- Extract generated files from the container
- Structure them in the standard sandbox format:
  ```
  selfinstruct-00001/
  ├── task.toml          
  ├── instruction.md     
  ├── environment/       
  ├── solution/          
  └── tests/             
  ```