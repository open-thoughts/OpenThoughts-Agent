# SWE-Bench Docker Snippets

Dockerfile snippets for SWE-Bench agent environments, derived from SkyRL-Agent's
evaluation utilities (`utils.py:158-355`) and `instance_swe_entry.sh`.

## Available Snippets

### `git_config.dockerfile`
Configures git for non-interactive operation:
- Disables pager to prevent hangs in automated environments
- Disables binary diff display for cleaner output

### `env_setup.dockerfile`
Sets up environment variables and aliases:
- PIP cache directory for faster package installations
- Git no-pager alias

### `search_tools_deps.dockerfile`
Installs dependencies for SkyRL-Agent's custom search tools:
- `chardet`: Character encoding detection
- `networkx`: Graph library for code entity relationships
- `rank-bm25`: BM25 ranking for semantic code search

### `swe_util_dirs.dockerfile`
Creates SWE-Bench utility directory structure:
- `/swe_util/eval_data/instances`: For instance JSON files

### `skyrl_full_setup.dockerfile` (Recommended)
Combined snippet including all of the above configurations.
This is the recommended snippet for most use cases.

## Usage with Harbor

These snippets can be applied to task Dockerfiles using Harbor's `dockerfile_override` feature:

```yaml
# In your Harbor YAML config
dockerfile_override:
  method: append
  path: docker/snippets/swebench/skyrl_full_setup.dockerfile
```

## Override Methods

- `prepend`: Add snippet content BEFORE the original Dockerfile
- `append`: Add snippet content AFTER the original Dockerfile (recommended)
- `replace`: Replace the entire Dockerfile with the snippet (use with caution)

## Source

These configurations are derived from:
- [SkyRL-Agent](https://github.com/NovaSky-AI/SkyRL) `skyrl_agent/tasks/swebench/utils.py`
- Specifically the `initialize_runtime()` function (lines 158-355)
- And `scripts/setup/instance_swe_entry.sh`

## Notes

1. **Git Config**: The git pager and binary diff settings prevent common issues
   where agents hang waiting for user input or produce unreadable diff output.

2. **Search Tools**: Even if not using SkyRL-Agent's custom search tool, these
   dependencies can be useful for code analysis tasks.

3. **Directory Structure**: The `/swe_util` directory is standard for SWE-Bench
   evaluation infrastructure.

4. **Runtime vs Build Time**: These snippets handle BUILD-TIME configuration.
   For RUNTIME cleanup (like removing nested .git directories), agents should
   execute appropriate commands during task execution.
