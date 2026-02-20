# SWE-Bench Full Setup Snippet (SkyRL-Agent Style)
# Purpose: Complete build-time setup for SWE-Bench agent environments
# Source: SkyRL-Agent utils.py:158-355, instance_swe_entry.sh
#
# This snippet combines all recommended configurations from SkyRL-Agent:
# 1. Git configuration for non-interactive operation
# 2. Environment variables and aliases
# 3. Search tools dependencies (for BM25-based code search)
# 4. SWE-Bench utility directories
#
# Usage: Append this to your task Dockerfile via dockerfile_override

# --- Git Configuration ---
# Disable pager to prevent interactive prompts that hang agents
# Disable binary diffs which can produce unreadable output
# Use conditional to handle cases where git isn't installed yet
RUN if command -v git >/dev/null 2>&1; then \
        git config --global core.pager "" && \
        git config --global diff.binary false; \
    fi

# --- Environment Setup ---
# Set up PIP cache for faster installs and git no-pager alias
RUN echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && \
    echo "alias git='git --no-pager'" >> ~/.bashrc && \
    mkdir -p ~/.cache/pip

# --- Search Tools Dependencies ---
# Required for SkyRL-Agent's custom BM25-based code search tool
# Even if not using the custom tool, these are useful for code analysis
RUN pip install --no-cache-dir chardet networkx 'rank-bm25>=0.2.0,<1.0.0' || true

# --- SWE-Bench Utility Directories ---
# Standard directory structure expected by SWE-Bench evaluation
RUN mkdir -p /swe_util/eval_data/instances
