# SWE-Bench Environment Setup Snippet
# Purpose: Set up environment variables and aliases for agent execution
# Source: SkyRL-Agent utils.py:167, instance_swe_entry.sh
#
# Creates .bashrc entries for:
# - PIP cache directory for faster package installations
# - Git no-pager alias to prevent interactive prompts

RUN echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && \
    echo "alias git='git --no-pager'" >> ~/.bashrc && \
    mkdir -p ~/.cache/pip
