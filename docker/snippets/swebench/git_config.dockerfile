# SWE-Bench Git Configuration Snippet
# Purpose: Configure git for cleaner output in agent environments
# Source: SkyRL-Agent utils.py:167
#
# These settings prevent git from using a pager (which can cause issues in
# non-interactive environments) and disable binary diff display.

# Use conditional to handle cases where git isn't installed yet
RUN if command -v git >/dev/null 2>&1; then \
        git config --global core.pager "" && \
        git config --global diff.binary false; \
    fi
