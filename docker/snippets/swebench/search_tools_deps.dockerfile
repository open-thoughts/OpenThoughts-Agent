# SWE-Bench Search Tools Dependencies Snippet
# Purpose: Install dependencies required by SkyRL-Agent custom search tools
# Source: SkyRL-Agent instance_swe_entry.sh:46-48
#
# These packages are used by the custom BM25-based code search tool:
# - chardet: Character encoding detection for reading source files
# - networkx: Graph library for code entity relationships
# - rank-bm25: BM25 ranking algorithm for semantic code search

RUN pip install --no-cache-dir chardet networkx 'rank-bm25>=0.2.0,<1.0.0'
