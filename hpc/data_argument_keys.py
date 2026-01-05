"""
Shared list of dataset-related override keys used when constructing
LLaMA-Factory configs (e.g., column/tag overrides that live under
`DataArguments` in the SFT submodule).
"""

DATA_ARGUMENT_KEYS: list[str] = [
    # Column overrides for Alpaca-style datasets
    "prompt_column",
    "query_column",
    "response_column",
    "history_column",
    # Column/tag overrides for ShareGPT-style datasets
    "messages",
    "system",
    "role_tag",
    "content_tag",
    "user_tag",
    "assistant_tag",
]

