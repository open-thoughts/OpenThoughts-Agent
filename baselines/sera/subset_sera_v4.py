#!/usr/bin/env python3
"""Build Sera-4.6-Lite-T2 v4 row-subsets with tool_calls PRE-RENDERED into
content, mirroring Ai2's `sera/datagen/data/postprocess/utils.py::transform_traj_hermes`.

Why v4? v3 fed structured OpenAI `tool_calls` through axolotl's `chat_template: chatml`,
which only reads `content` — the tool_calls were silently dropped during preprocess and
the SFT model never learned to emit tool calls (0/297 on SWE-bench).

Ai2's postprocess step (called via `format_and_save` with default `tool_call_format:
"hermes"`) renders each assistant turn's `tool_calls` into the content string as:

    <tool_call>
    {"name": "<func>", "arguments": {<args>}}
    </tool_call>

and wraps each `role:tool` observation as `<tool_response>...</tool_response>`,
converting role to `user`. Chatml then passes these wire-format tokens through
verbatim into input_ids + labels so they're in the training loss.

Source: allenai/Sera-4.6-Lite-T2 (36,083 rows, single jsonl, messages stored as
a JSON-encoded string). This is the dataset the upstream SERA-8B model was
actually trained on.

Target repos on HF (laion/), 316 and 1000 only this pass (per rollout plan):
  - Sera-4.6-Lite-T2-v4-316
  - Sera-4.6-Lite-T2-v4-1000
"""
import argparse
import copy
import json
import os
import random
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


SRC_REPO = "allenai/Sera-4.6-Lite-T2"
SRC_FILE = "sera-4.6-lite-t2_36083_string_enriched.jsonl"
SOURCE_TAG = SRC_REPO
TARGET_BASE = "laion/Sera-4.6-Lite-T2-v4"
SUBSET_SIZES = [316, 1000]
STAGING_DIR = Path("/Users/benjaminfeuer/Documents/scripts_dataset_build/_sera_v4_staging")
SEED = 42


# ---------------------------------------------------------------------------
# Faithful port of sera/datagen/data/postprocess/utils.py::transform_traj_hermes
# at github.com/allenai/SERA@main. Differences from upstream:
#   - Input is our dataset row (messages already a list after json.loads)
#     instead of a traj dict with a "history" key.
#   - We DO NOT override the system message with a parameter — we preserve the
#     original system content (our Sera-4.6-Lite-T2 rows already have the
#     canonical "You are a helpful assistant..." system prompt).
#   - `add_think` mode is unused (all 4.6-Lite-T2 assistant content already
#     contains explicit <think>...</think> blocks).
# ---------------------------------------------------------------------------


def _tool_call_to_action(tool_calls):
    """Render each OpenAI tool_call as a Hermes/Qwen3 <tool_call>...</tool_call> block."""
    actions = []
    if tool_calls is None:
        return actions
    for tc in tool_calls:
        try:
            args_raw = tc["function"]["arguments"]
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception:
            # Keep raw string if JSON parse fails (matches upstream behavior of
            # trusting the data — any parse failures will surface as training-
            # time tokenization errors, not silent drops).
            args = tc["function"].get("arguments")
        payload = {"name": tc["function"]["name"], "arguments": args}
        actions.append("<tool_call>\n" + json.dumps(payload) + "\n</tool_call>")
    return actions


def _wrap_tool_response(text):
    return "<tool_response>\n" + text + "\n</tool_response>"


def _flatten_content(content):
    """Tool + user messages can have content as a list[{type,text}] — extract the text."""
    if isinstance(content, list):
        assert len(content) == 1, f"Unexpected multi-part content: {content!r}"
        return content[0]["text"]
    if isinstance(content, str):
        return content
    raise ValueError(f"Message content type not recognized: {type(content)}")


def transform_hermes(messages):
    """Apply hermes rendering; returns a new flat messages list with
    keys {role, content, train}. Mirrors SERA's transform_traj_hermes +
    add_train_key in a single pass.

    Contract for each assistant turn:
      content = f"{original_content}\n\n{<tool_call>...</tool_call>}"
    For role:tool -> role:user with content wrapped in <tool_response>...</tool_response>.
    """
    out = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            text = _flatten_content(m["content"])
            out.append({
                "role": "user",
                "content": _wrap_tool_response(text),
                "train": False,
            })
            continue

        if role == "assistant":
            content = m.get("content") or ""
            if isinstance(content, list):
                content = _flatten_content(content)
            # Upstream edge case: "Exit due to cost limit" → synthetic submit call.
            # Keep behavior identical but we've never seen this in 4.6-Lite-T2.
            if content == "Exit due to cost limit":
                rendered = (
                    "Since we have successfully fixed the issue and verified it works, "
                    "let's submit the changes:\n\n"
                    '<tool_call>\n{"name": "submit", "arguments": {}}\n</tool_call>'
                )
            else:
                actions = _tool_call_to_action(m.get("tool_calls"))
                if actions:
                    rendered = content + "\n\n" + "\n".join(actions)
                else:
                    # No tool_calls on this assistant turn — pass content through as-is.
                    # (Rare: some upstream datasets have final assistant summary turns
                    # without tool calls.)
                    rendered = content
            out.append({"role": "assistant", "content": rendered, "train": True})
            continue

        if role == "system":
            content = _flatten_content(m["content"]) if not isinstance(m["content"], str) else m["content"]
            out.append({"role": "system", "content": content, "train": False})
            continue

        # user
        text = _flatten_content(m["content"])
        out.append({"role": "user", "content": text, "train": False})

    return out


# ---------------------------------------------------------------------------
# Subset builder
# ---------------------------------------------------------------------------


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def write_subsets(jsonl_path, staging_dir, sizes, full_size, seed):
    staging_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    idxs = list(range(full_size))
    random.shuffle(idxs)

    size_to_idxset = {}
    for s in sizes:
        if s > full_size:
            print(f"[skip] size={s} (only {full_size} rows)", flush=True)
            continue
        size_to_idxset[s] = set(idxs[:s])

    size_to_file = {}
    for s in size_to_idxset:
        p = staging_dir / f"sera-4.6-lite-t2_v4_{s}.jsonl"
        if p.exists():
            p.unlink()
        size_to_file[s] = open(p, "w")

    counts = {s: 0 for s in size_to_idxset}
    n_transformed, n_skipped = 0, 0

    with open(jsonl_path) as src:
        for i, line in enumerate(src):
            if i % 5000 == 0 and i > 0:
                print(f"  streamed {i}/{full_size} (transformed={n_transformed}, skipped={n_skipped})", flush=True)
            row = json.loads(line)
            # messages is stored as a JSON-encoded string
            try:
                raw_msgs = json.loads(row["messages"])
                new_msgs = transform_hermes(raw_msgs)
            except Exception as e:
                n_skipped += 1
                if n_skipped <= 5:
                    print(f"  [warn] line {i}: transform failed: {type(e).__name__}: {e}", flush=True)
                continue
            row["messages"] = new_msgs   # axolotl reads a list, not a string
            row["source"] = SOURCE_TAG
            # Trim upstream metadata fields that axolotl doesn't need and may confuse
            # schema inference (optional — all we NEED is `messages`).
            for drop in ("thought", "action", "agent", "message_type", "tool_call_ids", "cache_control"):
                row.pop(drop, None)
            n_transformed += 1
            out = json.dumps(row, ensure_ascii=False) + "\n"
            for s, idxset in size_to_idxset.items():
                if i in idxset:
                    size_to_file[s].write(out)
                    counts[s] += 1

    for f in size_to_file.values():
        f.close()

    print(f"[write] transformed={n_transformed}  skipped={n_skipped}", flush=True)
    for s, n in counts.items():
        print(f"[write] subset {s}: {n} rows", flush=True)

    return {s: staging_dir / f"sera-4.6-lite-t2_v4_{s}.jsonl" for s in size_to_idxset}


def make_readme(target_repo, size, full_size):
    return f"""---
license: apache-2.0
task_categories:
  - text-generation
tags:
  - sft
  - agent
  - swe-bench
  - axolotl
  - hermes-tool-calls
---

# {target_repo}

Row-subset of [allenai/Sera-4.6-Lite-T2](https://huggingface.co/datasets/allenai/Sera-4.6-Lite-T2)
(the dataset upstream SERA-8B was trained on), with OpenAI `tool_calls` **pre-rendered**
into the content string as Hermes/Qwen3-style `<tool_call>...</tool_call>` wire tokens
and tool responses wrapped as `<tool_response>...</tool_response>`.

This mirrors Ai2's `sera/datagen/data/postprocess/utils.py::transform_traj_hermes`
(default `tool_call_format: "hermes"`) which is the missing step between the public
Sera-4.6-Lite-T2 dataset and axolotl training. Without this pre-render, axolotl's
`chat_template: chatml` discards the structured `tool_calls` field and the SFT model
never learns to emit tool calls.

**Size**: {size:,} rows (source: {full_size:,} rows).

**Format**: Raw JSONL. Per row, `messages: list[{{role, content, train}}]`. Roles are
`system | user | assistant`. Tool observations are represented as `role: user` with
`<tool_response>...</tool_response>` wrapping (per SERA convention). `train: bool` on
each message is the per-message loss mask consumed by axolotl's `message_field_training: train`.

Sampling: deterministic random, seed=42, row-indexed into the full 36,083-row source.
Row subsets are nested.

## Usage (axolotl)

```yaml
datasets:
  - path: {target_repo}
    data_files:
      - sera-4.6-lite-t2_v4_{size}.jsonl
    type: chat_template
    field_messages: messages
    ds_type: json
    message_field_training: train
chat_template: chatml
```
"""


def push_dataset(api, target_repo, jsonl_path, size, full_size):
    print(f"[push] creating repo {target_repo}", flush=True)
    api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True)
    readme_tmp = jsonl_path.parent / f"README_{size}.md"
    readme_tmp.write_text(make_readme(target_repo, size, full_size))
    print(f"[push] uploading {jsonl_path.name} ({jsonl_path.stat().st_size / 1e6:.1f} MB) -> {target_repo}", flush=True)
    api.upload_file(
        path_or_fileobj=str(jsonl_path),
        path_in_repo=jsonl_path.name,
        repo_id=target_repo,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(readme_tmp),
        path_in_repo="README.md",
        repo_id=target_repo,
        repo_type="dataset",
    )
    readme_tmp.unlink()
    print(f"[push]  done {target_repo}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true", help="Write JSONL files locally, skip HF uploads")
    args = ap.parse_args()

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    print(f"[src] downloading {SRC_REPO}/{SRC_FILE}", flush=True)
    jsonl_path = hf_hub_download(SRC_REPO, SRC_FILE, repo_type="dataset", token=token)
    full_size = count_lines(jsonl_path)
    print(f"[src] {jsonl_path} has {full_size} rows", flush=True)

    paths = write_subsets(jsonl_path, STAGING_DIR, SUBSET_SIZES, full_size, SEED)

    if args.build_only:
        print("[build-only] skipping uploads", flush=True)
        return

    for size in SUBSET_SIZES:
        if size > full_size:
            continue
        push_dataset(api, f"{TARGET_BASE}-{size}", paths[size], size, full_size)


if __name__ == "__main__":
    main()
