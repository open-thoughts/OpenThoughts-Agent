#!/usr/bin/env python3
"""Verify how ReasoningTemplate handles thinking blocks in SFT training data.

Tests three scenarios:
1. RAW HF datasets → encode_multiturn (no preprocessing)
2. RAW HF datasets → prep_for_thinking → encode_multiturn (full pipeline)
3. Synthetic edge cases
"""

import sys
sys.path.insert(0, "/Users/benjaminfeuer/Documents/LLaMA-Factory/src")
sys.path.insert(0, "/Users/benjaminfeuer/Documents/OpenThoughts-Agent")

from copy import deepcopy
from transformers import AutoTokenizer
from datasets import load_dataset
from llamafactory.data.template import TEMPLATES
from scripts.datagen.prep_for_thinking import reformat_assistant_content

TOKENIZER_ID = "Qwen/Qwen3-8B"
TEMPLATE_NAME = "qwen3"

# Two SFT datasets to test
DATASETS = {
    "with_thinking": "DCAgent/exp-gfi-swesmith-embedding-mean-filtered-10K_glm_4.7_traces_jupiter",
    "kimi_k2": "penfever/kimi-k2-swegym-tasks-maxeps-32k",
}

THOUGHT_OPEN = "<think>\n"
THOUGHT_CLOSE = "\n</think>\n\n"


def extract_messages(convos):
    """Extract user/assistant messages from conversation list."""
    messages = []
    for msg in convos:
        role = msg.get("role", msg.get("from", ""))
        content = msg.get("content", msg.get("value", ""))
        if role in ("user", "human"):
            messages.append({"role": "user", "content": content})
        elif role in ("assistant", "gpt"):
            messages.append({"role": "assistant", "content": content})
    return messages


def encode_and_report(template, tokenizer, messages, label):
    """Encode messages with template and report think-block counts."""
    encoded_pairs = template.encode_multiturn(tokenizer, messages)
    print(f"\n  {label} ({len(encoded_pairs)} pairs):")
    for pair_idx, (prompt_ids, response_ids) in enumerate(encoded_pairs):
        response_text = tokenizer.decode(response_ids[:150], skip_special_tokens=False)
        response_preview = response_text[:300].replace("\n", "\\n")

        decoded_full = tokenizer.decode(response_ids, skip_special_tokens=False)
        think_count = decoded_full.count("<think>")
        close_count = decoded_full.count("</think>")
        decoded_full.lstrip().startswith("<think>")

        status = "OK" if think_count == 1 and close_count == 1 else "DOUBLE-THINK BUG"
        print(f"    Pair {pair_idx}: [{status}] prompt={len(prompt_ids)} tok, response={len(response_ids)} tok")
        print(f"      <think> count={think_count} </think> count={close_count}")
        print(f"      Response start: {response_preview}")
        print()
    return encoded_pairs


def preprocess_messages(messages):
    """Apply prep_for_thinking preprocessing to assistant messages."""
    preprocessed = deepcopy(messages)
    for i, msg in enumerate(preprocessed):
        if msg["role"] == "assistant":
            original = msg["content"]
            reformatted, fmt_label = reformat_assistant_content(original)
            preprocessed[i]["content"] = reformatted
            # Show what changed
            orig_preview = original[:80].replace("\n", "\\n")
            new_preview = reformatted[:80].replace("\n", "\\n")
            changed = "CHANGED" if reformatted != original else "UNCHANGED"
            print(f"    Turn {i//2} assistant: [{changed}] format={fmt_label}")
            if reformatted != original:
                print(f"      Before: {orig_preview}...")
                print(f"      After:  {new_preview}...")
    return preprocessed


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)

    template = deepcopy(TEMPLATES[TEMPLATE_NAME])
    template.enable_thinking = True  # match SFT config default
    print(f"Template: {TEMPLATE_NAME}, class: {type(template).__name__}")
    print(f"thought_words: {repr(template.thought_words)}")
    print(f"enable_thinking: {template.enable_thinking}")
    print()

    # =========================================================================
    # PHASE 1: Raw datasets (no preprocessing)
    # =========================================================================
    print("=" * 80)
    print("PHASE 1: RAW datasets → encode_multiturn (NO preprocessing)")
    print("=" * 80)

    for ds_label, ds_name in DATASETS.items():
        print(f"\n--- Dataset: {ds_label} ({ds_name}) ---")
        ds = load_dataset(ds_name, split="train")

        for row_idx in range(min(5, len(ds))):
            row = ds[row_idx]
            convos = row.get("conversations", [])
            if len(convos) < 4:
                continue

            messages = extract_messages(convos)
            if len(messages) < 4:
                continue

            print(f"\n  Row {row_idx}: {len(messages)} messages")
            for j in range(1, len(messages), 2):
                content = messages[j]["content"]
                has_think_with_nl = THOUGHT_OPEN in content
                has_think_bare = "<think>" in content
                preview = content[:100].replace("\n", "\\n")
                print(f"    Turn {j//2} assistant: has_think_nl={has_think_with_nl} has_think_bare={has_think_bare}")
                print(f"      Raw: {preview}...")

            encode_and_report(template, tokenizer, messages, "RAW encode_multiturn")
            break  # first matching row per dataset

    # =========================================================================
    # PHASE 2: Preprocessed datasets
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: RAW datasets → prep_for_thinking → encode_multiturn")
    print("=" * 80)

    for ds_label, ds_name in DATASETS.items():
        print(f"\n--- Dataset: {ds_label} ({ds_name}) ---")
        ds = load_dataset(ds_name, split="train")

        for row_idx in range(min(5, len(ds))):
            row = ds[row_idx]
            convos = row.get("conversations", [])
            if len(convos) < 4:
                continue

            messages = extract_messages(convos)
            if len(messages) < 4:
                continue

            print(f"\n  Row {row_idx}: {len(messages)} messages")
            print("\n  Preprocessing with reformat_assistant_content():")
            preprocessed = preprocess_messages(messages)

            # Verify guard check will pass
            for j in range(1, len(preprocessed), 2):
                content = preprocessed[j]["content"]
                guard_passes = THOUGHT_OPEN in content or THOUGHT_CLOSE in content
                print(f"    Turn {j//2}: guard_check_prevents_double_think={guard_passes}")

            encode_and_report(template, tokenizer, preprocessed, "PREPROCESSED encode_multiturn")
            break  # first matching row per dataset

    # =========================================================================
    # PHASE 3: Synthetic edge cases
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: Synthetic edge cases")
    print("=" * 80)

    # 3a: No thinking at all (should get empty think prepended)
    print("\n--- 3a: No thinking at all ---")
    messages_no_think = [
        {"role": "user", "content": "Hello, please solve this task."},
        {"role": "assistant", "content": '{"analysis": "I will solve the task", "commands": []}'},
        {"role": "user", "content": "The command ran successfully."},
        {"role": "assistant", "content": '{"analysis": "Task is done", "commands": [], "task_complete": true}'},
    ]
    print("  Before preprocessing:")
    preprocessed = preprocess_messages(messages_no_think)
    encode_and_report(template, tokenizer, preprocessed, "Preprocessed no-think")

    # 3b: Think tags without newlines (raw format)
    print("\n--- 3b: Think tags WITHOUT newlines (raw format) ---")
    messages_no_nl = [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": '<think>Let me think.</think>{"analysis": "done", "commands": []}'},
        {"role": "user", "content": "OK."},
        {"role": "assistant", "content": '<think>Almost done.</think>{"analysis": "finished", "commands": [], "task_complete": true}'},
    ]
    print("  Before preprocessing:")
    preprocessed = preprocess_messages(messages_no_nl)
    encode_and_report(template, tokenizer, preprocessed, "Preprocessed no-newline-think")

    # 3c: Already correct format (should be unchanged)
    print("\n--- 3c: Already correct Qwen3 format ---")
    messages_correct = [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": '<think>\nLet me think.\n</think>\n\n{"analysis": "done", "commands": []}'},
        {"role": "user", "content": "OK."},
        {"role": "assistant", "content": '<think>\nAlmost done.\n</think>\n\n{"analysis": "finished", "commands": [], "task_complete": true}'},
    ]
    print("  Before preprocessing:")
    preprocessed = preprocess_messages(messages_correct)
    encode_and_report(template, tokenizer, preprocessed, "Preprocessed already-correct")

    # 3d: Orphaned </think> (missing opening tag)
    print("\n--- 3d: Orphaned </think> (missing opening tag) ---")
    messages_orphan = [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": 'Let me think about this.</think>{"analysis": "done", "commands": []}'},
        {"role": "user", "content": "OK."},
        {"role": "assistant", "content": 'Done thinking.</think>{"analysis": "finished", "commands": [], "task_complete": true}'},
    ]
    print("  Before preprocessing:")
    preprocessed = preprocess_messages(messages_orphan)
    encode_and_report(template, tokenizer, preprocessed, "Preprocessed orphaned-think")


if __name__ == "__main__":
    main()
