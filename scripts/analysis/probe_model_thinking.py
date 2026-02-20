#!/usr/bin/env python3
"""Probe a HuggingFace model with real environment prompts to inspect thinking output.

Loads a model via the HF transformers pipeline, prompts it with the first
user message from each of the first N rows of a trace dataset, saves the
prompts and raw model responses to a JSON file.

Usage:
    python -m scripts.analysis.probe_model_thinking \
        --model laion/GLM-4_7-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-131k \
        --output probe_results.json

    # Custom dataset / number of prompts
    python -m scripts.analysis.probe_model_thinking \
        --model laion/GLM-4_7-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-131k \
        --dataset DCAgent2/some-other-dataset \
        --num-prompts 10 \
        --max-new-tokens 2048 \
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_DATASET = (
    "DCAgent2/DCAgent_dev_set_v2_laion_exp_tas_timeout_multiplier_4_0_traces_20260211_064438"
)
DEFAULT_MODEL = (
    "laion/GLM-4_7-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-131k"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe a HF model with real environment prompts",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset ID (default: {DEFAULT_DATASET})",
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompts to run (default: 5)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens to generate per prompt (default: 2048)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (default: probe_<model_short>_<timestamp>.json)",
    )
    p.add_argument(
        "--conversations-column",
        default="conversations",
        help="Column name for conversations (default: conversations)",
    )
    p.add_argument(
        "--role-tag",
        default="role",
        help="Key for role in message dicts (default: role)",
    )
    p.add_argument(
        "--content-tag",
        default="content",
        help="Key for content in message dicts (default: content)",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32", "auto"],
        help="Model dtype (default: bfloat16)",
    )
    # Sampling parameters — defaults match SkyRL RL rollout config
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0, matching RL rollout)",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p (default: 0.95, matching RL rollout)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling, -1 to disable (default: -1, matching RL rollout)",
    )
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (overrides temperature/top_p/top_k)",
    )
    return p.parse_args()


def extract_initial_prompts(
    dataset_name: str,
    num_prompts: int,
    conversations_col: str,
    role_tag: str,
    content_tag: str,
) -> list[dict]:
    """Extract the first user message from N evenly-spaced rows."""
    from datasets import load_dataset

    print(f"[probe] Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")

    # Pick rows evenly spaced across the dataset (stride = len // num_prompts,
    # min 1) so we sample diverse tasks rather than just the first few.
    n = len(ds)
    stride = max(n // num_prompts, 1)
    indices = [i * stride for i in range(num_prompts) if i * stride < n]
    print(f"[probe] Dataset has {n} rows, sampling indices: {indices}")

    prompts = []
    for i in indices:
        row = ds[i]
        convs = row[conversations_col]
        # Find the first user message (the environment/system prompt)
        first_user_content = None
        for msg in convs:
            if msg[role_tag] == "user":
                first_user_content = msg[content_tag]
                break

        if first_user_content is None:
            print(f"  Row {i}: no user message found, skipping")
            continue

        task_id = row.get("task", row.get("trial_name", f"row_{i}"))
        prompts.append({
            "row_index": i,
            "task": task_id,
            "messages": [{"role": "user", "content": first_user_content}],
        })
        print(f"  Row {i} ({task_id}): prompt length {len(first_user_content)} chars")

    return prompts


def _resolve_device_and_dtype(dtype: str):
    """Pick the best available device and a compatible torch dtype."""
    import torch

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = dtype_map[dtype]

    if torch.cuda.is_available():
        return "auto", torch_dtype, "cuda"

    if torch.backends.mps.is_available():
        # MPS (Apple Metal) does not support bfloat16; fall back to float16.
        if torch_dtype == torch.bfloat16:
            print("[probe] MPS does not support bfloat16, using float16 instead")
            torch_dtype = torch.float16
        return "mps", torch_dtype, "mps"

    if torch_dtype == "auto":
        torch_dtype = torch.float32
    return "cpu", torch_dtype, "cpu"


def _generate_once(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int,
    enable_thinking: bool,
    label: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = -1,
    greedy: bool = False,
):
    """Run a single generation pass and return result dict."""
    import torch
    from transformers import GenerationConfig

    # Some tokenizers don't support enable_thinking; fall back gracefully.
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        ).to(model.device)
    except TypeError:
        # Tokenizer chat template doesn't accept enable_thinking kwarg
        if enable_thinking:
            # Default call (thinking is the default for models that support it)
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
        else:
            print(f"  [{label}] Tokenizer does not support enable_thinking, skipping")
            return None

    input_len = inputs["input_ids"].shape[-1]
    print(f"  [{label}] Input tokens: {input_len}")

    if greedy:
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    else:
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        }
        # top_k=-1 means disabled; only set if actually limiting
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
        gen_config = GenerationConfig(**gen_kwargs)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=gen_config,
        )
    elapsed = time.time() - t0

    generated_ids = outputs[0][input_len:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    output_len = len(generated_ids)

    print(f"  [{label}] Output tokens: {output_len}, time: {elapsed:.1f}s")
    print(f"  [{label}] Response preview: {repr(response_text[:200])}")

    return {
        "input_tokens": input_len,
        "output_tokens": output_len,
        "generation_time_sec": round(elapsed, 2),
        "response": response_text,
    }


def run_inference(
    model_name: str,
    prompts: list[dict],
    max_new_tokens: int,
    dtype: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = -1,
    greedy: bool = False,
) -> list[dict]:
    """Run inference on each prompt with both thinking=True and thinking=False."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device_map, torch_dtype, device_label = _resolve_device_and_dtype(dtype)

    print(f"\n[probe] Loading model: {model_name} (dtype={dtype}, device={device_label})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    print(f"[probe] Model loaded on {model.device}")

    results = []
    for i, prompt_info in enumerate(prompts):
        messages = prompt_info["messages"]
        task = prompt_info["task"]

        print(f"\n[probe] Prompt {i + 1}/{len(prompts)} ({task})...")

        prompt_content = messages[0]["content"]

        result_entry = {
            "row_index": prompt_info["row_index"],
            "task": task,
            "prompt_content": prompt_content,
        }

        sampling_kwargs = dict(
            temperature=temperature, top_p=top_p, top_k=top_k, greedy=greedy,
        )

        # Run with thinking enabled
        thinking_on = _generate_once(
            model, tokenizer, messages, max_new_tokens,
            enable_thinking=True, label="thinking=ON", **sampling_kwargs,
        )
        if thinking_on is not None:
            result_entry["thinking_enabled"] = thinking_on
        else:
            result_entry["thinking_enabled"] = {"skipped": True}

        # Run with thinking disabled
        thinking_off = _generate_once(
            model, tokenizer, messages, max_new_tokens,
            enable_thinking=False, label="thinking=OFF", **sampling_kwargs,
        )
        if thinking_off is not None:
            result_entry["thinking_disabled"] = thinking_off
        else:
            result_entry["thinking_disabled"] = {"skipped": True}

        results.append(result_entry)

    return results


def _default_output_path(model: str) -> str:
    """Build a default output filename from the model name and current time."""
    from datetime import datetime
    # Use the part after the last '/' (or the whole string) as the short name,
    # replacing characters that are awkward in filenames.
    short = model.rsplit("/", 1)[-1]
    short = short.replace(" ", "_")[:60]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"probe_{short}_{ts}.json"


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = _default_output_path(args.model)

    prompts = extract_initial_prompts(
        args.dataset,
        args.num_prompts,
        args.conversations_column,
        args.role_tag,
        args.content_tag,
    )

    if not prompts:
        print("[probe] No prompts extracted, exiting.")
        sys.exit(1)

    sampling_mode = "greedy" if args.greedy else "sampling"
    print(f"[probe] Generation: {sampling_mode}"
          + (f" (temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k})"
             if not args.greedy else ""))

    results = run_inference(
        args.model,
        prompts,
        args.max_new_tokens,
        args.dtype,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        greedy=args.greedy,
    )

    output = {
        "model": args.model,
        "dataset": args.dataset,
        "num_prompts": len(results),
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "sampling": {
            "mode": sampling_mode,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        } if not args.greedy else {"mode": "greedy"},
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[probe] Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
