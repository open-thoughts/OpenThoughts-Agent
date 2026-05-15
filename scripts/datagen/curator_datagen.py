#!/usr/bin/env python3
"""
Non-agentic data generation using Bespoke Curator.

Reads a HuggingFace dataset of prompts (conversations format), sends each prompt
to a vLLM-served model via Curator's litellm backend (hosted_vllm provider).

Supports:
  - Checkpointing: --save-every N saves after every N completions
  - Sharding: --shard-index I --num-shards N for multi-node DP
  - Resume: automatically skips prompts that already have completions
    in the output directory (from a previous run or SIGTERM kill)

Usage:
    python scripts/datagen/curator_datagen.py \
        --input-dataset open-thoughts/OpenThoughts3-1.2M \
        --model Qwen/Qwen3-32B \
        --base-url http://localhost:8000/v1 \
        --api-key token-abc123 \
        --output-dir /stable/path/for/resume \
        --save-every 5000 \
        --output-repo my-org/my-completions

Env vars:
    HOSTED_VLLM_API_KEY  - API key for vLLM server (alternative to --api-key)
    HOSTED_VLLM_API_BASE - Base URL for vLLM server (alternative to --base-url)
    HF_TOKEN             - HuggingFace token for pushing results
"""

import argparse
import glob
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Optional

from bespokelabs import curator
from datasets import Dataset, concatenate_datasets, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="[curator-datagen] %(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Global flag for graceful shutdown
_shutdown_requested = False


def _handle_sigterm(signum, frame):
    """Handle SIGTERM (sent by SLURM before killing) by requesting graceful shutdown."""
    global _shutdown_requested
    log.warning(f"Received signal {signum} — will save checkpoint and exit after current chunk")
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------

def extract_prompt(conversations: List[Dict[str, str]]) -> str:
    """Extract the user prompt from a conversations list."""
    for msg in conversations:
        role = msg.get("from") or msg.get("role", "")
        content = msg.get("value") or msg.get("content", "")
        if role in ("human", "user"):
            return content
    if conversations:
        msg = conversations[-1]
        return msg.get("value") or msg.get("content", "")
    return ""


def build_system_prompt(conversations: List[Dict[str, str]]) -> Optional[str]:
    """Extract system prompt if present."""
    for msg in conversations:
        role = msg.get("from") or msg.get("role", "")
        content = msg.get("value") or msg.get("content", "")
        if role == "system":
            return content
    return None


# ---------------------------------------------------------------------------
# Curator prompt/parse functions (functional API, Curator >= 0.1.13)
# ---------------------------------------------------------------------------

def prompt_func(input: dict) -> list:
    """Build the prompt messages for the LLM."""
    messages = []
    system = input.get("_system_prompt")
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": input["_user_prompt"]})
    return messages


def parse_func(input: dict, response) -> dict:
    """Combine input with the model's response."""
    return {
        "source": input.get("source", ""),
        "domain": input.get("domain", ""),
        "difficulty": input.get("difficulty", None),
        "original_id": input.get("_original_id", ""),
        "prompt": input["_user_prompt"],
        "system_prompt": input.get("_system_prompt", ""),
        "completion": response,
        "model": input.get("_model_name", ""),
    }


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_existing_completions(out_path: str) -> Dataset:
    """Load all existing completions from checkpoint parquets in out_path.

    Returns a Dataset of already-completed rows (may be empty).
    """
    # Look for checkpoint parquets (checkpoint_NNNNN.parquet)
    ckpt_files = sorted(glob.glob(os.path.join(out_path, "checkpoint_*.parquet")))
    # Also check for final data.parquet
    final_parquet = os.path.join(out_path, "data.parquet")
    if os.path.exists(final_parquet):
        ckpt_files.append(final_parquet)

    if not ckpt_files:
        return Dataset.from_list([])

    # Use the largest checkpoint (most rows) — they're cumulative
    # But data.parquet is the final output if it exists
    best_file = None
    best_rows = 0
    for f in ckpt_files:
        try:
            ds = Dataset.from_parquet(f)
            if len(ds) > best_rows:
                best_rows = len(ds)
                best_file = f
        except Exception as e:
            log.warning(f"Could not read {f}: {e}")

    if best_file and best_rows > 0:
        log.info(f"Found existing checkpoint: {best_file} ({best_rows} rows)")
        return Dataset.from_parquet(best_file)

    return Dataset.from_list([])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Register SIGTERM handler for graceful shutdown
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    parser = argparse.ArgumentParser(
        description="Non-agentic datagen with Bespoke Curator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    parser.add_argument("--input-dataset", required=True,
                        help="HuggingFace dataset repo (e.g., open-thoughts/OpenThoughts3-1.2M)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of prompts to process")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling when --limit is set")
    parser.add_argument("--prompt-column", default="conversations",
                        help="Column containing conversations/prompts")

    # Model / serving
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., Qwen/Qwen3-32B)")
    parser.add_argument("--base-url", default=None,
                        help="vLLM server base URL (e.g., http://localhost:8000/v1)")
    parser.add_argument("--api-key", default=None,
                        help="API key for vLLM server")

    # Generation params (passed directly to Curator LLM)
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens per completion (set via litellm.max_tokens)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)

    # Output
    parser.add_argument("--output-repo", default=None,
                        help="HuggingFace repo to push results (e.g., org/dataset-name)")
    parser.add_argument("--output-dir", default=None,
                        help="Local directory to save results (stable path for resume)")

    # Checkpointing / sharding
    parser.add_argument("--save-every", type=int, default=None,
                        help="Save intermediate results every N prompts (for long runs / timeouts)")
    parser.add_argument("--max-requests-per-minute", type=int, default=None,
                        help="Rate limit for Curator requests (prevents overwhelming vLLM)")
    parser.add_argument("--shard-index", type=int, default=None,
                        help="Shard index for multi-node data parallelism (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards for data parallelism")

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load input dataset
    # -----------------------------------------------------------------------
    log.info(f"Loading dataset: {args.input_dataset} (split={args.split})")

    if args.limit:
        log.info(f"Streaming mode: sampling {args.limit} rows...")
        stream = load_dataset(args.input_dataset, split=args.split, streaming=True)
        stream = stream.shuffle(seed=args.seed, buffer_size=min(args.limit * 10, 10000))
        rows = []
        for row in stream:
            rows.append(row)
            if len(rows) >= args.limit:
                break
        ds = Dataset.from_list(rows)
        log.info(f"Sampled {len(ds)} rows (seed={args.seed})")
    else:
        ds = load_dataset(args.input_dataset, split=args.split)
        log.info(f"Loaded {len(ds)} rows")

    # Apply sharding if requested (each shard gets a disjoint slice)
    if args.shard_index is not None and args.num_shards is not None:
        total = len(ds)
        indices = list(range(args.shard_index, total, args.num_shards))
        ds = ds.select(indices)
        log.info(f"Shard {args.shard_index}/{args.num_shards}: selected {len(ds)} rows "
                 f"(of {total} total)")

    # -----------------------------------------------------------------------
    # Prepare input rows for Curator
    # -----------------------------------------------------------------------
    log.info("Extracting prompts...")
    input_rows = []
    for idx, row in enumerate(ds):
        convs = row.get(args.prompt_column, [])
        user_prompt = extract_prompt(convs)
        if not user_prompt:
            continue
        input_rows.append({
            "source": row.get("source", ""),
            "domain": row.get("domain", ""),
            "difficulty": row.get("difficulty", None),
            "_original_id": str(idx),
            "_user_prompt": user_prompt,
            "_system_prompt": build_system_prompt(convs),
            "_model_name": args.model,
        })

    log.info(f"Extracted {len(input_rows)} prompts")
    if not input_rows:
        log.error("No prompts extracted. Check --prompt-column and dataset format.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Set up output path and check for existing completions (resume)
    # -----------------------------------------------------------------------
    if args.output_dir:
        out_path = args.output_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = f"./curator_output/{timestamp}"
    os.makedirs(out_path, exist_ok=True)

    # Resume: load existing completions and skip already-done prompts
    existing_ds = load_existing_completions(out_path)
    prior_results = []
    if len(existing_ds) > 0:
        completed_ids = set(existing_ds["original_id"])
        original_count = len(input_rows)
        input_rows = [r for r in input_rows if r["_original_id"] not in completed_ids]
        log.info(f"Resume: {len(completed_ids)} already completed, "
                 f"{len(input_rows)} remaining (of {original_count})")
        prior_results.append(existing_ds)

        if not input_rows:
            log.info("All prompts already completed! Nothing to do.")
            output_ds = existing_ds
            elapsed = 0.0
            # Skip to save/upload
            _save_and_upload(output_ds, out_path, args, input_rows, elapsed)
            return
    else:
        log.info("No existing completions found — starting fresh")

    # -----------------------------------------------------------------------
    # Configure Curator (v0.1.13 functional API)
    # -----------------------------------------------------------------------
    if args.base_url:
        model_name = f"hosted_vllm/{args.model}"
        if args.api_key:
            os.environ["HOSTED_VLLM_API_KEY"] = args.api_key
        os.environ["HOSTED_VLLM_API_BASE"] = args.base_url
    else:
        model_name = args.model

    import litellm
    litellm.max_tokens = args.max_tokens
    # Long timeout for slow completions (16k tokens at ~30 tok/s = ~550s)
    # Default litellm timeout is too short and causes retry death spirals
    litellm.request_timeout = 3600

    log.info(f"Model: {model_name}")
    log.info(f"Temperature: {args.temperature}, Top-p: {args.top_p}, Max tokens: {args.max_tokens}")

    curator_kwargs = dict(
        model_name=model_name,
        prompt_func=prompt_func,
        parse_func=parse_func,
        temperature=args.temperature,
        top_p=args.top_p,
        require_all_responses=False,
    )
    if args.max_requests_per_minute:
        curator_kwargs["max_requests_per_minute"] = args.max_requests_per_minute
        log.info(f"Rate limit: {args.max_requests_per_minute} requests/minute")

    generator = curator.LLM(**curator_kwargs)

    # -----------------------------------------------------------------------
    # Run generation (with checkpointing and graceful shutdown)
    # -----------------------------------------------------------------------
    input_ds = Dataset.from_list(input_rows)

    # Always use chunked processing for resumability
    chunk_size = args.save_every or 5000  # Default 5k for auto-checkpointing
    all_results = list(prior_results)  # Start with any prior results
    total = len(input_rows)
    t0 = time.time()

    for chunk_start in range(0, total, chunk_size):
        if _shutdown_requested:
            log.warning("Shutdown requested — saving checkpoint and exiting")
            break

        chunk_end = min(chunk_start + chunk_size, total)
        chunk_ds = input_ds.select(range(chunk_start, chunk_end))
        log.info(f"Processing chunk {chunk_start}-{chunk_end} of {total}...")

        try:
            result_ds = generator(chunk_ds)
            all_results.append(result_ds)
            log.info(f"Chunk done: {len(result_ds)} completions")
        except Exception as e:
            log.error(f"Chunk {chunk_start}-{chunk_end} failed: {e}")
            log.warning("Saving what we have so far and exiting")
            break

        # Save checkpoint after every chunk
        merged = concatenate_datasets(all_results)
        ckpt_path = os.path.join(out_path, f"checkpoint_{len(merged):07d}.parquet")
        merged.to_parquet(ckpt_path)
        log.info(f"Checkpoint saved: {ckpt_path} ({len(merged)} total rows)")

        # Push intermediate results to HF if configured
        if args.output_repo:
            try:
                merged.push_to_hub(args.output_repo, private=False)
                log.info(f"Intermediate push to {args.output_repo} ({len(merged)} rows)")
            except Exception as e:
                log.warning(f"Intermediate HF push failed (will retry at end): {e}")

    elapsed = time.time() - t0

    if not all_results:
        log.error("No completions generated")
        sys.exit(1)

    output_ds = concatenate_datasets(all_results)
    log.info(f"Generation complete: {len(output_ds)} completions in {elapsed:.1f}s "
             f"({len(output_ds) / max(elapsed, 0.1):.1f} prompts/sec)")

    _save_and_upload(output_ds, out_path, args, input_rows, elapsed)


def _save_and_upload(output_ds, out_path, args, input_rows, elapsed):
    """Save final output and optionally upload to HF."""
    # -----------------------------------------------------------------------
    # Save output
    # -----------------------------------------------------------------------
    output_ds.save_to_disk(out_path)
    log.info(f"Saved to {out_path}")

    parquet_path = os.path.join(out_path, "data.parquet")
    output_ds.to_parquet(parquet_path)
    log.info(f"Parquet: {parquet_path}")

    if args.output_repo:
        log.info(f"Pushing to HuggingFace: {args.output_repo}")
        try:
            output_ds.push_to_hub(args.output_repo, private=False)
            log.info(f"Uploaded to https://huggingface.co/datasets/{args.output_repo}")
        except Exception as e:
            log.warning(f"Final HF push failed: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"Curator Datagen Summary")
    print(f"=" * 60)
    print(f"Input:       {args.input_dataset} ({len(input_rows)} prompts this run)")
    print(f"Model:       {args.model}")
    print(f"Output:      {len(output_ds)} completions (cumulative)")
    if elapsed > 0:
        print(f"Time:        {elapsed:.1f}s ({len(output_ds) / max(elapsed, 0.1):.1f} prompts/sec)")
    print(f"Local path:  {out_path}")
    if args.output_repo:
        print(f"HF repo:     {args.output_repo}")
    print("=" * 60)


if __name__ == "__main__":
    main()
