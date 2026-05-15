#!/usr/bin/env python3
"""
High-throughput async data generation against vLLM OpenAI-compatible servers.

Drop-in replacement for curator_datagen.py with proper concurrency control.
Uses asyncio + httpx with per-server semaphores to match max_num_seqs exactly,
avoiding the queue flooding and timeout death spirals that plague Curator.

Supports:
  - Concurrency control: --max-concurrent limits in-flight requests (default: 45)
  - Long timeouts: --request-timeout for slow completions (default: 1200s / 20 min)
  - Checkpointing: atomic JSONL append after each completion, --save-every for parquet
  - Sharding: --shard-index I --num-shards N for multi-node DP
  - Resume: skips prompts already in the checkpoint JSONL

Usage:
    python scripts/datagen/async_datagen.py \
        --input-dataset open-thoughts/OpenThoughts3-1.2M \
        --model Qwen/Qwen3-32B \
        --base-url http://localhost:8000/v1 \
        --api-key token-abc123 \
        --output-dir /stable/path/for/resume \
        --max-concurrent 45 \
        --request-timeout 1200 \
        --save-every 500
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from datasets import Dataset, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="[async-datagen] %(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Global flag for graceful shutdown
_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    log.warning(f"Received signal {signum} — finishing in-flight requests then saving")
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# Prompt extraction (same as curator_datagen.py)
# ---------------------------------------------------------------------------

def extract_prompt(conversations: List[Dict[str, str]]) -> str:
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
    for msg in conversations:
        role = msg.get("from") or msg.get("role", "")
        content = msg.get("value") or msg.get("content", "")
        if role == "system":
            return content
    return None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_completed_ids(jsonl_path: str) -> set:
    """Load IDs of already-completed prompts from checkpoint JSONL."""
    completed = set()
    if not os.path.exists(jsonl_path):
        return completed
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                completed.add(row["original_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    log.info(f"Loaded {len(completed)} completed IDs from {jsonl_path}")
    return completed


def save_parquet_checkpoint(jsonl_path: str, out_dir: str, count: int):
    """Convert checkpoint JSONL to parquet for compatibility with merge step."""
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        return
    ds = Dataset.from_list(rows)
    ckpt_path = os.path.join(out_dir, f"checkpoint_{count:07d}.parquet")
    ds.to_parquet(ckpt_path)
    log.info(f"Parquet checkpoint: {ckpt_path} ({len(ds)} rows)")


# ---------------------------------------------------------------------------
# Async completion worker
# ---------------------------------------------------------------------------

async def process_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    base_url: str,
    api_key: str,
    model: str,
    row: dict,
    params: dict,
    jsonl_path: str,
    jsonl_lock: asyncio.Lock,
    stats: dict,
) -> Optional[dict]:
    """Send one completion request with semaphore-based concurrency control."""
    messages = []
    if row.get("_system_prompt"):
        messages.append({"role": "system", "content": row["_system_prompt"]})
    messages.append({"role": "user", "content": row["_user_prompt"]})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": params["max_tokens"],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
    }

    # Enable thinking mode (e.g., for GLM-4.7, Qwen3)
    # vLLM accepts chat_template_kwargs as a top-level field in the raw HTTP payload.
    # (extra_body is an OpenAI Python SDK concept, not valid for raw httpx requests.)
    if params.get("enable_thinking"):
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    retries = 0
    max_retries = 5
    while retries <= max_retries:
        if _shutdown_requested:
            return None

        async with sem:
            try:
                url = f"{base_url}/chat/completions"
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()

                completion = ""
                if data.get("choices"):
                    msg_data = data["choices"][0].get("message", {})
                    content = msg_data.get("content", "")
                    reasoning = msg_data.get("reasoning_content", "")
                    if reasoning:
                        completion = f"<think>\n{reasoning}\n</think>\n{content}"
                    else:
                        completion = content

                result = {
                    "source": row.get("source", ""),
                    "domain": row.get("domain", ""),
                    "difficulty": row.get("difficulty", None),
                    "original_id": row["_original_id"],
                    "prompt": row["_user_prompt"],
                    "system_prompt": row.get("_system_prompt", ""),
                    "completion": completion,
                    "model": model,
                }

                # Atomic append to JSONL
                async with jsonl_lock:
                    with open(jsonl_path, "a") as f:
                        f.write(json.dumps(result) + "\n")

                stats["completed"] += 1
                return result

            except httpx.TimeoutException:
                retries += 1
                stats["timeouts"] += 1
                if retries <= max_retries:
                    wait = min(30, 5 * retries)
                    log.warning(
                        f"Timeout for {row['_original_id']} (attempt {retries}/{max_retries}), "
                        f"waiting {wait}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    log.error(f"Giving up on {row['_original_id']} after {max_retries} timeouts")
                    stats["failed"] += 1
                    return None

            except httpx.HTTPStatusError as e:
                retries += 1
                stats["http_errors"] += 1
                if e.response.status_code == 503 and retries <= max_retries:
                    wait = min(60, 10 * retries)
                    log.warning(
                        f"503 for {row['_original_id']} (attempt {retries}/{max_retries}), "
                        f"waiting {wait}s"
                    )
                    await asyncio.sleep(wait)
                elif e.response.status_code == 429 and retries <= max_retries:
                    wait = min(120, 20 * retries)
                    log.warning(f"429 rate limited, waiting {wait}s")
                    await asyncio.sleep(wait)
                elif retries <= max_retries:
                    wait = min(30, 5 * retries)
                    log.warning(
                        f"HTTP {e.response.status_code} for {row['_original_id']}, "
                        f"retrying in {wait}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    log.error(
                        f"Giving up on {row['_original_id']} after {max_retries} attempts: "
                        f"HTTP {e.response.status_code}"
                    )
                    stats["failed"] += 1
                    return None

            except Exception as e:
                retries += 1
                stats["errors"] += 1
                if retries <= max_retries:
                    wait = min(30, 5 * retries)
                    log.warning(f"Error for {row['_original_id']}: {e}, retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    log.error(f"Giving up on {row['_original_id']}: {e}")
                    stats["failed"] += 1
                    return None

    return None


# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------

async def progress_reporter(stats: dict, total: int, interval: float = 30.0):
    """Periodically log progress stats."""
    t0 = time.time()
    while not _shutdown_requested:
        await asyncio.sleep(interval)
        elapsed = time.time() - t0
        c = stats["completed"]
        rate = c / max(elapsed, 1) * 3600
        pct = c / max(total, 1) * 100
        remaining = (total - c) / max(rate, 0.01)
        log.info(
            f"Progress: {c}/{total} ({pct:.1f}%) | "
            f"{rate:.0f}/hr | "
            f"ETA: {remaining:.1f}h | "
            f"timeouts={stats['timeouts']} "
            f"http_err={stats['http_errors']} "
            f"failed={stats['failed']}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args):
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

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
    else:
        ds = load_dataset(args.input_dataset, split=args.split)
    log.info(f"Loaded {len(ds)} rows")

    # Apply sharding
    if args.shard_index is not None and args.num_shards is not None:
        total = len(ds)
        indices = list(range(args.shard_index, total, args.num_shards))
        ds = ds.select(indices)
        log.info(f"Shard {args.shard_index}/{args.num_shards}: {len(ds)} rows (of {total})")

    # -----------------------------------------------------------------------
    # Extract prompts
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
        })
    log.info(f"Extracted {len(input_rows)} prompts")
    if not input_rows:
        log.error("No prompts extracted")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Setup output and resume
    # -----------------------------------------------------------------------
    out_dir = args.output_dir or f"./async_output/{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "completions.jsonl")

    completed_ids = load_completed_ids(jsonl_path)
    if completed_ids:
        original_count = len(input_rows)
        input_rows = [r for r in input_rows if r["_original_id"] not in completed_ids]
        log.info(f"Resume: {len(completed_ids)} done, {len(input_rows)} remaining (of {original_count})")

    if not input_rows:
        log.info("All prompts already completed!")
        save_parquet_checkpoint(jsonl_path, out_dir, len(completed_ids))
        return

    # -----------------------------------------------------------------------
    # Run async completions
    # -----------------------------------------------------------------------
    sem = asyncio.Semaphore(args.max_concurrent)
    jsonl_lock = asyncio.Lock()
    stats = {"completed": 0, "timeouts": 0, "http_errors": 0, "errors": 0, "failed": 0}
    total = len(input_rows)
    params = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "enable_thinking": args.enable_thinking,
    }

    log.info(
        f"Starting async generation: {total} prompts, "
        f"max_concurrent={args.max_concurrent}, "
        f"timeout={args.request_timeout}s, "
        f"server={args.base_url}"
    )

    timeout = httpx.Timeout(
        connect=30.0,
        read=float(args.request_timeout),
        write=30.0,
        pool=60.0,
    )

    t0 = time.time()
    last_checkpoint = 0

    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(
        max_connections=args.max_concurrent + 10,
        max_keepalive_connections=args.max_concurrent,
    )) as client:
        # Start progress reporter
        reporter = asyncio.create_task(progress_reporter(stats, total))

        # Process in chunks to enable periodic checkpointing
        chunk_size = args.save_every or 500
        for chunk_start in range(0, total, chunk_size):
            if _shutdown_requested:
                break

            chunk_end = min(chunk_start + chunk_size, total)
            chunk = input_rows[chunk_start:chunk_end]
            log.info(f"Chunk {chunk_start}-{chunk_end} of {total}")

            tasks = [
                process_one(
                    client, sem, args.base_url, args.api_key, args.model,
                    row, params, jsonl_path, jsonl_lock, stats,
                )
                for row in chunk
            ]

            await asyncio.gather(*tasks)

            # Save parquet checkpoint
            total_done = len(completed_ids) + stats["completed"]
            if total_done > last_checkpoint:
                save_parquet_checkpoint(jsonl_path, out_dir, total_done)
                last_checkpoint = total_done

                # Push to HF if configured
                if args.output_repo:
                    try:
                        rows_list = []
                        with open(jsonl_path, "r") as f:
                            for line in f:
                                if line.strip():
                                    rows_list.append(json.loads(line))
                        if rows_list:
                            push_ds = Dataset.from_list(rows_list)
                            push_ds.push_to_hub(args.output_repo, private=False)
                            log.info(f"Pushed {len(push_ds)} rows to {args.output_repo}")
                    except Exception as e:
                        log.warning(f"HF push failed: {e}")

        reporter.cancel()

    elapsed = time.time() - t0
    total_done = len(completed_ids) + stats["completed"]

    # Final checkpoint
    save_parquet_checkpoint(jsonl_path, out_dir, total_done)

    # Final HF push
    if args.output_repo:
        try:
            rows_list = []
            with open(jsonl_path, "r") as f:
                for line in f:
                    if line.strip():
                        rows_list.append(json.loads(line))
            if rows_list:
                push_ds = Dataset.from_list(rows_list)
                push_ds.push_to_hub(args.output_repo, private=False)
                log.info(f"Final push: {len(push_ds)} rows to {args.output_repo}")
        except Exception as e:
            log.warning(f"Final HF push failed: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Async Datagen Summary")
    print(f"{'=' * 60}")
    print(f"Input:       {args.input_dataset}")
    print(f"Model:       {args.model}")
    print(f"Completed:   {stats['completed']} this run, {total_done} cumulative")
    print(f"Failed:      {stats['failed']}")
    print(f"Timeouts:    {stats['timeouts']}")
    print(f"HTTP errors: {stats['http_errors']}")
    if elapsed > 0:
        rate = stats["completed"] / elapsed * 3600
        print(f"Time:        {elapsed:.1f}s ({rate:.0f} completions/hr)")
    print(f"Local path:  {out_dir}")
    if args.output_repo:
        print(f"HF repo:     {args.output_repo}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="High-throughput async datagen against vLLM servers",
    )
    # Input
    parser.add_argument("--input-dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-column", default="conversations")

    # Model / serving
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True, help="vLLM server URL (e.g., http://localhost:8000/v1)")
    parser.add_argument("--api-key", default="token-placeholder")

    # Generation params
    parser.add_argument("--max-tokens", type=int, default=20480)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking mode (passes enable_thinking=True via extra_body)")

    # Concurrency / timeouts
    parser.add_argument("--max-concurrent", type=int, default=45,
                        help="Max in-flight requests (should be <= vLLM max_num_seqs)")
    parser.add_argument("--request-timeout", type=int, default=1200,
                        help="Read timeout per request in seconds (default: 1200 = 20 min)")

    # Output
    parser.add_argument("--output-repo", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-every", type=int, default=500)

    # Sharding
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
