#!/usr/bin/env python3
"""
Generate DCLM shell tasks - full pipeline in simple main().
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

from data.commons import create_task_from_dockerfiles_and_questions, create_standard_dockerfile
from Step1_filter_dclm_baseline.download_dclm.make_dclm_urls import DCLMShardDownloader
from Step1_filter_dclm_baseline.grep_terminal_like.bash_finder import BashDetector, process_single_shard
from Step2_classify.classify_tasks import TaskExtraction as ClassifySchema, SYSTEM_PROMPT as CLASSIFY_PROMPT

# =============================================================================
# Configuration
# =============================================================================
LIMIT = 50
BASH_THRESHOLD = 4.0
MODEL = "gpt-4o-mini"
MAX_CONCURRENT = 100


def main() -> None:
    # Step 1: Get bash sequences from DCLM
    print("Step 1a: Getting DCLM shard URLs...")
    downloader = DCLMShardDownloader()
    shard_paths = downloader.list_all_shards()
    urls = [f"https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/main/{p}" for p in shard_paths]

    print("Step 1b: Filtering bash sequences...")
    bash_sequences = filter_bash_sequences(urls, BASH_THRESHOLD, LIMIT)
    print(f"  -> {len(bash_sequences)} bash sequences")

    # Step 2: Classify shell tasks
    print("\nStep 2: Classifying shell tasks...")
    classified = asyncio.run(classify_all(bash_sequences, CLASSIFY_PROMPT, ClassifySchema))
    print(f"  -> {len(classified)} classified as shell tasks")

    if not classified:
        print("\nNo tasks found. Exiting.")
        return

    # Step 3: Generate task directories
    print("\nStep 3: Generating task directories...")
    dockerfile = create_standard_dockerfile()
    dockerfiles = [dockerfile] * len(classified)
    questions = [(c.get("task_description", ""), None, None) for c in classified]

    final_dir = create_task_from_dockerfiles_and_questions(dockerfiles, questions)
    print(f"\n{'='*60}")
    print(f"Generated {len(classified)} tasks at: {final_dir}")
    print(f"{'='*60}")


def filter_bash_sequences(urls, threshold, limit, max_concurrent=32):
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_dir = Path(tempfile.mkdtemp(prefix="dclm_cache_"))
    results = []

    def process_one(url):
        filename = url.split("/")[-1]
        local_path = cache_dir / filename

        if not local_path.exists():
            try:
                resp = requests.get(url, stream=True)
                resp.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
            except:
                return []

        try:
            result = process_single_shard(local_path, threshold)
            return result.bash_results
        except:
            return []

    # Process in batches to allow early stopping
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        url_iter = iter(urls)
        pending = set()

        # Seed initial batch
        for _ in range(max_concurrent):
            try:
                url = next(url_iter)
                pending.add(executor.submit(process_one, url))
            except StopIteration:
                break

        pbar = tqdm(total=limit or len(urls), desc="Processing shards")
        while pending:
            done_futures, pending = wait_first(pending)

            for future in done_futures:
                batch = future.result()
                results.extend(batch)
                pbar.update(len(batch))

                if limit and len(results) >= limit:
                    # Cancel remaining and exit
                    for f in pending:
                        f.cancel()
                    pbar.close()
                    return results[:limit]

                # Submit next URL
                try:
                    url = next(url_iter)
                    pending.add(executor.submit(process_one, url))
                except StopIteration:
                    pass

        pbar.close()

    return results[:limit] if limit else results


def wait_first(futures):
    from concurrent.futures import FIRST_COMPLETED, wait
    done, pending = wait(futures, return_when=FIRST_COMPLETED)
    return done, pending


async def classify_all(sequences, prompt, schema):
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def classify_one(seq):
        async with semaphore:
            try:
                resp = await client.beta.chat.completions.parse(
                    model=MODEL,
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": seq.get("text", "")}],
                    response_format=schema,
                )
                r = resp.choices[0].message.parsed
                if r.has_shell_task and r.task_description:
                    return {**seq, "task_description": r.task_description}
            except:
                pass
            return None

    results = await tqdm_asyncio.gather(*[classify_one(s) for s in sequences], desc="Classifying")
    return [r for r in results if r]


if __name__ == "__main__":
    main()
