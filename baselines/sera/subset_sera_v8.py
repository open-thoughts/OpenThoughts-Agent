"""
Generate size-ladder subsets of the SERA-author-blessed pre-rendered dataset
(ethanlshen/sera-subset) for the i9 / -v8 training run.

Source: ethanlshen/sera-subset (two JSONL files)
  - stage1: 22 972 rows (unresolved rollouts, threshold 0.88)
  - stage2: 25 244 rows (resolved soft-t0 rollouts)

Operation:
  1. Download both files from HF (or read from a local cache dir).
  2. Concatenate → 48 216 rows.
  3. Shuffle with seed=42.
  4. Write nested subsets (each smaller set is a prefix of the larger one):
       sera_subset_mixed_316.jsonl
       sera_subset_mixed_1000.jsonl
       sera_subset_mixed_3160.jsonl
       sera_subset_mixed_10000.jsonl

Rows are written verbatim — no re-rendering needed; the source dataset is
already in the pre-rendered Hermes wire format (messages with role/content/train).

Usage:
  python baselines/sera/subset_sera_v8.py
  python baselines/sera/subset_sera_v8.py --output-dir /e/data1/datasets/playground/ot-baf/sera_subset/subsets
  python baselines/sera/subset_sera_v8.py --sizes 316 1000   # quick test
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SRC_REPO = "ethanlshen/sera-subset"
SRC_FILES = [
    "22972_0.88_stage1_scaling_final_glm46_e2e_1ipf_swesmith_unresolved_ipf_1_atk_rft-think_SYSTEM_SIMPLE.jsonl",
    "25224_r0.88_stage2_scaling_final_glm46_e2e_1ipf_resolved_soft_t0_ipf_1_atk_rft-think_SYSTEM_SIMPLE.jsonl",
]

SUBSET_SIZES = [316, 1000, 3160, 10000]
SEED = 42


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def generate(
    output_dir: Path,
    sizes: list[int] = SUBSET_SIZES,
    seed: int = SEED,
    token: str | None = None,
    local_raw_dir: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for fname in SRC_FILES:
        if local_raw_dir is not None:
            src_path = local_raw_dir / fname
            print(f"[src] reading {src_path} …")
        else:
            from huggingface_hub import hf_hub_download
            print(f"[src] downloading {SRC_REPO}/{fname} …")
            src_path = Path(hf_hub_download(SRC_REPO, fname, repo_type="dataset", token=token))

        rows = _load_jsonl(src_path)
        print(f"      {len(rows)} rows")
        all_rows.extend(rows)

    print(f"[mix] total rows before shuffle: {len(all_rows)}")
    rng = random.Random(seed)
    rng.shuffle(all_rows)
    print(f"[mix] shuffled with seed={seed}")

    max_size = max(sizes)
    if max_size > len(all_rows):
        print(f"[warn] requested size {max_size} > dataset size {len(all_rows)}; clamping")

    for size in sorted(sizes):
        n = min(size, len(all_rows))
        out_path = output_dir / f"sera_subset_mixed_{size}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for row in all_rows[:n]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[out] {out_path} ({n} rows, {out_path.stat().st_size / 1e6:.1f} MB)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate size-ladder subsets of ethanlshen/sera-subset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("/e/data1/datasets/playground/ot-baf/sera_subset/subsets"),
        help="Directory to write sera_subset_mixed_*.jsonl files",
    )
    p.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=SUBSET_SIZES,
        help="Subset sizes to write",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    p.add_argument(
        "--local-raw-dir",
        type=Path,
        default=None,
        help="If set, read source files from this local directory instead of HF Hub",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (falls back to HF_TOKEN env var)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    import os
    token = args.token or os.environ.get("HF_TOKEN")
    generate(
        output_dir=args.output_dir,
        sizes=args.sizes,
        seed=args.seed,
        token=token,
        local_raw_dir=args.local_raw_dir,
    )


if __name__ == "__main__":
    main()
