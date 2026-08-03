"""Compare context length statistics across HuggingFace datasets.

Usage:
    python -m scripts.analysis.context_length_compare \
        DCAgent2/dataset-a DCAgent2/dataset-b \
        --tokenizer Qwen/Qwen3-8B \
        --filter 'trace_source==main' \
        --plot context_length_distribution.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis.trace_metrics import load_and_filter, tokenize_dataset
from scripts.analysis.utils import TOKEN_REPRESENTATIONS


def print_stats(name: str, counts: np.ndarray) -> None:
    """Print a summary table for *counts*."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  rows   = {len(counts):,}")
    print(f"  mean   = {np.mean(counts):,.0f} tokens")
    print(f"  median = {np.median(counts):,.0f}")
    print(f"  std    = {np.std(counts):,.0f}")
    print(f"  min    = {np.min(counts):,.0f}")
    print(f"  p10    = {np.percentile(counts, 10):,.0f}")
    print(f"  p25    = {np.percentile(counts, 25):,.0f}")
    print(f"  p75    = {np.percentile(counts, 75):,.0f}")
    print(f"  p90    = {np.percentile(counts, 90):,.0f}")
    print(f"  p95    = {np.percentile(counts, 95):,.0f}")
    print(f"  max    = {np.max(counts):,.0f}")


def distribution_bin_edges(results: list[tuple[str, np.ndarray]], bin_count: int) -> np.ndarray:
    """Build the log-scale histogram edges used for a multi-dataset comparison."""
    if bin_count < 2:
        raise ValueError("--plot-bins must be at least 2")
    values = np.concatenate([counts for _, counts in results if len(counts)])
    positive_values = values[values > 0]
    if not len(positive_values):
        raise ValueError("Cannot plot context lengths without positive token counts")
    return np.logspace(
        np.log10(max(100, positive_values.min())),
        np.log10(max(100, positive_values.max())),
        bin_count,
    )


def plot_context_length_distributions(
    results: list[tuple[str, np.ndarray]],
    output_path: Path,
    *,
    bin_count: int,
    title: str | None,
) -> None:
    """Write a log-scale context-length overlay plot for the selected datasets."""
    if not results:
        raise ValueError("No datasets with token counts are available to plot")
    bins = distribution_bin_edges(results, bin_count)
    sorted_results = sorted(results, key=lambda item: np.median(item[1]))
    color_maps = ("tab20", "Set1", "tab20b", "tab20c")
    colors = [
        plt.colormaps[color_map](index / max(plt.colormaps[color_map].N - 1, 1))
        for color_map in color_maps
        for index in range(plt.colormaps[color_map].N)
    ]

    figure, axis = plt.subplots(figsize=(28, 18))
    for index, (name, counts) in enumerate(sorted_results):
        median = np.median(counts)
        p90 = np.percentile(counts, 90)
        label = f"{name} (n={len(counts):,}, med={median:,.0f}, p90={p90:,.0f})"
        axis.hist(
            counts,
            bins=bins,
            alpha=0.7,
            label=label,
            histtype="step",
            linewidth=1.8,
            color=colors[index % len(colors)],
        )

    axis.set_xscale("log")
    axis.set_xlabel("Token Count (log scale)", fontsize=14)
    axis.set_ylabel("Frequency", fontsize=14)
    axis.set_title(
        title or f"Context Length Distribution Across {len(results)} Datasets", fontsize=18
    )
    for cutoff, label in ((8192, "8k"), (16384, "16k"), (32768, "32k"), (65536, "64k"), (131072, "131k")):
        axis.axvline(cutoff, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        axis.text(cutoff, axis.get_ylim()[1] * 0.95, f" {label}", fontsize=9, color="gray", va="top")
    axis.legend(
        loc="upper left",
        fontsize=8,
        ncol=1,
        framealpha=0.9,
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare context length statistics across HF datasets.",
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="HuggingFace dataset repo IDs to compare",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-8B",
        help="HF tokenizer to use (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--representation",
        choices=TOKEN_REPRESENTATIONS,
        default="conversation_text",
        help=(
            "Token input to measure: concatenated conversation_text (the prior "
            "default), serialized JSON, or tokenizer chat_template."
        ),
    )
    parser.add_argument(
        "--filter",
        dest="filter_spec",
        default=None,
        help="Row filter in 'column==value' format (e.g. 'trace_source==main')",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional path for a log-scale distribution overlay PNG.",
    )
    parser.add_argument(
        "--plot-bins",
        type=int,
        default=80,
        help="Number of log-scale histogram bin edges when --plot is set (default: 80).",
    )
    parser.add_argument(
        "--plot-title",
        default=None,
        help="Optional title for the distribution plot.",
    )
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )

    all_results: list[tuple[str, np.ndarray]] = []

    for repo_id in args.datasets:
        ds, _ = load_and_filter(repo_id, args.split, args.filter_spec)
        print(f"  Tokenizing {len(ds):,} rows...")
        counts = tokenize_dataset(ds, tokenizer, representation=args.representation)
        if not len(counts):
            print("  Warning: no rows available after filtering; skipping.")
            continue
        print_stats(repo_id, counts)
        all_results.append((repo_id, counts))

    # Print comparison table if multiple datasets
    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print("  COMPARISON SUMMARY")
        print(f"{'=' * 80}")
        header = f"{'Dataset':<65} {'Rows':>6} {'Mean':>8} {'Median':>8} {'P90':>8} {'Max':>8}"
        print(header)
        print("-" * len(header))
        for name, counts in all_results:
            short = name.split("/")[-1][:62]
            print(
                f"{short:<65} {len(counts):>6,} "
                f"{np.mean(counts):>8,.0f} {np.median(counts):>8,.0f} "
                f"{np.percentile(counts, 90):>8,.0f} {np.max(counts):>8,.0f}"
            )

    if args.plot is not None:
        plot_context_length_distributions(
            all_results,
            args.plot,
            bin_count=args.plot_bins,
            title=args.plot_title,
        )
        print(f"\nSaved distribution plot to {args.plot}")


if __name__ == "__main__":
    main()
