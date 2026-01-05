"""Plot episode-count distributions for Harbor trace datasets on Hugging Face."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None


_EPISODE_PATTERN = re.compile(r"(\d+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch one or more Hugging Face datasets containing an 'episode' "
            "column and plot the distribution of episode indices."
        )
    )
    parser.add_argument(
        "repos",
        nargs="+",
        help="Hugging Face dataset repo IDs (e.g., org/dataset).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load (default: train).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Standard deviation for Gaussian smoothing (default: 2.0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional filename for the plot (saved in this script's directory).",
    )
    parser.add_argument(
        "--token-bin-size",
        type=int,
        default=200,
        help="Bin size (in tokens) for the per-turn token histogram (default: 200).",
    )
    parser.add_argument(
        "--token-sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma for token histograms (default: 1.5).",
    )
    parser.add_argument(
        "--token-max-percentile",
        type=float,
        default=99.5,
        help="Percentile cap for max token counts to drop outliers (default: 99.5).",
    )
    return parser.parse_args()


def _extract_episode_numbers(values: Iterable) -> List[int]:
    episodes: List[int] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, float)):
            episodes.append(int(value))
            continue
        if isinstance(value, str):
            cleaned = value.replace("-", " ").replace("_", " ")
            match = _EPISODE_PATTERN.search(cleaned)
            if match:
                episodes.append(int(match.group(1)))
            continue
        # Fallback: try to interpret as dict with 'episode' key or skip silently
        if isinstance(value, dict):
            inner = value.get("episode")
            if isinstance(inner, (int, float)):
                episodes.append(int(inner))
            elif isinstance(inner, str):
                cleaned_inner = inner.replace("-", " ").replace("_", " ")
                match = _EPISODE_PATTERN.search(cleaned_inner)
                if match:
                    episodes.append(int(match.group(1)))
    return episodes


def _get_token_encoder():
    if tiktoken is None:
        return None
    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, encoder) -> int:
    if not text:
        return 0
    if encoder is not None:
        return len(encoder.encode(text))
    return len(text.split())


def _extract_token_counts(conversations_col: Iterable, encoder) -> List[int]:
    counts: List[int] = []
    for convo in conversations_col:
        if not isinstance(convo, list) or not convo:
            continue
        last_message = convo[-1]
        if not isinstance(last_message, dict):
            continue
        content = last_message.get("content")
        if not isinstance(content, str):
            continue
        counts.append(_count_tokens(content, encoder))
    return counts


def _gaussian_kernel(sigma: float) -> np.ndarray:
    # Ensure kernel is large enough to capture most of the Gaussian mass
    radius = max(int(3 * sigma), 1)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def _smooth_counts(counts: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel(sigma)
    return np.convolve(counts, kernel, mode="same")


def _prepare_series(episodes: Sequence[int], sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    if not episodes:
        raise ValueError("No valid episodes found in dataset")
    episodes_arr = np.array(episodes, dtype=int)
    min_ep = episodes_arr.min()
    shifted = episodes_arr - min_ep
    max_shifted = shifted.max()
    bins = np.arange(0, max_shifted + 1)
    counts = np.zeros_like(bins, dtype=float)
    for ep in shifted:
        counts[ep] += 1
    smoothed = _smooth_counts(counts, sigma)
    return bins, smoothed


def _load_dataset(repo_id: str, split: str):
    return load_dataset(repo_id, split=split)


def main() -> None:
    args = _parse_args()
    token_encoder = _get_token_encoder()
    if token_encoder is None:
        print(
            "tiktoken not available; falling back to whitespace token counts.",
        )

    repo_data = []
    for repo in args.repos:
        dataset = _load_dataset(repo, split=args.split)
        if "episode" not in dataset.column_names:
            raise ValueError(f"Dataset {repo} split {args.split} lacks an 'episode' column")
        if "conversations" not in dataset.column_names:
            raise ValueError(f"Dataset {repo} split {args.split} lacks a 'conversations' column")

        episodes = _extract_episode_numbers(dataset["episode"])
        if not episodes:
            raise ValueError(f"Dataset {repo} split {args.split} has no parseable episodes")

        token_counts = _extract_token_counts(dataset["conversations"], token_encoder)
        if not token_counts:
            raise ValueError(f"Dataset {repo} split {args.split} has no tokenizable turns")

        repo_data.append(
            {
                "repo": repo,
                "episodes": episodes,
                "token_counts": token_counts,
            }
        )

    plt.figure(figsize=(14, 7))
    max_x = 0

    for entry in repo_data:
        x, y = _prepare_series(entry["episodes"], sigma=args.sigma)
        max_x = max(max_x, x.max())
        label = entry["repo"].split("/")[-1]
        plt.plot(x, y, label=label)

    plt.title("Episode Count Distribution")
    plt.xlabel("Episode index")
    plt.ylabel("Smoothed count")
    if max_x > 0:
        step = 5 if max_x >= 5 else 1
        xticks = np.arange(0, max_x + 1, step)
        plt.xticks(xticks)
    else:
        plt.xticks([0])
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.8, 1])

    script_dir = Path(__file__).resolve().parent
    if args.output:
        base_stem = args.output.stem or "episode_distribution"
        ext = args.output.suffix or ".png"
    else:
        base_stem = "episode_distribution"
        ext = ".png"
    episode_plot_path = script_dir / f"{base_stem}{ext}"
    plt.savefig(episode_plot_path)
    print(f"Saved episode distribution plot to {episode_plot_path}")

    # Token histogram figure
    plt.figure(figsize=(14, 7))
    n_repos = len(repo_data)
    if n_repos == 0:
        raise ValueError("No repositories processed.")
    combined_counts = np.concatenate([np.asarray(entry["token_counts"], dtype=float) for entry in repo_data])
    percentile = min(max(args.token_max_percentile, 0.0), 100.0)
    if percentile < 100.0:
        token_cap = float(np.percentile(combined_counts, percentile))
    else:
        token_cap = float(combined_counts.max())
    token_cap = max(token_cap, 1.0)
    for entry in repo_data:
        filtered = [c for c in entry["token_counts"] if c <= token_cap]
        if not filtered:
            filtered = [min(entry["token_counts"])]
        entry["token_counts"] = filtered
    global_max_tokens = max(max(entry["token_counts"]) for entry in repo_data)
    bin_size = max(args.token_bin_size, 1)
    bins = np.arange(0, global_max_tokens + bin_size, bin_size)
    if len(bins) < 2:
        bins = np.array([0, bin_size])
    centers = bins[:-1] + bin_size / 2
    for entry in repo_data:
        counts, _ = np.histogram(entry["token_counts"], bins=bins)
        counts = counts.astype(float)
        counts = _smooth_counts(counts, sigma=args.token_sigma)
        counts = np.where(counts <= 0, 1e-6, counts)
        plt.plot(centers, counts, label=entry["repo"].split("/")[-1])

    plt.title("Token Count per Turn (Binned)")
    plt.xlabel(f"Tokens per turn (bin size = {bin_size})")
    plt.ylabel("Number of turns")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.8, 1])

    token_plot_path = script_dir / f"{base_stem}_token_hist{ext}"
    plt.savefig(token_plot_path)
    print(f"Saved token histogram plot to {token_plot_path}")


if __name__ == "__main__":
    main()
