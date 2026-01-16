#!/usr/bin/env python3
"""
Token counting utilities for datasets

Provides utilities to count tokens in text using tiktoken (OpenAI's tokenizer)
which is commonly used for estimating API costs and dataset sizes.
"""

import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not installed. Install with: pip install tiktoken")


class TokenCounter:
    """Count tokens in text using various tokenizers"""

    def __init__(self, model: str = "gpt-4"):
        """
        Initialize token counter

        Args:
            model: Model name for tokenizer (e.g., "gpt-4", "gpt-3.5-turbo", "text-embedding-ada-002")
        """
        self.model = model
        self.encoding = None

        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                print(f"Warning: Model {model} not found, using cl100k_base encoding")
                self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        if not TIKTOKEN_AVAILABLE or self.encoding is None:
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4

        return len(self.encoding.encode(text))

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts

        Args:
            texts: List of texts to tokenize

        Returns:
            List of token counts
        """
        return [self.count_tokens(text) for text in texts]

    def get_token_stats(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get statistics about token counts

        Args:
            texts: List of texts to analyze

        Returns:
            Dictionary with statistics
        """
        counts = self.count_tokens_batch(texts)

        if not counts:
            return {
                'total_tokens': 0,
                'mean_tokens': 0,
                'min_tokens': 0,
                'max_tokens': 0,
                'num_texts': 0
            }

        return {
            'total_tokens': sum(counts),
            'mean_tokens': sum(counts) / len(counts),
            'min_tokens': min(counts),
            'max_tokens': max(counts),
            'num_texts': len(counts),
            'counts': counts
        }


def count_tokens_in_dataset(
    dataset_path: str,
    instruction_filename: str = "instruction.md",
    model: str = "gpt-4"
) -> Dict[str, Any]:
    """
    Count tokens in a sandboxes-format dataset

    Args:
        dataset_path: Path to dataset directory containing task subdirectories
        instruction_filename: Name of instruction file in each task directory
        model: Model name for tokenizer

    Returns:
        Dictionary with token statistics
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # Find all task directories
    task_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    if not task_dirs:
        print(f"Warning: No task directories found in {dataset_path}")
        return {'total_tokens': 0, 'num_tasks': 0}

    counter = TokenCounter(model=model)
    all_instructions = []
    task_stats = []

    for task_dir in task_dirs:
        instruction_file = task_dir / instruction_filename

        if not instruction_file.exists():
            print(f"Warning: {instruction_file} not found, skipping")
            continue

        instruction_text = instruction_file.read_text()
        token_count = counter.count_tokens(instruction_text)

        all_instructions.append(instruction_text)
        task_stats.append({
            'task_dir': str(task_dir.name),
            'tokens': token_count,
            'chars': len(instruction_text)
        })

    # Get overall statistics
    overall_stats = counter.get_token_stats(all_instructions)

    return {
        'overall': overall_stats,
        'per_task': task_stats,
        'num_tasks': len(task_stats),
        'model': model
    }


def format_token_stats(stats: Dict[str, Any]) -> str:
    """
    Format token statistics as human-readable string

    Args:
        stats: Statistics dictionary from count_tokens_in_dataset

    Returns:
        Formatted string
    """
    if not stats or 'overall' not in stats:
        return "No statistics available"

    overall = stats['overall']

    output = []
    output.append("Token Statistics")
    output.append("=" * 50)
    output.append(f"Model: {stats.get('model', 'unknown')}")
    output.append(f"Number of tasks: {stats['num_tasks']}")
    output.append(f"Total tokens: {overall['total_tokens']:,}")
    output.append(f"Mean tokens per task: {overall['mean_tokens']:.1f}")
    output.append(f"Min tokens: {overall['min_tokens']:,}")
    output.append(f"Max tokens: {overall['max_tokens']:,}")

    # Estimate costs (rough estimates for GPT-4)
    cost_per_1k_input = 0.03  # $0.03 per 1K tokens for GPT-4 input
    cost_per_1k_output = 0.06  # $0.06 per 1K tokens for GPT-4 output

    estimated_input_cost = (overall['total_tokens'] / 1000) * cost_per_1k_input
    estimated_output_cost = (overall['total_tokens'] / 1000) * cost_per_1k_output

    output.append("")
    output.append("Estimated Costs (GPT-4 rates):")
    output.append(f"  Input tokens:  ${estimated_input_cost:.2f}")
    output.append(f"  Output tokens: ${estimated_output_cost:.2f}")
    output.append(f"  Total (input + output): ${estimated_input_cost + estimated_output_cost:.2f}")
    output.append("")
    output.append("Note: Actual costs depend on model used and provider rates")
    output.append("=" * 50)

    return "\n".join(output)


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Count tokens in a dataset")
    parser.add_argument("dataset_path", help="Path to dataset directory")
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="Model name for tokenizer (default: gpt-4)"
    )
    parser.add_argument(
        "--instruction-filename",
        default="instruction.md",
        help="Name of instruction file (default: instruction.md)"
    )

    args = parser.parse_args()

    print(f"Counting tokens in: {args.dataset_path}")
    print(f"Using model: {args.model}")
    print()

    stats = count_tokens_in_dataset(
        args.dataset_path,
        instruction_filename=args.instruction_filename,
        model=args.model
    )

    print(format_token_stats(stats))

    # Optionally save to JSON
    import json
    output_file = Path(args.dataset_path) / "token_stats.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nDetailed statistics saved to: {output_file}")


if __name__ == "__main__":
    main()
