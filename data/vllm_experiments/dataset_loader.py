#!/usr/bin/env python3
"""
Dataset loader for vLLM benchmark prompts.

Generates or loads benchmark datasets with controlled prompt/output length distributions.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class BenchmarkPrompt:
    """A single benchmark prompt with metadata."""
    prompt: str
    prompt_length: int  # tokens (approximate)
    expected_output_length: int  # tokens
    metadata: Dict[str, Any]


class DatasetLoader:
    """Load or generate benchmark datasets for vLLM experiments."""

    # Synthetic prompt templates for different complexity levels
    PROMPT_TEMPLATES = {
        "short": [
            "Write a Python function to calculate {task}.",
            "Explain the concept of {concept} in simple terms.",
            "What is {question}?",
            "Summarize the key points about {topic}.",
        ],
        "medium": [
            "Write a detailed Python class that implements {task}. Include error handling, "
            "type hints, and docstrings. Make sure the code follows best practices.",
            "Provide a comprehensive explanation of {concept}, including its history, "
            "applications, advantages, and disadvantages. Use examples to illustrate key points.",
            "Compare and contrast {topic1} and {topic2}. Discuss their similarities, "
            "differences, use cases, and when to choose one over the other.",
        ],
        "long": [
            "Design and implement a complete Python module for {task}. The module should include: "
            "1) A well-structured class hierarchy with base classes and derived classes, "
            "2) Comprehensive error handling and validation, "
            "3) Unit tests using pytest, "
            "4) Complete docstrings and type hints, "
            "5) Example usage demonstrating all major features. "
            "Follow PEP 8 style guidelines and best practices for production code.",
            "Write a detailed technical blog post about {concept}. The post should cover: "
            "1) Introduction and motivation (why this matters), "
            "2) Historical context and evolution, "
            "3) Technical deep-dive with code examples, "
            "4) Common pitfalls and how to avoid them, "
            "5) Real-world applications and case studies, "
            "6) Future trends and developments, "
            "7) Conclusion with actionable takeaways. "
            "Target audience: intermediate to advanced developers.",
        ],
    }

    TASK_EXAMPLES = [
        "fibonacci numbers", "binary search", "linked list reversal", "merge sort",
        "breadth-first search", "depth-first search", "dynamic programming solution",
        "LRU cache", "hash table", "binary tree traversal", "graph algorithms",
        "string matching", "matrix operations", "recursive backtracking",
    ]

    CONCEPT_EXAMPLES = [
        "machine learning", "neural networks", "transformers", "attention mechanisms",
        "gradient descent", "backpropagation", "regularization", "cross-validation",
        "reinforcement learning", "generative models", "transfer learning",
        "distributed computing", "MapReduce", "consensus algorithms", "CAP theorem",
    ]

    TOPIC_EXAMPLES = [
        "Python decorators", "context managers", "generators", "coroutines",
        "metaclasses", "descriptors", "async/await", "type hints",
        "REST APIs", "GraphQL", "microservices", "serverless architecture",
        "Docker containers", "Kubernetes", "CI/CD pipelines", "test automation",
    ]

    def __init__(self, output_dir: str = "datasets", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
        return len(text) // 4

    def generate_synthetic_prompt(self, length_category: str = "medium") -> BenchmarkPrompt:
        """Generate a synthetic prompt of specified length category."""
        if length_category not in self.PROMPT_TEMPLATES:
            length_category = "medium"

        template = random.choice(self.PROMPT_TEMPLATES[length_category])

        # Fill template with random examples
        prompt = template.format(
            task=random.choice(self.TASK_EXAMPLES),
            concept=random.choice(self.CONCEPT_EXAMPLES),
            topic=random.choice(self.TOPIC_EXAMPLES),
            topic1=random.choice(self.TOPIC_EXAMPLES),
            topic2=random.choice(self.TOPIC_EXAMPLES),
            question=f"the difference between {random.choice(self.CONCEPT_EXAMPLES)} and {random.choice(self.CONCEPT_EXAMPLES)}",
        )

        # Estimate lengths
        prompt_length = self.estimate_token_count(prompt)

        # Expected output length based on prompt category
        output_length_map = {
            "short": (50, 150),
            "medium": (150, 500),
            "long": (500, 2000),
        }
        expected_output_length = random.randint(*output_length_map[length_category])

        return BenchmarkPrompt(
            prompt=prompt,
            prompt_length=prompt_length,
            expected_output_length=expected_output_length,
            metadata={
                "category": length_category,
                "template_type": "synthetic",
            }
        )

    def generate_dataset(
        self,
        num_prompts: int = 500,
        length_distribution: Optional[Dict[str, float]] = None,
        max_output_tokens: int = 512,
    ) -> List[BenchmarkPrompt]:
        """
        Generate a synthetic dataset with controlled characteristics.

        Args:
            num_prompts: Total number of prompts to generate
            length_distribution: Distribution of prompt lengths
                e.g., {"short": 0.3, "medium": 0.5, "long": 0.2}
            max_output_tokens: Maximum tokens to generate per prompt

        Returns:
            List of BenchmarkPrompt objects
        """
        if length_distribution is None:
            length_distribution = {"short": 0.3, "medium": 0.5, "long": 0.2}

        # Normalize distribution
        total = sum(length_distribution.values())
        length_distribution = {k: v/total for k, v in length_distribution.items()}

        # Calculate counts for each category
        counts = {}
        for category, proportion in length_distribution.items():
            counts[category] = int(num_prompts * proportion)

        # Adjust for rounding errors
        counts["medium"] += num_prompts - sum(counts.values())

        # Generate prompts
        prompts = []
        for category, count in counts.items():
            for _ in range(count):
                prompt = self.generate_synthetic_prompt(category)
                # Cap output length
                prompt.expected_output_length = min(prompt.expected_output_length, max_output_tokens)
                prompts.append(prompt)

        # Shuffle
        random.shuffle(prompts)

        return prompts

    def save_dataset(
        self,
        prompts: List[BenchmarkPrompt],
        filename: str,
        format: str = "jsonl"
    ) -> Path:
        """
        Save dataset to file.

        Args:
            prompts: List of BenchmarkPrompt objects
            filename: Output filename
            format: Format ('jsonl' or 'json')

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        if format == "jsonl":
            with open(output_path, 'w') as f:
                for prompt in prompts:
                    record = {
                        "prompt": prompt.prompt,
                        "prompt_length": prompt.prompt_length,
                        "expected_output_length": prompt.expected_output_length,
                        "metadata": prompt.metadata,
                    }
                    f.write(json.dumps(record) + "\n")
        elif format == "json":
            records = [
                {
                    "prompt": p.prompt,
                    "prompt_length": p.prompt_length,
                    "expected_output_length": p.expected_output_length,
                    "metadata": p.metadata,
                }
                for p in prompts
            ]
            with open(output_path, 'w') as f:
                json.dump(records, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

        return output_path

    def load_dataset(self, filename: str) -> List[BenchmarkPrompt]:
        """Load dataset from file."""
        file_path = self.output_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        prompts = []

        if filename.endswith(".jsonl"):
            with open(file_path, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    prompts.append(BenchmarkPrompt(
                        prompt=record["prompt"],
                        prompt_length=record["prompt_length"],
                        expected_output_length=record["expected_output_length"],
                        metadata=record.get("metadata", {}),
                    ))
        elif filename.endswith(".json"):
            with open(file_path, 'r') as f:
                records = json.load(f)
                for record in records:
                    prompts.append(BenchmarkPrompt(
                        prompt=record["prompt"],
                        prompt_length=record["prompt_length"],
                        expected_output_length=record["expected_output_length"],
                        metadata=record.get("metadata", {}),
                    ))
        else:
            raise ValueError(f"Unknown file format: {filename}")

        return prompts

    def create_benchmark_suite(self) -> Dict[str, Path]:
        """
        Create a suite of benchmark datasets with different characteristics.

        Returns:
            Dictionary mapping dataset name to file path
        """
        datasets = {}

        # Small quick test (100 prompts)
        print("Generating quick test dataset (100 prompts)...")
        quick_test = self.generate_dataset(
            num_prompts=100,
            length_distribution={"short": 0.4, "medium": 0.5, "long": 0.1},
            max_output_tokens=256,
        )
        datasets["quick_test"] = self.save_dataset(quick_test, "quick_test.jsonl")

        # Medium benchmark (500 prompts) - balanced
        print("Generating medium benchmark dataset (500 prompts)...")
        medium = self.generate_dataset(
            num_prompts=500,
            length_distribution={"short": 0.3, "medium": 0.5, "long": 0.2},
            max_output_tokens=512,
        )
        datasets["medium_balanced"] = self.save_dataset(medium, "medium_balanced.jsonl")

        # Large benchmark (1000 prompts) - for throughput testing
        print("Generating large benchmark dataset (1000 prompts)...")
        large = self.generate_dataset(
            num_prompts=1000,
            length_distribution={"short": 0.3, "medium": 0.5, "long": 0.2},
            max_output_tokens=512,
        )
        datasets["large_throughput"] = self.save_dataset(large, "large_throughput.jsonl")

        # Short prompts (500 prompts) - for latency testing
        print("Generating short prompts dataset (500 prompts)...")
        short = self.generate_dataset(
            num_prompts=500,
            length_distribution={"short": 0.9, "medium": 0.1, "long": 0.0},
            max_output_tokens=256,
        )
        datasets["short_latency"] = self.save_dataset(short, "short_latency.jsonl")

        # Long prompts (300 prompts) - for context testing
        print("Generating long prompts dataset (300 prompts)...")
        long_prompts = self.generate_dataset(
            num_prompts=300,
            length_distribution={"short": 0.0, "medium": 0.3, "long": 0.7},
            max_output_tokens=1024,
        )
        datasets["long_context"] = self.save_dataset(long_prompts, "long_context.jsonl")

        return datasets

    def print_dataset_stats(self, prompts: List[BenchmarkPrompt]):
        """Print statistics about a dataset."""
        prompt_lengths = [p.prompt_length for p in prompts]
        output_lengths = [p.expected_output_length for p in prompts]

        print(f"\nDataset Statistics:")
        print(f"  Total prompts: {len(prompts)}")
        print(f"\n  Prompt lengths (tokens):")
        print(f"    Mean: {np.mean(prompt_lengths):.1f}")
        print(f"    Median: {np.median(prompt_lengths):.1f}")
        print(f"    Min: {np.min(prompt_lengths)}")
        print(f"    Max: {np.max(prompt_lengths)}")
        print(f"\n  Expected output lengths (tokens):")
        print(f"    Mean: {np.mean(output_lengths):.1f}")
        print(f"    Median: {np.median(output_lengths):.1f}")
        print(f"    Min: {np.min(output_lengths)}")
        print(f"    Max: {np.max(output_lengths)}")

        # Category distribution
        categories = [p.metadata.get("category", "unknown") for p in prompts]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1

        print(f"\n  Category distribution:")
        for cat, count in sorted(category_counts.items()):
            print(f"    {cat}: {count} ({count/len(prompts)*100:.1f}%)")


def main():
    """Generate benchmark datasets."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate vLLM benchmark datasets")
    parser.add_argument("--output-dir", default="datasets", help="Output directory")
    parser.add_argument("--num-prompts", type=int, default=500, help="Number of prompts")
    parser.add_argument("--max-output-tokens", type=int, default=512, help="Max output tokens")
    parser.add_argument("--create-suite", action="store_true", help="Create full benchmark suite")
    parser.add_argument("--show-stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    loader = DatasetLoader(args.output_dir, seed=args.seed)

    if args.create_suite:
        print("Creating benchmark suite...")
        datasets = loader.create_benchmark_suite()
        print(f"\nCreated {len(datasets)} datasets:")
        for name, path in datasets.items():
            print(f"  {name}: {path}")

            if args.show_stats:
                prompts = loader.load_dataset(path.name)
                loader.print_dataset_stats(prompts)
    else:
        print(f"Generating dataset with {args.num_prompts} prompts...")
        prompts = loader.generate_dataset(
            num_prompts=args.num_prompts,
            max_output_tokens=args.max_output_tokens,
        )

        output_path = loader.save_dataset(prompts, f"custom_{args.num_prompts}.jsonl")
        print(f"Saved dataset to: {output_path}")

        if args.show_stats:
            loader.print_dataset_stats(prompts)


if __name__ == "__main__":
    main()
