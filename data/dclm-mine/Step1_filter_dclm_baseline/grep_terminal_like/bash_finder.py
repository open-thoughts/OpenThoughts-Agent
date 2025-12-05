#!/usr/bin/env python3
"""
DCLM Bash Sequence Finder - Parallel Version

Processes all DCLM shards in parallel using multiprocessing.
Optimized for processing 20k+ files on compute nodes.
"""

import re
import json
import random
import zstandard as zstd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from urllib.parse import urlparse
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time


@dataclass
class BashScore:
    """Detailed breakdown of bash detection score"""
    total_score: float
    bash_block: bool = False
    shebang: bool = False
    prompt_pattern: int = 0
    pipe_pattern: int = 0
    redirect_pattern: int = 0

    def to_dict(self):
        return asdict(self)


class BashDetector:
    """Detects and scores bash-related content in text sequences"""

    def __init__(self):
        # Compile all regex patterns
        self.bash_block_re = re.compile(r"```(bash|sh|zsh|shell)", re.IGNORECASE)
        self.shebang_re = re.compile(r"#!/(usr/bin/env )?(ba)?sh")
        self.prompt_re = re.compile(
            r"(?m)^[ \t]*(?:"
            r"\$ +\S+"                         # normal user shell prompt: "$ ls"
            r"|"
            r"[A-Za-z0-9._-]+@[A-Za-z0-9._-]+:" # user@host:
            r"[^\n$#]*# +\S+"                  # root-style prompt: user@host:~/dir# apt update
            r")"
        )
        self.pipe_re = re.compile(r"\|\s+(grep|awk|sed|xargs|sort|uniq|cut|head|tail|wc)\b")
        self.redir_re = re.compile(r"(>/dev/null 2>&1|2>\s*&1|<<\s*EOF|<<-EOF|<<\s*\"EOF\")")

        # Additional patterns for more robust detection
        self.common_commands_re = re.compile(r"\b(cd|ls|mkdir|rm|cp|mv|chmod|chown|cat|echo|export|source)\b")
        self.script_patterns_re = re.compile(r"\b(if\s+\[|while\s+\[|for\s+\w+\s+in|function\s+\w+|case\s+\S+\s+in)\b")

    def score_text(self, text: str) -> BashScore:
        """Score text for bash-related content with detailed breakdown"""
        score = BashScore(total_score=0.0)

        # High confidence indicators (10 points each)
        if self.bash_block_re.search(text):
            score.bash_block = True
            score.total_score += 10.0

        if self.shebang_re.search(text):
            score.shebang = True
            score.total_score += 10.0

        # Medium confidence patterns (2 points each, count occurrences)
        prompt_matches = len(self.prompt_re.findall(text))
        if prompt_matches > 0:
            score.prompt_pattern = prompt_matches
            score.total_score += min(prompt_matches * 2.0, 8.0)  # Cap at 8 points

        pipe_matches = len(self.pipe_re.findall(text))
        if pipe_matches > 0:
            score.pipe_pattern = pipe_matches
            score.total_score += min(pipe_matches * 2.0, 6.0)  # Cap at 6 points

        redir_matches = len(self.redir_re.findall(text))
        if redir_matches > 0:
            score.redirect_pattern = redir_matches
            score.total_score += min(redir_matches * 2.0, 6.0)  # Cap at 6 points

        # Additional patterns (1-2 points)
        if self.common_commands_re.search(text):
            score.total_score += 1.0

        if self.script_patterns_re.search(text):
            score.total_score += 2.0

        return score


@dataclass
class WorkerResult:
    """Result from processing a single shard file"""
    file_path: str
    total_sequences: int
    bash_sequences: int
    bash_results: List[Dict]
    score_distribution: Dict[int, int]
    domain_counts: Dict[str, int]
    error: Optional[str] = None


def extract_base_domain(url: str) -> str:
    """
    Extract base domain from URL (e.g., 'apple.stackexchange.com' from full URL)
    Returns empty string if URL is invalid
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc or parsed.path.split('/')[0]
        # Remove 'www.' prefix if present
        if netloc.startswith('www.'):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def process_single_shard(file_path: Path, threshold: float) -> WorkerResult:
    """
    Process a single shard file - worker function for multiprocessing

    Args:
        file_path: Path to .jsonl.zst file
        threshold: Score threshold for bash classification

    Returns:
        WorkerResult with statistics and bash sequences
    """
    detector = BashDetector()
    bash_results = []
    total_sequences = 0
    bash_sequences = 0
    score_distribution = defaultdict(int)
    domain_counts = defaultdict(int)

    try:
        with open(file_path, 'rb') as compressed:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed) as reader:
                text_stream = reader.read().decode('utf-8')

                for line_idx, line in enumerate(text_stream.splitlines()):
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        text = data.get('text', '')
                        url = data.get('url', '')
                        metadata = data.get('metadata', {})

                        if not text:
                            continue

                        total_sequences += 1

                        # Score the text
                        score = detector.score_text(text)

                        # Track score distribution
                        score_bucket = int(score.total_score)
                        score_distribution[score_bucket] += 1

                        # If it looks like bash, save it
                        if score.total_score >= threshold:
                            # Extract domain for frequency tracking
                            domain = extract_base_domain(url)
                            if domain:
                                domain_counts[domain] += 1
                            else:
                                domain_counts["<no_domain>"] += 1

                            result = {
                                'text': text,
                                'score': score.to_dict(),
                                'source_file': file_path.name,
                                'sequence_index': line_idx,
                                'url': url,
                                'metadata': metadata,
                            }
                            bash_results.append(result)
                            bash_sequences += 1

                    except json.JSONDecodeError:
                        continue

        return WorkerResult(
            file_path=str(file_path),
            total_sequences=total_sequences,
            bash_sequences=bash_sequences,
            bash_results=bash_results,
            score_distribution=dict(score_distribution),
            domain_counts=dict(domain_counts),
            error=None
        )

    except Exception as e:
        return WorkerResult(
            file_path=str(file_path),
            total_sequences=0,
            bash_sequences=0,
            bash_results=[],
            score_distribution={},
            domain_counts={},
            error=str(e)
        )


class ParallelShardProcessor:
    """Processes DCLM shards in parallel"""

    def __init__(self, output_dir: Path, threshold: float = 4.0, num_workers: Optional[int] = None,
                 sequences_per_shard: int = 80000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.num_workers = num_workers or mp.cpu_count()
        self.sequences_per_shard = sequences_per_shard

        # Aggregated statistics
        self.stats = {
            'total_sequences': 0,
            'bash_sequences': 0,
            'files_processed': 0,
            'files_failed': 0,
            'score_distribution': defaultdict(int),
            'domain_counts': Counter()
        }

        # Output file sharding
        self.current_shard_idx = 0
        self.current_shard_count = 0
        self.current_shard_file = None
        self.progress_file = self.output_dir / 'processing_progress.txt'

    def _get_next_shard_file(self):
        """Open next output shard file"""
        if self.current_shard_file:
            self.current_shard_file.close()

        shard_filename = self.output_dir / f'bash_sequences_shard_{self.current_shard_idx:05d}.jsonl'
        self.current_shard_file = open(shard_filename, 'w')
        self.current_shard_count = 0
        print(f"\nCreating new output shard: {shard_filename.name}")
        return self.current_shard_file

    def process_all_shards(self, shard_files: List[Path], checkpoint_interval: int = 100) -> None:
        """
        Process all shard files in parallel with checkpointing

        Args:
            shard_files: List of paths to .jsonl.zst files
            checkpoint_interval: Save statistics every N files
        """
        print(f"Processing {len(shard_files)} shard files using {self.num_workers} workers")
        print(f"Threshold: {self.threshold}")
        print(f"Output directory: {self.output_dir}")
        print(f"Sequences per output shard: {self.sequences_per_shard:,}")

        start_time = time.time()

        # Create progress file
        with open(self.progress_file, 'w') as f:
            f.write(f"Started processing at {time.ctime()}\n")
            f.write(f"Total files: {len(shard_files)}\n")
            f.write(f"Workers: {self.num_workers}\n")
            f.write(f"Threshold: {self.threshold}\n")
            f.write(f"Sequences per output shard: {self.sequences_per_shard:,}\n\n")

        # Process in parallel with progress bar
        worker_func = partial(process_single_shard, threshold=self.threshold)

        # Start with first shard file
        out_file = self._get_next_shard_file()

        try:
            with mp.Pool(processes=self.num_workers) as pool:
                # Use imap_unordered for better performance (results come as they complete)
                results_iter = pool.imap_unordered(worker_func, shard_files, chunksize=1)

                # Process results as they come in
                for i, result in enumerate(tqdm(results_iter, total=len(shard_files),
                                               desc="Processing shards"), 1):
                    # Handle errors
                    if result.error:
                        self.stats['files_failed'] += 1
                        print(f"\nError processing {result.file_path}: {result.error}")
                        continue

                    # Update statistics
                    self.stats['total_sequences'] += result.total_sequences
                    self.stats['bash_sequences'] += result.bash_sequences
                    self.stats['files_processed'] += 1

                    # Merge score distributions
                    for score, count in result.score_distribution.items():
                        self.stats['score_distribution'][score] += count

                    # Merge domain counts
                    for domain, count in result.domain_counts.items():
                        self.stats['domain_counts'][domain] += count

                    # Write bash sequences to file with sharding
                    for seq in result.bash_results:
                        # Check if we need to start a new shard
                        if self.current_shard_count >= self.sequences_per_shard:
                            self.current_shard_idx += 1
                            out_file = self._get_next_shard_file()

                        json.dump(seq, out_file)
                        out_file.write('\n')
                        self.current_shard_count += 1

                    # Checkpoint progress
                    if i % checkpoint_interval == 0:
                        self._save_checkpoint(i, len(shard_files), time.time() - start_time)
        finally:
            # Close the last shard file
            if self.current_shard_file:
                self.current_shard_file.close()

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"\n{'='*80}")
        print(f"Processing complete!")
        print(f"Time elapsed: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files failed: {self.stats['files_failed']}")
        print(f"Output shards created: {self.current_shard_idx + 1}")
        print(f"Average time per file: {elapsed/len(shard_files):.2f} seconds")
        print(f"{'='*80}")

    def _save_checkpoint(self, files_done: int, total_files: int, elapsed: float) -> None:
        """Save progress checkpoint"""
        with open(self.progress_file, 'a') as f:
            f.write(f"\nCheckpoint at {time.ctime()}\n")
            f.write(f"Files processed: {files_done}/{total_files} ({files_done/total_files*100:.1f}%)\n")
            f.write(f"Time elapsed: {elapsed/60:.1f} minutes\n")
            f.write(f"Total sequences: {self.stats['total_sequences']:,}\n")
            f.write(f"Bash sequences: {self.stats['bash_sequences']:,}\n")
            if self.stats['total_sequences'] > 0:
                f.write(f"Proportion: {self.stats['bash_sequences']/self.stats['total_sequences']:.4%}\n")

    def save_final_results(self, sample_size: int = 20) -> None:
        """Save final statistics and random samples"""
        # Save statistics
        stats_file = self.output_dir / 'statistics.json'
        stats_output = {
            **self.stats,
            'proportion_bash': (self.stats['bash_sequences'] / self.stats['total_sequences']
                               if self.stats['total_sequences'] > 0 else 0),
            'score_distribution': dict(self.stats['score_distribution']),
            'domain_counts': dict(self.stats['domain_counts'])
        }

        with open(stats_file, 'w') as f:
            json.dump(stats_output, f, indent=2)

        print(f"\nFinal Statistics:")
        print(f"  Total sequences processed: {self.stats['total_sequences']:,}")
        print(f"  Bash sequences found: {self.stats['bash_sequences']:,}")
        print(f"  Proportion: {stats_output['proportion_bash']:.4%}")
        print(f"  Files processed: {self.stats['files_processed']}")
        print(f"  Files failed: {self.stats['files_failed']}")
        print(f"  Unique domains: {len(self.stats['domain_counts'])}")
        print(f"\nStatistics saved to {stats_file}")

        # Save domain frequency analysis
        self._save_domain_analysis()

        # Create random samples
        self._create_random_samples(sample_size)

    def _save_domain_analysis(self) -> None:
        """Save top 100 domains with their frequencies"""
        print(f"\nAnalyzing domain frequencies...")

        total_bash = self.stats['bash_sequences']
        if total_bash == 0:
            print("No bash sequences to analyze")
            return

        # Get top 100 domains
        top_domains = self.stats['domain_counts'].most_common(100)

        # Calculate count for "others"
        top_100_count = sum(count for _, count in top_domains)
        others_count = total_bash - top_100_count

        # Build the output list
        domain_analysis = []
        for domain, count in top_domains:
            frequency = (count / total_bash) * 100
            domain_analysis.append({
                'domain': domain,
                'count': count,
                'frequency_percent': round(frequency, 2)
            })

        # Add "others" entry
        if others_count > 0:
            others_frequency = (others_count / total_bash) * 100
            domain_analysis.append({
                'domain': 'others',
                'count': others_count,
                'frequency_percent': round(others_frequency, 2)
            })

        # Save to file
        domain_file = self.output_dir / 'domain_frequency_top100.json'
        output = {
            'total_bash_sequences': total_bash,
            'unique_domains': len(self.stats['domain_counts']),
            'top_100_domains': domain_analysis
        }

        with open(domain_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Domain frequency analysis saved to {domain_file}")
        print(f"  Top domain: {top_domains[0][0]} ({top_domains[0][1]:,} sequences, "
              f"{(top_domains[0][1]/total_bash)*100:.2f}%)")
        if len(top_domains) >= 2:
            print(f"  2nd domain: {top_domains[1][0]} ({top_domains[1][1]:,} sequences, "
                  f"{(top_domains[1][1]/total_bash)*100:.2f}%)")
        if len(top_domains) >= 3:
            print(f"  3rd domain: {top_domains[2][0]} ({top_domains[2][1]:,} sequences, "
                  f"{(top_domains[2][1]/total_bash)*100:.2f}%)")

    def _create_random_samples(self, n: int) -> None:
        """Create random samples from all bash sequences across all shards"""
        print(f"\nCreating {n} random samples...")

        # Read from all shard files
        all_sequences = []
        shard_files = sorted(self.output_dir.glob('bash_sequences_shard_*.jsonl'))

        print(f"Reading from {len(shard_files)} output shard(s)...")
        for shard_file in shard_files:
            with open(shard_file, 'r') as f:
                for line in f:
                    if line.strip():
                        all_sequences.append(json.loads(line))

        if len(all_sequences) == 0:
            print("No bash sequences found to sample")
            return

        sample_size = min(n, len(all_sequences))
        samples = random.sample(all_sequences, sample_size)

        # Save samples
        samples_file = self.output_dir / f'bash_sequences_sample_{sample_size}.jsonl'
        with open(samples_file, 'w') as f:
            for seq in samples:
                json.dump(seq, f)
                f.write('\n')

        # Save human-readable version
        readable_file = self.output_dir / f'bash_sequences_sample_{sample_size}_readable.txt'
        with open(readable_file, 'w') as f:
            for i, seq in enumerate(samples, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"SAMPLE {i}/{sample_size}\n")
                f.write(f"Source: {seq['source_file']}, Sequence: {seq['sequence_index']}\n")
                f.write(f"Score: {seq['score']['total_score']:.1f}\n")
                f.write(f"Score breakdown: {seq['score']}\n")
                f.write(f"{'='*80}\n\n")
                f.write(seq['text'])
                f.write("\n\n")

        print(f"Saved {sample_size} random samples to:")
        print(f"  {samples_file}")
        print(f"  {readable_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Find bash sequences in DCLM dataset (parallel version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all shards in a directory with 16 workers
  python3 bash_sequence_finder_parallel.py --shard-dir ./shards --output-dir ./results --workers 16

  # Process with custom threshold and sample size
  python3 bash_sequence_finder_parallel.py --shard-dir ./shards --output-dir ./results \\
      --threshold 3.5 --sample-size 50 --workers 32
        """
    )
    parser.add_argument('--shard-dir', type=str, required=True,
                       help='Directory containing downloaded .jsonl.zst shard files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--threshold', type=float, default=4.0,
                       help='Minimum score threshold to classify as bash (default: 4.0)')
    parser.add_argument('--sample-size', type=int, default=20,
                       help='Number of random sequences to sample for inspection (default: 20)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count)')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                       help='Save progress every N files (default: 100)')
    parser.add_argument('--sequences-per-shard', type=int, default=80000,
                       help='Number of sequences per output shard file (default: 80000)')

    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    output_dir = Path(args.output_dir)

    if not shard_dir.exists():
        print(f"Error: Shard directory does not exist: {shard_dir}")
        return

    # Find all .jsonl.zst files (including in subdirectories)
    shard_files = list(shard_dir.glob('**/*.jsonl.zst'))

    if not shard_files:
        print(f"No .jsonl.zst files found in {shard_dir}")
        return

    print(f"Found {len(shard_files)} shard files")

    # Process shards in parallel
    processor = ParallelShardProcessor(
        output_dir=output_dir,
        threshold=args.threshold,
        num_workers=args.workers,
        sequences_per_shard=args.sequences_per_shard
    )

    processor.process_all_shards(
        shard_files=shard_files,
        checkpoint_interval=args.checkpoint_interval
    )

    # Save final results
    processor.save_final_results(sample_size=args.sample_size)


if __name__ == '__main__':
    main()
