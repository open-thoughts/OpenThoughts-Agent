#!/usr/bin/env python3
"""
Analyze domain frequencies from statistics.json and output top 200 domains
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def analyze_domain_counts(stats_file: Path, output_file: Path, top_n: int = 200):
    """
    Read domain counts from statistics.json and output top N domains with proportions

    Args:
        stats_file: Path to statistics.json
        output_file: Path to output JSON file
        top_n: Number of top domains to include (default: 200)
    """
    # Read statistics file
    print(f"Reading {stats_file}...")
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    domain_counts = stats.get('domain_counts', {})

    if not domain_counts:
        print("Error: No domain_counts found in statistics.json")
        return

    # Convert to Counter for easy sorting
    counter = Counter(domain_counts)
    total_count = sum(counter.values())

    print(f"Total bash sequences: {total_count:,}")
    print(f"Unique domains: {len(counter):,}")

    # Get top N domains
    top_domains = counter.most_common(top_n)

    # Calculate proportion for top domains
    top_domains_count = sum(count for _, count in top_domains)
    others_count = total_count - top_domains_count

    # Build output list
    domain_analysis = []
    for domain, count in top_domains:
        proportion = (count / total_count) * 100
        domain_analysis.append({
            'domain': domain,
            'count': count,
            'proportion_percent': round(proportion, 3)
        })

    # Add "others" if there are more domains
    if others_count > 0:
        others_proportion = (others_count / total_count) * 100
        domain_analysis.append({
            'domain': 'others',
            'count': others_count,
            'proportion_percent': round(others_proportion, 3)
        })

    # Create output
    output = {
        'total_bash_sequences': total_count,
        'unique_domains': len(counter),
        'top_domains_included': top_n,
        'top_domains': domain_analysis
    }

    # Write output file
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nAnalysis complete!")
    print(f"Output written to: {output_file}")
    print(f"\nTop 5 domains:")
    for i, (domain, count) in enumerate(top_domains[:5], 1):
        proportion = (count / total_count) * 100
        print(f"  {i}. {domain}: {count:,} ({proportion:.3f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze domain frequencies from statistics.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze statistics.json in a results directory
  python3 analyze_statistics.py /path/to/results_dir/statistics.json

  # Specify custom output file and top N
  python3 analyze_statistics.py /path/to/statistics.json --output custom_output.json --top-n 500
        """
    )

    parser.add_argument('statistics_file', type=str,
                       help='Path to statistics.json file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: domain_analysis_top200.json in same directory)')
    parser.add_argument('--top-n', type=int, default=200,
                       help='Number of top domains to include (default: 200)')

    args = parser.parse_args()

    stats_file = Path(args.statistics_file)

    if not stats_file.exists():
        print(f"Error: Statistics file does not exist: {stats_file}")
        return

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = stats_file.parent / f'domain_analysis_top{args.top_n}.json'

    analyze_domain_counts(stats_file, output_file, args.top_n)


if __name__ == '__main__':
    main()
