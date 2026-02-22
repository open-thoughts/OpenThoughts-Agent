#!/usr/bin/env python3
"""
Analyze ablation experiment results from the database.
Creates CSV of results and plots of score vs sampling parameter for each ablation type.
"""

import os
import csv
import json
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from supabase import create_client, Client
import matplotlib.pyplot as plt
import numpy as np

# Set environment variables
os.environ['SUPABASE_URL'] = 'https://rpzmyuapoqilpghynmza.supabase.co/'
os.environ['SUPABASE_SERVICE_ROLE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJwem15dWFwb3FpbHBnaHlubXphIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDI3ODI0OCwiZXhwIjoyMDY5ODU0MjQ4fQ.uKmQ_WpxT2npuy_pQeq2tQ4nu685CH9srzTikShCmMQ'

# Define ablation experiments by category
# All patterns use exp_tas_ prefix for the ablation experiments
# baseline_x_value is the x-axis value from baseline.sbatch for plotting baseline star marker
ABLATION_CATEGORIES = {
    'temperature': {
        'patterns': ['exp_tas_temp_0.25', 'exp_tas_temp_0.5', 'exp_tas_temp_2.0', 'exp_tas_temp_4.0'],
        'baseline_x_value': 1.0,  # from baseline.sbatch: temperature=1.0
        'label': 'Temperature',
        'values': [0.25, 0.5, 2.0, 4.0]
    },
    'max_tokens': {
        'patterns': ['exp_tas_max_tokens_1024', 'exp_tas_max_tokens_2048', 'exp_tas_max_tokens_4096', 'exp_tas_max_tokens_8192'],
        'baseline_x_value': 16384,  # default max_tokens (not explicitly set in baseline.sbatch)
        'label': 'Max Tokens',
        'values': [1024, 2048, 4096, 8192]
    },
    'max_episodes': {
        'patterns': ['exp_tas_max_episodes_32', 'exp_tas_max_episodes_64', 'exp_tas_max_episodes_256', 'exp_tas_max_episodes_512'],
        'baseline_x_value': 128,  # from baseline.sbatch: max_episodes=128
        'label': 'Max Episodes',
        'values': [32, 64, 256, 512]
    },
    'top_k': {
        'patterns': ['exp_tas_top_k_16', 'exp_tas_top_k_32', 'exp_tas_top_k_64', 'exp_tas_top_k_128'],
        'baseline_x_value': None,  # not set in baseline (disabled by default)
        'label': 'Top-K',
        'values': [16, 32, 64, 128]
    },
    'min_p': {
        'patterns': ['exp_tas_min_p_0.01', 'exp_tas_min_p_0.05', 'exp_tas_min_p_0.1'],
        'baseline_x_value': 0.0,  # default (not set)
        'label': 'Min-P',
        'values': [0.01, 0.05, 0.1]
    },
    'frequency_penalty': {
        'patterns': ['exp_tas_frequency_penalty_0.25', 'exp_tas_frequency_penalty_0.5', 'exp_tas_frequency_penalty_1.0'],
        'baseline_x_value': 0.0,  # default (not set)
        'label': 'Frequency Penalty',
        'values': [0.25, 0.5, 1.0]
    },
    'presence_penalty': {
        'patterns': ['exp_tas_presence_penalty_0.25', 'exp_tas_presence_penalty_0.5', 'exp_tas_presence_penalty_1.0'],
        'baseline_x_value': 0.0,  # default (not set)
        'label': 'Presence Penalty',
        'values': [0.25, 0.5, 1.0]
    },
    'repetition_penalty': {
        'patterns': ['exp_tas_repetition_penalty_1.05', 'exp_tas_repetition_penalty_1.1', 'exp_tas_repetition_penalty_1.2'],
        'baseline_x_value': 1.0,  # default (not set)
        'label': 'Repetition Penalty',
        'values': [1.05, 1.1, 1.2]
    },
    'summarize_threshold': {
        'patterns': ['exp_tas_summarize_threshold_2048', 'exp_tas_summarize_threshold_4096', 'exp_tas_summarize_threshold_16384', 'exp_tas_summarize_threshold_32768'],
        'baseline_x_value': 8192,  # from baseline.sbatch: proactive_summarization_threshold=8192
        'label': 'Summarize Threshold',
        'values': [2048, 4096, 16384, 32768]
    },
    'timeout_multiplier': {
        'patterns': ['exp_tas_timeout_multiplier_0.25', 'exp_tas_timeout_multiplier_1.0', 'exp_tas_timeout_multiplier_4.0', 'exp_tas_timeout_multiplier_8.0'],
        'baseline_x_value': 2.0,  # from baseline.sbatch: timeout-multiplier 2.0
        'label': 'Timeout Multiplier',
        'values': [0.25, 1.0, 4.0, 8.0]
    },
    'binary_flags': {
        'patterns': ['exp_tas_summarize_off', 'exp_tas_interleaved_thinking', 'exp_tas_linear_history_off', 'exp_tas_raw_content_off', 'exp_tas_full_thinking', 'exp_tas_parser_xml', 'exp_tas_tmux_large', 'exp_tas_high_diversity', 'exp_tas_low_diversity'],
        'baseline_x_value': None,
        'label': 'Binary Flags',
        'values': None  # categorical
    }
}


def create_supabase_client() -> Client:
    """Create and return Supabase client."""
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment")

    return create_client(url, key)


def get_benchmarks_map(client: Client) -> Dict[str, str]:
    """Get mapping of benchmark_id to benchmark name."""
    response = client.table('benchmarks').select('id, name').execute()
    return {b['id']: b['name'] for b in response.data}


def extract_accuracy_metrics(metrics_data) -> Tuple[Optional[float], Optional[float]]:
    """Extract accuracy and std from metrics data."""
    try:
        if not metrics_data:
            return None, None

        if isinstance(metrics_data, str):
            metrics = json.loads(metrics_data)
        elif isinstance(metrics_data, (list, dict)):
            metrics = metrics_data
        else:
            return None, None

        accuracy = None
        std = None

        if isinstance(metrics, list):
            for item in metrics:
                if isinstance(item, dict) and 'name' in item and 'value' in item:
                    name = item['name'].lower()
                    value = item['value']

                    if 'accuracy' in name and 'stderr' not in name:
                        accuracy = value
                    elif 'stderr' in name or 'std' in name:
                        std = value

        elif isinstance(metrics, dict):
            if 'accuracy' in metrics:
                accuracy = metrics['accuracy']
            elif 'acc' in metrics:
                accuracy = metrics['acc']
            elif 'score' in metrics:
                accuracy = metrics['score']

            if 'std' in metrics:
                std = metrics['std']
            elif 'accuracy_stderr' in metrics:
                std = metrics['accuracy_stderr']

        return accuracy, std

    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        return None, None


def search_ablation_jobs(client: Client, pattern: str) -> List[dict]:
    """Search for jobs matching an ablation pattern."""
    jobs = []
    offset = 0
    batch_size = 1000

    while True:
        response = client.table('sandbox_jobs').select(
            'id, job_name, model_id, benchmark_id, metrics, created_at, ended_at'
        ).ilike('job_name', f'%{pattern}%').range(offset, offset + batch_size - 1).execute()

        if not response.data:
            break
        jobs.extend(response.data)
        if len(response.data) < batch_size:
            break
        offset += batch_size

    return jobs


def search_baseline_jobs(client: Client) -> List[dict]:
    """Search for baseline experiment jobs (exp_tas_baseline specifically)."""
    jobs = []
    offset = 0
    batch_size = 1000

    # Search for exp_tas_baseline specifically (the actual baseline experiment)
    while True:
        response = client.table('sandbox_jobs').select(
            'id, job_name, model_id, benchmark_id, metrics, created_at, ended_at'
        ).ilike('job_name', '%exp_tas_baseline%').range(offset, offset + batch_size - 1).execute()

        if not response.data:
            break
        jobs.extend(response.data)
        if len(response.data) < batch_size:
            break
        offset += batch_size

    return jobs


def parse_experiment_name(job_name: str) -> Tuple[str, Optional[str]]:
    """Parse experiment name and shard from job name."""
    # Pattern: experiment_name_fingerprint_shardN or experiment_name_timestamp_shardN
    parts = job_name.split('_shard')
    if len(parts) >= 2:
        exp_name = parts[0]
        shard = parts[1].split('_')[0] if parts[1] else None
        return exp_name, shard
    return job_name, None


def extract_param_value(pattern: str) -> Optional[float]:
    """Extract numeric parameter value from pattern name."""
    # Match patterns like temp_0.5, max_tokens_1024, etc.
    match = re.search(r'_(\d+\.?\d*)$', pattern)
    if match:
        return float(match.group(1))
    return None


def main():
    """Main analysis function."""
    print("=" * 80)
    print("ABLATION EXPERIMENT ANALYSIS")
    print("=" * 80)

    # Create client
    client = create_supabase_client()
    benchmarks_map = get_benchmarks_map(client)

    # Results storage
    all_results = []
    category_results = defaultdict(list)

    # Get baseline results first (exp_tas)
    print("\nFetching baseline (exp_tas) results...")
    baseline_jobs = search_baseline_jobs(client)
    baseline_scores = {}

    for job in baseline_jobs:
        benchmark_id = job.get('benchmark_id')
        benchmark_name = benchmarks_map.get(benchmark_id, 'unknown')
        accuracy, std = extract_accuracy_metrics(job.get('metrics'))

        if accuracy is not None:
            if benchmark_name not in baseline_scores:
                baseline_scores[benchmark_name] = []
            baseline_scores[benchmark_name].append(accuracy)

    # Average baseline scores per benchmark
    baseline_avg = {k: sum(v)/len(v) for k, v in baseline_scores.items() if v}
    print(f"Found exp_tas baseline results for {len(baseline_avg)} benchmarks")
    if baseline_jobs:
        print(f"  Sample job name: {baseline_jobs[0].get('job_name', 'N/A')[:80]}")

    # Add baseline (exp_tas) to results
    for benchmark, score in baseline_avg.items():
        all_results.append({
            'category': 'baseline',
            'experiment': 'exp_tas (baseline)',
            'param_value': 'baseline',
            'benchmark': benchmark,
            'accuracy': score,
            'std': None,
            'num_jobs': len(baseline_scores.get(benchmark, []))
        })

    # Process each ablation category
    for category, config in ABLATION_CATEGORIES.items():
        print(f"\nProcessing category: {category}")

        for pattern in config['patterns']:
            print(f"  Searching for: {pattern}")
            jobs = search_ablation_jobs(client, pattern)
            print(f"    Found {len(jobs)} jobs")

            # Group jobs by benchmark
            benchmark_scores = defaultdict(list)

            for job in jobs:
                benchmark_id = job.get('benchmark_id')
                benchmark_name = benchmarks_map.get(benchmark_id, 'unknown')
                accuracy, std = extract_accuracy_metrics(job.get('metrics'))

                if accuracy is not None:
                    benchmark_scores[benchmark_name].append({
                        'accuracy': accuracy,
                        'std': std,
                        'job_id': job.get('id'),
                        'job_name': job.get('job_name')
                    })

            # Get parameter value
            param_value = extract_param_value(pattern)
            if param_value is None:
                param_value = pattern  # For binary flags

            # Average scores per benchmark
            for benchmark, scores in benchmark_scores.items():
                avg_acc = sum(s['accuracy'] for s in scores) / len(scores)

                result = {
                    'category': category,
                    'experiment': pattern,
                    'param_value': param_value,
                    'benchmark': benchmark,
                    'accuracy': avg_acc,
                    'std': None,
                    'num_jobs': len(scores)
                }
                all_results.append(result)
                category_results[category].append(result)

    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(output_dir, 'analysis_results')
    os.makedirs(results_dir, exist_ok=True)

    # Write main CSV with all results
    csv_path = os.path.join(results_dir, 'ablation_results.csv')
    print(f"\nWriting results to: {csv_path}")

    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['category', 'experiment', 'param_value', 'benchmark', 'accuracy', 'std', 'num_jobs']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(all_results, key=lambda x: (x['category'], str(x['param_value']), x['benchmark'])):
            writer.writerow(row)

    print(f"Total results: {len(all_results)}")

    # Write separate CSV per category (including baseline for comparison)
    for category, results in category_results.items():
        cat_csv_path = os.path.join(results_dir, f'{category}_results.csv')
        with open(cat_csv_path, 'w', newline='') as f:
            fieldnames = ['experiment', 'param_value', 'benchmark', 'accuracy', 'baseline_accuracy', 'delta_vs_baseline', 'std', 'num_jobs']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # First add baseline (exp_tas) rows for this category's benchmarks
            cat_benchmarks = set(r['benchmark'] for r in results)
            for benchmark in sorted(cat_benchmarks):
                baseline_acc = baseline_avg.get(benchmark)
                if baseline_acc is not None:
                    writer.writerow({
                        'experiment': 'exp_tas (baseline)',
                        'param_value': 'baseline',
                        'benchmark': benchmark,
                        'accuracy': baseline_acc,
                        'baseline_accuracy': baseline_acc,
                        'delta_vs_baseline': 0.0,
                        'std': None,
                        'num_jobs': len(baseline_scores.get(benchmark, []))
                    })

            # Then add the ablation results with delta
            for row in sorted(results, key=lambda x: (str(x['param_value']), x['benchmark'])):
                baseline_acc = baseline_avg.get(row['benchmark'])
                delta = None
                if baseline_acc is not None and row['accuracy'] is not None:
                    delta = row['accuracy'] - baseline_acc
                out_row = {k: v for k, v in row.items() if k != 'category'}
                out_row['baseline_accuracy'] = baseline_acc
                out_row['delta_vs_baseline'] = delta
                writer.writerow(out_row)
        print(f"  Written: {cat_csv_path}")

    # Create summary pivot table (experiment x benchmark)
    print("\nCreating summary pivot table...")

    # Get unique experiments and benchmarks
    experiments = sorted(set(r['experiment'] for r in all_results))
    benchmarks = sorted(set(r['benchmark'] for r in all_results))

    # Build pivot
    pivot_data = {exp: {} for exp in experiments}
    for r in all_results:
        pivot_data[r['experiment']][r['benchmark']] = r['accuracy']

    # Write pivot CSV
    pivot_csv_path = os.path.join(results_dir, 'ablation_pivot.csv')
    with open(pivot_csv_path, 'w', newline='') as f:
        fieldnames = ['experiment'] + benchmarks
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiments:
            row = {'experiment': exp}
            row.update(pivot_data[exp])
            writer.writerow(row)
    print(f"Written pivot table: {pivot_csv_path}")

    # Create plots for numeric ablation categories
    print("\nGenerating plots...")
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    for category, config in ABLATION_CATEGORIES.items():
        if config['values'] is None:  # Skip binary flags for line plots
            continue

        results = category_results.get(category, [])
        if not results:
            continue

        # Get benchmarks present in this category
        cat_benchmarks = sorted(set(r['benchmark'] for r in results))

        if not cat_benchmarks:
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot each benchmark
        colors = plt.cm.tab10(np.linspace(0, 1, len(cat_benchmarks)))

        # Get baseline x-value for this category
        baseline_x = config.get('baseline_x_value')

        for idx, benchmark in enumerate(cat_benchmarks):
            bench_results = [r for r in results if r['benchmark'] == benchmark]

            # Sort by parameter value
            try:
                bench_results_sorted = sorted(bench_results, key=lambda x: float(x['param_value']))
            except (ValueError, TypeError):
                continue

            x_vals = [float(r['param_value']) for r in bench_results_sorted]
            y_vals = [r['accuracy'] for r in bench_results_sorted]

            # Add baseline point to the data if we have baseline accuracy
            if benchmark in baseline_avg and baseline_x is not None:
                baseline_val = baseline_avg[benchmark]
                # Insert baseline point into the sorted lists
                x_vals_with_baseline = x_vals.copy()
                y_vals_with_baseline = y_vals.copy()

                # Find insertion point to keep sorted order
                insert_idx = 0
                for i, x in enumerate(x_vals_with_baseline):
                    if x > baseline_x:
                        break
                    insert_idx = i + 1

                x_vals_with_baseline.insert(insert_idx, baseline_x)
                y_vals_with_baseline.insert(insert_idx, baseline_val)
            else:
                x_vals_with_baseline = x_vals
                y_vals_with_baseline = y_vals

            if x_vals_with_baseline and y_vals_with_baseline:
                # Plot the line with all points including baseline
                ax.plot(x_vals_with_baseline, y_vals_with_baseline, 'o-', color=colors[idx],
                       label=benchmark, markersize=8, linewidth=2)

                # Add STAR marker at baseline point to highlight it
                if benchmark in baseline_avg and baseline_x is not None:
                    baseline_val = baseline_avg[benchmark]
                    ax.plot(baseline_x, baseline_val, '*', color=colors[idx], markersize=25,
                           markeredgecolor='black', markeredgewidth=2, zorder=10)

        # Add vertical line at baseline x-value
        if baseline_x is not None:
            ax.axvline(x=baseline_x, color='gray', linestyle=':', alpha=0.5, linewidth=2)
            # Get y-axis limits after plotting
            ylim = ax.get_ylim()
            ax.annotate(f'baseline\n({baseline_x})',
                       xy=(baseline_x, ylim[1]),
                       xytext=(0, -5), textcoords='offset points',
                       fontsize=10, ha='center', va='top',
                       fontweight='bold', color='gray')

        ax.set_xlabel(config['label'], fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        baseline_str = f" (baseline={baseline_x})" if baseline_x is not None else ""
        ax.set_title(f'Ablation: {config["label"]} vs Accuracy\n(★ marks baseline at {baseline_x})', fontsize=13)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f'{category}_ablation.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {plot_path}")

    # Create bar plot for binary flags (including baseline/exp_tas)
    binary_results = category_results.get('binary_flags', [])
    if binary_results:
        # Get unique experiments and benchmarks
        binary_experiments = ['exp_tas\n(baseline)'] + sorted(set(r['experiment'] for r in binary_results))
        binary_benchmarks = sorted(set(r['benchmark'] for r in binary_results))

        if binary_experiments and binary_benchmarks:
            fig, ax = plt.subplots(figsize=(16, 7))

            x = np.arange(len(binary_experiments))
            width = 0.8 / len(binary_benchmarks)

            colors = plt.cm.tab10(np.linspace(0, 1, len(binary_benchmarks)))

            for idx, benchmark in enumerate(binary_benchmarks):
                values = []
                for exp in binary_experiments:
                    if 'baseline' in exp:
                        # Use baseline value (exp_tas)
                        values.append(baseline_avg.get(benchmark, 0))
                    else:
                        exp_results = [r for r in binary_results if r['experiment'] == exp and r['benchmark'] == benchmark]
                        if exp_results:
                            values.append(exp_results[0]['accuracy'])
                        else:
                            values.append(0)

                offset = (idx - len(binary_benchmarks)/2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, label=benchmark, color=colors[idx])

                # Highlight baseline bars with edge
                ax.bar(0 + offset, values[0], width, color=colors[idx], edgecolor='black', linewidth=2)

            ax.set_xlabel('Experiment', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title('Binary Flag Ablations vs exp_tas Baseline', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(binary_experiments, rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

            # Add vertical line to separate baseline
            ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)

            plt.tight_layout()
            plot_path = os.path.join(plots_dir, 'binary_flags_ablation.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved plot: {plot_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Calculate average across all benchmarks per experiment
    experiment_avgs = defaultdict(list)
    for r in all_results:
        experiment_avgs[r['experiment']].append(r['accuracy'])

    print("\nAverage accuracy per experiment (across all benchmarks):")
    print("-" * 60)
    for exp in sorted(experiment_avgs.keys()):
        avg = sum(experiment_avgs[exp]) / len(experiment_avgs[exp])
        print(f"  {exp:<40} {avg:.4f}")

    print(f"\nOutput files saved to: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
