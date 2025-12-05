#!/usr/bin/env python3
"""
Aggregate and analyze vLLM experiment results.

Collects results from multiple experiments and generates comparative analysis.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime


class ResultsAggregator:
    """Aggregate and analyze experiment results."""

    METRIC_COLUMNS = [
        "config_hash",
        "dataset_name",
        "num_requests",
        "successful_requests",
        "failed_requests",
        "success_rate",
        "throughput_tokens_per_sec",
        "throughput_requests_per_sec",
        "ttft_mean",
        "ttft_p50",
        "ttft_p95",
        "ttft_p99",
        "e2e_latency_mean",
        "e2e_latency_p50",
        "e2e_latency_p95",
        "e2e_latency_p99",
        "itl_mean",
        "itl_p50",
        "itl_p95",
        "itl_p99",
        "total_tokens_generated",
        "total_time",
    ]

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)

    def load_result(self, result_file: Path) -> Optional[Dict[str, Any]]:
        """Load a single result file."""
        try:
            with open(result_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {result_file}: {e}")
            return None

    def load_all_results(self) -> List[Dict[str, Any]]:
        """Load all result files from results directory."""
        result_files = list(self.results_dir.rglob("results_*.json"))
        print(f"Found {len(result_files)} result files")

        results = []
        for result_file in result_files:
            result = self.load_result(result_file)
            if result:
                results.append(result)

        print(f"Successfully loaded {len(results)} results")
        return results

    def results_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        rows = []

        for result in results:
            row = {col: result.get(col) for col in self.METRIC_COLUMNS}

            # Add Prometheus metrics if available
            if result.get("prometheus_metrics"):
                prom = result["prometheus_metrics"]
                row["prom_preemptions"] = prom.get("num_preemptions_total", 0)
                row["prom_cache_usage"] = prom.get("gpu_cache_usage_perc", 0)
                row["prom_requests_running"] = prom.get("num_requests_running", 0)
                row["prom_requests_waiting"] = prom.get("num_requests_waiting", 0)

            rows.append(row)

        return pd.DataFrame(rows)

    def load_config(self, config_hash: str, config_dir: str = "configs") -> Optional[Dict[str, Any]]:
        """Load configuration by hash."""
        config_path = Path(config_dir)

        # Search for config file with this hash
        config_files = list(config_path.rglob(f"*{config_hash}*.json"))

        if not config_files:
            return None

        try:
            with open(config_files[0], 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config {config_hash}: {e}")
            return None

    def enrich_with_configs(
        self,
        df: pd.DataFrame,
        config_dir: str = "configs"
    ) -> pd.DataFrame:
        """Add configuration parameters to results dataframe."""
        config_params = [
            "gpu_memory_utilization",
            "max_num_batched_tokens",
            "max_num_seqs",
            "max_model_len",
            "kv_cache_dtype",
            "enable_prefix_caching",
            "all2all_backend",
            "enable_eplb",
            "enable_chunked_prefill",
            "max_num_partial_prefills",
            "num_scheduler_steps",
        ]

        for param in config_params:
            df[param] = None

        # Load configs for each result
        for idx, row in df.iterrows():
            config_hash = row["config_hash"]
            config = self.load_config(config_hash, config_dir)

            if config:
                for param in config_params:
                    df.at[idx, param] = config.get(param)

        return df

    def generate_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        summary = df[numeric_cols].describe()
        return summary

    def find_best_configs(
        self,
        df: pd.DataFrame,
        metric: str = "throughput_tokens_per_sec",
        top_n: int = 10
    ) -> pd.DataFrame:
        """Find best configurations for a given metric."""
        return df.nlargest(top_n, metric)

    def find_worst_configs(
        self,
        df: pd.DataFrame,
        metric: str = "throughput_tokens_per_sec",
        bottom_n: int = 10
    ) -> pd.DataFrame:
        """Find worst configurations for a given metric."""
        return df.nsmallest(bottom_n, metric)

    def compare_parameter(
        self,
        df: pd.DataFrame,
        parameter: str,
        metric: str = "throughput_tokens_per_sec"
    ) -> pd.DataFrame:
        """Compare metric across different values of a parameter."""
        if parameter not in df.columns:
            print(f"Warning: Parameter '{parameter}' not found in results")
            return pd.DataFrame()

        grouped = df.groupby(parameter)[metric].agg([
            'count', 'mean', 'std', 'min', 'max',
            ('p50', lambda x: np.percentile(x, 50)),
            ('p95', lambda x: np.percentile(x, 95)),
        ])

        return grouped.sort_values('mean', ascending=False)

    def generate_report(
        self,
        df: pd.DataFrame,
        output_file: str = "experiment_report.txt"
    ):
        """Generate a comprehensive text report."""
        output_path = self.results_dir / output_file

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("vLLM EXPERIMENT RESULTS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total experiments: {len(df)}\n")
            f.write("\n")

            # Summary statistics
            f.write("-" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")

            key_metrics = [
                "throughput_tokens_per_sec",
                "throughput_requests_per_sec",
                "ttft_p95",
                "itl_p95",
                "success_rate",
            ]

            for metric in key_metrics:
                if metric in df.columns:
                    f.write(f"\n{metric}:\n")
                    f.write(f"  Mean: {df[metric].mean():.2f}\n")
                    f.write(f"  Std: {df[metric].std():.2f}\n")
                    f.write(f"  Min: {df[metric].min():.2f}\n")
                    f.write(f"  Max: {df[metric].max():.2f}\n")
                    f.write(f"  P50: {df[metric].quantile(0.5):.2f}\n")
                    f.write(f"  P95: {df[metric].quantile(0.95):.2f}\n")

            # Top configurations
            f.write("\n" + "-" * 80 + "\n")
            f.write("TOP 10 CONFIGURATIONS (by throughput)\n")
            f.write("-" * 80 + "\n")

            top_10 = self.find_best_configs(df, "throughput_tokens_per_sec", 10)

            for i, (idx, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"\n{i}. Config: {row['config_hash']}\n")
                f.write(f"   Throughput: {row['throughput_tokens_per_sec']:.2f} tok/s\n")
                f.write(f"   TTFT P95: {row['ttft_p95'] * 1000:.2f}ms\n")
                f.write(f"   ITL P95: {row['itl_p95'] * 1000:.2f}ms\n")
                f.write(f"   Success Rate: {row['success_rate'] * 100:.1f}%\n")

                # Add config params if available
                if "max_num_batched_tokens" in row:
                    f.write(f"   max_num_batched_tokens: {row['max_num_batched_tokens']}\n")
                if "max_num_seqs" in row:
                    f.write(f"   max_num_seqs: {row['max_num_seqs']}\n")
                if "kv_cache_dtype" in row:
                    f.write(f"   kv_cache_dtype: {row['kv_cache_dtype']}\n")

            # Parameter comparisons
            f.write("\n" + "-" * 80 + "\n")
            f.write("PARAMETER COMPARISONS\n")
            f.write("-" * 80 + "\n")

            important_params = [
                "max_num_batched_tokens",
                "max_num_seqs",
                "kv_cache_dtype",
                "all2all_backend",
                "enable_eplb",
                "enable_chunked_prefill",
            ]

            for param in important_params:
                if param in df.columns and df[param].notna().any():
                    f.write(f"\n{param}:\n")
                    comparison = self.compare_parameter(df, param, "throughput_tokens_per_sec")
                    f.write(comparison.to_string())
                    f.write("\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Report saved to: {output_path}")
        return output_path

    def export_to_csv(self, df: pd.DataFrame, filename: str = "experiment_results.csv"):
        """Export results to CSV."""
        output_path = self.results_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Results exported to: {output_path}")
        return output_path

    def create_visualization_data(self, df: pd.DataFrame, output_file: str = "viz_data.json"):
        """Create data file for visualization tools."""
        output_path = self.results_dir / output_file

        viz_data = {
            "summary": {
                "total_experiments": len(df),
                "metrics": {
                    "throughput": {
                        "mean": float(df["throughput_tokens_per_sec"].mean()),
                        "max": float(df["throughput_tokens_per_sec"].max()),
                        "min": float(df["throughput_tokens_per_sec"].min()),
                    },
                    "ttft_p95": {
                        "mean": float(df["ttft_p95"].mean() * 1000),  # Convert to ms
                        "min": float(df["ttft_p95"].min() * 1000),
                        "max": float(df["ttft_p95"].max() * 1000),
                    },
                },
            },
            "experiments": df.to_dict(orient="records"),
        }

        with open(output_path, 'w') as f:
            json.dump(viz_data, f, indent=2)

        print(f"Visualization data saved to: {output_path}")
        return output_path


def main():
    """Run results aggregation from command line."""
    parser = argparse.ArgumentParser(description="Aggregate vLLM experiment results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--config-dir", default="configs", help="Configs directory")
    parser.add_argument("--output-csv", default="experiment_results.csv", help="Output CSV file")
    parser.add_argument("--output-report", default="experiment_report.txt", help="Output report file")
    parser.add_argument("--enrich-configs", action="store_true", help="Enrich with config parameters")
    parser.add_argument("--compare-param", help="Compare specific parameter")
    parser.add_argument("--metric", default="throughput_tokens_per_sec", help="Metric for comparisons")

    args = parser.parse_args()

    aggregator = ResultsAggregator(args.results_dir)

    # Load all results
    print("Loading results...")
    results = aggregator.load_all_results()

    if not results:
        print("No results found!")
        return

    # Convert to dataframe
    df = aggregator.results_to_dataframe(results)

    # Enrich with configs if requested
    if args.enrich_configs:
        print("Enriching with configuration parameters...")
        df = aggregator.enrich_with_configs(df, args.config_dir)

    # Export to CSV
    aggregator.export_to_csv(df, args.output_csv)

    # Generate report
    aggregator.generate_report(df, args.output_report)

    # Create visualization data
    aggregator.create_visualization_data(df)

    # Compare specific parameter if requested
    if args.compare_param:
        print(f"\nComparing parameter: {args.compare_param}")
        comparison = aggregator.compare_parameter(df, args.compare_param, args.metric)
        print(comparison)

    # Show summary
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)

    print(f"\nTotal experiments: {len(df)}")
    print(f"Successful experiments: {(df['success_rate'] > 0.95).sum()}")

    print(f"\nThroughput (tokens/sec):")
    print(f"  Best: {df['throughput_tokens_per_sec'].max():.2f}")
    print(f"  Worst: {df['throughput_tokens_per_sec'].min():.2f}")
    print(f"  Mean: {df['throughput_tokens_per_sec'].mean():.2f}")

    print(f"\nTTFT P95 (ms):")
    print(f"  Best: {df['ttft_p95'].min() * 1000:.2f}")
    print(f"  Worst: {df['ttft_p95'].max() * 1000:.2f}")
    print(f"  Mean: {df['ttft_p95'].mean() * 1000:.2f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
