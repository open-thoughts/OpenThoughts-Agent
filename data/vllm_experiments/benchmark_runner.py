#!/usr/bin/env python3
"""
Benchmark runner for vLLM inference experiments.

Sends requests to vLLM server and collects performance metrics:
- Throughput (tokens/sec, requests/sec)
- Latency (TTFT, ITL, end-to-end)
- System metrics (preemptions, cache usage)
"""

import asyncio
import aiohttp
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    prompt: str
    prompt_length: int
    output_length: int
    ttft: float  # Time to first token (seconds)
    end_to_end_latency: float  # Total time (seconds)
    success: bool
    error_msg: Optional[str] = None
    tokens_per_second: float = 0.0
    inter_token_latency: float = 0.0  # Average ITL


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    # Configuration
    config_hash: str
    num_requests: int
    request_rate: float
    dataset_name: str
    timestamp: str

    # Request-level metrics
    successful_requests: int
    failed_requests: int
    success_rate: float

    # Throughput metrics
    total_tokens_generated: int
    total_time: float
    throughput_tokens_per_sec: float
    throughput_requests_per_sec: float

    # Latency metrics (successful requests only)
    ttft_mean: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float

    e2e_latency_mean: float
    e2e_latency_p50: float
    e2e_latency_p95: float
    e2e_latency_p99: float

    itl_mean: float
    itl_p50: float
    itl_p95: float
    itl_p99: float

    # System metrics (if available)
    prometheus_metrics: Optional[Dict[str, Any]] = None

    # Raw request data
    request_metrics: Optional[List[Dict[str, Any]]] = None


class BenchmarkRunner:
    """Run benchmarks against a vLLM server."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        prometheus_url: str = "http://localhost:8001",
        output_dir: str = "results",
    ):
        self.server_url = server_url.rstrip("/")
        self.prometheus_url = prometheus_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def send_request(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> RequestMetrics:
        """Send a single inference request and measure metrics."""
        start_time = time.time()
        ttft = None
        tokens_generated = 0
        prompt_length = len(prompt.split())  # Rough estimate

        try:
            async with session.post(
                f"{self.server_url}/v1/completions",
                json={
                    "model": "glm",  # Matches model_id in dp_debug.py
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                },
                timeout=aiohttp.ClientTimeout(total=None),  # No timeout
            ) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    return RequestMetrics(
                        prompt=prompt,
                        prompt_length=prompt_length,
                        output_length=0,
                        ttft=0.0,
                        end_to_end_latency=time.time() - start_time,
                        success=False,
                        error_msg=f"HTTP {response.status}: {error_msg}",
                    )

                # Process streaming response
                async for line in response.content:
                    if not line:
                        continue

                    line = line.decode('utf-8').strip()
                    if not line.startswith('data: '):
                        continue

                    if line == 'data: [DONE]':
                        break

                    try:
                        data = json.loads(line[6:])  # Skip 'data: '

                        # Record TTFT on first token
                        if ttft is None and 'choices' in data and len(data['choices']) > 0:
                            ttft = time.time() - start_time

                        # Count tokens
                        if 'choices' in data and len(data['choices']) > 0:
                            choice = data['choices'][0]
                            if 'text' in choice and choice['text']:
                                tokens_generated += 1

                    except json.JSONDecodeError:
                        continue

            end_time = time.time()
            total_time = end_time - start_time

            # Calculate ITL (if more than 1 token)
            itl = 0.0
            if tokens_generated > 1 and ttft is not None:
                decode_time = total_time - ttft
                itl = decode_time / (tokens_generated - 1)

            return RequestMetrics(
                prompt=prompt,
                prompt_length=prompt_length,
                output_length=tokens_generated,
                ttft=ttft or 0.0,
                end_to_end_latency=total_time,
                success=True,
                tokens_per_second=tokens_generated / total_time if total_time > 0 else 0.0,
                inter_token_latency=itl,
            )

        except asyncio.TimeoutError:
            return RequestMetrics(
                prompt=prompt,
                prompt_length=prompt_length,
                output_length=0,
                ttft=0.0,
                end_to_end_latency=time.time() - start_time,
                success=False,
                error_msg="Request timeout",
            )
        except Exception as e:
            return RequestMetrics(
                prompt=prompt,
                prompt_length=prompt_length,
                output_length=0,
                ttft=0.0,
                end_to_end_latency=time.time() - start_time,
                success=False,
                error_msg=str(e),
            )

    async def run_benchmark(
        self,
        prompts: List[str],
        max_tokens: List[int],
        request_rate: Optional[float] = None,
        temperature: float = 0.7,
    ) -> List[RequestMetrics]:
        """
        Run benchmark with specified prompts and request rate.

        Args:
            prompts: List of prompts to send
            max_tokens: List of max_tokens for each prompt
            request_rate: Requests per second (None = unlimited)
            temperature: Sampling temperature

        Returns:
            List of RequestMetrics
        """
        results = []

        async with aiohttp.ClientSession() as session:
            if request_rate is None or request_rate <= 0:
                # Send all requests as fast as possible
                tasks = [
                    self.send_request(session, prompt, max_tok, temperature)
                    for prompt, max_tok in zip(prompts, max_tokens)
                ]
                results = await asyncio.gather(*tasks)
            else:
                # Send requests at specified rate (concurrent execution)
                delay = 1.0 / request_rate
                tasks = []
                total_requests = len(prompts)
                start_time = time.time()
                completed = [0]  # Use list to allow modification in callback

                def on_complete(future):
                    completed[0] += 1
                    elapsed = time.time() - start_time
                    if completed[0] > 0:
                        avg_time = elapsed / completed[0]
                        remaining = total_requests - completed[0]
                        eta = avg_time * remaining
                        print(f"Progress: {completed[0]}/{total_requests} requests completed "
                              f"({completed[0]*100/total_requests:.1f}%) - "
                              f"ETA: {eta:.1f}s ({eta/60:.1f}min)")

                for i, (prompt, max_tok) in enumerate(zip(prompts, max_tokens)):
                    task = asyncio.create_task(
                        self.send_request(session, prompt, max_tok, temperature)
                    )
                    task.add_done_callback(on_complete)
                    tasks.append(task)

                    # Progress on launch
                    if (i + 1) % 10 == 0 or i == 0:
                        print(f"Launched {i + 1}/{total_requests} requests...")

                    # Wait before launching next request
                    if i < len(prompts) - 1:
                        await asyncio.sleep(delay)

                print(f"All {total_requests} requests launched. Waiting for completion...")
                # Wait for all requests to complete
                results = await asyncio.gather(*tasks)

        return results

    async def fetch_prometheus_metrics(self) -> Optional[Dict[str, Any]]:
        """Fetch metrics from Prometheus endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.prometheus_url}/metrics",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status != 200:
                        return None

                    text = await response.text()

                    # Parse key metrics
                    metrics = {}
                    for line in text.split('\n'):
                        if line.startswith('#') or not line.strip():
                            continue

                        # Simple parsing (not comprehensive)
                        if 'vllm:num_requests_running' in line:
                            metrics['num_requests_running'] = float(line.split()[-1])
                        elif 'vllm:num_requests_waiting' in line:
                            metrics['num_requests_waiting'] = float(line.split()[-1])
                        elif 'vllm:num_preemptions_total' in line:
                            metrics['num_preemptions_total'] = float(line.split()[-1])
                        elif 'vllm:gpu_cache_usage_perc' in line:
                            metrics['gpu_cache_usage_perc'] = float(line.split()[-1])
                        elif 'vllm:num_generation_tokens_from_beam_sampler_total' in line:
                            metrics['generation_tokens_total'] = float(line.split()[-1])

                    return metrics

        except Exception as e:
            print(f"Warning: Could not fetch Prometheus metrics: {e}")
            return None

    def aggregate_results(
        self,
        request_metrics: List[RequestMetrics],
        config_hash: str,
        dataset_name: str,
        request_rate: float,
        total_time: float,
        prometheus_metrics: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResults:
        """Aggregate request metrics into benchmark results."""
        successful = [m for m in request_metrics if m.success]
        failed = [m for m in request_metrics if not m.success]

        if not successful:
            # All requests failed
            return BenchmarkResults(
                config_hash=config_hash,
                num_requests=len(request_metrics),
                request_rate=request_rate,
                dataset_name=dataset_name,
                timestamp=datetime.now().isoformat(),
                successful_requests=0,
                failed_requests=len(failed),
                success_rate=0.0,
                total_tokens_generated=0,
                total_time=total_time,
                throughput_tokens_per_sec=0.0,
                throughput_requests_per_sec=0.0,
                ttft_mean=0.0,
                ttft_p50=0.0,
                ttft_p95=0.0,
                ttft_p99=0.0,
                e2e_latency_mean=0.0,
                e2e_latency_p50=0.0,
                e2e_latency_p95=0.0,
                e2e_latency_p99=0.0,
                itl_mean=0.0,
                itl_p50=0.0,
                itl_p95=0.0,
                itl_p99=0.0,
                prometheus_metrics=prometheus_metrics,
            )

        # Extract metrics from successful requests
        ttfts = [m.ttft for m in successful if m.ttft > 0]
        e2e_latencies = [m.end_to_end_latency for m in successful]
        itls = [m.inter_token_latency for m in successful if m.inter_token_latency > 0]
        total_tokens = sum(m.output_length for m in successful)

        return BenchmarkResults(
            config_hash=config_hash,
            num_requests=len(request_metrics),
            request_rate=request_rate,
            dataset_name=dataset_name,
            timestamp=datetime.now().isoformat(),
            successful_requests=len(successful),
            failed_requests=len(failed),
            success_rate=len(successful) / len(request_metrics),
            total_tokens_generated=total_tokens,
            total_time=total_time,
            throughput_tokens_per_sec=total_tokens / total_time if total_time > 0 else 0.0,
            throughput_requests_per_sec=len(successful) / total_time if total_time > 0 else 0.0,
            ttft_mean=float(np.mean(ttfts)) if ttfts else 0.0,
            ttft_p50=float(np.percentile(ttfts, 50)) if ttfts else 0.0,
            ttft_p95=float(np.percentile(ttfts, 95)) if ttfts else 0.0,
            ttft_p99=float(np.percentile(ttfts, 99)) if ttfts else 0.0,
            e2e_latency_mean=float(np.mean(e2e_latencies)),
            e2e_latency_p50=float(np.percentile(e2e_latencies, 50)),
            e2e_latency_p95=float(np.percentile(e2e_latencies, 95)),
            e2e_latency_p99=float(np.percentile(e2e_latencies, 99)),
            itl_mean=float(np.mean(itls)) if itls else 0.0,
            itl_p50=float(np.percentile(itls, 50)) if itls else 0.0,
            itl_p95=float(np.percentile(itls, 95)) if itls else 0.0,
            itl_p99=float(np.percentile(itls, 99)) if itls else 0.0,
            prometheus_metrics=prometheus_metrics,
            request_metrics=[asdict(m) for m in request_metrics],
        )

    def save_results(self, results: BenchmarkResults, filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{results.config_hash}_{timestamp}.json"

        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(asdict(results), f, indent=2)

        return output_path

    def print_results(self, results: BenchmarkResults):
        """Print formatted benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print(f"\nConfiguration: {results.config_hash}")
        print(f"Dataset: {results.dataset_name}")
        print(f"Timestamp: {results.timestamp}")
        print(f"\nRequests:")
        print(f"  Total: {results.num_requests}")
        print(f"  Successful: {results.successful_requests}")
        print(f"  Failed: {results.failed_requests}")
        print(f"  Success Rate: {results.success_rate * 100:.2f}%")

        print(f"\nThroughput:")
        print(f"  Tokens/sec: {results.throughput_tokens_per_sec:.2f}")
        print(f"  Requests/sec: {results.throughput_requests_per_sec:.2f}")
        print(f"  Total tokens: {results.total_tokens_generated}")
        print(f"  Total time: {results.total_time:.2f}s")

        print(f"\nTime to First Token (TTFT):")
        print(f"  Mean: {results.ttft_mean * 1000:.2f}ms")
        print(f"  P50: {results.ttft_p50 * 1000:.2f}ms")
        print(f"  P95: {results.ttft_p95 * 1000:.2f}ms")
        print(f"  P99: {results.ttft_p99 * 1000:.2f}ms")

        print(f"\nEnd-to-End Latency:")
        print(f"  Mean: {results.e2e_latency_mean * 1000:.2f}ms")
        print(f"  P50: {results.e2e_latency_p50 * 1000:.2f}ms")
        print(f"  P95: {results.e2e_latency_p95 * 1000:.2f}ms")
        print(f"  P99: {results.e2e_latency_p99 * 1000:.2f}ms")

        print(f"\nInter-Token Latency (ITL):")
        print(f"  Mean: {results.itl_mean * 1000:.2f}ms")
        print(f"  P50: {results.itl_p50 * 1000:.2f}ms")
        print(f"  P95: {results.itl_p95 * 1000:.2f}ms")
        print(f"  P99: {results.itl_p99 * 1000:.2f}ms")

        if results.prometheus_metrics:
            print(f"\nPrometheus Metrics:")
            for key, value in results.prometheus_metrics.items():
                print(f"  {key}: {value}")

        print("=" * 80 + "\n")


async def main():
    """Run benchmark from command line."""
    parser = argparse.ArgumentParser(description="Run vLLM benchmark")
    parser.add_argument("--server-url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--prometheus-url", default="http://localhost:8001", help="Prometheus metrics URL")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSONL file")
    parser.add_argument("--config-hash", default="unknown", help="Configuration hash")
    parser.add_argument("--request-rate", type=float, default=None, help="Requests per second (unlimited if not set)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit number of prompts")

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    prompts = []
    max_tokens_list = []

    with open(args.dataset, 'r') as f:
        for i, line in enumerate(f):
            if args.max_prompts and i >= args.max_prompts:
                break

            record = json.loads(line)
            prompts.append(record['prompt'])
            max_tokens_list.append(record.get('expected_output_length', 512))

    print(f"Loaded {len(prompts)} prompts")

    # Run benchmark
    runner = BenchmarkRunner(
        server_url=args.server_url,
        prometheus_url=args.prometheus_url,
        output_dir=args.output_dir,
    )

    print(f"\nRunning benchmark...")
    if args.request_rate:
        print(f"  Request rate: {args.request_rate} req/s")
    else:
        print(f"  Request rate: unlimited")

    start_time = time.time()
    request_metrics = await runner.run_benchmark(
        prompts=prompts,
        max_tokens=max_tokens_list,
        request_rate=args.request_rate,
        temperature=args.temperature,
    )
    total_time = time.time() - start_time

    # Fetch Prometheus metrics
    prometheus_metrics = await runner.fetch_prometheus_metrics()

    # Aggregate results
    results = runner.aggregate_results(
        request_metrics=request_metrics,
        config_hash=args.config_hash,
        dataset_name=Path(args.dataset).stem,
        request_rate=args.request_rate or 0.0,
        total_time=total_time,
        prometheus_metrics=prometheus_metrics,
    )

    # Print and save results
    runner.print_results(results)
    output_path = runner.save_results(results)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
