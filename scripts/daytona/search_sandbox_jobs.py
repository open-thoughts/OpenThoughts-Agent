#!/usr/bin/env python3
"""
Search sandbox_jobs table for job names with include/exclude filters
and extract accuracy metrics to CSV.
"""

import os
import csv
import json
import argparse
from typing import List, Optional, Dict, Tuple
from supabase import create_client, Client
from dotenv import load_dotenv

SECRET_ENV_PATH = os.environ.get("DC_AGENT_SECRET_ENV")
if SECRET_ENV_PATH and os.path.isfile(SECRET_ENV_PATH):
    load_dotenv(SECRET_ENV_PATH)
else:
    if SECRET_ENV_PATH:
        print(f"Warning: DC_AGENT_SECRET_ENV set to '{SECRET_ENV_PATH}' but file not found.", flush=True)
    print("Warning: DC_AGENT_SECRET_ENV not set. Ensure Supabase credentials are available in the environment.", flush=True)

def create_supabase_client() -> Client:
    """Create and return Supabase client."""
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment")
    
    return create_client(url, key)

def get_benchmarks_map(client: Client) -> Dict[str, str]:
    """
    Get mapping of benchmark_id to benchmark name.
    
    Args:
        client: Supabase client
    
    Returns:
        Dictionary mapping benchmark_id to benchmark name
    """
    response = client.table('benchmarks').select('id, name').execute()
    return {b['id']: b['name'] for b in response.data}

def get_models_map(client: Client) -> Dict[str, str]:
    """
    Get mapping of model_id to model name.
    
    Args:
        client: Supabase client
    
    Returns:
        Dictionary mapping model_id to model name
    """
    response = client.table('models').select('id, name').execute()
    return {m['id']: m['name'] for m in response.data}

def search_jobs(
    client: Client,
    include_substrings: List[str],
    exclude_substrings: Optional[List[str]] = None
) -> List[dict]:
    """
    Search sandbox_jobs table for jobs matching criteria.
    
    Args:
        client: Supabase client
        include_substrings: List of substrings that must be in job_name
        exclude_substrings: List of substrings that must NOT be in job_name
    
    Returns:
        List of matching job records
    """
    exclude_substrings = exclude_substrings or []
    
    # Start with base query
    query = client.table('sandbox_jobs').select('id, job_name, metrics')
    
    # Apply include filters (all must match)
    for substring in include_substrings:
        query = query.ilike('job_name', f'%{substring}%')
    
    # Execute query to get initial results
    response = query.execute()
    jobs = response.data
    
    # Filter out excluded substrings (client-side filtering for exclude logic)
    if exclude_substrings:
        filtered_jobs = []
        for job in jobs:
            job_name = job.get('job_name', '').lower()
            if not any(exclude_sub.lower() in job_name for exclude_sub in exclude_substrings):
                filtered_jobs.append(job)
        jobs = filtered_jobs
    
    return jobs

def search_all_jobs_for_model(
    client: Client,
    model_id: str
) -> List[dict]:
    """
    Search sandbox_jobs table for ALL jobs for a specific model using model_id UUID.
    
    Args:
        client: Supabase client
        model_id: Model UUID to search for
    
    Returns:
        List of all job records for the model with evaluation names extracted
    """
    # Query by model_id UUID field
    response = client.table('sandbox_jobs').select('id, job_name, metrics, model_id').eq('model_id', model_id).execute()
    jobs = response.data
    
    print(f"Found {len(jobs)} jobs for model_id: {model_id}")
    
    # Extract evaluation name from job_name for each job
    for job in jobs:
        job_name = job.get('job_name', '')
        
        # Extract eval name from job_name pattern
        # Common patterns:
        # - modelname__evalname__uuid
        # - traces-terminus-2__uuid
        # - gsm8k__uuid
        # - humaneval__uuid
        # - gpqa_diamond__uuid
        # - math500__uuid
        
        # Look for common evaluation patterns
        eval_name = 'unknown'
        job_lower = job_name.lower()
        
        # Check for specific evaluation types
        if '__' in job_name:
            parts = job_name.split('__')
            # The eval name is typically the second-to-last part
            if len(parts) >= 2:
                eval_part = parts[-2]
                # Clean common patterns
                if 'traces-terminus' in eval_part:
                    eval_name = 'traces-terminus-2'
                elif 'gsm8k' in eval_part:
                    eval_name = 'gsm8k'
                elif 'humaneval' in eval_part:
                    eval_name = 'humaneval'
                elif 'gpqa' in eval_part:
                    eval_name = 'gpqa_diamond'
                elif 'math500' in eval_part or 'math' in eval_part:
                    eval_name = 'math500'
                elif 'dev_set' in eval_part:
                    eval_name = 'dev_set'
                elif 'llm_verifier' in eval_part:
                    eval_name = 'llm_verifier'
                else:
                    eval_name = eval_part
        else:
            # Check in the full job name
            if 'traces-terminus' in job_lower:
                eval_name = 'traces-terminus-2'
            elif 'gsm8k' in job_lower:
                eval_name = 'gsm8k'
            elif 'humaneval' in job_lower:
                eval_name = 'humaneval'
            elif 'gpqa' in job_lower:
                eval_name = 'gpqa_diamond'
            elif 'math500' in job_lower or 'math' in job_lower:
                eval_name = 'math500'
            elif 'dev_set' in job_lower or 'dev-set' in job_lower:
                eval_name = 'dev_set'
            elif 'llm_verifier' in job_lower:
                eval_name = 'llm_verifier'
            elif 'terminal' in job_lower and ('bench' in job_lower or 'benchmark' in job_lower):
                eval_name = 'terminal_bench_2.0'
        
        job['eval_name'] = eval_name
    
    return jobs

def extract_accuracy_metrics(metrics_data) -> tuple:
    """
    Extract accuracy and std from metrics data.
    
    Args:
        metrics_data: Metrics data (could be JSON string, list, or dict)
    
    Returns:
        Tuple of (accuracy, std) or (None, None) if not found
    """
    try:
        if not metrics_data:
            return None, None
        
        # Handle different data types
        if isinstance(metrics_data, str):
            metrics = json.loads(metrics_data)
        elif isinstance(metrics_data, (list, dict)):
            metrics = metrics_data
        else:
            return None, None
        
        accuracy = None
        std = None
        
        # If it's a list of name/value pairs (like the taskmaster data)
        if isinstance(metrics, list):
            for item in metrics:
                if isinstance(item, dict) and 'name' in item and 'value' in item:
                    name = item['name'].lower()
                    value = item['value']
                    
                    if 'accuracy' in name and 'stderr' not in name:
                        accuracy = value
                    elif 'stderr' in name or 'std' in name:
                        std = value
        
        # If it's a dictionary
        elif isinstance(metrics, dict):
            # Common patterns for accuracy
            if 'accuracy' in metrics:
                accuracy = metrics['accuracy']
            elif 'acc' in metrics:
                accuracy = metrics['acc']
            elif 'score' in metrics:
                accuracy = metrics['score']
            
            # Common patterns for std
            if 'std' in metrics:
                std = metrics['std']
            elif 'standard_deviation' in metrics:
                std = metrics['standard_deviation']
            elif 'accuracy_std' in metrics:
                std = metrics['accuracy_std']
            elif 'accuracy_stderr' in metrics:
                std = metrics['accuracy_stderr']
        
        return accuracy, std
        
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        return None, None

def display_model_results_by_eval(jobs: List[dict]):
    """
    Display results grouped by evaluation type with accuracy and std.
    
    Args:
        jobs: List of job records with eval_name field
    """
    # Group by evaluation name
    eval_groups = {}
    for job in jobs:
        eval_name = job.get('eval_name', 'unknown')
        if eval_name not in eval_groups:
            eval_groups[eval_name] = []
        eval_groups[eval_name].append(job)
    
    # Display results by evaluation
    print("\n" + "=" * 80)
    print("RESULTS BY EVALUATION TYPE")
    print("=" * 80)
    
    for eval_name in sorted(eval_groups.keys()):
        eval_jobs = eval_groups[eval_name]
        print(f"\n{eval_name}:")
        print("-" * 40)
        
        for job in eval_jobs:
            job_id = job.get('id')
            job_name = job.get('job_name')
            metrics = job.get('metrics')
            
            accuracy, std = extract_accuracy_metrics(metrics)
            
            if accuracy is not None:
                print(f"  Accuracy: {accuracy:.6f}" + (f" ± {std:.6f}" if std is not None else ""))
            else:
                print(f"  Accuracy: N/A")
            print(f"  Job ID: {job_id}")
            print(f"  Full name: {job_name[:80]}..." if len(job_name) > 80 else f"  Full name: {job_name}")
            print()

def extract_model_name_from_job(job_name: str) -> str:
    """
    Extract model name from job_name - it's typically the last meaningful part.
    
    Args:
        job_name: Full job name string
    
    Returns:
        Extracted model name
    """
    if not job_name:
        return 'unknown'
    
    model_name = 'unknown'
    
    # First check if there's a __ pattern (usually before UUID)
    if '__' in job_name:
        parts = job_name.split('__')
        # The model name is usually in the part before the UUID
        if len(parts) >= 2:
            # Take the second to last part which contains the model name
            model_part = parts[-2]
            # This might still have prefixes, so extract the actual model name
            if '_' in model_part:
                # Get the last segment after underscores
                model_name = model_part.split('_')[-1]
            else:
                model_name = model_part
    else:
        # No __ pattern, look for the model name after the eval type
        if 'traces-terminus-2_' in job_name:
            # Everything after traces-terminus-2_ is the model name
            model_name = job_name.split('traces-terminus-2_')[-1]
            # Remove any timestamp suffixes (like _20251114_044434)
            if '_202' in model_name:  # Assuming timestamps start with 202x
                model_name = model_name.split('_202')[0]
        elif 'gsm8k_' in job_name:
            model_name = job_name.split('gsm8k_')[-1]
            if '_202' in model_name:
                model_name = model_name.split('_202')[0]
        elif '_' in job_name:
            # Just take the last part
            parts = job_name.split('_')
            # Find the part that looks like a model name (not a date/time)
            for i in range(len(parts)-1, -1, -1):
                part = parts[i]
                # Skip parts that look like dates or times
                if not part.isdigit() and not part.startswith('202'):
                    model_name = part
                    break
    
    return model_name

def generate_filtered_model_benchmark_matrix(client: Client, include_substrings: List[str], 
                                           exclude_substrings: List[str], output_file: str, or_mode: bool = False):
    """
    Generate a CSV matrix with models as rows and benchmarks as columns, filtered by job name patterns.
    
    Args:
        client: Supabase client
        include_substrings: List of substrings that must be in job_name
        exclude_substrings: List of substrings that must NOT be in job_name
        output_file: Output CSV file path
    
    Returns:
        Path to generated CSV file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get mappings
    benchmarks_map = get_benchmarks_map(client)
    models_map = get_models_map(client)
    
    # Get filtered jobs with all necessary fields including timestamps
    print("Fetching filtered jobs...")
    response = client.table('sandbox_jobs').select('id, job_name, model_id, benchmark_id, metrics, created_at, ended_at').execute()
    all_jobs = response.data
    
    # Filter by include/exclude patterns
    filtered_jobs = []
    for job in all_jobs:
        job_name = job.get('job_name', '')
        
        # Apply include logic (AND or OR)
        if or_mode:
            # OR mode: any substring must be present
            include_match = any(sub.lower() in job_name.lower() for sub in include_substrings)
        else:
            # AND mode: all substrings must be present (default)
            include_match = all(sub.lower() in job_name.lower() for sub in include_substrings)
        
        if include_match:
            # Check if none of the exclude substrings are present
            if not any(ex.lower() in job_name.lower() for ex in exclude_substrings):
                filtered_jobs.append(job)
    
    print(f"Found {len(filtered_jobs)} matching jobs")
    
    # Build matrix: Keep only earliest job for each (model, benchmark) combination
    # Track earliest job by (model_id, benchmark_id)
    earliest_jobs = {}
    
    for job in filtered_jobs:
        model_id = job.get('model_id')
        benchmark_id = job.get('benchmark_id')
        
        if not model_id or not benchmark_id:
            continue
        
        # Use ended_at if available, otherwise created_at for timestamp
        job_timestamp = job.get('ended_at') or job.get('created_at')
        if not job_timestamp:
            continue
            
        key = (model_id, benchmark_id)
        
        # Keep only the earliest job for this combination
        if key not in earliest_jobs or job_timestamp < earliest_jobs[key]['timestamp']:
            earliest_jobs[key] = {
                'job': job,
                'timestamp': job_timestamp
            }
    
    # Now build the matrix from earliest jobs only
    model_benchmark_matrix = {}
    
    for (model_id, benchmark_id), job_data in earliest_jobs.items():
        job = job_data['job']
        
        benchmark_name = benchmarks_map.get(benchmark_id, 'unknown')
        model_name = models_map.get(model_id, 'unknown_model')
        
        # Extract accuracy
        accuracy, std = extract_accuracy_metrics(job.get('metrics'))
        
        if accuracy is not None:
            if model_id not in model_benchmark_matrix:
                model_benchmark_matrix[model_id] = {
                    'model_name': model_name
                }
            
            # Store single value (from earliest job) not a list
            model_benchmark_matrix[model_id][benchmark_name] = {
                'accuracy': accuracy,
                'std': std,
                'job_id': job.get('id')
            }
    
    # Get unique benchmark names from filtered jobs
    benchmark_names = sorted(set(benchmarks_map.get(job.get('benchmark_id'), 'unknown') 
                                for job in filtered_jobs if job.get('benchmark_id')))
    
    # Create cleaner column names
    column_name_map = {
        'clean-sandboxes-tasks-eval-set': 'clean_sandboxes_eval',
        'clean-sandboxes-tasks-recleaned': 'clean_sandboxes_recleaned', 
        'dev_set_71_tasks': 'dev_set_71',
        'terminal_bench_2': 'terminal_bench_2'
    }
    
    # Build matrix rows with earliest job accuracy for each model-benchmark pair
    matrix_data = []
    
    for model_id, data in model_benchmark_matrix.items():
        row = {
            'model_id': model_id,
            'model_name': data.get('model_name', 'unknown')
        }
        
        # Add accuracy for each benchmark with cleaner column names
        for benchmark in benchmark_names:
            clean_name = column_name_map.get(benchmark, benchmark.replace('-', '_'))
            if benchmark in data and benchmark != 'model_name':
                benchmark_data = data[benchmark]
                # Now it's a dict with accuracy, std, job_id
                row[clean_name] = benchmark_data['accuracy']
                # Optionally could add std column: row[f"{clean_name}_std"] = benchmark_data['std']
            else:
                row[clean_name] = None
        
        matrix_data.append(row)
    
    # Sort by model name
    matrix_data.sort(key=lambda x: x['model_name'])
    
    # Write CSV with cleaner column names
    clean_benchmark_names = [column_name_map.get(b, b.replace('-', '_')) for b in benchmark_names]
    fieldnames = ['model_id', 'model_name'] + clean_benchmark_names
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in matrix_data:
            writer.writerow(row)
    
    print(f"\nFiltered model-benchmark matrix saved to: {output_file}")
    print(f"Total models: {len(matrix_data)}")
    print(f"Benchmarks: {', '.join(benchmark_names)}")
    
    return output_file

def generate_model_benchmark_matrix(client: Client, output_file: str = "results/model_benchmark_matrix.csv"):
    """
    Generate a CSV with models as rows and benchmarks as columns showing accuracy.
    
    Args:
        client: Supabase client
        output_file: Output CSV file path
    
    Returns:
        Path to generated CSV file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get all benchmarks and models
    benchmarks_map = get_benchmarks_map(client)
    models_map = get_models_map(client)
    
    # Get all jobs with model_id, benchmark_id, and metrics
    print("Fetching all jobs with benchmark and model data...")
    response = client.table('sandbox_jobs').select('id, job_name, model_id, benchmark_id, metrics').execute()
    jobs = response.data
    
    # Build matrix: model_id -> benchmark_name -> accuracies list
    model_benchmark_matrix = {}
    
    for job in jobs:
        model_id = job.get('model_id')
        benchmark_id = job.get('benchmark_id')
        
        if not model_id or not benchmark_id:
            continue
            
        benchmark_name = benchmarks_map.get(benchmark_id, 'unknown')
        
        # Extract accuracy
        accuracy, _ = extract_accuracy_metrics(job.get('metrics'))
        
        if accuracy is not None:
            if model_id not in model_benchmark_matrix:
                model_benchmark_matrix[model_id] = {
                    'model_name': models_map.get(model_id, 'unknown_model')
                }
            
            if benchmark_name not in model_benchmark_matrix[model_id]:
                model_benchmark_matrix[model_id][benchmark_name] = []
            
            model_benchmark_matrix[model_id][benchmark_name].append(accuracy)
    
    # Calculate average accuracy for each model-benchmark pair
    matrix_data = []
    benchmark_names = sorted(set(benchmarks_map.values()))
    
    # Create cleaner column names
    column_name_map = {
        'clean-sandboxes-tasks-eval-set': 'clean_sandboxes_eval',
        'clean-sandboxes-tasks-recleaned': 'clean_sandboxes_recleaned', 
        'dev_set_71_tasks': 'dev_set_71',
        'terminal_bench_2': 'terminal_bench_2'
    }
    
    for model_id, data in model_benchmark_matrix.items():
        row = {
            'model_id': model_id,
            'model_name': data.get('model_name', 'unknown')
        }
        
        # Add average accuracy for each benchmark with cleaner column names
        for benchmark in benchmark_names:
            clean_name = column_name_map.get(benchmark, benchmark.replace('-', '_'))
            if benchmark in data and benchmark != 'model_name':
                accuracies = data[benchmark]
                avg_accuracy = sum(accuracies) / len(accuracies)
                row[clean_name] = avg_accuracy
            else:
                row[clean_name] = None
        
        matrix_data.append(row)
    
    # Sort by model name
    matrix_data.sort(key=lambda x: x['model_name'])
    
    # Write CSV with cleaner column names
    clean_benchmark_names = [column_name_map.get(b, b.replace('-', '_')) for b in benchmark_names]
    fieldnames = ['model_id', 'model_name'] + clean_benchmark_names
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in matrix_data:
            writer.writerow(row)
    
    print(f"\nModel-Benchmark matrix saved to: {output_file}")
    print(f"Total models: {len(matrix_data)}")
    print(f"Benchmarks: {', '.join(benchmark_names)}")
    
    # Show top models per benchmark
    print("\nTop models per benchmark:")
    for benchmark in benchmark_names:
        # Find best model for this benchmark
        best_model = None
        best_accuracy = -1
        
        for row in matrix_data:
            if row.get(benchmark) is not None and row[benchmark] > best_accuracy:
                best_accuracy = row[benchmark]
                best_model = row
        
        if best_model:
            print(f"\n{benchmark}:")
            print(f"  Best: {best_model['model_name']} - {best_accuracy:.4f}")
    
    return output_file

def generate_comprehensive_csv(jobs: List[dict], eval_type: str, output_dir: str = "results"):
    """
    Generate detailed and summary CSVs for evaluation results grouped by model.
    
    Args:
        jobs: List of job records with metrics
        eval_type: Type of evaluation (e.g., 'traces-terminus-2', 'gsm8k')
        output_dir: Directory to save CSV files
    
    Returns:
        Tuple of (detailed_file, summary_file) paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare detailed data
    detailed_data = []
    model_summary = {}
    
    for job in jobs:
        job_id = job.get('id')
        job_name = job.get('job_name', '')
        model_id = job.get('model_id', 'no_model_id')
        metrics = job.get('metrics')
        
        # Extract model name from job_name
        model_name = extract_model_name_from_job(job_name)
        
        # Extract accuracy and std
        accuracy, std = extract_accuracy_metrics(metrics)
        
        # Add to detailed data
        detailed_data.append({
            'job_id': job_id,
            'model_id': model_id,
            'model_name': model_name,
            'accuracy': accuracy,
            'std': std,
            'job_name': job_name
        })
        
        # Add to model summary
        if model_id not in model_summary:
            model_summary[model_id] = {
                'model_id': model_id,
                'model_name': model_name,
                'runs': [],
                'accuracies': [],
                'stds': []
            }
        
        model_summary[model_id]['runs'].append(job_id)
        if accuracy is not None:
            model_summary[model_id]['accuracies'].append(accuracy)
        if std is not None:
            model_summary[model_id]['stds'].append(std)
    
    # Sort detailed data by model_id and accuracy
    detailed_data.sort(key=lambda x: (x['model_id'], -(x['accuracy'] if x['accuracy'] is not None else 0)))
    
    # Write detailed CSV
    clean_eval_type = eval_type.replace(' ', '_').replace('/', '_')
    detailed_file = os.path.join(output_dir, f"{clean_eval_type}_detailed.csv")
    with open(detailed_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['job_id', 'model_id', 'model_name', 'accuracy', 'std', 'job_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in detailed_data:
            writer.writerow(row)
    
    # Calculate summary statistics
    summary_data = []
    for model_id, data in model_summary.items():
        avg_acc = sum(data['accuracies']) / len(data['accuracies']) if data['accuracies'] else None
        avg_std = sum(data['stds']) / len(data['stds']) if data['stds'] else None
        max_acc = max(data['accuracies']) if data['accuracies'] else None
        min_acc = min(data['accuracies']) if data['accuracies'] else None
        
        summary_data.append({
            'model_id': model_id,
            'model_name': data['model_name'],
            'num_runs': len(data['runs']),
            'avg_accuracy': avg_acc,
            'max_accuracy': max_acc,
            'min_accuracy': min_acc,
            'avg_std': avg_std
        })
    
    # Sort by average accuracy
    summary_data.sort(key=lambda x: x['avg_accuracy'] if x['avg_accuracy'] is not None else 0, reverse=True)
    
    # Write summary CSV
    summary_file = os.path.join(output_dir, f"{clean_eval_type}_summary.csv")
    with open(summary_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['model_id', 'model_name', 'num_runs', 'avg_accuracy', 'max_accuracy', 'min_accuracy', 'avg_std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_data:
            writer.writerow(row)
    
    # Print summary statistics
    print(f"\n" + "=" * 80)
    print(f"COMPREHENSIVE RESULTS FOR {eval_type.upper()}")
    print("=" * 80)
    print(f"Total jobs: {len(detailed_data)}")
    print(f"Unique models: {len(summary_data)}")
    print(f"\nTop 5 models by average accuracy:")
    for i, row in enumerate(summary_data[:5], 1):
        if row['avg_accuracy'] is not None:
            print(f"{i}. {row['model_name']}")
            print(f"   Model ID: {row['model_id'][:8]}...")
            avg_std_val = row['avg_std'] if row['avg_std'] else 0
            print(f"   Runs: {row['num_runs']}, Avg: {row['avg_accuracy']:.4f} ± {avg_std_val:.4f}")
            print(f"   Max: {row['max_accuracy']:.4f}, Min: {row['min_accuracy']:.4f}")
    
    print(f"\nDetailed results saved to: {detailed_file}")
    print(f"Summary results saved to: {summary_file}")
    
    return detailed_file, summary_file

def display_formatted_results(csv_data: List[dict]):
    """Display results with bold highlighting for highest accuracy."""
    if not csv_data:
        return
    
    # Find highest accuracy
    valid_rows = [row for row in csv_data if row['accuracy'] is not None]
    if not valid_rows:
        print("\nNo valid accuracy data to display.")
        return
    
    highest_row = max(valid_rows, key=lambda x: x['accuracy'])
    highest_uuid = highest_row['uuid']
    
    print(f"\nFormatted Results:")
    print("=" * 100)
    print(f"{'UUID':<38} {'Job Name':<40} {'Accuracy':<15} {'Std Error':<15}")
    print("-" * 100)
    
    for row in csv_data:
        uuid = row['uuid'] or ''
        job_name = row['job_name'][:37] + "..." if row['job_name'] and len(row['job_name']) > 40 else (row['job_name'] or '')
        
        is_highest = (uuid == highest_uuid)
        
        # Format accuracy
        if row['accuracy'] is not None:
            acc_str = f"**{row['accuracy']:.6f}**" if is_highest else f"{row['accuracy']:.6f}"
        else:
            acc_str = ""
        
        # Format std
        if row['std'] is not None:
            std_str = f"**{row['std']:.6f}**" if is_highest else f"{row['std']:.6f}"
        else:
            std_str = ""
        
        print(f"{uuid:<38} {job_name:<40} {acc_str:<15} {std_str:<15}")
    
    if highest_row['accuracy'] is not None:
        print(f"\nHighest accuracy: **{highest_row['accuracy']:.6f}** (UUID: {highest_uuid})")

def main(include_substrings: List[str] = None, exclude_substrings: List[str] = None, 
         output_file: str = None, model_id: str = None, or_mode: bool = False):
    """Main function to search jobs and create CSV."""
    
    exclude_substrings = exclude_substrings or []
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    try:
        # Create Supabase client
        client = create_supabase_client()
        
        # Determine search mode
        if model_id:
            # Model-specific search mode using UUID
            print(f"Searching for ALL jobs for model_id: {model_id}")
            print("Including Terminal Bench 2.0, dev set, and all benchmarks")
            print("=" * 60)
            
            jobs = search_all_jobs_for_model(client, model_id)
            
            # Generate output filename if not provided
            if not output_file:
                # Use first 8 chars of UUID for filename
                short_id = model_id[:8] if len(model_id) > 8 else model_id
                output_file = f"results/model_{short_id}_all_jobs.csv"
            elif not output_file.startswith("results/"):
                output_file = f"results/{output_file}"
            
            # Display results grouped by evaluation
            display_model_results_by_eval(jobs)
                
        else:
            # Include/exclude search mode - generate matrix with filtered jobs
            if not include_substrings:
                print("Error: Either --model or --include must be specified")
                return
                
            print(f"Searching for jobs with include: {include_substrings}")
            print(f"Excluding jobs with: {exclude_substrings}")
            
            # Generate matrix with filtered jobs  
            include_str = "_".join(include_substrings).replace(" ", "_").replace("/", "_")[:50]
            suffix = "_OR_matrix.csv" if or_mode else "_matrix.csv"
            output_file = output_file or f"results/{include_str}{suffix}"
            
            generate_filtered_model_benchmark_matrix(
                client, include_substrings, exclude_substrings, output_file, or_mode
            )
            return
        
        print(f"\nTotal jobs found: {len(jobs)}")
        
        # Prepare CSV data with eval_name for model search
        csv_data = []
        for job in jobs:
            job_id = job.get('id')
            job_name = job.get('job_name')
            metrics_json = job.get('metrics')
            eval_name = job.get('eval_name', '')  # Will be empty for non-model searches
            
            accuracy, std = extract_accuracy_metrics(metrics_json)
            
            csv_row = {
                'uuid': job_id,
                'job_name': job_name,
                'accuracy': accuracy,
                'std': std
            }
            
            # Add eval_name column if doing model search
            if model_id:
                csv_row['eval_name'] = eval_name
            
            csv_data.append(csv_row)
        
        # Write to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['uuid', 'eval_name', 'job_name', 'accuracy', 'std'] if model_id else ['uuid', 'job_name', 'accuracy', 'std']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        
        print(f"Results saved to {output_file}")
        
        # Print summary
        valid_accuracy_count = sum(1 for row in csv_data if row['accuracy'] is not None)
        print(f"Jobs with valid accuracy: {valid_accuracy_count}/{len(csv_data)}")
        
        # Display formatted results for non-model searches (unless comprehensive CSV was generated)
        if not model_id and not detected_eval_type:
            display_formatted_results(csv_data)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search sandbox_jobs table and extract metrics")
    
    # Create mutually exclusive group for search modes
    search_group = parser.add_mutually_exclusive_group(required=False)
    
    search_group.add_argument(
        '--model', '-m',
        help='Model UUID to search for ALL jobs (from models table)'
    )
    
    search_group.add_argument(
        '--include', '-i',
        nargs='+',
        help='Substrings that must be present in job_name (space-separated)'
    )
    
    search_group.add_argument(
        '--matrix',
        action='store_true',
        help='Generate model-benchmark matrix CSV with all models and benchmarks'
    )
    
    parser.add_argument(
        '--exclude', '-e', 
        nargs='+',
        default=[],
        help='Substrings that must NOT be present in job_name (space-separated)'
    )
    
    parser.add_argument(
        '--or-mode',
        action='store_true',
        help='Use OR logic for --include (any substring matches instead of all)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output CSV filename (default: auto-generated based on search terms)'
    )
    
    args = parser.parse_args()
    
    print("Sandbox Jobs Search Tool")
    print("=" * 40)
    
    # Default to matrix if no arguments provided
    if not args.model and not args.include and not args.matrix:
        args.matrix = True
    
    if args.matrix:
        # Generate model-benchmark matrix
        client = create_supabase_client()
        output_file = args.output or "results/model_benchmark_matrix.csv"
        generate_model_benchmark_matrix(client, output_file)
    elif args.model:
        # Model search mode
        main(model_id=args.model, output_file=args.output)
    else:
        # Include/exclude search mode
        main(include_substrings=args.include, exclude_substrings=args.exclude, output_file=args.output, or_mode=args.or_mode)
