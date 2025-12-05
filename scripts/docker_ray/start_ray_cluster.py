import argparse
import logging
import os
import ray
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

def main():
    parser = argparse.ArgumentParser(description='Start Ray Serve with vLLM')
    parser.add_argument('--min-replicas', type=int, required=True, help='Minimum number of replicas')
    parser.add_argument('--max-replicas', type=int, required=True, help='Maximum number of replicas')
    parser.add_argument('--tensor-parallel-size', type=int, required=True, help='Tensor parallel size')
    parser.add_argument('--model', type=str, required=True, help='Model name/path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--num-cpus', type=int, default=8, help='Number of CPU cores for Ray (default: 8)')
    parser.add_argument('--num-gpus', type=int, default=2, help='Number of GPUs for Ray (default: 2)')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        # Enable Ray logging
        os.environ["RAY_LOG_LEVEL"] = "INFO"
        os.environ["RAY_SERVE_LOG_LEVEL"] = "INFO"
        os.environ["VLLM_LOG_LEVEL"] = "INFO"
        # Enable HuggingFace progress bars and logging
        os.environ["HF_HUB_VERBOSITY"] = "info"
        os.environ["TRANSFORMERS_VERBOSITY"] = "info"
        print(f"Starting Ray Serve with verbose logging...")
    
    # Extract model ID from full model path for deployment naming
    model_id = args.model.split('/')[-1].replace('-', '_').replace('.', '_')
    
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Model ID: {model_id}")
    print(f"  Min replicas: {args.min_replicas}")
    print(f"  Max replicas: {args.max_replicas}")
    print(f"  Tensor parallel size: {args.tensor_parallel_size}")
    print(f"  Verbose: {args.verbose}")
    print(f"  CPU cores: {args.num_cpus}")
    
    llm_config = LLMConfig(
        model_loading_config=dict(
            model_id=model_id,
            model_source=args.model,
        ),
        runtime_env={"env_vars": {
            "VLLM_USE_V1": "1",
        }},
        deployment_config=dict(
            autoscaling_config=dict(
                min_replicas=args.min_replicas, 
                max_replicas=args.max_replicas,
            ),
            ray_actor_options={
                "num_cpus": 4,  # CPUs per replica
                "num_gpus": args.tensor_parallel_size,
            },        
        ),
        engine_kwargs=dict(
            tensor_parallel_size=args.tensor_parallel_size,
        ),
    )

    app = build_openai_app({"llm_configs": [llm_config]})
    
    serve.run(app, blocking=True)

if __name__ == "__main__":
    main()