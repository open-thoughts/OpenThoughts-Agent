from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-0.5b",
        model_source="Qwen/Qwen2.5-7B-Instruct",
    ),
    runtime_env={"env_vars": {"VLLM_USE_V1": "1"}},
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=16, max_replicas=16,
        )
    ),
    # Pass the desired accelerator type (e.g. A10G, L4, etc.)
    # accelerator_type="GH200",
    # You can customize the asdfengine arguments (e.g. vLLM engine kwargs)
    engine_kwargs=dict(
        #enforce_eager=True,
        tensor_parallel_size=1,
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, blocking=True)

