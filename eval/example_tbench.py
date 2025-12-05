from terminal_bench.cli.tb.runs import create, BackendType
from terminal_bench.agents import AgentName

results = create(
    dataset_name="terminal-bench-core",
    dataset_version="0.1.1",
    backend=BackendType.DAYTONA,
    agent=AgentName.TERMINUS,
    model_name="hosted_vllm/qwen-0.5b",
    agent_kwargs=["api_base=http://127.0.0.1:8000/v1", "key=fake_key"],
    # task_ids=["hello-world"],
    n_concurrent_trials=16,
)

print(results)