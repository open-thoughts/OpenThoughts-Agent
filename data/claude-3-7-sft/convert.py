from datasets import Dataset


def openai_to_sharegpt(messages: list) -> list:
    role_map = {"user": "human", "assistant": "gpt", "system": "system"}
    return [
        {"from": role_map[message["role"]], "value": message["content"]}
        for message in messages
    ]


def convert_openai_to_sharegpt(
    dataset: Dataset, conversations_column: str, output_column: str
) -> Dataset:
    def f(row: dict):
        try:
            row[output_column] = openai_to_sharegpt(row[conversations_column])
        except Exception as e:
            raise e
        return row
    dataset = dataset.map(f)
    return dataset

from datasets import load_dataset
dataset = load_dataset("mlfoundations-dev/terminal-bench-traces-local", split="train")
# ds = dataset.filter(lambda x: x['model'] == "claude-3-7-sonnet-20250219")
ds = dataset
ds = convert_openai_to_sharegpt(ds, "conversations", "conversations")
ds.push_to_hub("mlfoundations-dev/tbench_traces_local_sharegptv1")