from datasets import Dataset

def subsample_traces(traces: Dataset, num_episodes: int) -> Dataset:
    def f(x):
        x['conversations'] =x['conversations'][:num_episodes * 2]
        return x
    
    dataset = traces.map(f)
    return dataset