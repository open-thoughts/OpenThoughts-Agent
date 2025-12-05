# DCLM Bash Sequence Finder

Finds bash-related text sequences in the DCLM dataset using graduated scoring.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Test on Random Subset

```bash
./run_pipeline.sh [num_shards] [sample_size] [threshold] [workers]

# Examples:
./run_pipeline.sh 2 20 8 2   # 2 shards, 20 samples for inspection, threshold 8, 2 workers
```

This selects random shards from `/data/horse/ws/rehe951g-dcagent/dclm-baseline/`, processes them in parallel, and saves results to `test_results_*/`.

### Process All Shards

```bash
./run_all_shards.sh [workers] [threshold] [sample_size]

# Examples:
./run_all_shards.sh              # All CPUs, threshold 4.0, 100 samples
./run_all_shards.sh 32 4.0 200  # 32 workers, threshold 4.0, 200 samples
```

## Configuration

Set custom DCLM directory:
```bash
DCLM_DIR=/path/to/dclm ./run_pipeline.sh
DCLM_DIR=/path/to/dclm ./run_all_shards.sh 32
```

## Scoring System

Text is scored for bash-related content:
- **Bash blocks/shebangs**: 10 points
- **Shell prompts, pipes, redirects**: 2 points each
- **Variables, commands**: 1-2 points

Default threshold: 4.0 (balanced detection)

## Output Files

Results are saved to timestamped directories:

```
test_results_*/  or  results_all_*/
├── bash_sequences_all.jsonl              # All bash sequences
├── bash_sequences_sample_N_readable.txt  # Human-readable samples
├── statistics.json                        # Overall statistics
└── processing_progress.txt                # Checkpoints
```

## Slurm script 

slurm_process_shards.sh 

## Analyzing results 

statistics on the results:

python3 analyze_statistics.py /data/horse/ws/rehe951g-dcagent/dclm-terminal-like/results_all_20251117_155756/statistics.json --output /data/horse/ws/rehe951g-dcagent/dclm-terminal-like/results_all_20251117_155756/output.json
