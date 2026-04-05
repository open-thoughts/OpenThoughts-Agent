# Agent Setup Prompt — Eval System on New Cluster

Copy the prompt below and paste it into a Claude Code session on the target cluster. It will read the onboarding doc, detect the cluster, and complete the setup.

---

## Prompt

```
I need you to set up the eval system on this cluster. Follow the guide at eval/docs/CLUSTER_ONBOARDING.md exactly.

Context:
- This repo is OpenThoughts-Agent, an eval system that runs LLM evals on HPC clusters via SLURM
- The eval listener (eval/unified_eval_listener.py) submits sbatch jobs that start vLLM + harbor evals
- Each cluster needs: a cluster config YAML (eval/clusters/<name>.yaml), a dotenv (hpc/dotenv/<name>.env), conda env(s) with vLLM + harbor, and ~/secrets.env
- Existing cluster configs to reference: eval/clusters/m2.yaml (MBZ H200), eval/clusters/jupiter.yaml (JSC GH200)

Steps:
1. Read eval/docs/CLUSTER_ONBOARDING.md for the full setup guide
2. Detect this cluster:
   - Run: hostname, sinfo -N --format="%.30N %.6t %.5c %.10G %.10m" | head -5, sinfo --format="%P %D %G" --noheader, uname -m, nvidia-smi | head -3, ls /usr/local/cuda*
   - Run: conda env list
   - Check if ~/secrets.env exists and has DAYTONA_API_KEY, SUPABASE_URL, HF_TOKEN
3. Check if eval/clusters/<detected_cluster>.yaml already exists. If yes, verify it matches hardware. If no, create it using m2.yaml as template.
4. Check if hpc/dotenv/<detected_cluster>.env already exists. If no, create it using m2.env as template. Use $USER for paths.
5. Verify harbor is installed: python -c "import harbor; print(harbor.__version__)"
   - If not installed: check if harbor repo exists locally, install with pip install -e /path/to/harbor
   - Verify harbor is pinned to commit 6fdb92e7 or later (but NOT e371289f)
6. Verify hf_transfer is installed: python -c "import hf_transfer"
   - If not: pip install hf_transfer
7. Pre-download datasets:
   - python eval/snapshot_download.py DCAgent/dev_set_v2
   - python eval/snapshot_download.py DCAgent2/terminal_bench_2
   - python eval/snapshot_download.py DCAgent2/swebench-verified-random-100-folders
8. Create log directories: mkdir -p eval/local/<cluster>/logs experiments/listener_logs
9. Dry-run the listener:
   source ~/secrets.env && PYTHONPATH=$PWD python eval/unified_eval_listener.py \
     --cluster-config eval/clusters/<cluster>.yaml \
     --preset v2 \
     --priority-file eval/lists/a1_nl2bash.txt \
     --require-priority-list \
     --baseline-model-config eval/baseline_model_configs.yaml \
     --timeout-multiplier 2.0 --tp-size 2 --enable-thinking \
     --dry-run --once --verbose
10. Report the results: what cluster was detected, what was created/verified, and whether the dry-run passed.

If anything is missing (secrets, harbor, conda env), tell me what's needed and stop — don't guess at secrets or install packages without confirming.
```
