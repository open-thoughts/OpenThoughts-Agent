# Student difficulty on teacher trajectories

Which tokens of a strong teacher's agent trajectory does a weaker student find surprising? This directory scores
teacher-only CalibForge episodes (Qwen3.8-27B, mini-swe-agent 2.4.6 tool mode) under a student checkpoint by teacher
forcing, caches per-token log-probs, and serves a local viewer. No rollouts, no training.

## Pipeline (Horizon)

| step | script | output (under `$E=/scratch/11584/lukedhlee/experiments/student_difficulty`) |
|---|---|---|
| sample | `sample.py --out $E/sample.json` | 20 episodes, seeded and domain-stratified, with source paths and exclusions |
| render | `render_full.py --manifest $E/sample.json --tokenizer-dir <student> --out-dir $E/renders` | one token sequence per episode in the student's chat template; per-turn prefix checks |
| serve | `STUDENT_MODEL=<dir> sbatch -o $E/logs/serve_%j.out serve_hold.sbatch` | the student on one GB200 node (eval serve args, no draft, 131,072 context) |
| score | `srun --jobid=<job> --overlap ... score.py --renders $E/renders --out $E/scores/<set> --tag <set> [--checks] [--condition norefeed]` | per-token log-prob, rank, top-5 per target token; `run.json` with provenance and checks |
| site | `build_site.py --renders $E/renders --scores $E/scores --tags <sets...> --manifest $E/sample.json --out $E/site` | static site + `tables/turns.csv`, `tables/spans.csv` |

Score sets so far: `dk0921`, `rlvr1s12` (full history) and `dk0921_norefeed`, `rlvr1s12_norefeed`.

**Conditions.** `full_history` scores the episode in one causal pass. Every earlier turn keeps its reasoning, which is
what the teacher conditioned on: the collection ran `interleaved_thinking: true`, and Qwen3.8's template keeps earlier
thinking. `norefeed` sends one request per turn, with earlier assistant turns stripped of their reasoning. That is the
student's own MSA2 eval view. The same body tokens are scored in both conditions, so positions line up.

**Correctness checks** (`score.py --checks`, first episode, recorded in `run.json`):
- **Render:** every turn's history renders, token for token, as the prefix of the full sequence. Target tokens decode
  to exactly the assistant bodies.
- **Generation:** at sampled positions, generation from `ids[:j]` agrees with `prompt_logprobs[j]`. On 28 of 28
  positions top-1 matches, with mean |Δ| 0.12 nats. Shifting by ±1 gives 0–1 of 28 and 4.6–4.9 nats.
- **No-future:** re-scoring a prefix that ends at a turn leaves that turn's scores unchanged up to noise.
- **Repeat:** repeating the identical request gives the serving noise floor. vLLM on this MoE is not bit-deterministic:
  median 0.0005, mean 0.02, p99 0.27–0.30, max 1.6–2.1 nats per token.

**Target classes.** reasoning / prose / command (the bash string inside the tool-call JSON) are content. think_tag /
call_fmt / eot are format. Tool calls appear in the student's `<tool_call>{json}` form: the teacher wrote XML, so
call_fmt measures the conversion rather than the teacher.

**Adding a checkpoint** (e.g. SFT steps): serve it, then `score.py --tag <new>` into `$E/scores/<new>` and rebuild the
site with the extra tag. Scores join renders by token position and are checked against each render's text hash.

## Viewer

```
python3 -m http.server 8765 -d <site dir>     # then open http://localhost:8765
```

Deep links: `#t05` opens a trajectory, `#t05.12` opens turn 12, and `#spans` / `#bookmarks` / `#about` open those
views. Bookmarks live in browser localStorage, with JSON export and import.

The method-benchmark report is `compare.html`: `python compare_page.py --bench <benchmark dir> --text report_text.html
--out <site dir>/compare.html` fills the figures in `report_text.html` (the prose, written after reading the results)
from `benchmark.json` and the judged windows.

The same site can be published as a private claude.ai Artifact: `python artifact_bundle.py --site <site dir> --out <dir>`
writes the benchmark page as the artifact page plus `viewer.html`, its JS/CSS and the per-trajectory JSON (the
38 MB `tables/` CSVs stay local). Bookmark export is a download, which the Artifact frame blocks.
