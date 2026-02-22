# Research Brainstorming: Sandbox-Based Data for Training Code Agents

## Context

DC-Agent has **43+ dataset generators** spanning 5 sandbox strategies (test-based, Dockerfile-based, self-documented, bug-fix/patch, complex/integration), 10+ languages, and a full RL pipeline (SkyRL + GRPO). The binary pass/fail reward comes from test execution in Docker sandboxes. Current findings: instruction quality matters hugely (+62% from adding example values), and ablations have explored sampling parameters extensively. This doc proposes **research directions** to push beyond the current setup.

---

## Theme 1: Data Quality & Curation

### 1.1 Difficulty-Aware Data Mixing

**Hypothesis:** There is an optimal difficulty distribution for RL training data, and the current uniform 10K-per-dataset mixing is suboptimal.

**Motivation:** Tasks that are always solved produce no gradient. Tasks never solved produce only zero reward. The "Goldilocks zone" (30-50% pass@8) should dominate the training mix.

**Experiment:**
- **Baseline:** Uniform 10K sample from all 43 datasets
- **Treatments:** (a) "Goldilocks" -- only tasks with pass@8 between 25-75%; (b) 70/20/10 medium/hard/very_hard; (c) 50/30/20 very_hard/hard/medium; (d) equal representation per difficulty band
- Same Qwen3-8B + GRPO, evaluate on tiered benchmarks (S/A/B/C/D)

**Expected outcome:** Goldilocks mix outperforms uniform by 5-15%, especially on B-tier+. Would establish reward signal density as the key predictor of data utility.

**Effort:** Medium | **Paper potential:** High (data-centric RL story)

---

### 1.2 Instruction Specificity vs. Generalization

**Hypothesis:** Training on overly specified instructions (with exact I/O examples) hurts generalization to underspecified real-world tasks.

**Motivation:** The +62% improvement from better instructions is great for benchmarks, but real GitHub issues are vague. There's a tension between learnability and transfer.

**Experiment:**
- Generate 3 versions of each dataset: (a) Minimal (function signature only), (b) Standard (current with examples), (c) Rich (3+ I/O examples + edge cases)
- Evaluate on both well-specified benchmarks AND a new "underspecified" suite from real GitHub issues
- **Baseline:** Standard instructions only

**Expected outcome:** A mix of specificity levels generalizes best. Standard wins on well-specified benchmarks; mixed wins on underspecified tasks.

**Effort:** Medium-High | **Paper potential:** High (practical insight for the field)

---

### 1.3 Test Quality Scoring & Filtering

**Hypothesis:** Filtering out low-quality tests (trivially satisfiable, low coverage, buggy) improves downstream agent performance significantly.

**Motivation:** Reward signal quality is everything in RL. Some LLM-generated tests only check file existence. Some mined tests have weak assertions. A "deception rate" metric (how often a random solution passes) would expose this.

**Experiment:**
- Score all tests: assertion count, mutation testing score, deception rate (run 10 random solutions, measure false positive rate)
- Partition into quality quartiles. Train on each.
- **Baseline:** All tests included
- **Treatments:** Top-50%, top-25%, quality-weighted sampling

**Expected outcome:** High-quality tests -> significantly better on hard benchmarks. Gap smaller on easy tasks.

**Effort:** High | **Paper potential:** Very high (test quality for RL is underexplored)

---

### 1.4 Cross-Dataset Deduplication

**Hypothesis:** 5-15% semantic duplication exists across the 43 generators (all mining overlapping sources), and deduplication improves training efficiency.

**Motivation:** `generate_bash_tasks.py` and `generate_bash_tasks_with_tests.py` both mine The Stack bash subset. Self-documented generator also mines The Stack. Overlap wastes training budget and could inflate benchmarks.

**Experiment:**
- MinHash + CodeBERT embedding similarity across all datasets
- Measure duplication rate, near-duplicate rate (>0.8 cosine), benchmark contamination
- **Baseline:** Full dataset. **Treatment:** Deduplicated dataset
- Compare convergence speed and final performance

**Expected outcome:** Dedup improves training efficiency (fewer steps to same performance).

**Effort:** Medium

---

## Theme 2: Reward Signal Design

### 2.1 Dense Rewards from Partial Test Passage

**Hypothesis:** Replacing binary pass/fail with fraction-of-tests-passed improves RL training stability and final performance, especially on hard tasks.

**Motivation:** Currently passing 4/5 tests gets the same reward (0) as passing 0/5. With GRPO using 8 samples/prompt, hard tasks often produce all-zero batches -- no gradient. Partial credit preserves learning signal from near-misses.

**Experiment:**
- Modify `test.sh` templates to count individual assertions and report fraction passed
- **Baseline:** Binary reward (current)
- **Treatments:** (a) Proportional (fraction passed), (b) Threshold (1.0 if >= 80% pass, else 0.0)
- Same GRPO setup, Qwen3-8B

**Expected outcome:** Proportional rewards -> faster learning, +5-10% on hard tasks. Binary may still be competitive on easy tasks.

**Effort:** Low-Medium | **Paper potential:** High (simple change, large impact)

---

### 2.2 Process Reward Models for Agent Trajectories

**Hypothesis:** A PRM that scores individual agent steps enables credit assignment over 128-step trajectories, improving sample efficiency 2-3x.

**Motivation:** With max_episodes=128, the agent takes many steps before getting any reward. It doesn't know which steps contributed to success. A PRM provides intermediate signal.

**Experiment:**
- Collect 50K success + 50K failure trajectories
- Train PRM (Qwen2.5-1.5B) on step-level labels from heuristics: file reads in successful traces, compilation checks, test runs before submission
- Use PRM as auxiliary reward during GRPO
- **Baseline:** Binary outcome reward only

**Expected outcome:** 2-3x faster convergence. Agents explore more efficiently.

**Effort:** High | **Paper potential:** Very high (novel contribution)

---

### 2.3 Self-Play Task Generation

**Hypothesis:** Having the agent generate its own tasks creates a self-play loop where difficulty naturally scales with model capability.

**Motivation:** Mined tasks have a fixed difficulty distribution. Self-generated tasks at the model's frontier (~30-50% success rate) keep training in the optimal zone automatically.

**Experiment:**
- Phase 1: Train agent to generate (instruction, Dockerfile, test.sh) tuples
- Phase 2: Agent generates N tasks, attempts to solve them, keeps frontier tasks (30-50% success)
- Phase 3: RL on frontier tasks. Iterate phases 2-3 for K rounds.
- **Baseline:** Standard mined tasks

**Expected outcome:** Progressive improvement per round. Agents eventually solve tasks harder than any in the original training set.

**Effort:** Very High (moonshot) | **Paper potential:** Top-tier venue

---

## Theme 3: Curriculum Learning

### 3.1 Multi-Armed Bandit Dataset Sampling

**Hypothesis:** Dynamically adjusting sampling weights across the 43 datasets during RL (based on current learning signal) outperforms fixed uniform sampling.

**Motivation:** Early in training, easy bash tasks provide dense reward. Later, only SWE-bench tasks remain unsolved. The fixed 10K/dataset mix doesn't adapt to the model's changing frontier.

**Experiment:**
- UCB1 or Thompson sampling bandit over 43 datasets
- Bandit reward = variance of outcome reward within a batch (high variance = informative)
- **Baseline:** Uniform sampling
- Metric: benchmark score vs. wall-clock time

**Expected outcome:** Same benchmark score in 30-50% less training time.

**Effort:** Medium | **Paper potential:** High

---

### 3.2 Language-Based Curriculum

**Hypothesis:** Training on simpler languages first (bash, Python) before complex ones (Rust, C++) leads to better cross-language transfer than uniform mixing.

**Experiment:**
- Phase 1: bash + Python. Phase 2: + Go + JS. Phase 3: + Java + C++ + Rust. Phase 4: all.
- **Baseline:** All languages from step 0
- Evaluate per-language AND cross-language transfer

**Expected outcome:** Curriculum model better on complex languages, matches or exceeds baseline on scripting languages. Key insight: does "learning to debug" transfer across languages?

**Effort:** Medium

---

### 3.3 Sandbox Strategy Curriculum (Easy to Hard)

**Hypothesis:** Training by sandbox complexity (test-based -> self-documented -> Dockerfile -> bug-fix -> integration) outperforms random ordering.

**Experiment:**
- Complexity order: test-based (simplest) -> self-documented -> Dockerfile -> bug-fix/patch -> integration (hardest)
- Train each strategy for fixed steps before introducing next
- **Baseline:** All strategies from step 0
- **Ablation:** Reverse order (hard to easy)

**Expected outcome:** Easy-to-hard produces best final performance. Hard-to-easy may show faster initial gains but lower final score.

**Effort:** Low-Medium

---

## Theme 4: Multi-Task & Transfer

### 4.1 Dataset Attribution (Which Datasets Actually Matter?)

**Hypothesis:** Not all 43 datasets contribute equally. Some may even hurt performance. The optimal subset is likely 10-15 datasets.

**Motivation:** 43 x 10K = 430K tasks. Some (QuixBugs with 40 puzzles, heavily upsampled) may be noise. Data Shapley analysis would identify what matters.

**Experiment:**
- Train 43 single-dataset RL models. Evaluate each on full benchmark suite.
- Compute influence matrix: which training datasets improve which benchmark categories.
- Leave-one-out analysis for marginal contribution.
- Train "curated" model on top-K datasets.
- **Baseline:** All 43 datasets

**Expected outcome:** 60-80% of performance from 10-15 datasets. Curated mix matches full while being ~3x cheaper.

**Effort:** High | **Paper potential:** Very high (would be a reference for the field)

---

### 4.2 Transfer Matrix Across Sandbox Types

**Hypothesis:** Transfer is asymmetric -- some strategies are better "donors" than others.

**Experiment:**
- Train 5 specialist models (one per sandbox strategy). Evaluate each on all strategy types.
- Construct 5x5 transfer matrix.

**Expected outcome:** Bug-fix and integration tasks have highest positive transfer (require broadest skill set).

**Effort:** Medium | **Paper potential:** Medium-High

---

### 4.3 Skill Distillation (1K Tasks = 10K Tasks?)

**Hypothesis:** Language-agnostic meta-skills (explore -> read test -> write code -> run test -> debug -> iterate) can be distilled into a compact training set.

**Experiment:**
- Cluster 10K successful trajectories by action sequence patterns (ignoring language content)
- Identify universal action patterns
- Construct 1K-task set covering all patterns. Compare vs. random 10K.

**Expected outcome:** 1K curated set achieves 80-90% of 10K random set. Task diversity > task count.

**Effort:** Medium-High | **Paper potential:** High

---

## Theme 5: Synthetic Data Improvements

### 5.1 LLM Consistency Verification

**Hypothesis:** 10-20% of LLM-generated (instruction, test) pairs are broken -- the test checks something the instruction doesn't describe, or vice versa.

**Experiment:**
- After generating (instruction, test.sh), run a third LLM call to solve the task
- Classify: "solvable" (LLM solves it), "hard" (legitimate but unsolved), "broken" (no reasonable solution passes)
- Track broken rate per generator. Train with and without filtering.
- **Baseline:** All tasks

**Expected outcome:** Removing broken tasks improves RL stability (fewer impossible-reward batches).

**Effort:** Medium

---

### 5.2 Adversarial Test Generation

**Hypothesis:** Tests targeting common agent mistakes (off-by-one, wrong paths, empty input edge cases) provide more discriminative reward signal than happy-path tests.

**Experiment:**
- Analyze 100 most common failure patterns from existing traces
- Prompt LLM to generate tests targeting these failure modes
- Compare: standard tests vs. adversarial tests on false positive rate and RL training quality
- **Baseline:** Standard test generation

**Expected outcome:** Adversarial tests reduce false positive rate from ~15% to ~3%.

**Effort:** Medium

---

### 5.3 Test Synthesis for Arbitrary Code (10-100x Scale)

**Hypothesis:** Generating tests for arbitrary high-quality code (rather than mining code that already has tests) expands the task pool 10-100x.

**Motivation:** Current mining requires existing test patterns -- most code doesn't have tests. LLM-generated tests validated by mutation testing could unlock vast new sources.

**Experiment:**
- Mine 100K high-quality Python files from The Stack (filtered by stars)
- LLM generates test suites. Validate against original code (should pass).
- Mutation testing as quality check (should fail on mutants).
- Compare resulting dataset vs. existing test-mined datasets.

**Expected outcome:** 5-10x more tasks with comparable quality after validation filtering.

**Effort:** Medium-High | **Paper potential:** High

---

### 5.4 Compositional Multi-Skill Tasks

**Hypothesis:** Tasks combining 2-4 skills (file I/O + parsing + DB + testing) train better agents for complex real-world tasks than single-skill tasks.

**Experiment:**
- Define 8-10 atomic skills. Generate 2-skill, 3-skill, 4-skill compositions.
- Compare: (a) single-skill only, (b) 2-skill, (c) 3-4 skill, (d) mix
- **Baseline:** Single-skill tasks

**Expected outcome:** Mix performs best overall; compositions critical for B-tier+ benchmarks.

**Effort:** Medium

---

## Theme 6: Scaling Laws

### 6.1 Task Count vs. Rollout Depth

**Hypothesis:** There exist Chinchilla-like scaling laws for agentic RL: an optimal balance between unique tasks and rollouts per task.

**Motivation:** Is 10K tasks x 8 rollouts better than 40K tasks x 2 rollouts? Currently n_samples_per_prompt=8 is chosen without theoretical justification.

**Experiment:**
- Fix total compute. Sweep: 1K x 32, 5K x 16, 10K x 8, 20K x 4, 40K x 2
- Fit power-law curves to results.
- **Baseline:** Current 10K x 8

**Expected outcome:** Optimal point near 10K x 8-16 for 8B models, shifting toward more tasks for larger models.

**Effort:** High | **Paper potential:** Very high (foundational result)

---

### 6.2 Per-Generator Saturation Analysis

**Hypothesis:** Each generator has a saturation point. Some (bash, pytest mining The Stack millions) saturate slowly; others (QuixBugs with 40 puzzles) saturate instantly.

**Experiment:**
- For top-10 generators, train with 1K, 2.5K, 5K, 10K, 20K tasks
- Fit saturation curves (log or power-law)

**Expected outcome:** Identifies where to invest data collection effort: generators far from saturation benefit most from more mining.

**Effort:** Medium-High

---

## Theme 7: Novel Sandbox Designs

### 7.1 Interactive Debugging Sandboxes

**Hypothesis:** Training agents to use pdb/gdb develops diagnostic capabilities that transfer to all task types.

**Experiment:**
- Create "debug-based" tasks: buggy code + failing test, must use debugger to identify and fix
- Verify agent used debugger (check command history)
- 5K debug tasks from Defects4J, BugsInPy
- **Baseline:** Standard training without debug tasks

**Expected outcome:** Debug-trained agents better at diagnosing failures across all task types.

**Effort:** Medium-High

---

### 7.2 Multi-File Repository Sandboxes

**Hypothesis:** Single-file tasks don't transfer to repository-scale reasoning. There's a critical transition at 3-5 file modifications.

**Experiment:**
- Create "repo task" generator: clone repos, introduce bugs across 2-3 files
- Vary scope: 1 file, 2-3 files, 5+ files
- Measure transfer from single-file training to multi-file evaluation
- **Baseline:** Current mostly-single-file training

**Expected outcome:** Multi-file training necessary to cross the "navigation threshold."

**Effort:** High | **Paper potential:** High

---

### 7.3 Multi-Stage Verification Sandboxes

**Hypothesis:** Tasks with 3-5 sequential verification stages (compile -> unit test -> integration test -> edge cases) teach better planning.

**Experiment:**
- test.sh runs stages sequentially, reports progress. Reward proportional to stages passed.
- Agent receives intermediate feedback between stages.
- 5K multi-stage tasks across domains.
- **Baseline:** Single-stage verification

**Expected outcome:** Better planning, fewer wasted attempts at later stages.

**Effort:** Medium-High

---

### 7.4 DevOps / SysAdmin Sandboxes

**Hypothesis:** Training on system tasks (configure nginx, set up cron, debug Docker) unlocks a new capability dimension.

**Experiment:**
- Generate tasks from Ansible roles, Terraform configs, K8s manifests
- Tasks: "configure nginx to proxy port 8080," "debug why container fails to start"
- Mix into training at 10-20% of total

**Expected outcome:** Unlocks DevOps capabilities without sacrificing coding performance.

**Effort:** Medium

---

## Theme 8: Evaluation

### 8.1 Decomposed Capability Evaluation

**Hypothesis:** Pass/fail conflates multiple capabilities. Decomposition reveals which training data improves which capability.

**Experiment:**
- Annotate benchmarks with required capabilities
- Instrument trajectories: "did agent read test file?" (exploration), "syntactically correct code?" (quality), "ran test before submitting?" (verification)
- Build capability radar chart per model

**Expected outcome:** RL primarily improves exploration and verification; SFT primarily improves code quality.

**Effort:** Medium | **Paper potential:** Medium-High

---

### 8.2 Adaptive Difficulty Benchmarks

**Hypothesis:** Static benchmarks saturate. Adaptive benchmarks targeting the model's 30-50% success zone provide more informative evaluation.

**Experiment:**
- Build generator that targets the model's frontier difficulty band
- Compare information gain of adaptive vs. static evaluation

**Expected outcome:** Detects capability improvements earlier, more actionable feedback.

**Effort:** Medium

---

## Theme 9: Agent Reasoning & Planning

### 9.1 Optimal Thinking Budget

**Hypothesis:** The optimal ratio of thinking tokens to action tokens varies by task difficulty (~20-30% for easy, ~40-50% for hard).

**Experiment:**
- Instrument trajectories to measure thinking vs. action tokens per step
- Correlate with success rate, controlling for difficulty
- Test fixed (10%, 25%, 50%) vs. adaptive thinking budgets
- **Baseline:** Current default allocation

**Expected outcome:** Adaptive allocation outperforms fixed; thinking budget is an underexplored hyperparameter.

**Effort:** Medium

---

### 9.2 Explicit Plan-Then-Execute

**Hypothesis:** Requiring an upfront plan before taking actions improves multi-step task success by 10-15%.

**Experiment:**
- Add planning phase: first N steps must be planning (read files, outline approach) before code writing
- Compare: (a) no planning, (b) free-form, (c) structured (enumerate files, tests, steps)
- SFT on planning demonstrations, then RL
- **Baseline:** No explicit planning phase

**Expected outcome:** Structured planning improves C-tier+ by 10-15%, minimal effect on easy tasks.

**Effort:** Medium

---

## Theme 10: Exploration & Tool Use

### 10.1 Exploration Reward Bonus

**Hypothesis:** A small reward bonus for information-gathering actions (cat test.sh, ls, checking imports) improves success on hard tasks.

**Motivation:** Many failures show the agent writing code immediately without reading the test. Incentivizing exploration fixes this.

**Experiment:**
- Define "good exploration": reading test files, listing directories, checking available libraries
- +0.05 per unique exploration action (cap +0.3)
- **Baseline:** Outcome-only reward
- **Treatments:** (a) Outcome + exploration bonus, (b) bonus that decays over training

**Expected outcome:** Agents explore more systematically, +5-10% on hard tasks. Bonus can be removed at test time.

**Effort:** Low-Medium | **Paper potential:** Medium-High

---

### 10.2 Error Recovery Training

**Hypothesis:** Training on tasks with injected errors (broken import, wrong permissions, conflicting library) makes agents more robust.

**Experiment:**
- Create error-injection variants: (a) broken import, (b) wrong permissions, (c) conflicting library, (d) corrupt config
- Agent must diagnose and fix before solving actual task
- **Baseline:** Standard training. **Treatment:** + 30% error-injection tasks

**Expected outcome:** +5-10% across all tiers from better error diagnosis and recovery.

**Effort:** Low-Medium

---

### 10.3 Tool Discovery Tasks

**Hypothesis:** Training agents to discover available tools in their sandbox (rather than assuming a fixed toolset) improves generalization to new environments.

**Experiment:**
- Create tasks requiring non-standard tools installed in Dockerfile but not mentioned in instruction
- Agent must discover via `which`, `pip list`, `dpkg -l`, etc.
- **Baseline:** Standard-tool tasks only. **Treatment:** + 20% tool-discovery tasks

**Expected outcome:** Better generalization when sandbox environment is unfamiliar.

**Effort:** Medium

---

## Theme 11: Moonshots

### 11.1 Self-Improving Data Pipeline

The trained agent analyzes its own failures, generates targeted training data, and iterates. Each round produces 2-5% improvement. Key question: do improvements compound or plateau?

### 11.2 Multi-Agent Collaborative Training

Train specialized agents (planner, coder, tester, reviewer) that take turns in a shared sandbox. Team should outperform generalist on hard tasks where review/debugging has highest value.

### 11.3 Sandbox Comprehension Module

Pre-train a small adapter specifically on (terminal output -> interpretation) pairs. Terminal output has very different statistics from natural language; a specialized module could improve error diagnosis.

---

## Priority Ranking (Impact / Effort)

| Rank | Idea | Effort | Impact | Quick Win? |
|------|------|--------|--------|------------|
| 1 | **2.1** Dense rewards (partial test) | Low-Med | High | Yes |
| 2 | **10.1** Exploration reward bonus | Low-Med | High | Yes |
| 3 | **1.1** Difficulty-aware mixing | Medium | High | |
| 4 | **10.2** Error recovery training | Low-Med | Med-High | Yes |
| 5 | **3.1** Bandit curriculum | Medium | High | |
| 6 | **5.1** LLM consistency check | Medium | Med-High | |
| 7 | **3.3** Sandbox strategy curriculum | Low-Med | Medium | Yes |
| 8 | **6.1** Scaling laws (tasks vs rollouts) | High | Very High | Paper |
| 9 | **4.1** Dataset attribution | High | Very High | Paper |
| 10 | **1.3** Test quality scoring | High | High | Paper |
| 11 | **1.2** Instruction specificity study | Med-High | High | Paper |
| 12 | **5.3** Test synthesis for arbitrary code | Med-High | High | |
| 13 | **2.2** Process reward models | High | Very High | Paper |
| 14 | **2.3** Self-play | Very High | Transformative | Top venue |

---

## Suggested First Experiments (Low-Hanging Fruit)

1. **Dense rewards (2.1)** -- modify test.sh templates to report fraction passed. Minimal code change, potentially large RL improvement.
2. **Exploration bonus (10.1)** -- add +0.05 for `cat test.sh` style actions. Quick to implement, addresses known failure mode.
3. **Error injection (10.2)** -- create broken variants of existing tasks. Reuses existing infrastructure.
4. **Sandbox strategy curriculum (3.3)** -- just reorder existing data. No new data needed.

## Suggested Paper-Worthy Projects

1. **Scaling laws (6.1)** -- foundational result, widely applicable
2. **Dataset attribution (4.1)** -- practical impact, reference for the community
3. **Test quality for RL (1.3)** -- underexplored area, clean story
4. **Self-play (2.3)** -- ambitious but transformative if it works
