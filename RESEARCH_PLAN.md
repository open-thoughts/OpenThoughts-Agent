
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

## Theme 4: Multi-Task & Transfer

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

## Theme 7: RL Data (Harbor Task) Innovations

### 7.1 Dockerfile Curriculum: Environment Complexity as Difficulty Knob

**Hypothesis:** Varying the Dockerfile from batteries-included to bare-bones for the same (instruction, test.sh) pair creates a natural difficulty curriculum that improves RL training.

**Motivation:** The Dockerfile is a uniquely controllable variable in Harbor tasks. A task that's easy with all dependencies pre-installed becomes hard when the agent must figure out what to install. This is free difficulty variation without changing the core task.

**Experiment:**
- For each task, generate 5 Dockerfile variants: D1 (full deps) -> D5 (bare OS image)
- Leverages existing `perturbed_docker` pipeline (`data/perturbed_docker/utils.py`)
- **Treatments:** (a) D1-only, (b) D5-only, (c) uniform mix, (d) curriculum (D1->D5 over training)
- **Baseline:** Current single Dockerfile per task

**Expected outcome:** Curriculum (D1->D5) outperforms all static mixes. Agent learns to solve tasks first, then learns environment setup -- mirroring real-world developer experience.

**Effort:** Medium | **Impact:** High

---

### 7.2 Instruction Information Dropout for Robustness

**Hypothesis:** Systematically removing information from instructions during RL training forces the agent to explore rather than rely on spoon-fed details, improving generalization.

**Motivation:** Real-world tasks (GitHub issues, Slack requests) are vague. Training only on well-specified instructions creates a distribution mismatch. Information dropout bridges this gap.

**Experiment:**
- Define info types: file paths, function signatures, I/O examples, error specifications, library references
- Progressive dropout: start with full instructions, increase dropout rate over training
- **Baseline:** Full instructions throughout
- **Treatments:** (a) Random 30% dropout, (b) Progressive 0%->50%, (c) Type-specific dropout
- Evaluate on both well-specified benchmarks AND vague GitHub issues

**Expected outcome:** Progressive dropout produces agents that handle both well-specified and underspecified tasks. Static dropout hurts benchmark performance.

**Effort:** Medium | **Impact:** High

---

### 7.3 Cross-Language Task Transplantation

**Hypothesis:** Transplanting the same task logic into different languages (rewrite instruction + Dockerfile + test.sh) expands the task pool without new mining and improves cross-language transfer.

**Motivation:** A "sort a linked list" task in Python can become the same task in Rust, Go, TypeScript. The core skill is language-agnostic; the implementation challenge varies.

**Experiment:**
- Select 1K tasks with language-agnostic logic
- LLM transplants each into 3-5 target languages
- Leverages language-parameterized Dockerfiles in `commons.py`
- **Baseline:** Python-only tasks. **Treatments:** (a) Transplanted single language, (b) Multi-language mix
- Validate by running transplanted tests

**Expected outcome:** Multi-language mix improves performance on underrepresented languages without hurting Python. Expands effective task pool 3-5x.

**Effort:** Medium | **Impact:** Medium-High

---

### 7.4 Test-First Task Generation (Reverse Direction)

**Hypothesis:** Generating the instruction FROM the test (instead of test from instruction) produces higher-quality task pairs because the test is ground-truth by construction.

**Motivation:** Current pipeline: instruction -> test (LLM generates both, consistency not guaranteed). Reverse: mine high-quality test files from The Stack, LLM generates matching instruction. Test validity is guaranteed since it's mined from working code.

**Experiment:**
- Mine 10K high-quality test files (pytest, Jest, Go test) from The Stack
- LLM generates instruction that the test validates
- Verify by having a second LLM solve instruction and run test
- **Baseline:** Current forward generation. **Treatment:** Reverse generation
- Measure broken pair rate and downstream RL performance

**Expected outcome:** Broken pair rate drops from 10-20% to <5%. Better reward signal -> better RL training.

**Effort:** Medium | **Impact:** High

---

### 7.5 Staged Reward Decomposition

**Hypothesis:** Splitting monolithic test.sh into ordered checkpointed stages and rewarding max stage reached provides denser signal than binary pass/fail while preserving ordering.

**Motivation:** Unlike naive fraction-of-tests-passed (Theme 2.1), staged rewards respect task structure: you can't test correctness before the code compiles, can't test edge cases before happy path works. This is a structured dense reward.

**Experiment:**
- LLM-based test decomposer splits integration-style tests into 3-7 ordered stages
- Reward = max_stage_reached / total_stages
- **Baseline:** Binary reward. **Treatments:** (a) Staged reward, (b) Staged + binary bonus for full completion
- Focus on complex multi-step tasks where binary reward is most limiting

**Expected outcome:** Staged reward improves learning on hard tasks (C/D-tier) by 10-20% while maintaining performance on easy tasks.

**Effort:** Medium-High | **Impact:** High

---

### 7.6 Temporal Reward (Speed Bonus)

**Hypothesis:** Adding a time-based bonus to the reward (R = passed * (1 + alpha * (1 - time_used/timeout))) teaches efficient exploration and reduces timeout failures.

**Motivation:** Many RL failures are timeouts from inefficient exploration (reading entire codebases, running redundant tests). A speed bonus incentivizes the agent to be decisive.

**Experiment:**
- **Baseline:** Binary reward. **Treatments:** (a) alpha=0.2, (b) alpha=0.5, (c) alpha=1.0
- Measure: pass rate, average solve time, timeout rate
- Evaluate whether speed bonus hurts performance on genuinely hard tasks

**Expected outcome:** Moderate alpha (0.2-0.5) reduces timeout rate by 20-30% with no drop in pass rate. High alpha hurts hard tasks.

**Effort:** Low | **Impact:** Medium

---

## Theme 8: Prompt Diversity & Difficulty Control

### 8.1 Instruction Style Transfer (Communication Format Diversity)

**Hypothesis:** Rewriting the same task instruction in different communication formats (formal spec, GitHub issue, Slack message, SO question, code review comment, error report) trains robust intent extraction across styles.

**Motivation:** Real-world coding requests arrive in wildly different formats. Training only on clean, well-structured instructions creates a distribution mismatch with messy real-world inputs.

**Experiment:**
- For each task, LLM rewrites the instruction in 5+ styles (one API call per variant)
- Tests unchanged -- only instruction surface form changes
- **Baseline:** Original instructions only
- **Treatments:** (a) Single alternate style, (b) Uniform mix of all styles, (c) Curriculum from formal -> informal
- Evaluate on SWE-bench and real GitHub issue benchmarks

**Expected outcome:** Style diversity improves performance on real-world tasks (SWE-bench, GitHub issues) by 5-10% with minimal cost to well-specified benchmarks.

**Effort:** Low | **Impact:** High

---

### 8.2 Specification Modality Mixing

**Hypothesis:** Expressing the same requirement in different specification modalities -- (a) natural language prose, (b) pseudocode, (c) I/O examples only, (d) type signatures + docstrings, (e) failing test description -- forces the agent to extract intent from diverse specification formats.

**Motivation:** Developers communicate requirements in many modalities. An agent trained only on natural language instructions struggles when given pseudocode or just I/O examples.

**Experiment:**
- For each task, generate requirement in 5 modalities (tests unchanged; spec modality is the only variable)
- **Baseline:** Natural language only
- **Treatments:** (a) Uniform mix of all modalities, (b) Curriculum from easiest (NL prose) to hardest (I/O examples only)
- Evaluate on benchmarks with diverse specification styles

**Expected outcome:** Modality mixing improves generalization to non-standard specifications. Curriculum ordering outperforms uniform mix.

**Effort:** Medium | **Impact:** High

---

### 8.3 Decomposition Granularity Spectrum

**Hypothesis:** Presenting the same complex task at varying granularity levels -- (a) single vague sentence, (b) 3-5 requirement bullets, (c) step-by-step with file structure hints -- trains agents to handle both detailed specs and vague requests.

**Motivation:** Real tasks range from "make it work" to detailed PRDs. Agents need both "follow detailed spec" and "figure it out from a vague request" capabilities.

**Experiment:**
- For each task, generate 3 granularity levels (tests identical across all three)
- **Baseline:** Single granularity level (current)
- **Treatments:** (a) Uniform mix, (b) Curriculum from detailed (c) -> vague (a) over training
- Evaluate on both well-specified and underspecified benchmarks

**Expected outcome:** Curriculum from detailed->vague outperforms all static approaches. Agent learns to solve tasks first, then learns to handle ambiguity.

**Effort:** Low-Medium | **Impact:** High

---

### 8.4 Constraint Ladder (Difficulty Escalation)

**Hypothesis:** Adding escalating constraints to the same base task -- +no external libs -> +single file -> +memory limit -> +time limit -> +specific API style -- creates a natural difficulty gradient without changing core logic.

**Motivation:** Real-world coding involves constraints (performance, style, compatibility). An agent trained only on unconstrained tasks struggles with constrained environments.

**Distinct from 7.1:** Dockerfile curriculum varies the *environment*; constraint ladder varies the *requirements*.

**Experiment:**
- For each task, generate 5 constraint levels (0 constraints = easy, 4 constraints = very hard)
- **Baseline:** Unconstrained tasks only
- **Treatments:** (a) Random constraint level, (b) Curriculum 0->4 over training, (c) Uniform mix
- Tests extended to verify constraint compliance

**Expected outcome:** Curriculum (0->4) outperforms static approaches. Agent learns core problem-solving first, then constraint satisfaction.

**Effort:** Medium | **Impact:** High

---

### 8.5 Adversarial Context Padding

**Hypothesis:** Adding irrelevant context (backstory, red herring requirements, verbose logs, unrelated code snippets) to instructions forces the agent to identify what actually matters, improving robustness to noisy real-world inputs.

**Motivation:** Real GitHub issues are 80% context / 20% actionable request. Agents trained on clean, minimal instructions are easily distracted by noise.

**Distinct from 7.2:** Information dropout *removes* useful info; context padding *adds* noise. They are complementary -- one trains robustness to missing info, the other to irrelevant info.

**Experiment:**
- For each task, add 3 levels of padding: (a) Mild (1-2 irrelevant sentences), (b) Moderate (paragraph of context + red herrings), (c) Heavy (verbose logs + unrelated code + misleading hints)
- Tests unchanged
- **Baseline:** Clean instructions. **Treatments:** (a) Single padding level, (b) Uniform mix, (c) Curriculum mild->heavy
- Difficulty knob: amount and subtlety of padding

**Expected outcome:** Context padding training improves performance on real-world GitHub issues by 5-15%. Curriculum ordering helps.

**Effort:** Low | **Impact:** Medium-High

---

### 8.6 Error-State Initialization (Broken Code Start)

**Hypothesis:** Seeding the workspace with partially broken code (instead of starting from empty) forces the agent to diagnose, fix, and extend existing code -- a harder and more realistic task.

**Motivation:** Real-world coding rarely starts from scratch. Debugging requires understanding someone else's code, which is fundamentally different from greenfield development. Most benchmarks don't test this.

**Experiment:**
- For each task, generate 3 starting states: (a) Empty (current), (b) Partially correct with 1-2 subtle bugs (off-by-one, wrong import, missing edge case), (c) Structurally correct but logically broken
- Test verifies final correctness regardless of starting state
- **Baseline:** Empty workspace. **Treatments:** (a) Broken start only, (b) Mix of empty + broken, (c) Curriculum empty->broken
- Difficulty knob: number and subtlety of seeded bugs

**Expected outcome:** Mixed training (empty + broken starts) produces agents significantly better at debugging tasks while maintaining greenfield performance.

**Effort:** Medium | **Impact:** High

---

### 8.7 Ambiguity Injection (Underspecified Requirements)

**Hypothesis:** Deliberately removing precise specification -- so that multiple valid solutions exist -- trains agents to make design decisions rather than just follow specs.

**Motivation:** Real tasks are often underspecified. An agent that can only follow exact specs fails when it must make design choices (data structure selection, error handling strategy, API design).

**Experiment:**
- For each task, create 3 ambiguity levels: (a) Fully specified (current), (b) Partially specified (some choices left open), (c) Minimal ("just make it work")
- Replace exact-output tests with property-based tests that accept any correct implementation
- **Baseline:** Fully specified. **Treatments:** (a) Mixed ambiguity levels, (b) Curriculum specified->ambiguous
- Difficulty: amount of ambiguity

**Expected outcome:** Ambiguity training improves performance on underspecified real-world benchmarks by 10-15%. Property-based tests are key enabler.

**Effort:** Medium | **Impact:** High

---

### 8.8 Domain-Shifted Task Cloning

**Hypothesis:** Transplanting algorithm-centric tasks into concrete application domains (e.g., "implement a parser" -> "parse patient records from HL7" / "parse financial transactions from MT940") exposes the agent to domain vocabulary and real-world framing.

**Motivation:** Pure algorithmic tasks don't prepare agents for domain-specific language (medical, financial, DevOps). Same underlying algorithm, different domain context.

**Distinct from 7.3:** Cross-language transplant changes the *programming language*; domain shift changes the *application domain* and vocabulary while keeping the same language.

**Experiment:**
- Select 1K algorithm-centric tasks, transplant each into 3-5 application domains
- Tests adapted to domain-specific I/O formats
- **Baseline:** Algorithm-centric tasks only. **Treatments:** (a) Single domain, (b) Multi-domain mix
- Evaluate on domain-specific benchmarks

**Expected outcome:** Multi-domain mix improves performance on real-world domain-specific tasks without hurting algorithmic performance. Effective task pool expansion 3-5x.

**Effort:** Low-Medium | **Impact:** Medium-High

---

### 8.9 Multi-Objective Task Stacking

**Hypothesis:** Combining 2-3 independent micro-tasks into one compound instruction (e.g., "fix the CSV parser AND add logging AND write a migration script") trains agents in project management and context-switching.

**Motivation:** Real-world development involves juggling multiple objectives simultaneously. Difficulty comes from managing multiple concerns, not individual task hardness.

**Experiment:**
- Compose compound tasks from existing micro-tasks (2-task, 3-task combinations)
- Tests check all objectives independently; partial credit natural
- **Baseline:** Single-objective tasks. **Treatments:** (a) 2-task compounds, (b) 3-task compounds, (c) Mix of single + compound
- Evaluate on complex multi-objective benchmarks

**Expected outcome:** Compound task training improves performance on multi-file, multi-objective benchmarks by 10-20%. Partial credit reward (Theme 2.1) is synergistic.

**Effort:** Medium | **Impact:** Medium-High

---

### 8.10 Prerequisite Knowledge Gradient

**Hypothesis:** Varying how much domain background the instruction provides -- from zero-context expert-level (assumes knowledge) to fully explained (spoon-fed algorithm) -- creates a natural difficulty gradient that trains adaptable agents.

**Motivation:** Real tasks vary wildly in how much context they provide. "Implement AES-128" assumes crypto knowledge; a detailed spec with pseudocode assumes nothing. Agents need to handle both.

**Experiment:**
- For each task, generate 3 knowledge levels:
  - Level 0: Expert ("Implement AES-128 encryption") -- assumes domain knowledge
  - Level 1: Intermediate ("Implement a block cipher with SubBytes, ShiftRows, MixColumns, AddRoundKey steps") -- explains algorithm
  - Level 2: Novice ("Encrypt data using the provided pseudocode") -- spoon-fed
- Tests identical; only instruction detail varies
- **Baseline:** Single knowledge level. **Treatments:** (a) Uniform mix, (b) Curriculum Level 2 -> Level 0
- Evaluate on tasks requiring varying domain expertise

**Expected outcome:** Curriculum (novice->expert) produces agents that handle both well-explained and terse expert-level instructions.

**Effort:** Medium | **Impact:** Medium-High

---

## Priority Ranking (Impact / Effort)

| Rank | Idea | Effort | Impact | Quick Win? |
|------|------|--------|--------|------------|
| 1 | **2.1** Dense rewards (partial test) | Low-Med | High | Yes |
| 2 | **8.1** Instruction style transfer | Low | High | Yes |
| 3 | **7.6** Temporal reward (speed bonus) | Low | Medium | Yes |
| 4 | **8.5** Adversarial context padding | Low | Med-High | Yes |
| 5 | **8.3** Decomposition granularity spectrum | Low-Med | High | Yes |
| 6 | **1.1** Difficulty-aware mixing | Medium | High | |
| 7 | **7.2** Instruction information dropout | Medium | High | |
| 8 | **7.4** Test-first task generation | Medium | High | |
| 9 | **8.2** Specification modality mixing | Medium | High | |
| 10 | **8.4** Constraint ladder | Medium | High | |
| 11 | **8.6** Error-state initialization | Medium | High | |
| 12 | **8.7** Ambiguity injection | Medium | High | |
| 13 | **5.1** LLM consistency check | Medium | Med-High | |
| 14 | **7.1** Dockerfile curriculum | Medium | High | |
| 15 | **7.3** Cross-language transplantation | Medium | Med-High | |
| 16 | **8.8** Domain-shifted task cloning | Low-Med | Med-High | |
| 17 | **8.9** Multi-objective task stacking | Medium | Med-High | |
| 18 | **8.10** Prerequisite knowledge gradient | Medium | Med-High | |
| 19 | **7.5** Staged reward decomposition | Med-High | High | |
| 20 | **1.3** Test quality scoring | High | High | Paper |
| 21 | **1.2** Instruction specificity study | Med-High | High | Paper |
| 22 | **5.3** Test synthesis for arbitrary code | Med-High | High | |
| 23 | **4.3** Skill distillation | Med-High | High | |

---

## Suggested First Experiments (Low-Hanging Fruit)

1. **Dense rewards (2.1)** -- modify test.sh templates to report fraction passed. Minimal code change, potentially large RL improvement.
2. **Instruction style transfer (8.1)** -- rewrite instructions in different communication formats (GitHub issue, Slack, SO question). One LLM call per variant, tests unchanged. Cheapest path to prompt diversity.
3. **Adversarial context padding (8.5)** -- add irrelevant context to instructions. Easy to generate, trains robustness to real-world noise.
4. **Decomposition granularity spectrum (8.3)** -- same task at 3 detail levels. Tests unchanged, cheap to generate, trains handling of vague requests.
5. **LLM consistency check (5.1)** -- filter broken instruction/test pairs. Improves RL stability.

## Suggested Paper-Worthy Projects

1. **Prompt diversity & difficulty control (Theme 8 combined)** -- comprehensive study of how varying instruction format, specificity, constraints, and noise affects RL-trained coding agents. Novel contribution: systematic taxonomy of prompt variation axes with controlled experiments. Clean story connecting to real-world distribution shift.
2. **Test quality for RL (1.3)** -- underexplored area, clean story about reward signal quality.
3. **Instruction specificity vs. generalization (1.2)** -- practical insight for the field on the tension between learnability and transfer.
4. **Error-state initialization (8.6)** -- novel training paradigm: debugging-first instead of greenfield-first. Connects to real-world developer experience.
