# A1 Dataset Documentation
# Generated: 2026-03-26
# Total datasets in CSV: 115
# Total sandbox repos in registry: 104
# Total generator scripts found: 196

## a1_agenttuning_alfworld
- Status: ON HF
- Trace dataset: DCAgent/neulab-agenttuning-alfworld-sandboxes_glm_4.7_traces_jupiter
- Trace count: 11117
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-agenttuning-alfworld-sandboxes
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_abstract.py
    -> DCAgent/neulab-agenttuning-alfworld-sandboxes
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_agenttuning-alfworld.py
    -> DCAgent/neulab-agenttuning-alfworld-sandboxes

## a1_agenttuning_db
- Status: ON HF
- Trace dataset: DCAgent/neulab-agenttuning-db-sandboxes_glm_4.7_traces_jupiter
- Trace count: 10001
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-agenttuning-db-sandboxes
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_agenttuning-db.py
    -> DCAgent/neulab-agenttuning-db-sandboxes

## a1_agenttuning_kg
- Status: ON HF
- Trace dataset: DCAgent/neulab-agenttuning-kg-sandboxes_glm_4.7_traces_jupiter
- Trace count: 10413
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-agenttuning-kg-sandboxes
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_agenttuning-kg.py
    -> DCAgent/neulab-agenttuning-kg-sandboxes

## a1_agenttuning_mind2web
- Status: ON HF
- Trace dataset: DCAgent/neulab-agenttuning-mind2web-sandboxes_glm_4.7_traces_jupiter
- Trace count: 9999
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-agenttuning-mind2web-sandboxes
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_agenttuning-mind2web.py
    -> DCAgent/neulab-agenttuning-mind2web-sandboxes

## a1_agenttuning_os
- Status: ON HF
- Trace dataset: DCAgent/neulab-agenttuning-os-sandboxes_glm_4.7_traces_jupiter
- Trace count: 10000
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-agenttuning-os-sandboxes
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_agenttuning-os.py
    -> DCAgent/neulab-agenttuning-os-sandboxes

## a1_agenttuning_webshop
- Status: ON HF
- Trace dataset: DCAgent/neulab-agenttuning-webshop-sandboxes_glm_4.7_traces_jupiter
- Trace count: 9178
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-agenttuning-webshop-sandboxes
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_agenttuning-webshop.py
    -> DCAgent/neulab-agenttuning-webshop-sandboxes

## a1_all_puzzles
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/All_Puzzles_5k-sandboxes
- Task count: 5000
- Sandbox dataset: DCAgent/All_Puzzles_5k-sandboxes
- Notes: needs upsample to 10k
- Generator: NOT FOUND

## a1_bash_textbook
- Status: IN QUEUE
- Trace dataset: DCAgent/bash_textbook_tasks
- Task count: 10000
- Sandbox dataset: DCAgent/bash_textbook_tasks
- Notes: sandbox ready - 1 env (f1a4b41235ee)
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_stack-bash
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_nemotron-bash
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/bash_textbook/generate.py
    -> DCAgent/bash_textbook_tasks

## a1_bespoke
- Status: NEEDS SANDBOX REGEN
- Trace dataset: mlfoundations-dev/bespokelabs-sky-t1-*
- Sandbox dataset: 
- Notes: no sandboxes found
- Generator: NOT FOUND

## a1_bigcodebench
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/exp_rpt_bigcodebench-10k
- Task count: N/A
- Sandbox dataset: DCAgent/exp_rpt_bigcodebench
- Notes: HF dataset generation error
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bigcodebench_tasks.py

## a1_bugsinpy
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_bugsinpy-v4_10k_glm_4.7_traces_jupiter
- Trace count: 12622
- Sandbox dataset: DCAgent/exp_rpt_bugsinpy
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bugsinpy_mf_tasks.py
    -> DCAgent/exp_rpt_bugsinpy-mf
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bugsinpy_tasks.py
    -> DCAgent/exp_rpt_bugsinpy

## a1_bugsinpy_mf
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/exp_rpt_bugsinpy-mf-v3
- Task count: 500
- Sandbox dataset: DCAgent/exp_rpt_bugsinpy-mf
- Notes: 1439 unique envs - needs single-env regen
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bugsinpy_mf_tasks.py
    -> DCAgent/exp_rpt_bugsinpy-mf
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bugsinpy_tasks.py
    -> DCAgent/exp_rpt_bugsinpy

## a1_bugswarm
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_bugswarm_10k_glm_4.7_traces_jupiter
- Trace count: 5330
- Sandbox dataset: DCAgent/exp_rpt_bugswarm
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bugswarm_tasks.py

## a1_code_contests
- Status: IN QUEUE
- Trace dataset: DCAgent/code-contests-sandboxes-with-tests
- Task count: 9644
- Sandbox dataset: DCAgent/code-contests-sandboxes-with-tests
- Notes: sandbox ready - 1 env (6284aa0dd1b9) needs snapshot
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/code_contests/generate_new_code_contests_questions.py
    -> DCAgent/code_contests_new_questions_gpt-5-mini-sandboxes
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/code_contests/generate_with_tests.py
    -> DCAgent/code-contests-sandboxes-with-tests
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/llm_verifier/generate_code_contests.py
    -> DCAgent/exp_llmve_llm-verifier-code-contests

## a1_code_feedback
- Status: ON HF
- Trace dataset: DCAgent/neulab-code-feedback-sandboxes_glm_4.7_traces_jupiter
- Trace count: 9925
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-code-feedback-sandboxes
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_code-feedback.py
    -> DCAgent/neulab-code-feedback-sandboxes

## a1_codeactinstruct
- Status: ON HF
- Trace dataset: DCAgent/neulab-codeactinstruct-sandboxes_glm_4.7_traces_jupiter
- Trace count: 10651
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-codeactinstruct-sandboxes
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_codeactinstruct.py
    -> DCAgent/neulab-codeactinstruct-sandboxes

## a1_codeelo
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_codeelo-v2_10k_glm_4.7_traces_jupiter
- Trace count: 6610
- Sandbox dataset: DCAgent/exp_rpt_codeelo-v2
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_codeelo_tasks.py

## a1_codeforces
- Status: IN QUEUE
- Trace dataset: mlfoundations-dev/codeforces-sandboxes-1
- Task count: 9957
- Sandbox dataset: mlfoundations-dev/codeforces-sandboxes-1
- Notes: correct sandbox format
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/codeforces/generate.py
    -> mlfoundations-dev/codeforces-sandboxes-1
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/codeforces/generate_abstract.py
    -> mlfoundations-dev/codeforces-sandboxes-1

## a1_codenet_python
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_codenet-python_glm_4.7_traces_jupiter
- Trace count: 9912
- Sandbox dataset: DCAgent/exp_rpt_codenet-python
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_codenet_tasks.py

## a1_codereval
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/exp_rpt_codereval-python-v2
- Task count: 230
- Sandbox dataset: DCAgent/exp_rpt_codereval-python
- Notes: needs upsample to 10k
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_codereval_tasks.py

## a1_codex_chats
- Status: NEEDS SANDBOX REGEN
- Trace dataset: 
- Sandbox dataset: 
- Notes: no sandbox on HF
- Generator: NOT FOUND

## a1_crosscodeeval
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_crosscodeeval-{python/java/csharp/typescript}
- Sandbox dataset: 
- Notes: covered by per-language variants below
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_crosscodeeval_tasks.py

## a1_crosscodeeval_csharp
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_crosscodeeval-csharp_10k_glm_4.7_traces_jupiter
- Trace count: 10023
- Sandbox dataset: DCAgent/exp_rpt_crosscodeeval-csharp
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_crosscodeeval_tasks.py
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_csharp_test_tasks.py
    -> DCAgent/exp_rpt_stack-csharp-v2-test
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_csharp_test_tasks.py
    -> DCAgent/exp_rpt_nemotron-csharp

## a1_crosscodeeval_java
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_crosscodeeval-java_10k_glm_4.7_traces_jupiter
- Trace count: 10010
- Sandbox dataset: DCAgent/exp_rpt_crosscodeeval-java
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_crosscodeeval_tasks.py

## a1_crosscodeeval_python
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_crosscodeeval-python-v2_10k_glm_4.7_traces_jupiter
- Trace count: 9635
- Sandbox dataset: DCAgent/exp_rpt_crosscodeeval-python-v2
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_crosscodeeval_tasks.py

## a1_crosscodeeval_typescript
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_crosscodeeval-typescript_10k_glm_4.7_traces_jupiter
- Trace count: 11476
- Sandbox dataset: DCAgent/exp_rpt_crosscodeeval-typescript
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_crosscodeeval_tasks.py

## a1_curriculum_variants
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_curriculum-{easy/medium/hard}_10k_glm_4.7_traces_jupiter
- Sandbox dataset: DCAgent/exp_rpt_curriculum-easy
- Notes: 3 variants
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_curriculum_tasks.py

## a1_defects4j
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_defects4j-v3_10k_glm_4.7_traces_jupiter
- Trace count: 10473
- Task count: 10000
- Sandbox dataset: DCAgent/exp_rpt_defects4j
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_defects4j_tasks.py
    -> DCAgent/exp_rpt_defects4j

## a1_e2egit
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/exp_rpt_e2egit-large
- Task count: 5000
- Sandbox dataset: DCAgent/exp_rpt_e2egit
- Notes: needs upsample to 10k
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_e2egit_tasks.py

## a1_exercism_python
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_exercism-python_10k_glm_4.7_traces_jupiter
- Trace count: 10156
- Sandbox dataset: DCAgent/exp_rpt_exercism-python
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_exercism_tasks.py

## a1_freelancer
- Status: ON HF
- Trace dataset: DCAgent/perturbed-docker-exp-freelancer-tasks_glm_4.7_traces
- Trace count: 4626
- Sandbox dataset: DCAgent/perturbed-docker-exp-freelancer-tasks
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/claude_docker/generate_freelancer.py
    -> DCAgent/claude-docker-exp-freelancer-traces
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/github_dockerfiles/generate_freelancer.py
    -> DCAgent/github-dockerfiles-docker-exp-freelancer-tasks
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/llm_verifier/generate_freelancer.py
    -> DCAgent/exp_llmve_llm-verifier-freelancer-sandboxes

## a1_ghactions
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_ghactions_glm_4.7_traces_jupiter
- Trace count: 7933
- Sandbox dataset: DCAgent/exp_rpt_ghactions
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_ghactions_tasks.py
    -> DCAgent/exp_rpt_ghactions

## a1_github_dockerfiles
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/github-dockerfiles-docker-exp-taskmaster2-tasks
- Task count: N/A
- Sandbox dataset: DCAgent/github-dockerfiles-docker-exp-taskmaster2-tasks
- Notes: no glm traces
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_dockerfile_tasks.py
    -> DCAgent/exp_rpt_stack-dockerfile
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/github_dockerfiles/generate_taskmaster.py
    -> DCAgent/github-dockerfiles-docker-exp-taskmaster2-tasks

## a1_glaive_code_assistant
- Status: ON HF
- Trace dataset: DCAgent/glaive-code-assistant-sandboxes_glm_4.7_traces_jupiter
- Trace count: 9801
- Task count: 10000
- Sandbox dataset: DCAgent/glaive-code-assistant-sandboxes
- Generator: NOT FOUND

## a1_go_browse_wa
- Status: ON HF
- Trace dataset: DCAgent/neulab-go-browse-wa-sandboxes_glm_4.7_traces_jupiter
- Trace count: 10735
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-go-browse-wa-sandboxes
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_go_test_tasks.py
    -> DCAgent/exp_rpt_stack-go-v3-test
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_go-browse-wa.py
    -> DCAgent/neulab-go-browse-wa-sandboxes

## a1_gsm8k
- Status: NEEDS SANDBOX REGEN
- Trace dataset: mlfoundations-dev/gsm8k_sandboxes
- Task count: 300
- Sandbox dataset: 
- Notes: wrong format (raw dirs not parquet) ~300 tasks
- Generator: NOT FOUND

## a1_inferredbugs
- Status: IN QUEUE
- Trace dataset: mlfoundations-dev/inferredbugs-sandboxes
- Task count: 10000
- Sandbox dataset: mlfoundations-dev/inferredbugs-sandboxes
- Notes: correct sandbox format
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/inferredbugs/generate.py
    -> mlfoundations-dev/inferredbugs-sandboxes
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/inferredbugs/generate_abstract.py
    -> mlfoundations-dev/inferredbugs-sandboxes

## a1_issue_tasks
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_issue_10k_glm_4.7_traces_jupiter
- Trace count: 10437
- Sandbox dataset: DCAgent/exp_rpt_issue
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_issue_tasks.py
    -> DCAgent/exp_rpt_issue

## a1_magicoder
- Status: ON HF
- Trace dataset: DCAgent/perturbed-docker-exp-magicoder-tasks-2_glm_4.7_traces_jupiter
- Trace count: 7820
- Sandbox dataset: DCAgent/perturbed-docker-exp-magicoder-tasks-2
- Generator: NOT FOUND

## a1_manybugs
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_manybugs-v2_10k_glm_4.7_traces_jupiter
- Trace count: 10314
- Sandbox dataset: DCAgent/exp_rpt_manybugs
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_manybugs_tasks.py
    -> DCAgent/exp_rpt_manybugs

## a1_methods2test
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/exp_rpt_methods2test-large
- Task count: 5000
- Sandbox dataset: DCAgent/exp_rpt_methods2test
- Notes: needs upsample to 10k
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_methods2test_tasks.py
    -> DCAgent/exp_rpt_methods2test
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_pymethods2test_tasks.py
    -> DCAgent/exp_rpt_pymethods2test

## a1_mind2web
- Status: ON HF
- Trace dataset: DCAgent/neulab-mind2web-sandboxes_glm_4.7_traces_jupiter
- Trace count: 10796
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-mind2web-sandboxes
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_agenttuning-mind2web.py
    -> DCAgent/neulab-agenttuning-mind2web-sandboxes
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_mind2web.py
    -> DCAgent/neulab-mind2web-sandboxes

## a1_multifile_composition
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_multifile_10k_glm_4.7_traces_jupiter
- Trace count: 9970
- Sandbox dataset: DCAgent/exp_rpt_multifile
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_multifile_tasks.py
    -> DCAgent/exp_rpt_multifile

## a1_nebius_swe_agent
- Status: ON HF
- Trace dataset: DCAgent/neulab-nebius-swe-agent-trajectories-sandboxes_glm_4.7_traces_jupiter
- Trace count: 12410
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-nebius-swe-agent-trajectories-sandboxes
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_nebius-swe-agent-trajectories.py
    -> DCAgent/neulab-nebius-swe-agent-trajectories-sandboxes

## a1_nemo_prism_math
- Status: ON HF
- Trace dataset: DCAgent/nemo-prism-math-sandboxes_glm_4.7_traces_jupiter
- Trace count: 8789
- Task count: 10000
- Sandbox dataset: DCAgent/nemo-prism-math-sandboxes
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_pr_tasks.py
    -> DCAgent/exp_rpt_pr

## a1_nemo_prism_math_v2
- Status: NEEDS SANDBOX REGEN
- Trace dataset: 
- Sandbox dataset: DCAgent/nemo-prism-math-sandboxes
- Notes: no sandbox on HF
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_pr_tasks.py
    -> DCAgent/exp_rpt_pr

## a1_nemotron_bash
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_nemotron-bash-v2_10k_glm_4.7_traces_jupiter
- Trace count: 10165
- Sandbox dataset: DCAgent/exp_rpt_nemotron-bash
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_stack-bash
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_nemotron-bash

## a1_nemotron_bash_withtests
- Status: NEEDS SANDBOX REGEN
- Trace dataset: 
- Sandbox dataset: DCAgent/exp_rpt_nemotron-bash-withtests
- Notes: no sandbox on HF
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_stack-bash
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_nemotron-bash
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_bash_tasks_with_tests.py
    -> DCAgent/exp_rpt_nemotron-bash-withtests

## a1_nemotron_bash_withtests_gpt5mini
- Status: NEEDS SANDBOX REGEN
- Trace dataset: 
- Sandbox dataset: DCAgent/exp_rpt_nemotron-bash-withtests-gpt5mini
- Notes: no sandbox on HF
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_stack-bash
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_nemotron-bash
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_bash_tasks_with_tests_gpt5mini.py
    -> DCAgent/exp_rpt_nemotron-bash-withtests-gpt5mini

## a1_nemotron_cpp
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_nemotron-cpp_10k_glm_4.7_traces_jupiter
- Trace count: 10160
- Sandbox dataset: DCAgent/exp_rpt_nemotron-cpp
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_cpp_test_tasks.py
    -> DCAgent/exp_rpt_stack-cpp-v2-test
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_cpp_test_tasks.py
    -> DCAgent/exp_rpt_nemotron-cpp

## a1_nemotron_csharp
- Status: NEEDS SANDBOX REGEN
- Trace dataset: 
- Sandbox dataset: DCAgent/exp_rpt_nemotron-csharp
- Notes: no sandbox on HF
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_csharp_test_tasks.py
    -> DCAgent/exp_rpt_stack-csharp-v2-test
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_csharp_test_tasks.py
    -> DCAgent/exp_rpt_nemotron-csharp

## a1_nemotron_junit
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_nemotron-junit_10k_glm_4.7_traces_jupiter
- Trace count: 10387
- Sandbox dataset: DCAgent/exp_rpt_nemotron-junit
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_junit_tasks.py
    -> DCAgent/exp_rpt_stack-junit
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_junit_tasks.py
    -> DCAgent/exp_rpt_nemotron-junit

## a1_nemotron_pytest
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_nemotron-pytest-gpt5mini-v2_10k_glm_4.7_traces_jupiter
- Trace count: 9615
- Task count: 10000
- Sandbox dataset: DCAgent/exp_rpt_nemotron-pytest-gpt5mini
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_pytest_tasks.py
    -> DCAgent/exp_rpt_stack-pytest
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_pytest_tasks_gpt5mini.py
    -> DCAgent/exp_rpt_nemotron-pytest-gpt5mini

## a1_nemotron_rspec
- Status: NEEDS SANDBOX REGEN
- Trace dataset: 
- Sandbox dataset: DCAgent/exp_rpt_nemotron-ruby
- Notes: no sandbox on HF
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_rspec_tasks.py
    -> DCAgent/exp_rpt_stack-ruby
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_rspec_tasks.py
    -> DCAgent/exp_rpt_nemotron-ruby

## a1_nemotron_rust
- Status: NEEDS SANDBOX REGEN
- Trace dataset: 
- Sandbox dataset: DCAgent/exp_rpt_nemotron-rust
- Notes: no sandbox on HF
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_rust_test_tasks.py
    -> DCAgent/exp_rpt_stack-rust-v3-test
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_rust_test_tasks.py
    -> DCAgent/exp_rpt_nemotron-rust

## a1_nl2bash
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/nl2bash
- Task count: 9230
- Sandbox dataset: DCAgent/nl2bash
- Notes: sandbox has 9230 tasks - needs upsample to 10k
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_stack-bash
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_nemotron-bash
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/nl2bash_etash/generate.py
    -> DCAgent/nl2bash

## a1_nnetnav_live
- Status: ON HF
- Trace dataset: DCAgent/neulab-nnetnav-live-sandboxes_glm_4.7_traces_jupiter
- Trace count: 11802
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-nnetnav-live-sandboxes
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_nnetnav-live.py
    -> DCAgent/neulab-nnetnav-live-sandboxes

## a1_nnetnav_wa
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/neulab-nnetnav-wa-sandboxes
- Sandbox dataset: DCAgent/neulab-nnetnav-wa-sandboxes
- Notes: broken/empty parquet
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_nnetnav-wa.py
    -> DCAgent/neulab-nnetnav-wa-sandboxes

## a1_openhands
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/neulab-openhands-sandboxes
- Sandbox dataset: DCAgent/neulab-openhands-sandboxes
- Notes: broken/empty parquet
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_openhands.py
    -> DCAgent/neulab-openhands-sandboxes
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_swe-gym-openhands-sampled-trajectories.py
    -> DCAgent/neulab-swe-gym-openhands-sampled-trajectories-sandboxes

## a1_orca_agentinstruct
- Status: ON HF
- Trace dataset: DCAgent/neulab-orca-agentinstruct-sandboxes_glm_4.7_traces_jupiter
- Trace count: 10029
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-orca-agentinstruct-sandboxes
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_orca-agentinstruct.py
    -> DCAgent/neulab-orca-agentinstruct-sandboxes

## a1_pr_mining
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_pr_10k_glm_4.7_traces_jupiter
- Trace count: 10333
- Sandbox dataset: DCAgent/exp_rpt_pr
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_pr_tasks.py
    -> DCAgent/exp_rpt_pr

## a1_pymethods2test
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_pymethods2test-v3_10k_glm_4.7_traces_jupiter
- Trace count: 9155
- Task count: 10000
- Sandbox dataset: DCAgent/exp_rpt_pymethods2test
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_methods2test_tasks.py
    -> DCAgent/exp_rpt_methods2test
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_pymethods2test_tasks.py
    -> DCAgent/exp_rpt_pymethods2test

## a1_qasper
- Status: IN QUEUE
- Trace dataset: mlfoundations-dev/qasper-sandboxes
- Task count: 10000
- Sandbox dataset: mlfoundations-dev/qasper-sandboxes
- Notes: correct sandbox format
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/qasper/generate.py
    -> mlfoundations-dev/qasper-sandboxes

## a1_quixbugs
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/exp_rpt_quixbugs-python-10k
- Task count: N/A
- Sandbox dataset: DCAgent/exp_rpt_quixbugs-python
- Notes: broken/empty parquet
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_quixbugs_tasks.py

## a1_r2egym
- Status: ON HF
- Trace dataset: DCAgent/exp-swd-r2egym-standard_glm_4.7_traces_locetash
- Trace count: 4578
- Sandbox dataset: DCAgent/exp-swd-r2egym-standard
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/swe_without_docker/generate_r2egym_standard.py
    -> DCAgent/exp-swd-r2egym-standard
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/swe_without_docker/generate_r2egym_wo_docker.py
    -> DCAgent/exp-swd-r2egym-wo-docker

## a1_refactoring_tasks
- Status: NEEDS SANDBOX REGEN
- Trace dataset: 
- Sandbox dataset: DCAgent/exp_rpt_refactor
- Notes: no sandbox on HF
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_refactoring_tasks.py
    -> DCAgent/exp_rpt_refactor

## a1_repo_scaffold
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_scaffold_10k_glm_4.7_traces_jupiter
- Trace count: 10262
- Sandbox dataset: DCAgent/exp_rpt_scaffold
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_scaffold_tasks.py
    -> DCAgent/exp_rpt_scaffold

## a1_sampled_swebench_verified
- Status: IN QUEUE
- Trace dataset: mlfoundations-dev/swebench-verified-sandboxes
- Task count: 500
- Sandbox dataset: mlfoundations-dev/swebench-verified-sandboxes
- Notes: no glm traces
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_swebench_tasks.py

## a1_self_instruct
- Status: IN QUEUE
- Trace dataset: DCAgent/selfinstruct-naive-sandboxes-1
- Task count: 9864
- Sandbox dataset: DCAgent/selfinstruct-naive-sandboxes-1
- Notes: no glm traces
- Generator: NOT FOUND

## a1_self_instruct_naive
- Status: IN QUEUE
- Trace dataset: DCAgent/selfinstruct-naive-sandboxes-2
- Task count: 9638
- Sandbox dataset: DCAgent/selfinstruct-naive-sandboxes-2
- Notes: no glm traces
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/self_instruct_naive/generate_v2_0.py
    -> DCAgent/selfinstruct-naive-sandboxes-2

## a1_softwareheritage
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/exp_rpt_softwareheritage-large
- Task count: 4998
- Sandbox dataset: DCAgent/exp_rpt_softwareheritage
- Notes: needs upsample to 10k
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_softwareheritage_tasks.py
    -> DCAgent/exp_rpt_softwareheritage

## a1_stack_bash
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_stack-bash_glm_4.7_traces_jupiter
- Trace count: 12726
- Sandbox dataset: DCAgent/exp_rpt_stack-bash
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_stack-bash
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_nemotron-bash

## a1_stack_bash_withtests
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_stack-bash-withtests_glm_4.7_traces_jupiter
- Trace count: 17624
- Sandbox dataset: DCAgent/exp_rpt_stack-bash-withtests
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_stack-bash
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bash_tasks_with_tests.py
    -> DCAgent/exp_rpt_stack-bash-withtests
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_nemotron-bash

## a1_stack_bash_withtests_gpt4o
- Status: IN QUEUE
- Trace dataset: 
- Sandbox dataset: 
- Notes: commercial model variant
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_stack-bash
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_nemotron-bash

## a1_stack_bash_withtests_gpt5mini
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_stack-bash-withtests-gpt5mini_glm_4.7_traces_jupiter
- Trace count: 13960
- Task count: 10000
- Sandbox dataset: DCAgent/exp_rpt_stack-bash-withtests-gpt5mini
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_stack-bash
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_bash_tasks_with_tests_gpt5mini.py
    -> DCAgent/exp_rpt_stack-bash-withtests-gpt5mini
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_bash_tasks.py
    -> DCAgent/exp_rpt_nemotron-bash

## a1_stack_cpp
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_stack-cpp_10k_glm_4.7_traces_jupiter
- Trace count: 10536
- Sandbox dataset: DCAgent/exp_rpt_stack-cpp-v2-test
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_cpp_test_tasks.py
    -> DCAgent/exp_rpt_stack-cpp-v2-test
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_cpp_test_tasks.py
    -> DCAgent/exp_rpt_nemotron-cpp

## a1_stack_csharp
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_stack-csharp_10k_glm_4.7_traces_jupiter
- Trace count: 17499
- Sandbox dataset: DCAgent/exp_rpt_stack-csharp-v2-test
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_csharp_test_tasks.py
    -> DCAgent/exp_rpt_stack-csharp-v2-test
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_csharp_test_tasks.py
    -> DCAgent/exp_rpt_nemotron-csharp

## a1_stack_dockerfile
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/exp_rpt_stack-dockerfile
- Task count: N/A
- Sandbox dataset: DCAgent/exp_rpt_stack-dockerfile
- Notes: broken/empty parquet
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_dockerfile_tasks.py
    -> DCAgent/exp_rpt_stack-dockerfile

## a1_stack_dockerfile_gpt4o
- Status: IN QUEUE
- Trace dataset: 
- Sandbox dataset: 
- Notes: commercial model variant
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_dockerfile_tasks.py
    -> DCAgent/exp_rpt_stack-dockerfile

## a1_stack_dockerfile_gpt5mini
- Status: IN QUEUE
- Trace dataset: DCAgent/exp_rpt_stack-dockerfile-gpt5mini
- Task count: 10000
- Sandbox dataset: 
- Notes: 10000 unique envs - cannot run
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_dockerfile_tasks.py
    -> DCAgent/exp_rpt_stack-dockerfile
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_dockerfile_tasks_gpt5mini.py
    -> DCAgent/exp_rpt_stack-dockerfile-gpt4o

## a1_stack_go
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/exp_rpt_stack-go
- Sandbox dataset: DCAgent/exp_rpt_stack-go-v3-test
- Notes: missing script cmd in container
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_go_test_tasks.py
    -> DCAgent/exp_rpt_stack-go-v3-test

## a1_stack_jest
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_stack-jest-large_10k_glm_4.7_traces_jupiter
- Trace count: 10440
- Sandbox dataset: DCAgent/exp_rpt_stack-jest
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_jest_tasks.py
    -> DCAgent/exp_rpt_stack-jest

## a1_stack_junit
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_stack-junit_glm_4.7_traces_jupiter
- Trace count: 12930
- Sandbox dataset: DCAgent/exp_rpt_stack-junit
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_junit_tasks.py
    -> DCAgent/exp_rpt_stack-junit
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_junit_tasks.py
    -> DCAgent/exp_rpt_nemotron-junit

## a1_stack_phpunit
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_stack-php-large_10k_glm_4.7_traces_jupiter
- Trace count: 13148
- Task count: 10000
- Sandbox dataset: DCAgent/exp_rpt_stack-php
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_phpunit_tasks.py
    -> DCAgent/exp_rpt_stack-php

## a1_stack_pytest
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_stack-pytest-large_10k_glm_4.7_traces_jupiter
- Trace count: 10270
- Sandbox dataset: DCAgent/exp_rpt_stack-pytest
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_pytest_tasks.py
    -> DCAgent/exp_rpt_stack-pytest

## a1_stack_pytest_gpt5mini
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_stack-pytest-gpt5mini_glm_4.7_traces_jupiter
- Trace count: 8563
- Task count: 10000
- Sandbox dataset: DCAgent/exp_rpt_stack-pytest-gpt5mini
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_pytest_tasks.py
    -> DCAgent/exp_rpt_stack-pytest
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_pytest_tasks_gpt5mini.py
    -> DCAgent/exp_rpt_stack-pytest-gpt5mini
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_pytest_tasks_gpt5mini.py
    -> DCAgent/exp_rpt_nemotron-pytest-gpt5mini

## a1_stack_pytest_synthetic_gpt5nano
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_stack-pytest-synthetic-gpt5nano_glm_4.7_traces_jupiter
- Trace count: 10397
- Task count: 10000
- Sandbox dataset: DCAgent/exp_rpt_stack-pytest-v2
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_pytest_tasks.py
    -> DCAgent/exp_rpt_stack-pytest

## a1_stack_pytest_withtests
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_stack-pytest-withtests_glm_4.7_traces_jupiter
- Trace count: 12603
- Task count: 10000
- Sandbox dataset: DCAgent/exp_rpt_stack-pytest-withtests
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_pytest_tasks.py
    -> DCAgent/exp_rpt_stack-pytest
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_pytest_tasks_with_tests.py
    -> DCAgent/exp_rpt_stack-pytest-withtests

## a1_stack_rspec
- Status: NEEDS SANDBOX REGEN
- Trace dataset: 
- Sandbox dataset: 
- Notes: no sandbox on HF
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_rspec_tasks.py
    -> DCAgent/exp_rpt_stack-ruby
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_rspec_tasks.py
    -> DCAgent/exp_rpt_nemotron-ruby

## a1_stack_ruby
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_stack-ruby_glm_4.7_traces_jupiter
- Trace count: 10499
- Sandbox dataset: DCAgent/exp_rpt_stack-ruby
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_rspec_tasks.py
    -> DCAgent/exp_rpt_stack-ruby

## a1_stack_rust
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_stack-rust_10k_glm_4.7_traces_jupiter
- Trace count: 10427
- Sandbox dataset: DCAgent/exp_rpt_stack-rust-v3-test
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_rust_test_tasks.py
    -> DCAgent/exp_rpt_stack-rust-v3-test
  - /e/scratch/jureap59/raoof1/datagen_repo/data/nemotron-mine/generate_rust_test_tasks.py
    -> DCAgent/exp_rpt_nemotron-rust

## a1_stack_selfdoc
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/exp_rpt_stack-selfdoc-large
- Task count: 4999
- Sandbox dataset: DCAgent/exp_rpt_stack-selfdoc
- Notes: needs upsample to 10k
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_self_documented_tasks.py
    -> DCAgent/exp_rpt_stack-selfdoc
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_selfdoc_test.py
    -> DCAgent/exp_rpt_stack-selfdoc-v2-test

## a1_stack_selfdoc_gpt4o
- Status: IN QUEUE
- Trace dataset: 
- Sandbox dataset: 
- Notes: commercial model variant
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_selfdoc_test.py
    -> DCAgent/exp_rpt_stack-selfdoc-v2-test

## a1_stack_selfdoc_gpt5mini
- Status: IN QUEUE
- Trace dataset: DCAgent/exp_rpt_stack-selfdoc-gpt5mini
- Task count: 10000
- Sandbox dataset: 
- Notes: job 296757 - not uploaded yet
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_selfdoc_test.py
    -> DCAgent/exp_rpt_stack-selfdoc-v2-test

## a1_stackexchange_codereview
- Status: NEEDS SANDBOX REGEN
- Trace dataset: mlfoundations-dev/stackexchange_codereview
- Task count: 50000
- Sandbox dataset: DCAgent/stackexchange-codereview-sandboxes
- Notes: wrong parquet format (no task_binary)
- Generator: NOT FOUND

## a1_stackexchange_overflow
- Status: ON HF
- Trace dataset: DCAgent/stackexchange-overflow-sandboxes-skywork_glm_4.7_traces_jupiter
- Trace count: 10150
- Sandbox dataset: DCAgent/stackexchange-overflow-sandboxes-skywork
- Generator: NOT FOUND

## a1_stackexchange_superuser
- Status: ON HF
- Trace dataset: DCAgent/stackexchange-superuser-sandboxes_glm_4.7_traces_jupiter
- Trace count: 10120
- Task count: 10000
- Sandbox dataset: DCAgent/stackexchange-superuser-sandboxes
- Generator: NOT FOUND

## a1_stackexchange_tezos
- Status: ON HF
- Trace dataset: DCAgent/stackexchange-tezos-sandboxes_glm_4.7_traces_locetash
- Trace count: 9955
- Sandbox dataset: DCAgent/stackexchange-tezos-sandboxes
- Generator: NOT FOUND

## a1_stackexchange_tor
- Status: ON HF
- Trace dataset: DCAgent/stackexchange-tor-sandboxes_glm_4.7_traces_jupiter
- Trace count: 10505
- Task count: 10000
- Sandbox dataset: DCAgent/stackexchange-tor-sandboxes
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/stackexchange/generate_tor.py
    -> DCAgent/stackexchange-tor-sandboxes

## a1_stackexchange_unix
- Status: ON HF
- Trace dataset: DCAgent/stackexchange-unix-sandboxes_glm_4.7_traces_jupiter
- Trace count: 10154
- Task count: 10000
- Sandbox dataset: DCAgent/stackexchange-unix-sandboxes
- Generator: NOT FOUND

## a1_staqc
- Status: ON HF
- Trace dataset: DCAgent/exp-gfi-staqc-askllm-filtered-10K_glm_4.7_traces_jupiter
- Trace count: 9954
- Sandbox dataset: DCAgent/exp-gfi-staqc-askllm-filtered-10K
- Notes: multiple variants
- Generator: NOT FOUND

## a1_swebench
- Status: SKIPPED
- Trace dataset: mlfoundations-dev/swebench-verified-sandboxes
- Task count: 500
- Sandbox dataset: mlfoundations-dev/swebench-verified-sandboxes
- Notes: eval-only benchmark
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_swebench_tasks.py

## a1_swegym
- Status: NEEDS SANDBOX REGEN
- Trace dataset: mlfoundations-dev/swe_gym
- Sandbox dataset: 
- Notes: raw; no sandboxes
- Generator: NOT FOUND

## a1_swegym_openhands
- Status: ON HF
- Trace dataset: DCAgent/neulab-swe-gym-openhands-sampled-trajectories-sandboxes_glm_4.7_traces_jupiter
- Trace count: 11254
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-swe-gym-openhands-sampled-trajectories-sandboxes
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_swe-gym-openhands-sampled-trajectories.py
    -> DCAgent/neulab-swe-gym-openhands-sampled-trajectories-sandboxes

## a1_swesmith
- Status: ON HF
- Trace dataset: DCAgent/swesmith-sandboxes-with_tests-gpt-5-mini-passed_glm_4.7_traces
- Trace count: 7173
- Sandbox dataset: DCAgent/swesmith-sandboxes-with_tests
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/swe_without_docker/generate_swesmith_standard.py
    -> DCAgent/exp-swd-swesmith-standard
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/swe_without_docker/generate_swesmith_wo_docker.py
    -> DCAgent/exp-swd-swesmith-wo-docker
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/unique_scale/generate_swesmith.py
    -> DCAgent/exp-swd-r2egym-wo-docker

## a1_swesmith_5k_trajectories
- Status: NEEDS SANDBOX REGEN
- Trace dataset: 
- Sandbox dataset: DCAgent/neulab-swe-smith-5ktrajectories-sandboxes
- Notes: no sandbox on HF
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_swe-smith-5ktrajectories.py
    -> DCAgent/neulab-swe-smith-5ktrajectories-sandboxes

## a1_synatra
- Status: IN QUEUE
- Trace dataset: DCAgent/neulab-synatra-sandboxes
- Task count: 10000
- Sandbox dataset: DCAgent/neulab-synatra-sandboxes
- Notes: correct sandbox format
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/neulab/generate_synatra.py
    -> DCAgent/neulab-synatra-sandboxes

## a1_taco
- Status: ON HF
- Trace dataset: DCAgent/exp_rpt_taco_glm_4.7_traces_jupiter
- Trace count: 8439
- Sandbox dataset: DCAgent/exp_rpt_taco
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_taco_tasks.py
    -> DCAgent/exp_rpt_taco

## a1_taskmaster2
- Status: ON HF
- Trace dataset: DCAgent/perturbed-docker-exp-taskmaster2-tasks_glm_4.7_traces_locetash
- Trace count: 9510
- Sandbox dataset: DCAgent/perturbed-docker-exp-taskmaster2-tasks
- Generator(s):
  - /e/scratch/jureap59/guha1/OpenThoughts-Agent/data/perturbed_docker/generate.py
    -> DCAgent/perturbed-docker-exp-taskmaster2-tasks

## a1_toolscale
- Status: NEEDS SANDBOX REGEN
- Trace dataset: 
- Task count: 4061
- Sandbox dataset: DCAgent/toolscale-sandboxes
- Notes: no sandbox found on HF
- Generator: NOT FOUND

## a1_travistorrent
- Status: NEEDS SANDBOX REGEN
- Trace dataset: 
- Sandbox dataset: DCAgent/exp_rpt_travistorrent
- Notes: no sandbox on HF
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_travistorrent_tasks.py
    -> DCAgent/exp_rpt_travistorrent

## a1_tulu3_sft_personas_math
- Status: ON HF
- Trace dataset: DCAgent/tulu3-sft-personas-math-sandboxes_glm_4.7_traces_jupiter
- Trace count: 9997
- Task count: 9998
- Sandbox dataset: DCAgent/tulu3-sft-personas-math-sandboxes
- Generator: NOT FOUND

## a1_ubuntu_package
- Status: NEEDS SANDBOX REGEN
- Trace dataset: 
- Sandbox dataset: DCAgent/ubuntu-package-sandboxes
- Notes: no sandbox on HF
- Generator: NOT FOUND

## a1_unitsyn_python
- Status: NEEDS SANDBOX REGEN
- Trace dataset: DCAgent/exp_rpt_unitsyn-python-v3
- Task count: 500
- Sandbox dataset: DCAgent/exp_rpt_unitsyn-python
- Notes: needs upsample to 10k
- Generator(s):
  - /e/scratch/jureap59/raoof1/datagen_repo/data/dclm-mine/generate_unitsyn_tasks.py

## a1_wizardlm_orca
- Status: ON HF
- Trace dataset: DCAgent/wizardlm-orca-sandboxes_glm_4.7_traces_jupiter
- Trace count: 9608
- Task count: 10000
- Sandbox dataset: DCAgent/wizardlm-orca-sandboxes
- Generator: NOT FOUND
