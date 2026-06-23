# Prepare the 4 Delphi-RL sweep datasets (RL_CONVENTION.md §2.2) as SkyRL-train
# GRPO parquet, CONTROLLED FOR n_steps via seeded random subsampling.
#
# Each dataset's train pool is subsampled to a common N prompts (default 500,
# seed 42) so that at the fixed train_batch_size/epochs/max_steps every dataset
# yields the same 100 steps over the same number of unique prompts — the dataset
# content is the only variable.
#
# Output rows mirror hpc/skyrl_yaml/leonardo/math_dataset.py (the `aime` boxed
# verifier contract, byte-identical to held-out eval):
#   prompt       : chat messages (problem + the shared boxed instruction)
#   env_class    : "aime" for the 3 math datasets (D1/D3/D4); "ifeval" for D2
#                  (wired in MarinSkyRL 430e443, §2.2)
#   reward_model : {"ground_truth": <normalized answer | IFEval constraint JSON>}
#
# Datasets (RL_CONVENTION §2.2):
#   D1 rlvr_math   allenai/RLVR-MATH                 (7,500;   boxed math → aime)
#   D2 rlvr_ifeval allenai/RLVR-IFeval              (14,973;  IF constraints → ifeval TODO)
#   D3 dapo_math   BytedTsinghua-SIA/DAPO-Math-17k  (17k uniq; "Answer:" → re-wrap boxed → aime)
#   D4 math500     HuggingFaceH4/MATH-500           (500 test-only; ⚠ TRAIN-ON-TEST vs held-out eval)
import argparse
import importlib.util
import os
import random

import datasets

# The `aime` env's answer-normalizer (same import pattern as math_dataset.py).
# Override on non-Leonardo hosts via MARINSKYRL_AIME_UTILS.
AIME_UTILS_PATH = os.environ.get(
    "MARINSKYRL_AIME_UTILS",
    "/leonardo_work/AIFAC_5C0_290/bfeuer00/code/MarinSkyRL/skyrl-gym/skyrl_gym/envs/aime/utils.py",
)
_spec = importlib.util.spec_from_file_location("aime_utils", AIME_UTILS_PATH)
U = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U)

# The shared boxed contract — IDENTICAL to math_dataset.py and held-out eval.
INSTRUCTION = (
    " Please reason step by step. At the very end, output your final answer on its "
    "own line in the exact format: 'Answer: \\boxed{ANSWER}'."
)

VAL_DATASET = "HuggingFaceH4/MATH-500"  # held-out math eval, shared by the math cells

# DAPO native prompts WRAP the problem in non-boxed "Answer:" boilerplate, BOTH a
# LEADING instruction sentence ("Solve the following math problem step by step. The
# last line of your response should be of the form Answer: $Answer ...") ending at the
# first blank line, AND a TRAILING directive ('Remember to put your answer on its own
# line after "Answer:".'). Strip BOTH so the shared boxed INSTRUCTION can be re-imposed
# cleanly. (The earlier truncate-at-first-marker logic cut at the LEADING sentence's
# "The last line of your response", collapsing every prompt to the identical preamble →
# only 1 unique prompt survived dedup. RL_CONVENTION §2.2 re-wrap.)
_DAPO_LEADING_MARKERS = ("The last line of your response", "Solve the following math problem")
_DAPO_TRAILING_MARKERS = ("Remember to put your answer", "The last line of your response")


def _strip_rlvr_math_fewshot(text):
    """RLVR-MATH wraps EVERY problem in a fixed 4-shot preamble: four worked
    'Question: ...\\nAnswer:...' exemplars, then the REAL problem as a final
    'Question: ...' block with NO trailing 'Answer:'. The shared preamble is
    identical across all 7,500 rows and pushes every prompt to ~1,600 chars
    (>512 tokens), so SkyRL's max_prompt_length filter dropped ALL rows →
    'dataset should be atleast as large as train_batch_size ... got size 0'.
    Keep only the LAST 'Question:' block (the real problem), drop its prefix.
    (RL_CONVENTION §2.2 — mirrors the bare-problem shape of DAPO/MATH-500.)"""
    marker = "Question:"
    idx = text.rfind(marker)
    body = text[idx + len(marker):] if idx != -1 else text
    return body.strip()


def _user_content(messages):
    """Last user-turn content from a chat-messages list."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return m["content"]
    return messages[-1]["content"]


def _strip_answer_boilerplate(text):
    """Extract the bare problem from a DAPO prompt that is wrapped in a leading
    instruction sentence + a trailing 'Answer:' directive. The leading sentence ends
    at the first blank line, so drop everything up to and including the FIRST '\n\n'
    when a known leading marker is present; then cut off the trailing directive."""
    body = text
    # Drop the leading instruction preamble (ends at the first blank line).
    if any(m in body[:200] for m in _DAPO_LEADING_MARKERS):
        i = body.find("\n\n")
        if i != -1:
            body = body[i + 2:]
    # Cut the trailing 'Answer:' directive.
    cut = len(body)
    for marker in _DAPO_TRAILING_MARKERS:
        j = body.find(marker)
        if j != -1:
            cut = min(cut, j)
    return body[:cut].strip()


def _math_row(problem, ground_truth, source, idx):
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": problem + INSTRUCTION}],
        "env_class": "aime",
        "reward_model": {"ground_truth": ground_truth if ground_truth is not None else ""},
        "extra_info": {"split": "train", "index": idx},
    }


def build_rlvr_math():
    ds = datasets.load_dataset("allenai/RLVR-MATH", split="train")

    def _map(ex, i):
        return _math_row(
            _strip_rlvr_math_fewshot(_user_content(ex["messages"])),
            U.normalize_final_answer(str(ex["ground_truth"])),
            "allenai/RLVR-MATH",
            i,
        )

    return ds.map(_map, with_indices=True, remove_columns=ds.column_names)


def build_dapo_math(unique_cap=20000):
    """DAPO-Math is ~1.79M rows = 17k unique problems repeated. Stream + dedup on
    the (stripped) prompt so the subsample draws from truly-unique prompts, and
    cap the unique pool to avoid materializing 1.79M rows."""
    stream = datasets.load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train", streaming=True)
    seen, rows = set(), []
    for ex in stream:
        prob = _strip_answer_boilerplate(_user_content(ex["prompt"]))
        if prob in seen:
            continue
        seen.add(prob)
        gt = U.normalize_final_answer(str(ex["reward_model"]["ground_truth"]))
        rows.append(_math_row(prob, gt, "BytedTsinghua-SIA/DAPO-Math-17k", len(rows)))
        if len(rows) >= unique_cap:
            break
    return datasets.Dataset.from_list(rows)


def build_math500(split="test"):
    ds = datasets.load_dataset(VAL_DATASET, split=split)

    def _map(ex, i):
        return _math_row(ex["problem"], U.normalize_final_answer(ex["answer"]), VAL_DATASET, i)

    return ds.map(_map, with_indices=True, remove_columns=ds.column_names)


def build_rlvr_ifeval():
    # D2: instruction-following, verifiable CONSTRAINTS (not a boxed-math answer).
    # ground_truth = the IFEval constraint spec JSON (func_name + kwargs), scored by the
    # `skyrl_gym/envs/ifeval` verifier (MarinSkyRL 430e443, now wired — RL_CONVENTION §2.2).
    # The RLVR-IFeval `messages` are SINGLE-TURN bare instructions (no few-shot preamble,
    # no boxed contract), so the user content is used as-is.
    ds = datasets.load_dataset("allenai/RLVR-IFeval", split="train")

    def _map(ex, i):
        return {
            "data_source": "allenai/RLVR-IFeval",
            "prompt": [{"role": "user", "content": _user_content(ex["messages"])}],
            "env_class": "ifeval",  # wired in MarinSkyRL 430e443
            "reward_model": {"ground_truth": ex["ground_truth"]},  # JSON constraint spec
            "extra_info": {"split": "train", "index": i, "constraint_type": ex.get("constraint_type")},
        }

    return ds.map(_map, with_indices=True, remove_columns=ds.column_names)


BUILDERS = {
    "rlvr_math": build_rlvr_math,
    "dapo_math": build_dapo_math,
    "math500": build_math500,
    "rlvr_ifeval": build_rlvr_ifeval,
}


def seeded_subsample(ds, n, seed):
    """Deterministic random subsample to n rows (all rows if pool <= n)."""
    if n is None or n >= len(ds):
        return ds
    idx = sorted(random.Random(seed).sample(range(len(ds)), n))
    return ds.select(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(BUILDERS))
    ap.add_argument("--subsample_n", type=int, default=500,
                    help="common prompt count across datasets (default 500 = MATH-500 floor; §2.2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "rlvr_ifeval":
        print("NOTE: rlvr_ifeval uses the `skyrl_gym/envs/ifeval` verifier wired in MarinSkyRL 430e443 "
              "(RL_CONVENTION.md §2.2).")

    train = seeded_subsample(BUILDERS[args.dataset](), args.subsample_n, args.seed)

    # Held-out val = MATH-500 test (== EVAL_CONVENTION held-out, boxed) for the math cells.
    # For math500 itself train == val → TRAIN-ON-TEST (RL_CONVENTION §2.2 contamination).
    # For ifeval MATH-500 is only a smoke placeholder (real IF metric = IFEval pass-rate).
    val = build_math500("test")

    train.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    val.to_parquet(os.path.join(args.output_dir, "validation.parquet"))
    print(f"[{args.dataset}] train rows (subsampled to <= {args.subsample_n}, seed {args.seed}): {len(train)}")
    print(f"[{args.dataset}] val rows (MATH-500 test): {len(val)}")
    print("sample train prompt:", train[0]["prompt"][0]["content"][:200])
    print("sample train gt:", repr(train[0]["reward_model"]["ground_truth"]))
    if args.dataset == "math500":
        print("NOTE: math500 train == MATH-500 test → TRAIN-ON-TEST (RL_CONVENTION §2.2).")


if __name__ == "__main__":
    main()
