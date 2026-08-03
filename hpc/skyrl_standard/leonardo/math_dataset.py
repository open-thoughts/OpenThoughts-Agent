# Prepare a hard-math (Hendrycks MATH) RL dataset for MarinSkyRL non-agentic GRPO.
# Reuses the EXISTING `aime` env verifier (Minerva / boxed answer-match, adapted
# from lm-eval-harness hendrycks_math + verl math_dapo) -- only the data differs.
#
# Output rows mirror examples/gsm8k/gsm8k_dataset.py but target the `aime` env:
#   prompt      : chat messages (hard math problem + "end with Answer: \boxed{}")
#   env_class   : "aime"
#   reward_model: {"ground_truth": <normalized boxed answer>}   <-- AIMEEnv reads this
#
# Train: DigitalLearningGmbH/MATH-lighteval (full Hendrycks MATH train, 7500),
#        filtered to Level 3-5 (harder -> naturally longer CoT).
# Val:   HuggingFaceH4/MATH-500.
import argparse
import importlib.util
import os
import datasets

# Import the aime env's verifier utils directly (avoid the skyrl_gym package
# __init__ which pulls heavy tool deps not installed in this prep env).
_UPATH = "/leonardo_work/AIFAC_5C0_290/bfeuer00/code/MarinSkyRL/skyrl-gym/skyrl_gym/envs/aime/utils.py"
_spec = importlib.util.spec_from_file_location("aime_utils", _UPATH)
U = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(U)  # noqa: E702

INSTRUCTION = (
    " Please reason step by step. At the very end, output your final answer on its "
    "own line in the exact format: 'Answer: \\boxed{ANSWER}'."
)


def gt_from_solution(solution_str):
    b = U.last_boxed_only_string(solution_str)
    if b is None:
        return None
    return U.normalize_final_answer(U.remove_boxed(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="/leonardo_work/AIFAC_5C0_290/bfeuer00/data/math")
    ap.add_argument("--min_level", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- TRAIN: MATH-lighteval (level is "Level N" string) ----
    train = datasets.load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")

    def train_map(ex, idx):
        gt = gt_from_solution(ex["solution"])
        return {
            "data_source": "DigitalLearningGmbH/MATH-lighteval",
            "prompt": [{"role": "user", "content": ex["problem"] + INSTRUCTION}],
            "env_class": "aime",
            "reward_model": {"ground_truth": gt if gt is not None else ""},
            "extra_info": {"split": "train", "index": idx,
                           "level": ex.get("level", ""), "type": ex.get("type", "")},
        }

    train = train.map(train_map, with_indices=True, remove_columns=train.column_names)

    def keep_train(row):
        gt = row["reward_model"]["ground_truth"]
        lvl = row["extra_info"]["level"]
        try:
            n = int(str(lvl).replace("Level", "").strip())
        except Exception:
            n = 0
        return bool(gt) and n >= args.min_level

    train = train.filter(keep_train)

    # ---- VAL: MATH-500 (has a clean `answer` field that is the boxed content) ----
    val = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")

    def val_map(ex, idx):
        gt = U.normalize_final_answer(ex["answer"])
        return {
            "data_source": "HuggingFaceH4/MATH-500",
            "prompt": [{"role": "user", "content": ex["problem"] + INSTRUCTION}],
            "env_class": "aime",
            "reward_model": {"ground_truth": gt},
            "extra_info": {"split": "test", "index": idx,
                           "level": ex.get("level", ""), "subject": ex.get("subject", "")},
        }

    val = val.map(val_map, with_indices=True, remove_columns=val.column_names)
    val = val.filter(lambda r: bool(r["reward_model"]["ground_truth"]))

    train.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    val.to_parquet(os.path.join(args.output_dir, "validation.parquet"))
    print(f"TRAIN rows (level>={args.min_level}): {len(train)}")
    print(f"VAL rows: {len(val)}")
    print("sample train prompt:", train[0]["prompt"][0]["content"][:200])
    print("sample train gt:", repr(train[0]["reward_model"]["ground_truth"]))


if __name__ == "__main__":
    main()
