#!/usr/bin/env python
"""Prepare a Delphi checkpoint for the `delphi` chat template (marin #6279).

Two jobs, both needed before cold-start SFT with `template: delphi`:

1. **Reserved-slot rename (no vocab growth).** The canonical chat tokens
   (`<|start_think|>`, `<|end_think|>`, `<|tool_call|>`, `<|tool_call_end|>`,
   `<|tool_result|>`, `<|tool_result_end|>`) are mapped ONTO existing Llama-3
   `<|reserved_special_token_N|>` slots by renaming the slot's `content` in
   `tokenizer.json` + `tokenizer_config.json`. Vocab size stays 128256 and the
   existing embedding rows are reused — `add_special_tokens` would instead append
   NEW rows at 128256+ (vocab growth + a model resize), which we do not want.

2. **Mean-init the reused embedding rows.** Reserved slots never appeared in
   pre/midtraining, so their input AND output (Delphi is `tie_word_embeddings:
   false`) embedding rows are effectively untrained. We overwrite each target row
   with the mean of the trained BPE rows (ids `< 128000`) so the tokens start at a
   sane centroid instead of their init values. They are still trained from data
   (see the synthetic-slice discussion in the chat_templating DESIGN doc).

This does NOT teach the tokens any behavior — that needs SFT/RL data that uses
them. It only makes them single tokens with non-garbage starting embeddings.

Usage:
  python sft/prepare_delphi_tokenizer.py \
      --model laion/delphi-9e19-p33m67-k0p20-lr83-a002 \
      --output /path/to/delphi-9e19-p33m67-delphi-tok
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# canonical string -> reserved slot id it is renamed onto (ids must be reserved_special_token_* in
# the Llama-3.1 / Delphi tokenizer; verified present 2026-06-08). Keep in sync with the `delphi`
# LLaMA-Factory template (thought_words + tool wrappers) and chat_templates/delphi_v0.jinja2.
CANONICAL_TOKEN_IDS = {
    "<|start_think|>": 128002,      # was <|reserved_special_token_0|>
    "<|end_think|>": 128003,        # was <|reserved_special_token_1|>
    "<|tool_call|>": 128005,        # was <|reserved_special_token_2|>
    "<|tool_call_end|>": 128011,    # was <|reserved_special_token_3|>
    "<|tool_result|>": 128012,      # was <|reserved_special_token_4|>
    "<|tool_result_end|>": 128013,  # was <|reserved_special_token_5|>
}
TRAINED_VOCAB_CEILING = 128000  # ids below this are regular trained BPE tokens (the mean-init pool)


def rename_reserved_slots(out_dir: Path, token_ids: dict[str, int]) -> None:
    """Rename reserved-slot `content` to the canonical strings, in place, keeping ids fixed."""
    id_to_new = {i: s for s, i in token_ids.items()}

    tok_json_path = out_dir / "tokenizer.json"
    tok_json = json.loads(tok_json_path.read_text())
    renamed = 0
    for entry in tok_json.get("added_tokens", []):
        if entry["id"] in id_to_new:
            assert "reserved_special_token" in entry["content"], (
                f"id {entry['id']} is {entry['content']!r}, not a reserved slot — refusing to clobber"
            )
            entry["content"] = id_to_new[entry["id"]]
            entry["special"] = True
            renamed += 1
    assert renamed == len(token_ids), f"renamed {renamed}/{len(token_ids)} in tokenizer.json"
    tok_json_path.write_text(json.dumps(tok_json, ensure_ascii=False, indent=2))

    cfg_path = out_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    atd = cfg.get("added_tokens_decoder", {})
    # transformers 5.x (`TokenizersBackend`) saves a minimal tokenizer_config.json with
    # NO `added_tokens_decoder` block — all added-tokens live in tokenizer.json (already
    # renamed above). Only rewrite the config block when the 4.x-style block is present.
    if atd:
        for sid, new in id_to_new.items():
            key = str(sid)
            assert key in atd and "reserved_special_token" in atd[key]["content"], (
                f"added_tokens_decoder[{key}] missing or not reserved"
            )
            atd[key]["content"] = new
            atd[key]["special"] = True
        cfg["added_tokens_decoder"] = atd
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))


def mean_init_rows(model, token_ids: dict[str, int]) -> None:
    """Overwrite the input and output embedding rows for token_ids with the trained-row mean."""
    ids = sorted(token_ids.values())
    with torch.no_grad():
        for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
            if emb is None:
                continue
            w = emb.weight
            centroid = w[:TRAINED_VOCAB_CEILING].mean(dim=0)
            for i in ids:
                w[i] = centroid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF repo id or local path of the Delphi checkpoint")
    ap.add_argument("--output", required=True, type=Path, help="output dir for the prepared model+tokenizer")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # 1. tokenizer: save, rename reserved slots, reload, verify single-token + no vocab growth
    tok = AutoTokenizer.from_pretrained(args.model)
    base_vocab = len(tok)
    tok.save_pretrained(args.output)
    rename_reserved_slots(args.output, CANONICAL_TOKEN_IDS)

    tok = AutoTokenizer.from_pretrained(args.output)
    assert len(tok) == base_vocab, f"vocab grew {base_vocab} -> {len(tok)} (rename failed, fell back to add)"
    for s, want_id in CANONICAL_TOKEN_IDS.items():
        got = tok.convert_tokens_to_ids(s)
        assert got == want_id, f"{s} -> {got}, expected {want_id}"
        assert len(tok.encode(s, add_special_tokens=False)) == 1, f"{s} did not tokenize to a single id"
    print(f"tokenizer OK: {len(CANONICAL_TOKEN_IDS)} tokens renamed onto reserved slots, vocab={len(tok)}")

    # 2. model: mean-init the reused rows, save
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    assert model.get_output_embeddings() is not None, "expected untied lm_head (Delphi is untied)"
    mean_init_rows(model, CANONICAL_TOKEN_IDS)
    model.save_pretrained(args.output)
    print(f"model OK: mean-init'd input+output rows for ids {sorted(CANONICAL_TOKEN_IDS.values())}")
    print(f"wrote prepared checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
