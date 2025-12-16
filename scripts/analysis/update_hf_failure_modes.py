#!/usr/bin/env python3
"""Augment a HF dataset with GPT-5 failure-mode summaries and push the update."""

from __future__ import annotations

import argparse
import json
import logging
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from openai import OpenAI


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate HF dataset rows with failure-mode analyses")
    parser.add_argument("repo_id", help="HF dataset repo id (e.g. penfever/DCAgent_...)")
    parser.add_argument("--split", default="train", help="Dataset split to load/push (default: train)")
    parser.add_argument(
        "--output-column",
        default="failure_mode_analysis",
        help="Name of the column that will store the GPT analysis",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Number of rows per OpenAI request",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for local testing",
    )
    parser.add_argument("--model", default="gpt-5.1", help="Judge model id")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16384,
        help="Maximum tokens for judge responses (default: 16384)",
    )
    parser.add_argument(
        "--conversation-char-limit",
        type=int,
        default=200000,
        help="Character cap per conversation excerpt (<=0 disables truncation)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows where the output column already exists",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push updated dataset back to HF (otherwise dry-run)",
    )
    parser.add_argument(
        "--commit-message",
        default="Add failure-mode analyses via GPT-5",
        help="Commit message when pushing to HF",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY env)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HF token (defaults to HF_TOKEN env)",
    )
    return parser.parse_args()


@dataclass
class RowContext:
    idx: int
    trial_name: str
    task: str
    instruction: str
    conversation_excerpt: str
    verifier_excerpt: str
    result: str


def summarize_conversation(conv: list, limit: int | None = 4000) -> str:
    if not conv:
        return "[missing conversation]"
    parts = []
    for turn in conv:
        role = turn.get("role", "unknown").upper()
        content = turn.get("content", "").strip()
        parts.append(f"[{role}] {content}")
    joined = "\n".join(parts)
    if limit and limit > 0 and len(joined) > limit:
        return f"{joined[:limit]}\n...[truncated]..."
    return joined


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert evaluator of autonomous LLM agents. For each trial you will receive:
    • the original instruction,
    • a condensed conversation transcript,
    • verifier output or scoring metadata.

    Your analysis must:
      1. Restate the concrete task requirements.
      2. Describe how the agent proceeded, citing specific actions / commands from the transcript.
      3. Explain the final outcome referencing the verifier evidence (scores, error text, etc.).

    Do NOT hedge with phrases like “seems” or “likely”; base conclusions on the provided text.
    Return a JSON array (no code fences). Each object must include:
        {"trial_name": "...", "analysis": "..."}
    where analysis is 2–4 sentences covering the three bullets above.
    """
).strip()


def build_prompt(batch: Sequence[RowContext]) -> str:
    sections = []
    for row in batch:
        sections.append(
            textwrap.dedent(
                f"""
                ### Trial: {row.trial_name or row.idx}
                Instruction / Task Descriptor:
                {row.instruction or row.task}

                Task metadata: {row.task}
                Result metadata: {row.result}

                Conversation Transcript (condensed):
                {row.conversation_excerpt}

                Verifier Output:
                {row.verifier_excerpt}
                """
            ).strip()
        )
    return (
        "For each trial above, cite concrete behaviors and verifier evidence when writing the analysis. "
        "Return JSON array with fields trial_name and analysis."
        "\n\n" + "\n\n".join(sections)
    )


def batched(seq: Sequence[RowContext], size: int) -> Iterable[Sequence[RowContext]]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def judge_batch(
    client: OpenAI,
    model: str,
    temperature: float,
    batch: Sequence[RowContext],
    max_output_tokens: int | None,
) -> list[str]:
    prompt = build_prompt(batch)
    completion_kwargs = {}
    if max_output_tokens:
        completion_kwargs["max_completion_tokens"] = max_output_tokens
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        **completion_kwargs,
    )
    content = resp.choices[0].message.content or "[]"
    content = content.strip()
    if content.startswith("```"):
        # strip code fence
        parts = content.split("```", 2)
        if len(parts) >= 2:
            snippet = parts[1]
            if "\n" in snippet:
                snippet = snippet.split("\n", 1)[1]
            content = snippet.strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse model output: %s\nPayload: %s", exc, content)
        raise

    analyses: dict[str, str] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                name = str(item.get("trial_name", "")).strip()
                analyses[name] = item.get("analysis", "")

    results = []
    for row in batch:
        key = row.trial_name or str(row.idx)
        results.append(analyses.get(key, ""))
    return results


def truncate_text(text: str, limit: int | None) -> str:
    if not limit or limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[truncated]..."


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ds = load_dataset(args.repo_id, split=args.split)
    df = ds.to_pandas()
    if args.max_rows:
        df = df.head(args.max_rows)

    # Prep contexts
    contexts: list[RowContext] = []
    conv_limit = args.conversation_char_limit if args.conversation_char_limit > 0 else None
    verifier_limit = conv_limit

    for idx, row in df.iterrows():
        if args.resume and isinstance(row.get(args.output_column), str) and row[args.output_column].strip():
            continue
        conv_data = row.get("conversations")
        if isinstance(conv_data, list):
            normalized = conv_data
        elif hasattr(conv_data, "tolist"):
            try:
                normalized = conv_data.tolist()
            except Exception:  # pragma: no cover - defensive
                normalized = []
        elif conv_data is None:
            normalized = []
        else:
            normalized = [conv_data]

        conv = summarize_conversation(normalized, limit=conv_limit)
        instruction = str(row.get("instruction") or row.get("task") or "")
        verifier_text = row.get("verifier_output")
        if isinstance(verifier_text, (list, tuple)):
            verifier_excerpt = "\n".join(str(x) for x in verifier_text)
        else:
            verifier_excerpt = str(verifier_text or "").strip()
        if not verifier_excerpt:
            verifier_excerpt = "[no verifier output]"
        else:
            verifier_excerpt = truncate_text(verifier_excerpt, verifier_limit)
        instruction_excerpt = truncate_text(instruction, conv_limit)
        contexts.append(
            RowContext(
                idx=idx,
                trial_name=str(row.get("trial_name") or row.get("run_id") or idx),
                task=str(row.get("task") or ""),
                instruction=instruction_excerpt,
                conversation_excerpt=conv,
                verifier_excerpt=verifier_excerpt,
                result=json.dumps(row.get("result")),
            )
        )

    if not contexts:
        logger.info("No rows require updates; exiting.")
        return

    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key missing.")
    client = OpenAI(api_key=api_key)

    analyses = ["" for _ in range(len(df))]
    for batch in batched(contexts, args.batch_size):
        logger.info("Judging batch of %d rows", len(batch))
        outputs = judge_batch(
            client,
            args.model,
            args.temperature,
            batch,
            max_output_tokens=args.max_output_tokens,
        )
        for row, text in zip(batch, outputs):
            analyses[row.idx] = text or ""

    df[args.output_column] = analyses
    updated = Dataset.from_pandas(df, preserve_index=False)

    if args.push:
        hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise RuntimeError("HF token required to push (set HF_TOKEN env or --hf-token)")
        api = HfApi(token=hf_token)
        logger.info("Pushing updated dataset to %s", args.repo_id)
        updated.push_to_hub(args.repo_id, token=hf_token, commit_message=args.commit_message)
        try:
            api.update_repo_settings(
                repo_id=args.repo_id,
                repo_type="dataset",
                private=False,
            )
        except AttributeError:
            logger.warning(
                "HfApi.update_repo_settings unavailable on this version; "
                "dataset visibility left unchanged.",
            )
    else:
        output_path = Path("./annotated_rows.parquet").resolve()
        df.to_parquet(output_path, index=False)
        logger.info("Wrote augmented parquet to %s (dry-run)")


if __name__ == "__main__":
    main()
