"""Canonical extractor of embedded Input/Output examples from codenet instruction.md.

The codenet instruction.md was LLM-generated from a CodeNet problem; it always (>99.9%)
embeds at least one worked Input/Output example, in one of several markdown shapes:

  (A) heading  '### Input' / '#### Output'        + fenced code block
  (B) bold      '**Input:**' / '**Output:**'       + fenced code block (possibly on next line)
  (C) single fence whose body holds 'Input:\\n... Output:\\n...'
  (D) bold/dash  '**Input:**\\n`B`' or '- Input: `B`'  -- value in inline backticks
  (E) single fence with no markers, preceded by '### Input' / '### Output' headings

This module returns an ORDERED list of (input_text, output_text) example pairs. The first
pair is the authoritative test case used by tests/test.sh (it diffs stdout against it).
"""
from __future__ import annotations
import re

FENCE_RE = re.compile(r'```[^\n]*\n(.*?)```', re.S)

# An example Input/Output marker. We explicitly DISALLOW the spec-section headings
# 'Input Format' / 'Output Format' / 'Input/Output Format' (those introduce prose, not
# a worked example) by forbidding the word 'format' after the keyword on the line.
INPUT_MARK = re.compile(r'(?i)(?:^|\n)[\s>*\-]*#{0,6}\s*\**\s*input\b(?![^\n]*\bformat\b)[^\n:`]*:?\s*\**')
OUTPUT_MARK = re.compile(r'(?i)(?:^|\n)[\s>*\-]*#{0,6}\s*\**\s*output\b(?![^\n]*\bformat\b)[^\n:`]*:?\s*\**')


def _label_for(prefix_text: str):
    lines = [l for l in prefix_text.split("\n") if l.strip()]  # noqa: E741
    if not lines:
        return None
    last = re.sub(r'[#*`:>\-]', '', lines[-1].strip().lower()).strip()
    # 'Input Format' / 'Output Format' headings introduce a prose spec, not an example.
    if "format" in last:
        return None
    if re.search(r'\boutput\b', last):
        return "output"
    if re.search(r'\binput\b', last):
        return "input"
    return None


def _split_inline(body: str):
    m = re.search(r'(?is)\binput\s*:?\s*\n(.*?)\n\s*output\s*:?\s*\n(.*)', body)
    if m:
        return m.group(1).strip("\n"), m.group(2).strip("\n")
    return None


def _fence_pairs(instr: str):
    fences = [(m.start(), m.group(1)) for m in FENCE_RE.finditer(instr)]
    if not fences:
        return []
    labeled = []
    prev_end = 0
    for start, body in fences:
        labeled.append((_label_for(instr[prev_end:start]), body))
        prev_end = start
    pairs = []
    i = 0
    while i < len(labeled):
        lab, body = labeled[i]
        if lab == "input" and i + 1 < len(labeled) and labeled[i + 1][0] == "output":
            pairs.append((body.strip("\n"), labeled[i + 1][1].strip("\n")))
            i += 2
            continue
        i += 1
    if pairs:
        return pairs
    # single-fence inline
    for _, body in labeled:
        si = _split_inline(body)
        if si:
            pairs.append(si)
    if pairs:
        return pairs
    # input fence followed by any next fence
    i = 0
    while i < len(labeled):
        lab, body = labeled[i]
        if lab == "input" and i + 1 < len(labeled):
            pairs.append((body.strip("\n"), labeled[i + 1][1].strip("\n")))
            i += 2
            continue
        i += 1
    return pairs


def _value_after(text: str, start: int) -> str | None:
    """Extract the example value beginning at offset `start` (just past a marker).

    Accepts: a fenced block, an inline-backtick run, or the next non-empty line(s)
    up to the next marker / blank-after-content.
    """
    rest = text[start:]
    # leading whitespace/newlines
    m = re.match(r'[ \t]*\n?', rest)
    rest2 = rest[m.end():]
    # fenced?
    fm = re.match(r'```[^\n]*\n(.*?)```', rest2, re.S)
    if fm:
        return fm.group(1).strip("\n")
    # inline backticks (may span the marker line or next line)
    bm = re.search(r'`([^`]+)`', rest2[:200])
    # only treat as inline-backtick if it appears before the next newline-blank gap
    if bm and bm.start() < 80:
        return bm.group(1).strip()
    # plain next non-empty line
    for ln in rest2.split("\n"):
        if ln.strip():
            return ln.strip().strip("`").strip()
    return None


def _marker_pairs(instr: str):
    """Scan Input/Output markers in order, pulling the value after each."""
    marks = []
    for m in INPUT_MARK.finditer(instr):
        marks.append((m.start(), m.end(), "input"))
    for m in OUTPUT_MARK.finditer(instr):
        marks.append((m.start(), m.end(), "output"))
    marks.sort()
    pairs = []
    i = 0
    while i < len(marks):
        s, e, kind = marks[i]
        if kind == "input" and i + 1 < len(marks) and marks[i + 1][2] == "output":
            inp = _value_after(instr, e)
            out = _value_after(instr, marks[i + 1][1])
            if inp is not None and out is not None:
                pairs.append((inp, out))
            i += 2
            continue
        i += 1
    return pairs


def extract_io_pairs(instr: str):
    """Return ordered list of (input, output) example pairs. Best-effort, multi-strategy."""
    pairs = _fence_pairs(instr)
    if not pairs:
        pairs = _marker_pairs(instr)
    # filter all-empty
    pairs = [(a, b) for a, b in pairs if (a.strip() or b.strip())]
    return pairs


if __name__ == "__main__":
    import pandas as pd
    import gzip
    import io
    import tarfile
    import sys
    import collections
    df = pd.read_parquet(sys.argv[1])
    n = len(df)
    dist = collections.Counter()
    zero = []
    for idx in range(n):
        row = df.iloc[idx]
        tar = tarfile.open(fileobj=io.BytesIO(gzip.decompress(row["task_binary"])))
        instr = tar.extractfile("instruction.md").read().decode()
        ps = extract_io_pairs(instr)
        dist[len(ps)] += 1
        if len(ps) == 0 and len(zero) < 20:
            zero.append(row["path"])
    print("total", n)
    print("pairs-per-task dist:", dict(sorted(dist.items())))
    print("zero-pair count:", dist[0])
    print("sample zero:", zero)
