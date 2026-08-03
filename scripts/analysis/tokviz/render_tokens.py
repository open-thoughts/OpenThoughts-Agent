"""render_tokens.py -- the heart of the deliverable.

Turn a text string (or token-id list) + an HF Qwen tokenizer into a fully
self-contained HTML document that shows, token-by-token, the EXACT literal
string the tokenizer feeds the model -- with special tokens highlighted and
every whitespace / control / non-UTF8 byte rendered with a visible glyph so a
human can see precisely what the model is prompted with.

Notes:
* Qwen uses a byte-level BPE tokenizer (``Qwen2TokenizerFast``). The raw
  ``convert_ids_to_tokens`` pieces contain GPT-2-style sentinel characters
  (``Ġ`` for a leading space, ``Ċ`` for newline, ``ĉ`` for tab, and ``Â``/``Ã``
  noise for the high bytes of multibyte UTF-8 sequences).
* To recover the real characters for a single token we call
  ``tokenizer.convert_tokens_to_string([piece])``. This correctly reassembles
  multibyte UTF-8 (e.g. ``ĠcafÃ©`` -> `` café``, ``æĹ¥æľ¬èªŀ`` -> ``日本語``).
* A single token can be a *fragment* of a multibyte character (the bytes are
  split across tokens). In that case the per-token decode yields the Unicode
  replacement char ``\\ufffd``; we detect that and fall back to showing the raw
  bytes as ``\\xHH`` so nothing is silently lost.
* Explicit byte-fallback tokens of the form ``<0x0A>`` are also rendered as
  ``\\x0A`` (these appear in some Qwen vocabs for raw bytes).

Emits inline-CSS, dependency-free HTML.
"""

from __future__ import annotations

import html
import re
from typing import List, Optional, Sequence

# Matches an explicit byte-fallback token like "<0x0A>".
_BYTE_TOKEN_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")

# Visible glyphs for otherwise-invisible characters.
GLYPH_SPACE = "·"      # middle dot  ·
GLYPH_NEWLINE = "↵"    # downwards arrow with corner leftwards  ↵
GLYPH_TAB = "→"        # rightwards arrow  →
GLYPH_CR = "␍"         # symbol for carriage return  ␍


# --------------------------------------------------------------------------- #
# Per-token byte recovery
# --------------------------------------------------------------------------- #
def decode_piece(tokenizer, token_id: int, piece: str) -> str:
    """Return the actual character string a single token represents.

    Handles three cases:
      1. Explicit byte-fallback token ``<0xHH>``  -> the single raw byte char.
      2. Normal byte-level BPE piece               -> convert_tokens_to_string.
      3. A piece that is a fragment of a multibyte char (decode gives U+FFFD).
         We do NOT try to merge across tokens here (each badge is independent);
         instead the caller's escaping turns the replacement char into \\xHH
         using the raw byte mapping recovered below.
    """
    m = _BYTE_TOKEN_RE.match(piece)
    if m:
        return chr(int(m.group(1), 16))
    try:
        return tokenizer.convert_tokens_to_string([piece])
    except Exception:
        return piece


def piece_raw_bytes(tokenizer, piece: str) -> Optional[bytes]:
    """Best-effort recovery of the raw bytes a byte-level piece encodes.

    GPT-2 / Qwen byte-level BPE maps each raw byte (0..255) to a printable
    Unicode codepoint. ``transformers`` exposes the *inverse* table on the slow
    tokenizer as ``byte_decoder``; the fast tokenizer does not. We rebuild the
    standard GPT-2 ``bytes_to_unicode`` table (it is fixed and identical across
    GPT-2-family tokenizers) and invert it. Returns None if the piece contains
    characters outside that table (e.g. it is an explicit ``<0xHH>`` token).
    """
    decoder = _gpt2_byte_decoder()
    out = bytearray()
    for ch in piece:
        if ch not in decoder:
            return None
        out.append(decoder[ch])
    return bytes(out)


_BYTE_DECODER_CACHE: Optional[dict] = None


def _gpt2_byte_decoder() -> dict:
    """The canonical GPT-2 unicode-char -> byte inverse table (cached)."""
    global _BYTE_DECODER_CACHE
    if _BYTE_DECODER_CACHE is not None:
        return _BYTE_DECODER_CACHE
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_encoder = {b: chr(c) for b, c in zip(bs, cs)}
    _BYTE_DECODER_CACHE = {v: k for k, v in byte_encoder.items()}
    return _BYTE_DECODER_CACHE


# --------------------------------------------------------------------------- #
# Escaping decoded text into visible HTML
# --------------------------------------------------------------------------- #
def escape_visible(text: str, raw_bytes: Optional[bytes]) -> str:
    """Escape ``text`` for HTML, replacing invisible/non-printable chars with
    visible glyphs. ``raw_bytes`` (if known) is used to render bytes that did
    not decode cleanly (replacement chars) as ``\\xHH``.
    """
    # If the clean decode produced replacement chars, the token is a fragment of
    # a multibyte sequence -> show the raw bytes literally so nothing is lost.
    if "�" in text and raw_bytes is not None:
        return "".join(f"<span class='byte'>\\x{b:02X}</span>" for b in raw_bytes)

    out: List[str] = []
    for ch in text:
        cp = ord(ch)
        if ch == " ":
            out.append(f"<span class='ws'>{GLYPH_SPACE}</span>")
        elif ch == "\n":
            # show the glyph AND an actual line break so multi-line prompts wrap
            out.append(f"<span class='ws'>{GLYPH_NEWLINE}</span><br>")
        elif ch == "\t":
            out.append(f"<span class='ws'>{GLYPH_TAB}</span>")
        elif ch == "\r":
            out.append(f"<span class='ws'>{GLYPH_CR}</span>")
        elif cp < 0x20 or cp == 0x7F:
            # other C0 control chars / DEL -> hex escape
            out.append(f"<span class='byte'>\\x{cp:02X}</span>")
        else:
            out.append(html.escape(ch))
    return "".join(out)


# --------------------------------------------------------------------------- #
# Special-token detection
# --------------------------------------------------------------------------- #
def build_special_id_set(tokenizer) -> set:
    """Union of all special + added token ids (im_start/im_end/endoftext/...)."""
    ids = set(tokenizer.all_special_ids or [])
    try:
        ids |= set(tokenizer.added_tokens_decoder.keys())
    except Exception:
        pass
    return ids


# --------------------------------------------------------------------------- #
# Core HTML builder
# --------------------------------------------------------------------------- #
_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0; padding: 24px; background: #f4f5f7; color: #1c1e21;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: #555; font-size: 13px; margin: 0 0 16px; }
.meta { font-size: 13px; color: #333; background:#fff; border:1px solid #e1e3e6;
        border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; }
.meta code { background:#eef0f3; padding:1px 5px; border-radius:4px; }
.legend { display:flex; flex-wrap:wrap; gap:10px; font-size:12px;
          background:#fff; border:1px solid #e1e3e6; border-radius:8px;
          padding:10px 14px; margin-bottom:18px; }
.legend .item { display:flex; align-items:center; gap:6px; }
.swatch { width:14px; height:14px; border-radius:3px; border:1px solid #0002; }
.section-label { font-size:12px; font-weight:700; text-transform:uppercase;
                 letter-spacing:.05em; color:#666; margin:18px 0 6px; }
.stream {
  font-family: "SF Mono", "Menlo", "Consolas", monospace; font-size: 14px;
  line-height: 2.1; background:#fff; border:1px solid #e1e3e6;
  border-radius:8px; padding:14px; word-break: break-word;
}
.tok {
  display: inline; padding: 2px 1px; border-radius: 3px;
  border: 1px solid #c9cdd3; margin: 0 0.5px;
}
.tok.shade0 { background: #eef2f7; }
.tok.shade1 { background: #e2e8f1; }
.tok.special { background: #ffe1b3; border-color: #e09b2d; font-weight: 600; }
.tok.special .lbl { font-size:10px; color:#9a5b00; }
.tok.gen { border-style: dashed; }
.gen-region { background: #eafbe7; border-radius:6px; padding: 2px 0; }
.ws { color:#c0392b; opacity:.85; }
.byte { color:#8e44ad; font-weight:600; }
.divider { display:inline-block; width:100%; border-top:1px dashed #aaa;
           margin:6px 0; }
.genhdr { color:#2e7d32; font-weight:700; font-size:12px; text-transform:uppercase;
          letter-spacing:.05em; }
"""


def _render_tokens_html(
    tokenizer,
    ids: Sequence[int],
    pieces: Sequence[str],
    special_ids: set,
    gen_start: Optional[int],
) -> str:
    """Render a run of tokens to badge spans. ``gen_start`` (if not None) marks
    the index at which the model's generated continuation begins."""
    spans: List[str] = []
    in_gen = False
    for idx, (tid, piece) in enumerate(zip(ids, pieces)):
        if gen_start is not None and idx == gen_start:
            spans.append(
                "<span class='divider'></span>"
                "<span class='genhdr'>&#9660; generated continuation below &#9660;</span>"
                "<span class='divider'></span>"
            )
            in_gen = True

        is_special = tid in special_ids
        decoded = decode_piece(tokenizer, tid, piece)
        raw = piece_raw_bytes(tokenizer, piece)
        title = html.escape(f"id={tid}  piece={piece!r}  decoded={decoded!r}")

        classes = ["tok", f"shade{idx % 2}"]
        if in_gen:
            classes.append("gen")
        if is_special:
            classes.append("special")
            body = (
                f"{html.escape(decoded)}"
                f"<span class='lbl'> #{tid}</span>"
            )
        else:
            body = escape_visible(decoded, raw)

        spans.append(
            f"<span class='{' '.join(classes)}' title='{title}'>{body}</span>"
        )

    rendered = "".join(spans)
    if gen_start is not None:
        return f"<div class='stream'>{rendered}</div>"
    return f"<div class='stream'>{rendered}</div>"


def render_to_html(
    tokenizer,
    *,
    model_id: str,
    setting_label: str,
    prompt_text: Optional[str] = None,
    prompt_ids: Optional[Sequence[int]] = None,
    generated_text: Optional[str] = None,
    generated_ids: Optional[Sequence[int]] = None,
    extra_note: str = "",
) -> str:
    """Build a standalone HTML document for one (prompt x model) sample.

    Provide either ``prompt_text`` (will be encoded) or ``prompt_ids``.
    Optionally provide the generated continuation as ``generated_text`` or
    ``generated_ids``; it is appended and visually distinguished.
    """
    special_ids = build_special_id_set(tokenizer)

    # ---- prompt tokens ----
    if prompt_ids is None:
        if prompt_text is None:
            raise ValueError("Provide prompt_text or prompt_ids")
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    prompt_ids = list(prompt_ids)

    # ---- generated tokens (optional) ----
    gen_ids: List[int] = []
    if generated_ids is not None:
        gen_ids = list(generated_ids)
    elif generated_text is not None and generated_text != "":
        gen_ids = list(tokenizer(generated_text, add_special_tokens=False)["input_ids"])

    all_ids = prompt_ids + gen_ids
    pieces = tokenizer.convert_ids_to_tokens(all_ids)
    gen_start = len(prompt_ids) if gen_ids else None

    stream_html = _render_tokens_html(
        tokenizer, all_ids, pieces, special_ids, gen_start
    )

    note_html = f"<div class='sub'>{html.escape(extra_note)}</div>" if extra_note else ""

    legend = """
    <div class='legend'>
      <div class='item'><span class='swatch' style='background:#eef2f7'></span>token (shade A)</div>
      <div class='item'><span class='swatch' style='background:#e2e8f1'></span>token (shade B)</div>
      <div class='item'><span class='swatch' style='background:#ffe1b3;border-color:#e09b2d'></span>special / added token (labelled with id)</div>
      <div class='item'><span class='swatch' style='background:#fff;border-style:dashed'></span>generated continuation (dashed border)</div>
      <div class='item'><span class='ws'>&middot; &rarr; &crarr;</span>&nbsp;space / tab / newline</div>
      <div class='item'><span class='byte'>\\xHH</span>&nbsp;raw / non-UTF8 byte</div>
    </div>
    """

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(setting_label)} — {html.escape(model_id)}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
  <h1>Exact tokenizer input — {html.escape(setting_label)}</h1>
  {note_html}
  <div class="meta">
    <div>Model: <code>{html.escape(model_id)}</code></div>
    <div>Prompt tokens: <code>{len(prompt_ids)}</code>
         &nbsp;|&nbsp; Generated tokens: <code>{len(gen_ids)}</code>
         &nbsp;|&nbsp; Total: <code>{len(all_ids)}</code></div>
  </div>
  {legend}
  <div class="section-label">Literal token stream (each badge = one token; hover for id &amp; raw piece)</div>
  {stream_html}
</div></body></html>
"""
