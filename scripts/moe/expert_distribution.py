#!/usr/bin/env python3
"""Reference profiler for MoE expert-call distribution (Qwen3-MoE family).

Runs the model through plain HuggingFace `transformers` and records, per
layer, which experts the router selects for every token — independent of
any vLLM capture path. Use as the trusted reference to compare against
vLLM-side routing captures on the same model + inputs.

Mechanism
---------
For a Qwen3-MoE model every sparse decoder layer holds a
`Qwen3MoeSparseMoeBlock` whose `.gate` is a Linear producing the per-token
router logits of shape (num_tokens, num_experts). We register a forward hook on
each `.gate` and recompute the top-k expert selection exactly as the model does
(softmax over logits -> topk). If a different MoE arch is loaded we fall back to
hooking any module whose name ends in `.gate` and emits a 2-D logits tensor
whose last dim == num_experts.

Usage
-----
    # Tiny local validation (no network, builds a random tiny MoE):
    python -m scripts.moe.expert_distribution --self-test

    # Profile a real model on an agentic-trace sample (chat messages):
    python -m scripts.moe.expert_distribution \
        --model Qwen/Qwen3-30B-A3B \
        --dataset DCAgent/dev_set --messages-field conversations \
        --num-samples 64 --max-tokens 4096 \
        --output report.json --plot

    # Profile on a local JSONL of raw prompts:
    python -m scripts.moe.expert_distribution \
        --model /path/to/qwen3-30b-a3b \
        --dataset prompts.jsonl --text-field text \
        --num-samples 64 --output report.json

See scripts/moe/README.md for the expected dataset schema and how to read the
entropy / Gini / dead-expert numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reference MoE expert-call distribution profiler (HF hooks).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B",
        help="HuggingFace model id or local path to a Qwen3-MoE checkpoint.",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Local .jsonl/.parquet path OR a HuggingFace dataset id. "
        "Required unless --self-test.",
    )
    p.add_argument(
        "--split",
        default="train",
        help="Dataset split (only used for HF dataset ids).",
    )
    p.add_argument(
        "--text-field",
        default=None,
        help="Column holding raw text. Mutually exclusive with --messages-field.",
    )
    p.add_argument(
        "--messages-field",
        default=None,
        help="Column holding a chat 'messages' list (role/content dicts); "
        "the tokenizer chat template is applied.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of dataset rows to profile (evenly spaced across the set).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Truncate each sample to this many tokens.",
    )
    p.add_argument(
        "--top-loaded",
        type=int,
        default=5,
        help="How many most/least-loaded experts to list per layer + globally.",
    )
    p.add_argument(
        "--collapsed-layers",
        type=int,
        default=5,
        help="How many most-collapsed (lowest-entropy) layers to flag.",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32", "auto"],
        help="Model dtype.",
    )
    p.add_argument(
        "--device-map",
        default="auto",
        help="transformers device_map (e.g. 'auto', 'cuda', 'cpu').",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Path to write the machine-readable JSON report.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save per-layer load heatmap + load histogram PNGs (needs matplotlib).",
    )
    p.add_argument(
        "--plot-prefix",
        default="moe_expert_dist",
        help="Filename prefix for the --plot PNGs.",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Build a tiny random Qwen3-MoE on CPU and validate hooks + stats math. "
        "No model/dataset download. Exits non-zero if any assertion fails.",
    )
    return p.parse_args(argv)


def log(msg: str) -> None:
    print(f"[moe] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Routing capture
# --------------------------------------------------------------------------- #
class RouterCapture:
    """Accumulates per-layer expert-selection counts via forward hooks.

    We hook each MoE block's `.gate` Linear. Its output is the router logits of
    shape (num_tokens, num_experts); we reproduce the model's selection
    (softmax -> top-k) and tally the selected expert indices per layer.
    """

    def __init__(self, num_experts: int, top_k: int, norm_topk_prob: bool = False):
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob  # unused for counts; kept for clarity
        # layer_name -> np.ndarray[num_experts] of selection counts
        self.counts: dict[str, np.ndarray] = {}
        # layer_name -> number of token-slots routed (tokens * top_k)
        self.token_slots: dict[str, int] = {}
        self._handles: list[Any] = []
        self._order: list[str] = []

    def _make_hook(self, name: str):
        import torch

        def hook(_module, _inputs, output):
            # `.gate` is a Linear -> output is the router-logit tensor.
            logits = output[0] if isinstance(output, tuple) else output
            if logits.dim() == 3:  # (batch, seq, num_experts)
                logits = logits.reshape(-1, logits.shape[-1])
            if logits.dim() != 2 or logits.shape[-1] != self.num_experts:
                return  # not a router gate we understand; ignore
            weights = torch.softmax(logits.float(), dim=-1)
            _, selected = torch.topk(weights, self.top_k, dim=-1)  # (tokens, top_k)
            sel = selected.reshape(-1).to("cpu").numpy()
            binc = np.bincount(sel, minlength=self.num_experts).astype(np.int64)
            if name not in self.counts:
                self.counts[name] = np.zeros(self.num_experts, dtype=np.int64)
                self.token_slots[name] = 0
                self._order.append(name)
            self.counts[name] += binc
            self.token_slots[name] += int(sel.shape[0])

        return hook

    def attach(self, model) -> int:
        """Hook every MoE gate. Returns the number of hooked layers."""
        gate_modules = _find_gate_modules(model, self.num_experts)
        for name, module in gate_modules:
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return len(gate_modules)

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def layer_names(self) -> list[str]:
        return list(self._order)


def _find_gate_modules(model, num_experts: int) -> list[tuple[str, Any]]:
    """Locate MoE router gate modules.

    Strategy: prefer the official Qwen3-MoE sparse block's `.gate`; otherwise
    fall back to any submodule whose name ends in '.gate' and is a Linear whose
    out_features == num_experts.
    """
    import torch.nn as nn

    gates: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if cls.endswith("SparseMoeBlock") and hasattr(module, "gate"):
            gates.append((name + ".gate", module.gate))
    if gates:
        return gates
    # Fallback: any '.gate' Linear with the right fan-out.
    for name, module in model.named_modules():
        if name.endswith(".gate") and isinstance(module, nn.Linear):
            if getattr(module, "out_features", None) == num_experts:
                gates.append((name, module))
    return gates


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def normalized_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of the load distribution, normalized to [0, 1].

    1.0 == perfectly uniform routing; 0.0 == all token-slots to one expert.
    """
    total = counts.sum()
    n = len(counts)
    if total == 0 or n <= 1:
        return 0.0
    p = counts.astype(np.float64) / total
    nz = p[p > 0]
    h = -np.sum(nz * np.log2(nz))
    return float(h / math.log2(n))


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of the load. 0 == perfectly equal; ->1 == maximally concentrated."""
    x = np.sort(counts.astype(np.float64))
    n = len(x)
    s = x.sum()
    if n == 0 or s == 0:
        return 0.0
    # G = (2 * sum(i * x_i) / (n * sum(x))) - (n + 1) / n, with i = 1..n
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * s) - (n + 1.0) / n)


def layer_stats(counts: np.ndarray, token_slots: int, top_loaded: int) -> dict:
    """Per-layer routing statistics from a selection-count vector."""
    n = len(counts)
    total = int(counts.sum())
    load = counts.astype(np.float64) / total if total else np.zeros(n)
    uniform = 1.0 / n if n else 0.0
    dead = int(np.sum(counts == 0))
    underused = int(np.sum(load < 0.1 * uniform))  # < 0.1x the uniform share
    order_desc = np.argsort(-counts)
    order_asc = np.argsort(counts)
    return {
        "num_experts": n,
        "token_slots": int(token_slots),
        "total_selections": total,
        "entropy_normalized": normalized_entropy(counts),
        "gini": gini(counts),
        "dead_experts": dead,
        "underused_experts": underused,  # load < 0.1x uniform (incl. dead)
        "load_max": float(load.max()) if n else 0.0,
        "load_mean": float(load.mean()) if n else 0.0,
        "load_min": float(load.min()) if n else 0.0,
        "max_uniform_ratio": float(load.max() / uniform) if uniform else 0.0,
        "mean_uniform_ratio": float(load.mean() / uniform) if uniform else 0.0,
        "top_experts": [
            {"expert": int(e), "count": int(counts[e]), "load": float(load[e])}
            for e in order_desc[:top_loaded]
        ],
        "bottom_experts": [
            {"expert": int(e), "count": int(counts[e]), "load": float(load[e])}
            for e in order_asc[:top_loaded]
        ],
        "counts": counts.astype(int).tolist(),
    }


def build_report(capture: RouterCapture, top_loaded: int, collapsed_layers: int) -> dict:
    per_layer: dict[str, dict] = {}
    global_counts = np.zeros(capture.num_experts, dtype=np.int64)
    global_slots = 0
    for name in capture.layer_names:
        c = capture.counts[name]
        per_layer[name] = layer_stats(c, capture.token_slots[name], top_loaded)
        global_counts += c
        global_slots += capture.token_slots[name]

    glob = layer_stats(global_counts, global_slots, top_loaded)

    # Distribution of per-layer entropy/Gini + flag most-collapsed layers.
    ent = {n: s["entropy_normalized"] for n, s in per_layer.items()}
    gin = {n: s["gini"] for n, s in per_layer.items()}
    ent_vals = np.array(list(ent.values())) if ent else np.array([0.0])
    gin_vals = np.array(list(gin.values())) if gin else np.array([0.0])
    most_collapsed = sorted(ent.items(), key=lambda kv: kv[1])[:collapsed_layers]

    return {
        "num_moe_layers": len(per_layer),
        "num_experts": capture.num_experts,
        "top_k": capture.top_k,
        "global": glob,
        "layer_entropy_distribution": {
            "min": float(ent_vals.min()),
            "mean": float(ent_vals.mean()),
            "max": float(ent_vals.max()),
            "std": float(ent_vals.std()),
        },
        "layer_gini_distribution": {
            "min": float(gin_vals.min()),
            "mean": float(gin_vals.mean()),
            "max": float(gin_vals.max()),
            "std": float(gin_vals.std()),
        },
        "most_collapsed_layers": [
            {"layer": n, "entropy_normalized": e, "gini": gin[n]}
            for n, e in most_collapsed
        ],
        "per_layer": per_layer,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(report: dict, top_loaded: int) -> None:
    g = report["global"]
    line = "=" * 72
    print(line)
    print("MoE EXPERT-CALL DISTRIBUTION  (HF reference profiler)")
    print(line)
    print(
        f"MoE layers: {report['num_moe_layers']}   "
        f"experts/layer: {report['num_experts']}   top_k: {report['top_k']}"
    )
    print(f"Total routed token-slots: {g['token_slots']:,}   "
          f"total selections: {g['total_selections']:,}")
    print()
    print("GLOBAL (all layers aggregated)")
    print(f"  normalized entropy : {g['entropy_normalized']:.4f}  (1.0 = uniform)")
    print(f"  gini               : {g['gini']:.4f}  (0 = equal, ->1 = concentrated)")
    print(f"  dead experts       : {g['dead_experts']} / {g['num_experts']}")
    print(f"  underused (<0.1x)  : {g['underused_experts']} / {g['num_experts']}")
    print(f"  load max/mean/min  : {g['load_max']:.4f} / {g['load_mean']:.4f} / {g['load_min']:.4f}")
    print(f"  max/mean uniform-x : {g['max_uniform_ratio']:.2f}x / {g['mean_uniform_ratio']:.2f}x")
    print(f"  top {top_loaded} experts     : "
          + ", ".join(f"e{e['expert']}({e['load']:.3f})" for e in g["top_experts"]))
    print(f"  bottom {top_loaded} experts  : "
          + ", ".join(f"e{e['expert']}({e['load']:.3f})" for e in g["bottom_experts"]))
    print()

    ed = report["layer_entropy_distribution"]
    gd = report["layer_gini_distribution"]
    print("PER-LAYER ENTROPY  min/mean/max/std : "
          f"{ed['min']:.4f} / {ed['mean']:.4f} / {ed['max']:.4f} / {ed['std']:.4f}")
    print("PER-LAYER GINI     min/mean/max/std : "
          f"{gd['min']:.4f} / {gd['mean']:.4f} / {gd['max']:.4f} / {gd['std']:.4f}")
    print()

    print("MOST-COLLAPSED LAYERS (lowest entropy):")
    for item in report["most_collapsed_layers"]:
        print(f"  {item['layer']:<48} H={item['entropy_normalized']:.4f}  G={item['gini']:.4f}")
    print()

    print("PER-LAYER TABLE")
    hdr = f"  {'layer':<46} {'H':>7} {'gini':>7} {'dead':>5} {'maxX':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, s in report["per_layer"].items():
        print(f"  {name:<46} {s['entropy_normalized']:>7.4f} {s['gini']:>7.4f} "
              f"{s['dead_experts']:>5} {s['max_uniform_ratio']:>6.2f}x")
    print(line)


def save_plots(report: dict, prefix: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        log(f"matplotlib unavailable, skipping plots: {exc}")
        return []

    saved: list[str] = []
    layers = list(report["per_layer"].keys())
    num_experts = report["num_experts"]

    # 1) per-layer load heatmap (layers x experts)
    mat = np.zeros((len(layers), num_experts), dtype=np.float64)
    for i, name in enumerate(layers):
        c = np.array(report["per_layer"][name]["counts"], dtype=np.float64)
        tot = c.sum()
        mat[i] = c / tot if tot else c
    fig, ax = plt.subplots(figsize=(max(6, num_experts / 16), max(4, len(layers) / 3)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xlabel("expert index")
    ax.set_ylabel("MoE layer")
    ax.set_title("Per-layer expert load (fraction of routed slots)")
    fig.colorbar(im, ax=ax, label="load")
    fig.tight_layout()
    heat_path = f"{prefix}_heatmap.png"
    fig.savefig(heat_path, dpi=120)
    plt.close(fig)
    saved.append(heat_path)

    # 2) global load histogram
    gc = np.array(report["global"]["counts"], dtype=np.float64)
    tot = gc.sum()
    load = gc / tot if tot else gc
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(num_experts), np.sort(load)[::-1])
    ax.axhline(1.0 / num_experts, color="red", ls="--", label="uniform")
    ax.set_xlabel("expert (sorted by load, desc)")
    ax.set_ylabel("global load fraction")
    ax.set_title("Global expert load (sorted)")
    ax.legend()
    fig.tight_layout()
    hist_path = f"{prefix}_histogram.png"
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)
    saved.append(hist_path)
    return saved


# --------------------------------------------------------------------------- #
# Model / data loading
# --------------------------------------------------------------------------- #
def _resolve_dtype(dtype: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }[dtype]


def load_model_and_tokenizer(model_id: str, dtype: str, device_map: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    torch_dtype = _resolve_dtype(dtype)
    if not torch.cuda.is_available() and torch_dtype == torch.bfloat16:
        # MPS/CPU bf16 support is spotty; fall back to fp32 for the reference run.
        log("no CUDA; using float32 instead of bfloat16 for numerical stability")
        torch_dtype = torch.float32
        device_map = "cpu" if device_map == "auto" else device_map

    log(f"loading model: {model_id}  (dtype={torch_dtype}, device_map={device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _read_local_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    rows.append(json.loads(ln))
        return rows
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"unsupported local dataset extension: {path.suffix}")


def load_rows(dataset: str, split: str, num_samples: int) -> list[dict]:
    """Load `num_samples` rows, evenly spaced across the dataset."""
    p = Path(dataset)
    if p.exists():
        log(f"loading local dataset: {dataset}")
        rows = _read_local_rows(p)
    else:
        from datasets import load_dataset

        log(f"loading HF dataset: {dataset} (split={split})")
        ds = load_dataset(dataset, split=split)
        rows = ds
    n = len(rows)
    stride = max(n // num_samples, 1)
    idxs = [i * stride for i in range(num_samples) if i * stride < n]
    log(f"dataset has {n} rows; sampling {len(idxs)} (stride {stride})")
    return [dict(rows[i]) for i in idxs]


def rows_to_input_ids(rows, tokenizer, text_field, messages_field, max_tokens):
    """Turn dataset rows into a list of 1-D token-id tensors (truncated)."""
    import torch

    encs: list[Any] = []
    skipped = 0
    for row in rows:
        if messages_field:
            msgs = row.get(messages_field)
            if not msgs:
                skipped += 1
                continue
            ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
            )
        else:
            txt = row.get(text_field)
            if not txt:
                skipped += 1
                continue
            ids = tokenizer(txt, add_special_tokens=True)["input_ids"]
        ids = ids[:max_tokens]
        if len(ids) == 0:
            skipped += 1
            continue
        encs.append(torch.tensor(ids, dtype=torch.long))
    if skipped:
        log(f"skipped {skipped} empty/invalid rows")
    return encs


def run_forward_passes(model, encodings, capture: RouterCapture) -> int:
    """Run each sample through the model (one at a time, no_grad). Returns token count."""
    import torch

    device = next(model.parameters()).device
    total_tokens = 0
    with torch.no_grad():
        for i, ids in enumerate(encodings, 1):
            input_ids = ids.unsqueeze(0).to(device)
            model(input_ids=input_ids, use_cache=False)
            total_tokens += int(ids.shape[0])
            if i % 8 == 0 or i == len(encodings):
                log(f"forward pass {i}/{len(encodings)}  ({total_tokens:,} tokens)")
    return total_tokens


def _model_moe_params(model) -> tuple[int, int, bool]:
    cfg = model.config
    num_experts = getattr(cfg, "num_experts", None)
    top_k = getattr(cfg, "num_experts_per_tok", None)
    norm = bool(getattr(cfg, "norm_topk_prob", False))
    if num_experts is None or top_k is None:
        raise ValueError(
            "model config lacks num_experts/num_experts_per_tok — not a recognized MoE."
        )
    return int(num_experts), int(top_k), norm


# --------------------------------------------------------------------------- #
# Self-test (tiny random MoE on CPU)
# --------------------------------------------------------------------------- #
def _build_tiny_moe():
    import torch
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,  # every layer is MoE
        max_position_embeddings=128,
        norm_topk_prob=False,
    )
    model = Qwen3MoeForCausalLM(cfg)
    model.eval()
    return model, cfg


def _approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def self_test() -> int:
    """Validate hooks + stats math on a tiny random MoE. Returns process exit code."""
    import torch

    log("=== SELF-TEST: tiny random Qwen3-MoE on CPU ===")
    model, cfg = _build_tiny_moe()
    num_experts, top_k, norm = _model_moe_params(model)
    log(f"tiny config: layers={cfg.num_hidden_layers} experts={num_experts} "
        f"top_k={top_k} hidden={cfg.hidden_size}")

    capture = RouterCapture(num_experts, top_k, norm)
    n_hooked = capture.attach(model)
    expected_moe_layers = cfg.num_hidden_layers  # decoder_sparse_step=1
    assert n_hooked == expected_moe_layers, (
        f"hooked {n_hooked} gates, expected {expected_moe_layers}")
    log(f"PASS: hooked {n_hooked} MoE gate modules (one per layer)")

    # Synthetic prompts.
    torch.manual_seed(1)
    seqs = [torch.randint(0, cfg.vocab_size, (length,)) for length in (12, 23, 7, 31)]
    total_tokens = run_forward_passes(model, seqs, capture)
    log(f"ran {len(seqs)} synthetic prompts, {total_tokens} tokens")

    # Hooks fire on every layer.
    assert len(capture.layer_names) == expected_moe_layers, (
        f"only {len(capture.layer_names)} layers recorded selections")
    log(f"PASS: all {expected_moe_layers} MoE layers recorded selections")

    # Count conservation: total selections == tokens * top_k * num_moe_layers.
    grand = sum(int(c.sum()) for c in capture.counts.values())
    expected = total_tokens * top_k * expected_moe_layers
    assert grand == expected, f"count mismatch: {grand} != {expected}"
    log(f"PASS: total selections {grand} == tokens*top_k*layers {expected}")
    # And per layer: tokens * top_k.
    for name, c in capture.counts.items():
        assert int(c.sum()) == total_tokens * top_k, f"layer {name} bad count"
    log(f"PASS: every layer has exactly tokens*top_k = {total_tokens * top_k} selections")

    # --- stats math sanity ---
    n = 16
    # forced uniform: equal counts -> entropy 1.0, gini 0, no dead.
    uniform = np.full(n, 100, dtype=np.int64)
    h_u, g_u = normalized_entropy(uniform), gini(uniform)
    assert _approx(h_u, 1.0, 1e-9), f"uniform entropy {h_u} != 1.0"
    assert _approx(g_u, 0.0, 1e-9), f"uniform gini {g_u} != 0.0"
    assert int(np.sum(uniform == 0)) == 0
    log(f"PASS: forced-uniform -> entropy={h_u:.6f} (=1.0), gini={g_u:.6f} (=0.0)")

    # forced single expert: entropy 0, gini -> (n-1)/n, dead == n-1.
    single = np.zeros(n, dtype=np.int64)
    single[3] = 1000
    h_s, g_s = normalized_entropy(single), gini(single)
    dead_s = int(np.sum(single == 0))
    assert _approx(h_s, 0.0, 1e-12), f"single entropy {h_s} != 0.0"
    assert _approx(g_s, (n - 1) / n, 1e-9), f"single gini {g_s} != {(n - 1) / n}"
    assert dead_s == n - 1, f"single dead {dead_s} != {n - 1}"
    log(f"PASS: forced-single -> entropy={h_s:.6f} (=0.0), gini={g_s:.6f} "
        f"(=(n-1)/n={ (n-1)/n:.6f}), dead={dead_s} (=n-1={n-1})")

    # Build + print + (round-trip) the full report on the real tiny run.
    report = build_report(capture, top_loaded=3, collapsed_layers=2)
    print_report(report, top_loaded=3)
    # JSON round-trip must not raise (machine-readable contract).
    json.loads(json.dumps(report))
    log("PASS: report serializes to JSON")

    log("=== SELF-TEST PASSED ===")
    return 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.dataset:
        log("ERROR: --dataset is required (or use --self-test)")
        return 2
    if bool(args.text_field) == bool(args.messages_field):
        log("ERROR: pass exactly one of --text-field / --messages-field")
        return 2

    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, args.device_map)
    num_experts, top_k, norm = _model_moe_params(model)
    log(f"MoE: num_experts={num_experts} top_k={top_k} norm_topk_prob={norm}")

    capture = RouterCapture(num_experts, top_k, norm)
    n_hooked = capture.attach(model)
    if n_hooked == 0:
        log("ERROR: found no MoE gate modules to hook — is this an MoE model?")
        return 3
    log(f"hooked {n_hooked} MoE gate modules")

    rows = load_rows(args.dataset, args.split, args.num_samples)
    encodings = rows_to_input_ids(
        rows, tokenizer, args.text_field, args.messages_field, args.max_tokens
    )
    if not encodings:
        log("ERROR: no usable samples after tokenization")
        return 4
    log(f"profiling {len(encodings)} samples (max_tokens={args.max_tokens})")

    total_tokens = run_forward_passes(model, encodings, capture)
    capture.detach()
    log(f"done: {total_tokens:,} tokens in {time.time() - t0:.1f}s")

    report = build_report(capture, args.top_loaded, args.collapsed_layers)
    report["meta"] = {
        "model": args.model,
        "dataset": args.dataset,
        "num_samples": len(encodings),
        "total_tokens": total_tokens,
        "max_tokens": args.max_tokens,
    }
    print_report(report, args.top_loaded)

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        log(f"wrote JSON report: {args.output}")

    if args.plot:
        saved = save_plots(report, args.plot_prefix)
        for s in saved:
            log(f"wrote plot: {s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
