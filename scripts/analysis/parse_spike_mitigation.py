#!/usr/bin/env python3
"""
Parse spike-mitigation (StaleClip / ZClip) metrics from SkyRL training logs.

Aggregates WANDB_MIRROR step lines across an entire chain-restart series (handles
multiple .out files in the same experiment), produces a per-step trajectory
table, and reports engagement statistics for either mechanism.

Outputs:
- A trajectory table (stdout) showing reward, grad_norm, entropy, log-ratio
  metrics, and the spike-mitigation engagement columns side-by-side.
- An engagement summary: how many steps the mechanism actually fired,
  the distribution of `scale` (StaleClip) or `effective_max` (ZClip)
  values, and the first step it engaged.
- Optional CSV export of the full per-step trajectory.

Use this as a complement to parse_skyrl_metrics.py when running ablations on
the spike-mitigation knobs (StaleClip / ZClip): parse_skyrl_metrics covers the
broad health view; this script answers "did the experimental mechanism
actually engage, and how often?".

Usage:
    python parse_spike_mitigation.py <log_dir>
    python parse_spike_mitigation.py <log_dir> --csv-out traj.csv
    python parse_spike_mitigation.py <log_dir> --mechanism stale_clip
    python parse_spike_mitigation.py <log_dir> --mechanism z_clip

The default --mechanism is "auto", which prefers stale_clip metrics if
present and falls back to z_clip.
"""

import argparse
import csv
import json
import os
import re
import sys
from glob import glob
from pathlib import Path
from typing import Any


WANDB_MIRROR_RE = re.compile(r"WANDB_MIRROR kind=train step=(\d+) metrics=(\{.*\})")


def collect_step_metrics(log_files: list[str]) -> dict[int, tuple[str, dict[str, Any]]]:
    """Parse WANDB_MIRROR lines from a list of log files.

    Returns ``{step: (jobid, metrics_dict)}``. Later writes win for the same
    step number, so the freshest chain-restart segment overrides earlier ones.
    Sort ``log_files`` by mtime before calling for the right precedence.
    """
    step_data: dict[int, tuple[str, dict[str, Any]]] = {}
    for lf in log_files:
        m_job = re.search(r"_(\d+)\.out$", lf)
        jobid = m_job.group(1) if m_job else "?"
        try:
            with open(lf, "r", errors="replace") as f:
                for line in f:
                    m = WANDB_MIRROR_RE.search(line)
                    if not m:
                        continue
                    step = int(m.group(1))
                    try:
                        metrics = json.loads(m.group(2))
                    except Exception:
                        continue
                    step_data[step] = (jobid, metrics)
        except Exception as e:
            print(f"# warning: could not read {lf}: {e}", file=sys.stderr)
    return step_data


def detect_mechanism(step_data: dict[int, tuple[str, dict[str, Any]]]) -> str:
    """Auto-detect which mechanism is in play by checking the latest record."""
    if not step_data:
        return "none"
    _, latest = step_data[max(step_data)]
    if any(k.startswith("policy/stale_clip/") for k in latest):
        return "stale_clip"
    if any(k.startswith("policy/z_clip/") for k in latest):
        return "z_clip"
    return "none"


def fmt(v, fmt_str: str = "{:.3f}") -> str:
    if v is None:
        return "    —"
    try:
        return fmt_str.format(float(v))
    except Exception:
        return str(v)


def print_trajectory(
    step_data: dict[int, tuple[str, dict[str, Any]]],
    mechanism: str,
) -> None:
    """Print a wide trajectory table to stdout."""
    if mechanism == "stale_clip":
        extra_hdr = ("sc_trig", "sc_scale", "sc_re", "sc_stmin")
        extra_keys = (
            "policy/stale_clip/triggered",
            "policy/stale_clip/scale",
            "policy/stale_clip/rolling_entropy",
            "policy/stale_clip/stale_min",
        )
        extra_fmts = ("{:.2f}", "{:.3f}", "{:.4f}", "{:.2f}")
    elif mechanism == "z_clip":
        extra_hdr = ("zc_trig", "zc_warm", "zc_emax", "zc_z")
        extra_keys = (
            "policy/z_clip/triggered",
            "policy/z_clip/warmup_remaining",
            "policy/z_clip/effective_max",
            "policy/z_clip/z_score",
        )
        extra_fmts = ("{:.2f}", "{:.0f}", "{:.3f}", "{:.2f}")
    else:
        extra_hdr = ()
        extra_keys = ()
        extra_fmts = ()

    base_hdr = ("step", "job", "reward", "p@8", "grad", "entropy", "lr_max", "stale_mean")
    base_keys = (
        None,  # step
        None,  # job
        "reward/avg_raw_reward",
        "reward/avg_pass_at_8",
        "policy/raw_grad_norm",
        "policy/policy_entropy",
        "policy/log_ratio_abs_max",
        "async/staleness_mean",
    )
    base_fmts = (None, None, "{:.3f}", "{:.3f}", "{:.3f}", "{:.4f}", "{:.3f}", "{:.3f}")

    hdr = base_hdr + extra_hdr
    print("  ".join(f"{c:>10}" for c in hdr))
    print("-" * (12 * len(hdr)))

    for step in sorted(step_data):
        job, m = step_data[step]
        row: list[str] = []
        for i, (h, k, f) in enumerate(zip(base_hdr, base_keys, base_fmts)):
            if k is None and h == "step":
                row.append(str(step))
            elif k is None and h == "job":
                row.append(job[-4:])
            else:
                row.append(fmt(m.get(k), f))
        for k, f in zip(extra_keys, extra_fmts):
            row.append(fmt(m.get(k), f))
        print("  ".join(f"{c:>10}" for c in row))


def engagement_summary(
    step_data: dict[int, tuple[str, dict[str, Any]]],
    mechanism: str,
) -> None:
    """Summarize how often + how strongly the spike-mitigation mechanism fired."""
    steps = sorted(step_data)
    if not steps or mechanism == "none":
        print("\n(No spike-mitigation engagement metrics found in these logs.)")
        return

    trig_key = f"policy/{mechanism}/triggered"
    triggered = [(s, m) for s in steps for _, m in [step_data[s]] if m.get(trig_key, 0) > 0]

    print()
    print(f"== {mechanism} ENGAGEMENT SUMMARY ==")
    print(f"steps covered: {steps[0]} -> {steps[-1]}  (n={len(steps)})")
    print(f"triggered>0:   {len(triggered)} of {len(steps)}  ({100*len(triggered)/len(steps):.1f}%)")

    if mechanism == "stale_clip":
        scale_key = "policy/stale_clip/scale"
        damped = [s for s in steps if (step_data[s][1].get(scale_key) or 1.0) < 1.0]
        if damped:
            print(f"steps with scale<1.0: {len(damped)}  (first: step {damped[0]})")
            # Scale histogram
            scale_buckets: dict[float, int] = {}
            for s in damped:
                v = round(step_data[s][1].get(scale_key, 1.0), 3)
                scale_buckets[v] = scale_buckets.get(v, 0) + 1
            print("scale distribution (damping_factor: n_steps):")
            for v in sorted(scale_buckets):
                print(f"  scale={v:.3f}: {scale_buckets[v]} steps")

    elif mechanism == "z_clip":
        warm_key = "policy/z_clip/warmup_remaining"
        warmup_done_step = next(
            (s for s in steps if (step_data[s][1].get(warm_key) or 0) == 0), None
        )
        if warmup_done_step is not None:
            print(f"warmup completed at step: {warmup_done_step}")
        else:
            last_warm = step_data[steps[-1]][1].get(warm_key)
            if last_warm is not None:
                print(f"warmup NOT YET COMPLETE — warmup_remaining={last_warm:.0f} at latest step {steps[-1]}")
        eff_max_key = "policy/z_clip/effective_max"
        clipped = [
            s for s in steps
            if (step_data[s][1].get(eff_max_key) or 1e9) < (
                step_data[s][1].get("policy/raw_grad_norm") or 0
            )
        ]
        print(f"steps where effective_max < raw_grad_norm (clipping active): {len(clipped)}")


def collapse_signals(step_data: dict[int, tuple[str, dict[str, Any]]]) -> None:
    """Report CLAUDE.md collapse-warning signals across the trajectory.

    Per CLAUDE.md "Collapse warning rule (combine signals — single metrics are noisy)":
    A run is at collapse risk when ≥2 of:
      - raw_grad_norm > 1.0
      - policy_entropy deviates >30% from 10-step rolling trend
      - log_ratio_abs_mean rises >2× recent window mean
      - log_ratio_abs_max > 0.5
    """
    steps = sorted(step_data)
    if not steps:
        return
    print()
    print("== CLAUDE.md COLLAPSE-SIGNAL SCAN ==")
    sustained_grad = 0
    flags = []
    for s in steps:
        m = step_data[s][1]
        g = m.get("policy/raw_grad_norm") or 0
        lrmax = m.get("policy/log_ratio_abs_max") or 0
        n_dp50 = m.get("policy/n_tokens_dp_gt_50pct") or 0
        cur_flags = []
        if g > 1.0:
            sustained_grad += 1
        else:
            sustained_grad = 0
        if sustained_grad >= 2:
            cur_flags.append(f"grad>1.0 ×{sustained_grad}")
        if lrmax > 0.5:
            cur_flags.append(f"lr_max={lrmax:.2f}")
        if n_dp50 > 50:
            cur_flags.append(f"dp_50pct={n_dp50:.0f}")
        if len(cur_flags) >= 2:
            flags.append((s, cur_flags))
    if flags:
        print(f"Steps with ≥2 collapse signals firing simultaneously: {len(flags)}")
        for s, fl in flags[:15]:
            print(f"  step={s}: {', '.join(fl)}")
        if len(flags) > 15:
            print(f"  ... ({len(flags) - 15} more)")
    else:
        print("No steps with ≥2 collapse signals firing simultaneously. Healthy.")


def write_csv(
    step_data: dict[int, tuple[str, dict[str, Any]]],
    out_path: Path,
) -> None:
    """Write the full per-step metrics dict to CSV."""
    steps = sorted(step_data)
    if not steps:
        return
    # Union of keys across all steps so columns line up.
    keys: list[str] = sorted({k for s in steps for k in step_data[s][1]})
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "jobid", *keys])
        for s in steps:
            jobid, m = step_data[s]
            w.writerow([s, jobid, *(m.get(k, "") for k in keys)])
    print(f"\nWrote {len(steps)} rows × {len(keys) + 2} cols to {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "log_dir",
        help="Directory containing the experiment's logs. Searches recursively for "
             "<dir>/**/logs/*.out (excludes anything under ray_logs/). Accepts a single .out file too.",
    )
    p.add_argument(
        "--mechanism",
        choices=["auto", "stale_clip", "z_clip"],
        default="auto",
        help="Which mechanism's metrics to highlight (default: auto-detect).",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to dump the full per-step metrics dict as CSV.",
    )
    args = p.parse_args()

    # Resolve log files.
    src = Path(args.log_dir)
    if src.is_file() and src.suffix == ".out":
        log_files = [str(src)]
    elif src.is_dir():
        log_files = [
            f for f in glob(str(src / "**" / "logs" / "*.out"), recursive=True)
            if "/ray_logs/" not in f
        ]
        if not log_files:
            # Fall back: any .out under the dir (excluding ray_logs).
            log_files = [
                f for f in glob(str(src / "**" / "*.out"), recursive=True)
                if "/ray_logs/" not in f
            ]
    else:
        print(f"error: {src} is neither a .out file nor a directory", file=sys.stderr)
        return 2

    # Sort by mtime so later writes win in the dedup.
    log_files.sort(key=lambda f: os.path.getmtime(f))
    if not log_files:
        print(f"error: no .out logs found under {src}", file=sys.stderr)
        return 2

    print(f"Scanning {len(log_files)} log files...")
    step_data = collect_step_metrics(log_files)
    if not step_data:
        print("No WANDB_MIRROR step records found.")
        return 1

    mechanism = args.mechanism
    if mechanism == "auto":
        mechanism = detect_mechanism(step_data)
    print(f"Detected mechanism: {mechanism}")
    print()

    print_trajectory(step_data, mechanism)
    engagement_summary(step_data, mechanism)
    collapse_signals(step_data)

    if args.csv_out:
        write_csv(step_data, args.csv_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
