#!/usr/bin/env python3
"""
Generate multi-round negotiation task suite in Harbor format.

Each task has a scenario (r_s, r_b, role, item context, K, p_min, p_max)
and includes a stateful LLM counterpart script. The agent negotiates over
up to K rounds using `python /app/counterpart.py offer|accept|reject`.

Verifier reads /app/negotiation_log.json and writes
  reward = A * (u_agent / (TS + eps))
to /logs/verifier/reward.txt.

Usage:
  python -m data.negotiation.generate --num-instances 100 --output-dir ./out --seed 42
  python -m data.negotiation.generate --num-instances 50 --source data/negotiation/craigslist_bargains/train.csv
  python -m data.negotiation.generate --num-instances 10 --output-dir ./out --upload-repo OpenThoughts-Agent/negotiation-tasks
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

# data.commons is imported only inside create_negotiation_task and main() so that
# tests can import this module without pulling in pyarrow/numpy/harbor.

# Verifier script: reads negotiation_log.json and computes reward
VERIFIER_SCRIPT = """\
#!/bin/bash
set -euo pipefail

# Multi-round negotiation verifier: reward = A * (u_agent / (TS + eps))
# Reads: /app/negotiation_log.json, /app/data/scenario.json
# Writes: /logs/verifier/reward.txt (always written, even on missing log)

mkdir -p /logs/verifier

python3 << 'VERIFIER'
import json
import os

REWARD_FILE = "/logs/verifier/reward.txt"
LOG_PATH = "/app/negotiation_log.json"
SCENARIO_PATH = "/app/data/scenario.json"
EPS = 1e-9

def main():
    try:
        with open(SCENARIO_PATH) as f:
            scenario = json.load(f)
        r_s = float(scenario["r_s"])
        r_b = float(scenario["r_b"])
        role = scenario.get("role", "seller").strip().lower()

        if not os.path.isfile(LOG_PATH):
            # Negotiation never started
            with open(REWARD_FILE, "w") as f:
                f.write("0\\n")
            return

        with open(LOG_PATH) as f:
            log = json.load(f)

        state = log.get("state", {})
        outcome = state.get("outcome")
        final_price = state.get("final_price")

        if outcome == "agreement" and final_price is not None:
            price = float(final_price)
            A = 1.0 if (r_s <= price <= r_b) else 0.0
            if A == 1.0:
                TS = r_b - r_s
                u_agent = (price - r_s) if role == "seller" else (r_b - price)
                reward = u_agent / (TS + EPS)
            else:
                reward = 0.0
        else:
            reward = 0.0
    except Exception:
        reward = 0.0

    with open(REWARD_FILE, "w") as f:
        f.write(f"{reward}\\n")

if __name__ == "__main__":
    main()
VERIFIER
"""

# Dockerfile: install openai, copy counterpart script
DOCKERFILE_TEMPLATE = """\
FROM ubuntu:24.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install openai --break-system-packages

COPY counterpart.py /app/counterpart.py
"""

# task.toml: longer agent timeout for multi-round negotiation
TASK_TOML_TEMPLATE = """\
version = "1.0"

[agent]
timeout_sec = 1800.0

[metadata]
author_name = "OpenThoughts-Agent"
author_email = "negotiation@openthoughts-agent.invalid"
difficulty = "medium"
category = "negotiation"
tags = ["negotiation", "bargaining", "multi-round"]

[verifier]
restart_environment = false
timeout_sec = 60.0
"""

# Placeholder items when no --source (synthetic)
DEFAULT_ITEMS: List[Dict[str, Any]] = [
    # --- Electronics (cheap) ---
    {
        "title": "Wireless Mouse",
        "description": "Like-new wireless mouse, 2.4GHz, lightly used.",
        "list_price": 25.0,
        "category": "electronics",
        "r_s": 18.0,   # ~72% of list
        "r_b": 24.0,   # ~96% of list; ZOPA = 6
    },
    {
        "title": "USB-C Hub",
        "description": "7-in-1 USB-C hub with HDMI, USB 3.0, SD card reader.",
        "list_price": 35.0,
        "category": "electronics",
        "r_s": 25.0,   # ~71% of list
        "r_b": 34.0,   # ~97% of list; ZOPA = 9
    },
    {
        "title": "Car Phone Mount",
        "description": "Magnetic dashboard phone mount, universal fit.",
        "list_price": 20.0,
        "category": "automotive",
        "r_s": 13.0,   # ~65% of list
        "r_b": 18.0,   # ~90% of list; ZOPA = 5
    },
    {
        "title": "Yoga Mat",
        "description": "Non-slip 6mm yoga mat with carrying strap.",
        "list_price": 30.0,
        "category": "sports-outdoors",
        "r_s": 20.0,   # ~67% of list
        "r_b": 28.0,   # ~93% of list; ZOPA = 8
    },
    {
        "title": "Strategy Board Game",
        "description": "Award-winning strategy board game, complete set.",
        "list_price": 40.0,
        "category": "toys-games",
        "r_s": 28.0,   # ~70% of list
        "r_b": 38.0,   # ~95% of list; ZOPA = 10
    },
    # --- Electronics / tools (mid-low) ---
    {
        "title": "Cordless Screwdriver",
        "description": "Compact cordless screwdriver with 20 bit set.",
        "list_price": 35.0,
        "category": "tools",
        "r_s": 25.0,   # ~71% of list
        "r_b": 33.0,   # ~94% of list; ZOPA = 8
    },
    {
        "title": "Coffee Maker",
        "description": "12-cup programmable drip coffee maker, stainless steel.",
        "list_price": 45.0,
        "category": "home-kitchen",
        "r_s": 32.0,   # ~71% of list
        "r_b": 42.0,   # ~93% of list; ZOPA = 10
    },
    {
        "title": "Bookshelf",
        "description": "5-tier bookshelf, wood finish, easy assembly.",
        "list_price": 60.0,
        "category": "furniture",
        "r_s": 45.0,   # ~75% of list
        "r_b": 55.0,   # ~92% of list; ZOPA = 10
    },
    # --- Mid-range ---
    {
        "title": "Webcam HD 1080p",
        "description": "Full HD webcam with built-in microphone and privacy cover.",
        "list_price": 55.0,
        "category": "electronics",
        "r_s": 40.0,   # ~73% of list
        "r_b": 52.0,   # ~95% of list; ZOPA = 12
    },
    {
        "title": "Bluetooth Speaker",
        "description": "Portable waterproof Bluetooth speaker, 20hr battery.",
        "list_price": 65.0,
        "category": "electronics",
        "r_s": 48.0,   # ~74% of list
        "r_b": 62.0,   # ~95% of list; ZOPA = 14
    },
    {
        "title": "Power Drill",
        "description": "18V cordless power drill with 2 batteries and charger.",
        "list_price": 65.0,
        "category": "tools",
        "r_s": 48.0,   # ~74% of list
        "r_b": 62.0,   # ~95% of list; ZOPA = 14
    },
    {
        "title": "Air Fryer",
        "description": "4-quart digital air fryer with 8 cooking presets.",
        "list_price": 75.0,
        "category": "home-kitchen",
        "r_s": 55.0,   # ~73% of list
        "r_b": 70.0,   # ~93% of list; ZOPA = 15
    },
    {
        "title": "Portable Jump Starter",
        "description": "2000A peak portable car jump starter with USB charging ports.",
        "list_price": 75.0,
        "category": "automotive",
        "r_s": 55.0,   # ~73% of list
        "r_b": 70.0,   # ~93% of list; ZOPA = 15
    },
    {
        "title": "Mechanical Keyboard",
        "description": "Tenkeyless mechanical keyboard, blue switches, RGB backlight.",
        "list_price": 80.0,
        "category": "electronics",
        "r_s": 60.0,   # ~75% of list
        "r_b": 78.0,   # ~98% of list; ZOPA = 18
    },
    {
        "title": "Hiking Backpack",
        "description": "40L hiking backpack with hydration bladder sleeve and rain cover.",
        "list_price": 85.0,
        "category": "sports-outdoors",
        "r_s": 62.0,   # ~73% of list
        "r_b": 80.0,   # ~94% of list; ZOPA = 18
    },
    {
        "title": "Bedside Table",
        "description": "2-drawer solid wood bedside table, walnut finish.",
        "list_price": 90.0,
        "category": "furniture",
        "r_s": 65.0,   # ~72% of list
        "r_b": 85.0,   # ~94% of list; ZOPA = 20
    },
    # --- Higher-price ---
    {
        "title": "Office Chair",
        "description": "Ergonomic office chair, lumbar support, adjustable armrests.",
        "list_price": 120.0,
        "category": "furniture",
        "r_s": 85.0,   # ~71% of list
        "r_b": 110.0,  # ~92% of list; ZOPA = 25
    },
    {
        "title": "External SSD 1TB",
        "description": "1TB portable external SSD, USB 3.2, 1000MB/s read speed.",
        "list_price": 120.0,
        "category": "electronics",
        "r_s": 90.0,   # ~75% of list
        "r_b": 115.0,  # ~96% of list; ZOPA = 25
    },
    {
        "title": "Smart Watch",
        "description": "Fitness smart watch with GPS, heart rate monitor, 7-day battery.",
        "list_price": 180.0,
        "category": "electronics",
        "r_s": 140.0,  # ~78% of list
        "r_b": 175.0,  # ~97% of list; ZOPA = 35
    },
    {
        "title": "Standing Desk",
        "description": "Electric height-adjustable standing desk, 55x28in, memory presets.",
        "list_price": 280.0,
        "category": "furniture",
        "r_s": 220.0,  # ~79% of list
        "r_b": 268.0,  # ~96% of list; ZOPA = 48
    },
]

# Path to the counterpart simulator script (copied into each task dir)
_COUNTERPART_SCRIPT_PATH = Path(__file__).resolve().parent / "counterpart.py"


def _resample_zopa_from_list_price(list_price: float, rng: random.Random) -> tuple[float, float]:
    """Sample r_s and r_b from list_price; ranges (0.7–0.95) and (1.05–1.3) ensure r_s < r_b."""
    r_s = float(list_price * rng.uniform(0.7, 0.95))
    r_b = float(list_price * rng.uniform(1.05, 1.3))
    return r_s, r_b


def load_items_from_csv(
    path: Path,
    zopa_mode: str = "filter",
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    """Load items from CraigslistBargain-style CSV.

    zopa_mode:
      - filter: skip rows where buyer_target/seller_target are missing or seller_target > buyer_target.
      - resample: for those rows set r_s = list_price * U(0.7, 0.95), r_b = list_price * U(1.05, 1.3).
    """
    if rng is None and zopa_mode == "resample":
        rng = random.Random()
    items: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                list_price = float(row.get("list_price", 0))
            except (ValueError, TypeError):
                continue
            if list_price <= 0:
                continue
            try:
                buyer_target = float(row.get("buyer_target", 0))
                seller_target = float(row.get("seller_target", 0))
                targets_ok = True
            except (ValueError, TypeError):
                buyer_target = seller_target = 0.0
                targets_ok = False
            missing = not targets_ok
            no_zopa = seller_target > buyer_target
            if zopa_mode == "filter":
                if missing or no_zopa:
                    continue
                r_s, r_b = seller_target, buyer_target
            else:  # resample
                if missing or no_zopa:
                    r_s, r_b = _resample_zopa_from_list_price(list_price, rng)
                else:
                    r_s, r_b = seller_target, buyer_target
                    if r_s > r_b:
                        r_s, r_b = _resample_zopa_from_list_price(list_price, rng)
            items.append({
                "title": (row.get("title") or "").strip() or "Item",
                "description": (row.get("description") or "").strip() or "",
                "list_price": float(list_price),
                "category": (row.get("category") or "").strip() or "general",
                "r_s": float(r_s),
                "r_b": float(r_b),
            })
    return items


def load_items(
    source: Optional[str],
    default: List[Dict[str, Any]],
    zopa_mode: str = "filter",
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load item list from --source (CSV or directory) or use default."""
    if not source:
        return default
    path = Path(source)
    rng = random.Random(seed) if seed is not None else None
    if path.is_dir():
        for name in ("train.csv", "items.csv"):
            p = path / name
            if p.exists():
                return load_items_from_csv(p, zopa_mode=zopa_mode, rng=rng)
        raise FileNotFoundError(f"No train.csv or items.csv in {path}")
    if path.exists():
        return load_items_from_csv(path, zopa_mode=zopa_mode, rng=rng)
    raise FileNotFoundError(f"Source not found: {path}")


def _derive_task_params(item: Dict[str, Any], role: str) -> Dict[str, Any]:
    """Derive K, p_min, p_max, delta_max, counterpart_opening from item + role."""
    list_price = float(item.get("list_price", 100.0))
    r_s = float(item["r_s"])
    r_b = float(item["r_b"])

    p_min = round(max(0.0, list_price * 0.5), 2)
    p_max = round(list_price * 2.0, 2)
    K = 10
    delta_max = round((p_max - p_min) * 0.1, 2)

    # Counterpart opens aggressively (20% away from their reservation)
    if role == "seller":
        # counterpart is buyer with reservation r_b
        counterpart_opening = round(max(p_min, r_b * 0.80), 2)
    else:
        # counterpart is seller with reservation r_s
        counterpart_opening = round(min(p_max, r_s * 1.20), 2)

    return {
        "K": K,
        "p_min": p_min,
        "p_max": p_max,
        "delta_max": delta_max,
        "counterpart_opening": counterpart_opening,
    }


def build_scenario(
    item: Dict[str, Any],
    role: str,
    seed: int,
) -> Dict[str, Any]:
    """Build scenario.json dict. Values coerced to JSON-serializable types."""
    params = _derive_task_params(item, role)
    return {
        "r_s": float(item["r_s"]),
        "r_b": float(item["r_b"]),
        "role": str(role).strip().lower() or "seller",
        "seed": int(seed),
        "K": params["K"],
        "p_min": params["p_min"],
        "p_max": params["p_max"],
        "delta_max": params["delta_max"],
        "counterpart_opening": params["counterpart_opening"],
        "item_context": {
            "title": str(item.get("title", "Item")),
            "description": str(item.get("description", "")),
            "list_price": float(item.get("list_price", 0)),
            "category": str(item.get("category", "general")),
        },
    }


def build_instruction(role: str, item: Dict[str, Any], scenario: Dict[str, Any]) -> str:
    """Build instruction.md content for a multi-round negotiation task."""
    title = item.get("title", "Item")
    description = item.get("description", "")
    list_price = item.get("list_price", 0)
    category = item.get("category", "general")

    r_s = float(scenario["r_s"])
    r_b = float(scenario["r_b"])
    K = scenario.get("K", 10)
    p_min = scenario.get("p_min", 0.0)
    p_max = scenario.get("p_max", list_price * 2)
    counterpart_opening = scenario.get("counterpart_opening", "?")

    agent_reservation = r_s if role == "seller" else r_b
    counterpart_role = "buyer" if role == "seller" else "seller"
    action_verb = "sell" if role == "seller" else "buy"
    reservation_label = "minimum acceptable price" if role == "seller" else "maximum acceptable price"
    reservation_note = (
        "Do not sell below this price." if role == "seller"
        else "Do not pay above this price."
    )
    score_direction = "higher" if role == "seller" else "lower"

    lines = [
        "# Bilateral Price Negotiation",
        "",
        f"You are a **{role}**. Negotiate to {action_verb} the item below.",
        "",
        "## Item",
        f"- **Title:** {title}",
        f"- **Category:** {category}",
        f"- **List price:** ${list_price}",
    ]
    if description:
        lines.extend(["", "**Description:**", description])
    lines.extend([
        "",
        "## Your position",
        f"- **Your {reservation_label} (reservation):** ${agent_reservation:.2f} — {reservation_note}",
        f"- **Allowed price range:** ${p_min:.2f} – ${p_max:.2f}",
        f"- **Max rounds:** {K}",
        "",
        f"The {counterpart_role}'s reservation price is **unknown to you**.",
        "",
        "## Current state",
        f"**Counterpart's opening offer:** ${counterpart_opening:.2f}",
        "It is your turn.",
        "",
        "## Commands",
        "",
        "**Make an offer** (must be within the allowed price range):",
        "```bash",
        "python /app/counterpart.py offer <price>",
        "```",
        "",
        f"**Accept the {counterpart_role}'s last offer:**",
        "```bash",
        "python /app/counterpart.py accept",
        "```",
        "",
        "**Walk away (no deal):**",
        "```bash",
        "python /app/counterpart.py reject",
        "```",
        "",
        "## Response format",
        "",
        "Each `offer` command returns JSON, for example:",
        "```json",
        json.dumps({
            "round": 1,
            "counterpart_offer": round(counterpart_opening * (1.05 if role == "seller" else 0.95), 2),
            "counterpart_message": "I can move a bit, but I need a better price from you.",
            "accepted": False,
            "done": False,
            "rounds_remaining": K - 1,
        }, indent=2),
        "```",
        "",
        "When `accepted: true` or `done: true`, the negotiation is over.",
        "",
        "## Scoring",
        "",
        "- No deal → reward = 0",
        "- Deal at price **p** → reward = your surplus / total surplus ∈ [0, 1]",
        "",
        f"Maximize your reward: drive the price **{score_direction}** while still reaching agreement.",
    ])
    return "\n".join(lines)


def create_negotiation_task(
    output_dir: Path,
    task_id: int,
    scenario: Dict[str, Any],
    instruction: str,
) -> Path:
    """Create one Harbor-format task dir with multi-round negotiation support."""
    from data.commons import create_task_directory_unified

    task_dir = create_task_directory_unified(
        output_dir=output_dir,
        task_id=task_id,
        instruction_content=instruction,
        dataset_prefix="negotiation",
        metadata={"scenario_seed": scenario.get("seed"), "role": scenario.get("role")},
        test_sh_content=VERIFIER_SCRIPT,
        task_toml_content=TASK_TOML_TEMPLATE,
        dockerfile_content=DOCKERFILE_TEMPLATE,
    )

    # Write scenario.json (counterpart.py reads this; agent can also read it —
    # honest-agent assumption as documented in README)
    data_dir = task_dir / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")

    # Copy counterpart.py into the task dir (Dockerfile COPYs it into /app/)
    if _COUNTERPART_SCRIPT_PATH.exists():
        shutil.copy2(_COUNTERPART_SCRIPT_PATH, task_dir / "environment" / "counterpart.py")

    return task_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-round negotiation task suite (Harbor format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--num-instances", "-n",
        type=int,
        default=10,
        help="Number of task instances to generate (default: 10)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for task dirs (default: data/negotiation/out)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=None,
        help="Path to CraigslistBargain CSV or directory containing train.csv (optional)",
    )
    parser.add_argument(
        "--zopa-mode",
        type=str,
        choices=("filter", "resample"),
        default="filter",
        help="filter: skip rows without ZOPA; resample: regenerate reservations from list_price (default: filter)",
    )
    parser.add_argument(
        "--upload-repo",
        type=str,
        default=None,
        help="If set, upload generated tasks to this HF repo (e.g. OpenThoughts-Agent/negotiation-tasks)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Do not upload even if --upload-repo is set (for testing)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (ROOT / "data" / "negotiation" / "out")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    items = load_items(
        args.source,
        DEFAULT_ITEMS,
        zopa_mode=args.zopa_mode,
        seed=args.seed,
    )
    if not items:
        raise SystemExit("No items loaded. Check --source or use default items.")

    roles = ["buyer", "seller"]
    created = []
    for i in range(args.num_instances):
        item = rng.choice(items)
        role = rng.choice(roles)
        seed = args.seed + i
        scenario = build_scenario(item, role, seed)
        instruction = build_instruction(role, item, scenario)
        task_dir = create_negotiation_task(output_dir, i, scenario, instruction)
        created.append(task_dir)

    print(f"Generated {len(created)} tasks under {output_dir}")

    if args.upload_repo and not args.no_upload:
        from data.commons import upload_tasks_to_hf
        print(f"Uploading to {args.upload_repo}...")
        upload_tasks_to_hf(str(output_dir), args.upload_repo)
        print("Upload complete.")
    elif args.upload_repo and args.no_upload:
        print("Skipping upload (--no-upload).")


if __name__ == "__main__":
    main()
