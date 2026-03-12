#!/usr/bin/env python3
"""
LLM-based counterpart for bilateral price negotiation (Harbor task format).

Called by the agent each round as a CLI tool. Maintains state between calls
in /app/negotiation_state.json. Uses GPT-4o (OPENAI_API_KEY env var)
to generate counterpart responses; falls back to a rule-based policy if no
API key is present.

Usage:
  python /app/counterpart.py offer <price>   # submit an offer
  python /app/counterpart.py accept          # accept counterpart's last offer
  python /app/counterpart.py reject          # reject and walk away

Paths (all under /app/):
  data/scenario.json        — read-only; holds r_s, r_b, role, K, p_min, p_max, etc.
  negotiation_state.json    — read/write; round-by-round state
  negotiation_log.json      — written on every call; verifier reads this for reward
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCENARIO_PATH = Path("/app/data/scenario.json")
STATE_PATH = Path("/app/negotiation_state.json")
LOG_PATH = Path("/app/negotiation_log.json")
EPS = 1e-9

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_scenario() -> Dict[str, Any]:
    with open(SCENARIO_PATH) as f:
        return json.load(f)


def _counterpart_opening(counterpart_role: str, counterpart_reservation: float,
                         p_min: float, p_max: float) -> float:
    """Rule-based aggressive opening offer for the counterpart."""
    if counterpart_role == "buyer":
        # Buyer opens 20% below their reservation (low-ball)
        return round(max(p_min, counterpart_reservation * 0.80), 2)
    else:
        # Seller opens 20% above their reservation (high-ball)
        return round(min(p_max, counterpart_reservation * 1.20), 2)


def init_state(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Build initial state from scenario. Called on first interaction."""
    role = scenario["role"]
    counterpart_role = "buyer" if role == "seller" else "seller"
    counterpart_reservation = float(scenario["r_b"]) if role == "seller" else float(scenario["r_s"])

    K = int(scenario.get("K", 10))
    p_min = float(scenario.get("p_min", 0.0))
    p_max = float(scenario.get("p_max", 200.0))

    # Use pre-computed opening if present, otherwise compute it
    opening = float(scenario.get("counterpart_opening",
                                  _counterpart_opening(counterpart_role, counterpart_reservation, p_min, p_max)))

    return {
        "round": 0,
        "done": False,
        "outcome": None,
        "final_price": None,
        "counterpart_role": counterpart_role,
        "counterpart_reservation": counterpart_reservation,
        "counterpart_last_offer": opening,
        "agent_offers": [],
        "counterpart_offers": [opening],
        "history": [
            {
                "turn": "counterpart",
                "action": "offer",
                "price": opening,
                "message": "Here is my opening offer.",
            }
        ],
        "K": K,
        "p_min": p_min,
        "p_max": p_max,
    }


def load_state(scenario: Dict[str, Any]) -> Dict[str, Any]:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    state = init_state(scenario)
    save_state(state)
    return state


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def save_log(state: Dict[str, Any], scenario: Dict[str, Any]) -> None:
    """Write full transcript + outcome for verifier."""
    LOG_PATH.write_text(json.dumps({"state": state, "scenario_role": scenario.get("role"),
                                    "r_s": scenario.get("r_s"), "r_b": scenario.get("r_b")},
                                   indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(price: float, r_s: float, r_b: float, role: str) -> float:
    A = 1.0 if (r_s <= price <= r_b) else 0.0
    if A == 0.0:
        return 0.0
    TS = r_b - r_s
    u_agent = (price - r_s) if role == "seller" else (r_b - price)
    return float(u_agent / (TS + EPS))

# ---------------------------------------------------------------------------
# Rule-based fallback counterpart
# ---------------------------------------------------------------------------

def _rule_based_response(counterpart_role: str, counterpart_reservation: float,
                          agent_offer: float, history: List[Dict[str, Any]],
                          rounds_remaining: int) -> Dict[str, Any]:
    """Simple fallback: accept if favorable, else concede 15% toward reservation."""
    if counterpart_role == "buyer":
        favorable = agent_offer <= counterpart_reservation
    else:
        favorable = agent_offer >= counterpart_reservation

    if favorable:
        return {"action": "accept", "price": None, "message": "That works for me."}

    # Counter-offer: move 15% toward reservation from last counterpart offer
    last_cp = next(
        (e["price"] for e in reversed(history) if e["turn"] == "counterpart" and e["action"] == "offer"),
        counterpart_reservation,
    )
    if counterpart_role == "buyer":
        new_price = round(min(counterpart_reservation, last_cp + 0.15 * (counterpart_reservation - last_cp)), 2)
    else:
        new_price = round(max(counterpart_reservation, last_cp - 0.15 * (last_cp - counterpart_reservation)), 2)

    if rounds_remaining <= 1:
        msg = "This is my final offer."
    else:
        msg = "I can move a bit, but I need a better price from you."

    return {"action": "offer", "price": new_price, "message": msg}

# ---------------------------------------------------------------------------
# LLM counterpart via OpenAI GPT-4o
# ---------------------------------------------------------------------------

def _build_system_prompt(counterpart_role: str, counterpart_reservation: float,
                          item: Dict[str, Any], K: int, rounds_remaining: int) -> str:
    title = item.get("title", "the item")
    category = item.get("category", "general")
    list_price = item.get("list_price", "unknown")
    if counterpart_role == "buyer":
        reservation_note = (
            f"Your reservation price (maximum you will pay) is ${counterpart_reservation:.2f}. "
            "Do not reveal this number directly."
        )
        accept_rule = (
            f"DECISION RULE (mandatory): if the seller's offer is <= ${counterpart_reservation:.2f}, "
            "you MUST respond with action=accept immediately — do not counter-offer."
        )
        counter_rule = (
            f"Only counter-offer when the seller's offer is strictly above ${counterpart_reservation:.2f}. "
            "Move your counter gradually upward toward your reservation over the remaining rounds."
        )
    else:
        reservation_note = (
            f"Your reservation price (minimum you will accept) is ${counterpart_reservation:.2f}. "
            "Do not reveal this number directly."
        )
        accept_rule = (
            f"DECISION RULE (mandatory): if the buyer's offer is >= ${counterpart_reservation:.2f}, "
            "you MUST respond with action=accept immediately — do not counter-offer."
        )
        counter_rule = (
            f"Only counter-offer when the buyer's offer is strictly below ${counterpart_reservation:.2f}. "
            "Move your counter gradually downward toward your reservation over the remaining rounds."
        )

    return (
        f"You are a {counterpart_role} negotiating the price of {title} ({category}). "
        f"List price: ${list_price}.\n\n"
        f"{reservation_note}\n\n"
        f"{accept_rule}\n\n"
        f"{counter_rule} "
        f"You have {rounds_remaining} of {K} round(s) remaining — factor in deadline pressure.\n\n"
        "Be concise (1-2 sentences). Do not reveal your reservation price.\n\n"
        "Respond ONLY with valid JSON (no markdown, no extra text):\n"
        '{"action": "offer" | "accept" | "reject", "price": <number or null>, "message": "<1-2 sentences>"}\n'
        'If action is "accept" or "reject", set price to null.'
    )


def _build_messages(history: List[Dict[str, Any]], agent_offer: float) -> List[Dict[str, Any]]:
    """Convert history to chat message format (last 8 turns for context)."""
    messages: List[Dict[str, Any]] = []
    recent = history[-8:]
    for entry in recent:
        if entry["turn"] == "agent" and entry["action"] == "offer":
            messages.append({
                "role": "user",
                "content": f"The other party offers ${entry['price']:.2f}. {entry.get('message', '')}".strip(),
            })
        elif entry["turn"] == "counterpart" and entry["action"] == "offer":
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "action": "offer",
                    "price": entry["price"],
                    "message": entry.get("message", ""),
                }),
            })
    # Current agent offer
    messages.append({"role": "user", "content": f"The other party offers ${agent_offer:.2f}."})
    return messages


def call_llm_counterpart(counterpart_role: str, counterpart_reservation: float,
                          item: Dict[str, Any], history: List[Dict[str, Any]],
                          agent_offer: float, K: int, rounds_remaining: int) -> Dict[str, Any]:
    """Call GPT-4o to generate counterpart response. Falls back to rule-based on any error."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _rule_based_response(counterpart_role, counterpart_reservation, agent_offer, history, rounds_remaining)

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return _rule_based_response(counterpart_role, counterpart_reservation, agent_offer, history, rounds_remaining)

    try:
        client = OpenAI(api_key=api_key)
        system = _build_system_prompt(counterpart_role, counterpart_reservation, item, K, rounds_remaining)
        messages = _build_messages(history, agent_offer)

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=150,
            temperature=0,  # deterministic for reproducible evaluation
            messages=[{"role": "system", "content": system}, *messages],
        )
        text = response.choices[0].message.content.strip()
        parsed: Dict[str, Any] = json.loads(text)
        # Validate required keys
        if "action" not in parsed or parsed["action"] not in ("offer", "accept", "reject"):
            raise ValueError(f"Invalid action in response: {parsed}")
        if parsed["action"] == "offer" and parsed.get("price") is None:
            raise ValueError("Offer action requires a price")
        return parsed
    except Exception as e:
        print(f"LLM counterpart failed: {e}. Falling back to rule-based policy.", file=sys.stderr)
        return _rule_based_response(counterpart_role, counterpart_reservation, agent_offer, history, rounds_remaining)

# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def handle_offer(args: argparse.Namespace, state: Dict[str, Any], scenario: Dict[str, Any]) -> None:
    price = args.price
    agent_message = getattr(args, "message", "") or ""
    p_min = state["p_min"]
    p_max = state["p_max"]
    K = state["K"]

    # Validate price bounds
    if not (p_min <= price <= p_max):
        _print({"error": f"Price {price:.2f} is outside the allowed range [{p_min:.2f}, {p_max:.2f}]."})
        return

    state["round"] += 1
    state["agent_offers"].append(round(price, 2))
    state["history"].append({
        "turn": "agent", "action": "offer",
        "price": round(price, 2), "message": agent_message,
    })

    rounds_remaining = K - state["round"]
    item = scenario.get("item_context", {})

    # Ask LLM counterpart (or fallback)
    response = call_llm_counterpart(
        state["counterpart_role"], state["counterpart_reservation"],
        item, state["history"], price, K, rounds_remaining,
    )

    cp_action = response.get("action", "offer")
    cp_message = response.get("message", "")
    cp_price_raw = response.get("price")

    if cp_action == "accept":
        # Counterpart accepts agent's offer
        _finalize(state, scenario, outcome="agreement", final_price=round(price, 2))
        save_log(state, scenario)
        reward = compute_reward(price, float(scenario["r_s"]), float(scenario["r_b"]), scenario["role"])
        _print({
            "round": state["round"],
            "accepted": True,
            "done": True,
            "outcome": "agreement",
            "final_price": round(price, 2),
            "reward": round(reward, 6),
            "counterpart_message": cp_message,
        })
        return

    if cp_action == "reject":
        _finalize(state, scenario, outcome="disagreement", final_price=None)
        state["history"].append({"turn": "counterpart", "action": "reject", "price": None, "message": cp_message})
        save_state(state)
        save_log(state, scenario)
        _print({"done": True, "outcome": "disagreement", "reward": 0.0, "counterpart_message": cp_message})
        return

    # Counterpart makes a counter-offer
    if cp_price_raw is None:
        # Malformed response from LLM — fall back
        fallback = _rule_based_response(
            state["counterpart_role"], state["counterpart_reservation"],
            price, state["history"], rounds_remaining,
        )
        cp_price_raw = fallback.get("price", state["counterpart_last_offer"])
        cp_message = fallback.get("message", cp_message)

    cp_price = round(float(max(p_min, min(p_max, cp_price_raw))), 2)
    state["counterpart_last_offer"] = cp_price
    state["counterpart_offers"].append(cp_price)
    state["history"].append({"turn": "counterpart", "action": "offer", "price": cp_price, "message": cp_message})

    if rounds_remaining <= 0:
        # Ran out of rounds after this exchange
        _finalize(state, scenario, outcome="timeout", final_price=None)
        save_log(state, scenario)
        _print({"done": True, "outcome": "timeout", "reward": 0.0,
                "counterpart_message": "We have run out of time. No deal."})
        return

    save_state(state)
    save_log(state, scenario)
    _print({
        "round": state["round"],
        "counterpart_offer": cp_price,
        "counterpart_message": cp_message,
        "accepted": False,
        "done": False,
        "rounds_remaining": rounds_remaining,
    })


def handle_accept(state: Dict[str, Any], scenario: Dict[str, Any]) -> None:
    price = state["counterpart_last_offer"]
    _finalize(state, scenario, outcome="agreement", final_price=price)
    state["history"].append({"turn": "agent", "action": "accept", "price": price, "message": ""})
    save_state(state)
    save_log(state, scenario)
    reward = compute_reward(price, float(scenario["r_s"]), float(scenario["r_b"]), scenario["role"])
    _print({
        "done": True,
        "outcome": "agreement",
        "final_price": price,
        "reward": round(reward, 6),
        "message": "You accepted the counterpart's offer.",
    })


def handle_reject(state: Dict[str, Any], scenario: Dict[str, Any]) -> None:
    _finalize(state, scenario, outcome="disagreement", final_price=None)
    state["history"].append({"turn": "agent", "action": "reject", "price": None, "message": ""})
    save_state(state)
    save_log(state, scenario)
    _print({"done": True, "outcome": "disagreement", "reward": 0.0,
            "message": "You rejected the negotiation. No deal."})


def _finalize(state: Dict[str, Any], scenario: Dict[str, Any],
              outcome: str, final_price: Optional[float]) -> None:
    state["done"] = True
    state["outcome"] = outcome
    state["final_price"] = final_price


def _print(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Negotiation counterpart CLI. Call once per round.",
    )
    subparsers = parser.add_subparsers(dest="action")

    offer_p = subparsers.add_parser("offer", help="Submit a price offer")
    offer_p.add_argument("price", type=float, help="Offered price")
    offer_p.add_argument("--message", default="", help="Optional natural language message")

    subparsers.add_parser("accept", help="Accept the counterpart's last offer")
    subparsers.add_parser("reject", help="Reject and end negotiation with no deal")

    args = parser.parse_args()

    if args.action is None:
        parser.print_help()
        sys.exit(1)

    if not SCENARIO_PATH.exists():
        _print({"error": f"Scenario file not found at {SCENARIO_PATH}"})
        sys.exit(1)

    scenario = load_scenario()
    state = load_state(scenario)

    if state["done"]:
        _print({"done": True, "outcome": state["outcome"], "final_price": state["final_price"],
                "message": "Negotiation already ended."})
        return

    if args.action == "offer":
        handle_offer(args, state, scenario)
    elif args.action == "accept":
        handle_accept(state, scenario)
    elif args.action == "reject":
        handle_reject(state, scenario)


if __name__ == "__main__":
    main()
