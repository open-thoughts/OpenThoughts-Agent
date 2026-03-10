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
                f.write("0\n")
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
        f.write(f"{reward}\n")

if __name__ == "__main__":
    main()
VERIFIER
