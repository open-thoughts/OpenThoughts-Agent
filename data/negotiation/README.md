# Negotiation Tasks

This module implements a sandbox-compatible **multi-round bilateral** negotiation task suite for OpenThoughts-Agent.

The tasks are inspired by classic bargaining benchmarks (DealOrNoDeal, CraigslistBargain) but are adapted to the Harbor/OpenThoughts-Agent task format with:

- **Multi-round alternating offers** (up to K rounds)
- **LLM counterpart** (GPT-4o, `temperature=0`) with a hidden reservation price
- **Deterministic reward** based on final agreed price vs. both parties' reservations
- **Scalable instance generation** and HF dataset upload
- **Terminal-native agent execution** via a CLI counterpart script

---

## Overview

Each task instance is a bilateral price negotiation between the agent and an LLM-driven counterpart.

### Roles

- **Seller:** wants the price high; knows their minimum acceptable price (`r_s`)
- **Buyer:** wants the price low; knows their maximum acceptable price (`r_b`)

The agent is assigned one role. The counterpart plays the other. **Each party knows their own reservation price but not the other party's.**

### Zone of Possible Agreement (ZOPA)

A deal is feasible when `r_s ≤ r_b`. Reward is non-zero only when the agreed price falls within ZOPA.

---

## Task Format

Each task directory contains:

```
negotiation-{id}/
├── instruction.md              # Agent-facing instructions (role, own reservation, CLI usage)
├── task.toml                   # Timeout and metadata
├── data/
│   └── scenario.json           # Full scenario (r_s, r_b, role, K, p_min, p_max, counterpart_opening, ...)
├── environment/
│   ├── Dockerfile              # ubuntu:24.04 + openai + counterpart.py
│   └── counterpart.py          # Stateful LLM counterpart CLI
└── tests/
    └── test.sh                 # Verifier: reads negotiation_log.json → reward.txt
```

**Integrity note:** `scenario.json` (including both reservation prices) is mounted at `/app/data/scenario.json` and is readable by the agent. This benchmark assumes an **honest agent** that does not read the counterpart's reservation from this file. For hardened evaluation, the runner would need to supply scenario data only to the verifier via a separate mount.

---

## Agent Interface

The agent interacts with the counterpart by calling a CLI script once per round:

### Commands

**Make an offer:**
```bash
python /app/counterpart.py offer <price>
```

**Accept the counterpart's last offer:**
```bash
python /app/counterpart.py accept
```

**Walk away (no deal):**
```bash
python /app/counterpart.py reject
```

### Response format

`offer` returns JSON:
```json
{
  "round": 3,
  "counterpart_offer": 92.50,
  "counterpart_message": "I can move a bit, but I need a better price from you.",
  "accepted": false,
  "done": false,
  "rounds_remaining": 7
}
```

If the counterpart accepts the agent's offer:
```json
{
  "round": 3,
  "accepted": true,
  "done": true,
  "outcome": "agreement",
  "final_price": 95.0,
  "reward": 0.25,
  "counterpart_message": "That works for me."
}
```

`accept` and `reject` also return JSON with `done: true`.

### Termination

The negotiation ends when any of the following occurs:

1. **Agent accepts** → agreement at counterpart's last offer
2. **Agent rejects** → disagreement (reward = 0)
3. **Counterpart accepts** agent's offer → agreement at agent's price
4. **Round limit K reached** → timeout / disagreement (reward = 0)

---

## Reward Definition

Reward is computed by the verifier from `negotiation_log.json` + `scenario.json`. It is always in [0, 1].

### Step 1: Agreement indicator

```
A = 1  if r_s ≤ final_price ≤ r_b
A = 0  otherwise (including no-deal)
```

### Step 2: Utilities (if agreement)

```
u_seller = final_price − r_s
u_buyer  = r_b − final_price
TS       = r_b − r_s           (total surplus)
u_agent  = u_seller if role == "seller" else u_buyer
```

### Final reward

```
reward = A * (u_agent / (TS + ε))
```

where ε = 1e-9 for numerical stability.

**Properties:**
- No deal → reward = 0
- Agent captures all surplus → reward ≈ 1
- Agent captures half surplus → reward ≈ 0.5

---

## Counterpart Behavior

The counterpart is an LLM (GPT-4o, `temperature=0` for reproducibility) given:

- Its role (buyer or seller)
- Its reservation price (hidden from agent)
- Item context (title, description, list price, category)
- Round count and deadline pressure

It opens aggressively (20% away from its reservation) and concedes gradually. It accepts once the agent's offer crosses its reservation.

**Fallback:** If `OPENAI_API_KEY` is not set or the `openai` package is unavailable, the counterpart uses a deterministic rule-based policy (concede 15% toward reservation per round).

### System Prompt Template

The prompt is constructed per-round by `_build_system_prompt()` in `counterpart.py`. For a **seller** counterpart with reservation `r_s`:

```
You are a seller negotiating the price of {title} ({category}).
List price: ${list_price}.

Your reservation price (minimum you will accept) is ${r_s}.
Do not reveal this number directly.

DECISION RULE (mandatory): if the buyer's offer is >= ${r_s},
you MUST respond with action=accept immediately — do not counter-offer.

Only counter-offer when the buyer's offer is strictly below ${r_s}.
Move your counter gradually downward toward your reservation over the
remaining rounds. You have {rounds_remaining} of {K} round(s) remaining
— factor in deadline pressure.

Be concise (1-2 sentences). Do not reveal your reservation price.

Respond ONLY with valid JSON (no markdown, no extra text):
{"action": "offer" | "accept" | "reject", "price": <number or null>, "message": "<1-2 sentences>"}
If action is "accept" or "reject", set price to null.
```

For a **buyer** counterpart the roles are mirrored: reservation is the maximum willing to pay, the mandatory accept rule triggers when the seller's offer is `<= r_b`, and counter-offers move upward toward `r_b`.

The explicit numeric threshold in the decision rule (`>= ${r_s}`) is intentional — without it GPT-4o at `temperature=0` greedily counter-offers even when the offer already exceeds the reservation.

---

## Scenario Schema

`data/scenario.json` in each task has the following shape:

```json
{
  "r_s": 85,
  "r_b": 110,
  "role": "seller",
  "seed": 42,
  "K": 10,
  "p_min": 60.0,
  "p_max": 240.0,
  "delta_max": 18.0,
  "counterpart_opening": 88.0,
  "item_context": {
    "title": "Office Chair",
    "description": "Ergonomic office chair, lumbar support, adjustable armrests.",
    "list_price": 120.0,
    "category": "furniture"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `r_s` | number | Seller reservation (minimum acceptable price) |
| `r_b` | number | Buyer reservation (maximum acceptable price) |
| `role` | string | Agent's role: `"buyer"` or `"seller"` |
| `seed` | integer | Random seed for reproducibility |
| `K` | integer | Maximum number of rounds |
| `p_min` | number | Minimum allowed offer price |
| `p_max` | number | Maximum allowed offer price |
| `delta_max` | number | Max per-round concession reversal (informational) |
| `counterpart_opening` | number | Counterpart's opening offer (shown in instruction) |
| `item_context` | object | Item metadata (title, description, list_price, category) |

---

## Synthetic Items

When no `--source` is provided, the generator draws from a built-in pool of **20 synthetic items** spanning 7 categories and four price tiers. Reservation prices are calibrated so that:

- **Seller floor (`r_s`):** ~70–79% of list price (seller's minimum acceptable price)
- **Buyer ceiling (`r_b`):** ~90–98% of list price (buyer's maximum acceptable price)
- **ZOPA width:** ~20–25% of list price — realistic but negotiable

| Category | Items | Price range |
|----------|-------|-------------|
| electronics | Wireless Mouse, USB-C Hub, Webcam, Bluetooth Speaker, Mechanical Keyboard, External SSD, Smart Watch | $25 – $180 |
| furniture | Bookshelf, Bedside Table, Office Chair, Standing Desk | $60 – $280 |
| tools | Cordless Screwdriver, Power Drill | $35 – $65 |
| home-kitchen | Coffee Maker, Air Fryer | $45 – $75 |
| sports-outdoors | Yoga Mat, Hiking Backpack | $30 – $85 |
| automotive | Car Phone Mount, Portable Jump Starter | $20 – $75 |
| toys-games | Strategy Board Game | $40 |

Items and roles are **sampled at random** (with replacement) per instance, so generating more instances than there are items yields varied scenarios with different seeds and role assignments.

To supply your own items, use `--source` with a CraigslistBargain-style CSV (see ZOPA Handling Modes below).

---

## ZOPA Handling Modes

When loading items from a CSV (e.g., CraigslistBargain), reservation values may be missing or form no ZOPA. The generator supports two modes via `--zopa-mode`:

- **filter (default):** Only keep rows where `seller_target ≤ buyer_target`. Rows without valid targets or without ZOPA are dropped.
- **resample:** For rows where targets are missing or `seller_target > buyer_target`, set:
  - `r_s = list_price × U(0.70, 0.95)`
  - `r_b = list_price × U(1.05, 1.30)`

---

## Generating Tasks

```bash
# Default: 10 instances, synthetic items, output to data/negotiation/out
python -m data.negotiation.generate

# Custom count and output
python -m data.negotiation.generate --num-instances 100 --output-dir ./my_tasks --seed 123

# From CraigslistBargain CSV
python -m data.negotiation.generate --num-instances 50 \
  --source data/negotiation/craigslist_bargains/train.csv --output-dir ./out

# Resample mode for rows without ZOPA
python -m data.negotiation.generate --num-instances 50 \
  --source data/negotiation/craigslist_bargains/train.csv \
  --zopa-mode resample --output-dir ./out
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--num-instances` | `-n` | 10 | Number of task instances |
| `--output-dir` | `-o` | `data/negotiation/out` | Output directory |
| `--seed` | | 42 | Random seed |
| `--source` | `-s` | (none) | CraigslistBargain CSV or directory |
| `--zopa-mode` | | filter | `filter` or `resample` |

---

## Hugging Face Upload

```bash
# Generate and upload in one go
python -m data.negotiation.generate --num-instances 200 --output-dir ./out \
  --upload-repo OpenThoughts-Agent/negotiation-tasks

# Generate only (no upload)
python -m data.negotiation.generate --num-instances 200 --output-dir ./out --no-upload
```

For HPC or launcher-driven runs, use the class-based generator:

```bash
python -m hpc.launch --job_type datagen \
  --datagen_script data/negotiation/generate_abstract.py \
  --datagen_target_repo OpenThoughts-Agent/negotiation-tasks \
  --datagen_extra_args "--stage tasks --limit 500 --seed 42"
```

See `data/README.md` for the full `BaseDataGenerator` workflow.

---

## Environment Requirements

The counterpart requires the `openai` Python package and `OPENAI_API_KEY` to be set in the container environment. The Dockerfile installs `openai` via pip. If the API key is absent, the counterpart falls back to a deterministic rule-based policy automatically — no crash.

### Injecting `OPENAI_API_KEY` into Harbor

Harbor does **not** automatically pass host environment variables into the Docker sandbox. The key must be present in the sbatch shell environment **before** Harbor is launched so it propagates through the process chain:

```
DC_AGENT_SECRET_ENV → sbatch sources secrets file → Harbor subprocess inherits → container reads os.environ
```

In practice, add the key to your cluster secrets file (or `~/.bashrc` on the login node) and ensure it is exported before launching any Harbor eval or tracegen job:

```bash
export OPENAI_API_KEY="sk-proj-..."
python eval/local/run_eval.py --datagen-config hpc/datagen_yaml/... \
  --dataset-path data/negotiation/example_task --harbor-config ...
```

If the key is unavailable at runtime, the counterpart silently falls back to a rule-based policy (concede 15% per round). Evaluation still produces valid rewards — the counterpart is just deterministic rather than LLM-driven.

---

## Example Task

A specimen task is provided under `data/negotiation/example_task/`. It models a seller negotiating an Office Chair with `r_s = 85`, `r_b = 110`, and counterpart opening at `88.0`.

Run locally (e.g., with Harbor):
```bash
harbor jobs start -p data/negotiation/example_task --agent terminus-2 --model <model> --env docker
```

---

## Running Tests

```bash
# Run all tests (from repo root)
python -m data.negotiation.test_generate

# Or with pytest
pytest data/negotiation/test_generate.py -v
```

Tests cover: reward formula, example task layout, scenario/instruction builders, CSV loading, and full task generation. Most tests run without `pyarrow`/`numpy`; only `TestFullGeneration` requires the full repo install.

---

## Verifier Contract

The verifier (`tests/test.sh`) must:

1. Read `/app/negotiation_log.json` (written by `counterpart.py` during negotiation)
2. Read `/app/data/scenario.json` (for `r_s`, `r_b`, `role`)
3. Compute agreement indicator A, agent utility, and reward
4. Write exactly one scalar to `/logs/verifier/reward.txt`

If `negotiation_log.json` is missing (agent never called counterpart.py), reward = 0.

---

## Invariant Rules

Every task must satisfy:

- Deterministic reward given a fixed negotiation log
- Exactly one scalar written to `/logs/verifier/reward.txt`
- No stochastic evaluation at test time (counterpart LLM uses `temperature=0`)
