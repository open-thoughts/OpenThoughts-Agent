# Bilateral Price Negotiation

You are a **seller**. Negotiate to sell the item below.

## Item

- **Title:** Office Chair
- **Category:** furniture
- **List price:** $120.0

**Description:**
Ergonomic office chair, adjustable height.

## Your position

- **Your minimum acceptable price (reservation):** $85.00 — Do not sell below this price.
- **Allowed price range:** $60.00 – $240.00
- **Max rounds:** 10

The buyer's reservation price is **unknown to you**.

## Current state

**Counterpart's opening offer:** $88.00
It is your turn.

## Commands

**Make an offer** (must be within the allowed price range):
```bash
python /app/counterpart.py offer <price>
```

**Accept the buyer's last offer:**
```bash
python /app/counterpart.py accept
```

**Walk away (no deal):**
```bash
python /app/counterpart.py reject
```

## Response format

Each `offer` command returns JSON, for example:
```json
{
  "round": 1,
  "counterpart_offer": 92.0,
  "counterpart_message": "I can move a bit, but I need a better price from you.",
  "accepted": false,
  "done": false,
  "rounds_remaining": 9
}
```

When `accepted: true` or `done: true`, the negotiation is over.

## Scoring

- No deal → reward = 0
- Deal at price **p** → reward = your surplus / total surplus ∈ [0, 1]

Maximize your reward: drive the price **higher** while still reaching agreement.
