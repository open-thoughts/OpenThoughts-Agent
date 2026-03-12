#!/bin/bash
set -euo pipefail
# Reference solution: seller negotiates from counterpart's opening (88.0) toward a deal.
# Counter at 100, then accept once counterpart reaches >= 85 (our reservation).

RESERVATION=85

# Parse response JSON, print counterpart_offer, and accept/exit if negotiation is done
# or counterpart's offer has reached our reservation.  Returns non-zero to continue.
handle_response() {
  local resp="$1"
  local round_name="$2"
  echo "$round_name: $resp"

  local done
  done=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('done',False))" 2>/dev/null || echo "False")
  if [ "$done" = "True" ]; then exit 0; fi

  local cp_offer
  cp_offer=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('counterpart_offer',0))" 2>/dev/null || echo "0")

  # Accept if counterpart has reached or exceeded our reservation
  if python3 -c "exit(0 if float('$cp_offer') >= $RESERVATION else 1)" 2>/dev/null; then
    python /app/counterpart.py accept
    exit 0
  fi
}

# Round 1: open at 100
handle_response "$(python /app/counterpart.py offer 100)" "Round 1"

# Round 2: counter at 95
handle_response "$(python /app/counterpart.py offer 95)" "Round 2"

# Fallback: accept counterpart's last offer (ensures agreement)
python /app/counterpart.py accept
