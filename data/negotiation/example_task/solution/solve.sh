#!/bin/bash
set -euo pipefail
# Reference solution: seller negotiates from counterpart's opening (88.0) toward a deal.
# Counter at 100, then accept once counterpart reaches >= 85 (our reservation).

# Round 1: open at 100
RESP=$(python /app/counterpart.py offer 100)
echo "Round 1: $RESP"

DONE=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('done',False))" 2>/dev/null || echo "False")
if [ "$DONE" = "True" ]; then exit 0; fi

CP_OFFER=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('counterpart_offer',0))" 2>/dev/null || echo "0")

# Accept if counterpart is at or above our reservation
if python3 -c "exit(0 if float('$CP_OFFER') >= 85 else 1)" 2>/dev/null; then
  python /app/counterpart.py accept
  exit 0
fi

# Round 2: counter at 95
RESP=$(python /app/counterpart.py offer 95)
echo "Round 2: $RESP"

DONE=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('done',False))" 2>/dev/null || echo "False")
if [ "$DONE" = "True" ]; then exit 0; fi

CP_OFFER=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('counterpart_offer',0))" 2>/dev/null || echo "0")

if python3 -c "exit(0 if float('$CP_OFFER') >= 85 else 1)" 2>/dev/null; then
  python /app/counterpart.py accept
  exit 0
fi

# Fallback: accept counterpart's last offer (ensures agreement)
python /app/counterpart.py accept
