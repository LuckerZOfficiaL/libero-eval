#!/bin/bash
set -e; set -o pipefail
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
for d in 2 3 4; do
  echo "=== [e80draw] DRAW $d START $(date) ==="
  if DRAW=$d ONLY_WEAK_FLEET=1 CHUNK=6 .venv/bin/python sweep/blind_pool.py; then echo "=== [e80draw] DRAW $d OK $(date) ==="; else echo "=== [e80draw] DRAW $d FAILED (exit $?) $(date) ==="; exit 1; fi
done
echo "=== [e80draw] ALL DRAWS DONE $(date) ==="
