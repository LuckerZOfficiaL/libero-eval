#!/bin/bash
set -e; set -o pipefail
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
echo "=== [e80draw] DRAW 4 (shiva, resumed from odin) START $(date) ==="
if DRAW=4 ONLY_WEAK_FLEET=1 CHUNK=6 .venv/bin/python sweep/blind_pool.py; then echo "=== [e80draw] DRAW 4 OK $(date) ==="; echo "=== [e80draw] ALL DRAWS DONE $(date) ==="; else echo "=== [e80draw] DRAW 4 FAILED (exit $?) $(date) ==="; exit 1; fi
