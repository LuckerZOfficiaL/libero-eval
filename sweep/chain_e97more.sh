#!/bin/bash
# shiva: third replication of std vs top4 (e97s), then top8 for all 15 targets and top2 for targets 7: (e97)
set -o pipefail
cd /home/lzhou/libero-eval
LA_PREFIX=e97s LA_ARMS=std,top4 CHUNK=6 .venv/bin/python sweep/la_operator_arms.py
LA_ARMS=top8 CHUNK=6 .venv/bin/python sweep/la_operator_arms.py
LA_ARMS=top2 LA_SLICE=7: CHUNK=6 .venv/bin/python sweep/la_operator_arms.py
echo "=== [e97more] EXIT $? $(date) ==="
