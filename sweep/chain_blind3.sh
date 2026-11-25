#!/bin/bash
# Odin queue v4 (09-16 04:10), after the quota crash at 03:39.
#  - CHUNK=3 caps peak build usage at ~5GB (builds are 1.66GB each, not the 0.55GB the guard assumed)
#  - set -e: a failed stage ABORTS the chain instead of printing ALL DONE (v3 hid two crashes)
#  - each stage reports its own exit status
set -e
cd /home/lzhou/libero-eval
export CHUNK=3
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 30; done
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null || true
rm -f results/analysis/decomp_adapters/*.safetensors 2>/dev/null || true
stage () {  # stage <name> <script>
  echo "=== [blind3] START $1 $(date) ==="
  if .venv/bin/python "$2"; then echo "=== [blind3] OK $1 $(date) ==="
  else echo "=== [blind3] FAILED $1 (exit $?) $(date) ==="; exit 1; fi
}
stage EXP-80 sweep/blind_pool.py
stage EXP-78 sweep/subset_probe.py
echo "=== [blind3] ALL STAGES OK $(date) ==="
