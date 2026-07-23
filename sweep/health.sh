#!/bin/bash
# Quick health check for the baseline sweep. Run: bash sweep/health.sh
cd /home/lzhou/libero-eval
R=results/baseline_results.jsonl
TOTAL=94

echo "===== SWEEP HEALTH  $(date '+%Y-%m-%d %H:%M:%S') ====="

# 1) progress
ok=$(grep -c '"status": "ok"' "$R" 2>/dev/null); ok=${ok:-0}
err=$(grep -cE '"status": "(error|server_died|parse_fail)"' "$R" 2>/dev/null); err=${err:-0}
echo "progress    : $ok / $TOTAL pairs done   |   errors: $err"
if [ -s "$R" ]; then
  last_ts=$(tail -1 "$R" | grep -o '"ts": "[^"]*"' | cut -d'"' -f4)
  echo "last result : $last_ts"
fi

# 2) orchestrator process
if pgrep -f 'sweep/[o]rchestrator.py' >/dev/null; then
  echo "orchestrator: ALIVE"
else
  echo "orchestrator: NOT RUNNING (finished, or crashed if progress < $TOTAL)"
fi

# 3) workers: servers listening + eval clients
up=0; for p in 8000 8001 8002 8003; do ss -ltn 2>/dev/null | grep -q ":$p " && up=$((up+1)); done
cl=$(pgrep -f 'libero/[m]ain.py' | wc -l)
echo "servers up  : $up / 4        eval clients: $cl / 4   (fewer is OK briefly during build/serve)"

# 4) GPUs  (healthy = K620 idle ~0 MiB; both P6000s ~16 GB used, util > 0)
echo "--- GPUs (K620 should be idle; P6000s busy) ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

# 5) resources (must not be exhausted)
echo "--- resources ---"
printf "shm free: %s   disk free: %s   ram avail: %s\n" \
  "$(df -h /dev/shm | awk 'NR==2{print $4}')" \
  "$(df -h /home/lzhou | awk 'NR==2{print $4}')" \
  "$(free -h | awk '/Mem:/{print $7}')"

# 6) any recent errors
rec=$(grep -E '"status": "(error|server_died)"' "$R" 2>/dev/null | tail -3)
[ -n "$rec" ] && { echo "--- recent errors ---"; echo "$rec"; }

# 7) latest orchestrator events
echo "--- last orchestrator events ---"
tail -6 results/logs/orchestrator.log 2>/dev/null
