#!/bin/bash
# Health check for a merge run.  Usage: bash sweep/merge_health.sh [plan.json]
cd /home/lzhou/libero-eval
PLAN=${1:-sweep/merge_plan.json}
# Count progress ONLY over this plan's (job,scene) evals (results/merge/ may hold files from other runs).
read RD TOTAL NW PORTS OK ERR < <(.venv/bin/python - "$PLAN" <<'PY'
import json, sys, glob
p = json.load(open(sys.argv[1]))
rd = p.get("results_dir", "results/merge")
want = {(j["name"], s) for j in p["jobs"] for s in j["eval_scenes"]}
ok = err = 0
for f in glob.glob(rd + "/*.jsonl"):
    for l in open(f):
        try:
            r = json.loads(l)
            if (r.get("job"), r.get("scene")) in want:
                if r.get("status") == "ok": ok += 1
                elif r.get("status") in ("error", "server_died", "parse_fail"): err += 1
        except Exception:
            pass
print(rd, len(want), p["n_workers"], ",".join(map(str, p["port_per_worker"])), ok, err)
PY
)
echo "===== MERGE HEALTH  $(date '+%Y-%m-%d %H:%M:%S')  (plan: $(basename "$PLAN")) ====="
echo "progress    : $OK / $TOTAL (pair,scene) evals   |   errors: $ERR"
pgrep -f 'merge_[o]rchestrator.py' >/dev/null && echo "orchestrator: ALIVE" || echo "orchestrator: NOT RUNNING (done, or crashed if < $TOTAL)"
up=0; for p in ${PORTS//,/ }; do ss -ltn 2>/dev/null | grep -q ":$p " && up=$((up+1)); done
echo "servers up  : $up / $NW        eval clients: $(pgrep -f 'libero/[m]ain.py' | wc -l)"
echo "--- GPUs (K620 idle; P6000s busy) ---"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo "--- resources ---"
printf "shm free: %s   disk free: %s   ram avail: %s\n" \
  "$(df -h /dev/shm | awk 'NR==2{print $4}')" "$(df -h /home/lzhou | awk 'NR==2{print $4}')" "$(free -h | awk '/Mem:/{print $7}')"
echo "--- this plan's results so far (pair / scene -> success) ---"
.venv/bin/python - "$PLAN" <<'PY'
import json, sys, glob
p = json.load(open(sys.argv[1]))
rd = p.get("results_dir", "results/merge")
want = {(j["name"], s) for j in p["jobs"] for s in j["eval_scenes"]}
for f in sorted(glob.glob(rd + "/*.jsonl")):
    for l in open(f):
        try:
            r = json.loads(l)
            if r.get("status") == "ok" and (r["job"], r["scene"]) in want:
                print(f"  {r['job']:28s} {r['scene'][:44]:44s} {r['success_rate']}  (n={r.get('n_trials')})")
        except Exception:
            pass
PY
echo "--- last events ---"
tail -6 results/merge_logs/orchestrator.log 2>/dev/null
echo "--- overall ---"
echo "progress    : $OK / $TOTAL (pair,scene) evals   |   errors: $ERR"
