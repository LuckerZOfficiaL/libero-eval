#!/bin/bash
# wait for odin's in-flight chunk (48-55) to finish, then relaunch fp so its todo excludes what shiva did
R=/home/lzhou/libero-eval; cd $R
have(){ ls results/merge/e92_base_f$1.jsonl >/dev/null 2>&1; }
until have 48 && have 50 && have 52 && have 53 && have 54 && have 55; do sleep 60; done
echo "odin chunk 48-55 complete $(date)"
tmux kill-session -t fp 2>/dev/null; pkill -f "sweep/fingerprint_pass.py" 2>/dev/null; pkill -f "merge_orchestrat[o]r.py" 2>/dev/null; sleep 5
rsync -a shiva:$R/results/merge/e92_base_f*.jsonl results/merge/
n=$(ls results/merge/e92_base_f*.jsonl | wc -l); echo "after merge: $n/76"
if [ "$n" -lt 76 ]; then
  CHUNK=2 .venv/bin/python sweep/fingerprint_pass.py --dry
  tmux new-session -d -s fp "cd $R && bash sweep/chain_fp.sh 2>&1 | tee -a $R/results/fp.log"
  until [ "$(ls results/merge/e92_base_f*.jsonl | wc -l)" -ge 76 ]; do sleep 120; rsync -aq shiva:$R/results/merge/e92_base_f*.jsonl results/merge/ 2>/dev/null; done
fi
echo "=== FINGERPRINT PASS 76/76 COMPLETE $(date) -> build EXP-85 inputs ==="
