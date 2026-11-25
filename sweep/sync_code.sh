#!/bin/bash
# Push code + small analysis inputs to shiva and verify. Run BEFORE launching anything there: a stale
# driver on shiva has already cost one run (unified_pool_driver.py was odin-only, EXP-39 died on import).
cd /home/lzhou/libero-eval
rsync -a --itemize-changes sweep/*.py sweep/*.sh shiva:/home/lzhou/libero-eval/sweep/ | grep -E '^[<>]' | awk '{print "  pushed", $2}'
rsync -a examples/libero/main.py examples/libero/main_dense.py shiva:/home/lzhou/libero-eval/examples/libero/
rsync -a EXPERIMENTS.md RESUME_WATCHERS.md shiva:/home/lzhou/libero-eval/
# deck filename tracks the MEETING DATE and is renamed as the date moves -- resolve, never hardcode
# newest by mtime, but never a deck whose header says SUPERSEDED (a cancelled meeting's file, kept for history)
DECK=$(for f in $(ls -t results/content_for_slides_*.md); do head -1 "$f" | grep -q SUPERSEDED || { echo "$f"; break; }; done)
rsync -a "$DECK" shiva:/home/lzhou/libero-eval/results/ 2>/dev/null
for f in sweep/plan.json sweep/plan106.json sweep/plan_ext.json sweep/e18e_plan.json \
         results/analysis/shared_donor_design.json results/analysis/confirm_design.json \
         results/analysis/probe_selector76_report.txt; do
  [ -f "$f" ] && rsync -a "$f" "shiva:/home/lzhou/libero-eval/$f"
done
d=$(rsync -ain --delete-excluded sweep/*.py shiva:/home/lzhou/libero-eval/sweep/ | grep -c '^[<>]')
echo "[sync] $(ls sweep/*.py | wc -l) python drivers; $d still differing after push (want 0)"
