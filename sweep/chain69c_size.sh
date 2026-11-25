#!/bin/bash
# 11-19 EXP-161 size control, order 0, eval only: the consfz correction kept in seeded random module subsets (exp147 RSUB): rap /
# rae = any part at the language backbone's share of parameters (42.6%) / of energy (71%), rle = language modules only at the
# vision + expert energy share (29%). Starts after chain69b; lowest priority (same gate as chain69b), port 10500.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
echo "[c69c] start $(date +%m-%d\ %H:%M)"
while tmux has-session -t c69b 2>/dev/null; do sleep 120; done
for v in rae rle rap; do
  gapwait 150 "$C67JOBS $C68JOBS $C70JOBS"; evalrun $v@0 10500 c69c
done
echo "[c69c] ALL EXIT $(date +%m-%d\ %H:%M)"
