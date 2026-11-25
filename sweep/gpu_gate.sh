# 11-18 shared GPU gating, sourced by the chains that run next to chain67 (chain68b / 69b / 70b). chain67 cannot be restarted
# (trainers inside) and its lanes see a GPU as busy only through a train_lora.py process bound to it (CUDA_VISIBLE_DEVICES) or
# >= 12 GB of memory: they miss another chain's exp147 train stage while it prepares (20:12: chain67 and chain70 both started a
# trainer on GPU0, erlc@19 OOM), and gap evals racing chain67's evals put 3 workers + a trainer + a foreign job on GPU0 (erl@0:
# 4 tasks lost). Here:
#   busy2 GPU     a train_lora.py process OR an exp147 train stage bound to GPU, or >= 12 GB in use
#   reserve GPU   10-min placeholder whose command line contains sweep/train_lora.py (env CUDA_VISIBLE_DEVICES=GPU), so
#                 chain67's busy() sees our claim before our trainer starts
#   lane2 GPU PFX DELAY JOBS   trainer lane: claims .<PFX>_<arm>_<order>; the free-GPU double check waits DELAY s (chain67
#                 waits 60 s -> a longer DELAY yields to chain67, then to the chain with the shorter DELAY)
#   pending JOBS  some arm@order has its adapter but no eval result yet
#   gapwait DELAY HIGHER   gap-eval gate: >= 16 GB free on both A6000s, no eval serving on another chain's port, no pending
#                 eval among HIGHER (higher-priority chains' jobs); true at t and t + DELAY
G_U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; G_U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
C67JOBS="erl@0 consb@1 erlc@19 erl@1 seq@0 consb@2 erlc@20 erl@2 seqc@19 consb@3 erlc@4 erl@3 seqc@20 consfa@0 seqc@4 consfac@19 consfac@20 consfac@4 consfa@1 consfa@2 consfa@3 consb64@0"
C68JOBS="ewc@0 ewcc@19 ewcc@20 ewcc@4"
C70JOBS="lrcfz@0 lrc@0"
dir() { [ $1 = 0 ] && echo $A/new_loras_cons || echo $A/new_loras_cons_s$1; }
out() { echo $(dir ${1#*@})/train_${1%@*}/adapter_model.safetensors; }
prev() { case $1 in *cm0@19|*cm0@21) echo ${1%cm0@*}@$(( (${1#*@} - 19) / 2 ));; *cm0@20) echo ${1%@20}@19;; *cm0@4) echo ${1%@4}@20;; *cm0@2[23]) echo ${1%@*}@$(( ${1#*@} - 1 ));;   # 11-23 EXP-172a
  *c@19) echo ${1%c@19}@0;; *c@20) echo ${1%@20}@19;; *c@4) echo ${1%@4}@20;;
  *c@21|*c@24|*c@27) echo ${1%c@*}@$(( (${1#*@} - 18) / 3 ));; *c@2[235689]) echo ${1%@*}@$(( ${1#*@} - 1 ));;   # 11-21 EXP-167 orders 1-3
  *l@30) echo ${1%l@30}6@5;; *l@3[12]) echo ${1%@*}@$(( ${1#*@} - 1 ));;   # 11-22 EXP-170 32-skill library stream
  *cs1@19) echo ${1%cs1@19}s1@0;; *cs1@20) echo ${1%@20}@19;; *cs1@4) echo ${1%@4}@20;;   # 11-22 EXP-171 seed-1 stream (order 0)
  *c@34) echo ${1%@34}@4;; *l@33) echo ${1%@33}@32;; *l@36) echo ${1%l@36}6@35;; *l@3[78]) echo ${1%@*}@$(( ${1#*@} - 1 ));; esac; }   # EXP-171 5th arrivals, order 1's 32
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
freemb() { nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g {print $2}'; }
busy2() {
  [ $(big $1) -gt 0 ] && return 0
  for p in $(pgrep -f "sweep/train_lora.py|sweep/exp147_consolidate.py"); do
    e=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null) || continue
    echo "$e" | grep -qx "CUDA_VISIBLE_DEVICES=$1" || continue
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "sweep/train_lora.py" && return 0
    echo "$e" | grep -qx "E147_STAGE=train" && return 0
  done; return 1
}
reserve() { ( export CUDA_VISIBLE_DEVICES=$1; exec -a "gpu-reserve sweep/train_lora.py" sleep 600 ) & }
lane2() {   # $1 GPU, $2 claim prefix, $3 delay, $4 jobs
  while true; do
    left=0; ran=0
    for j in $4; do
      [ -f $A/.$2_${j/@/_} ] && continue; left=1
      p=$(prev $j); [ -z "$p" ] || [ -f $(out $p) ] || continue
      until ! busy2 $1 && sleep $3 && ! busy2 $1; do sleep 30; done
      ( set -o noclobber; : > $A/.$2_${j/@/_} ) 2>/dev/null || continue
      reserve $1
      echo "[$2] trainer $j on $1 $(date +%H:%M)"
      CUDA_VISIBLE_DEVICES=$1 E145_SEQ=${j#*@} E147_VARS=${j%@*} E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
      ran=1; break
    done
    [ $left = 0 ] && break
    [ $ran = 0 ] && sleep 60
  done
}
# pending: j is local (11-19: it clobbered the callers' eval-loop j, so c70b evaluated ewcc@4 instead of lrcfz@0)
pending() { local j; for j in $1; do [ -f $(out $j) ] && ! ls results/merge/e112_c147s${j#*@}t*-${j%@*}_w0.25_* > /dev/null 2>&1 && return 0; done; return 1; }
gapok() {
  [ "$(freemb $G_U0)" -ge 16000 ] && [ "$(freemb $G_U2)" -ge 16000 ] || return 1
  ss -ltn | grep -q ":10\(35[0-3]\|36[01]\|40[0-3]\|45[01]\|50[01]\|55[01]\) " && return 1
  ! pending "$2"
}
gapwait() { until gapok $1 "$2" && sleep $1 && gapok $1 "$2"; do sleep 60; done; }
# evalrun holds $A/.evlock for the whole eval (11-19: chain67's sca@4 and chain69c's rae@0 started in the same 2-min window —
# the port check misses the merge-build phase before serving — servers died during load, rae@0 21/24); the lock fd is closed
# for the python children so a stray server cannot keep it
evalrun() {   # $1 arm@order, $2 port, $3 log tag, $4 workers per GPU (default 1): 3 tries (exp147 run skips completed jobs)
  ( flock 9; export ORCH_CUDA_MAP="0=$G_U0,2=$G_U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=${4:-1} CUDA_VISIBLE_DEVICES=$G_U0 E113_DEV=cuda:0
    for try in 1 2 3; do E112_PORT0=$2 E145_SEQ=${1#*@} E147_VARS=${1%@*} E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 9>&- 2>&1 | grep -v hostcfg 9>&-; done ) 9> $A/.evlock
  echo "[$3] $1 EVAL EXIT $(date +%m-%d\ %H:%M) ($(cat results/merge/e112_c147s${1#*@}t*-${1%@*}_w0.25_*.jsonl 2>/dev/null | grep -c '"status": "ok"') tasks ok)"
}
