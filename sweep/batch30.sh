#!/bin/bash
# EXP-36 batch: train the 30 new fleet LoRAs (goal/object/spatial), then eval each on its task.
# Recipe = the gate-validated trial exactly (1,500 steps, normalized actions, fleet norm stats).
# 3 shards, one per A6000; ~11h training + ~4h eval.
cd /home/lzhou/libero-eval
set -u
LIST=results/analysis/new_loras/batch30_tasks.txt
.venv/bin/python - <<'PY'
import pathlib
D = pathlib.Path("/media/pinas/palma/inspire_4suite_regenerated_hdf5")
out = []
for s in ("libero_goal_no_noops", "libero_object_no_noops", "libero_spatial_no_noops"):
    for f in sorted((D / s).glob("*_demo.hdf5")):
        out.append(f"{s} {f.name[:-len('_demo.hdf5')]}")
p = pathlib.Path("results/analysis/new_loras/batch30_tasks.txt")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text("\n".join(out) + "\n")
print(f"{len(out)} tasks")
PY

train_shard() {  # $1 = shard index (also GPU index)
  local i=0
  while read -r suite task; do
    if [ $((i % 3)) -eq "$1" ]; then
      local dir="results/analysis/new_loras/${suite}__${task}"
      if [ ! -f "$dir/adapter_model.safetensors" ]; then
        echo "[shard$1] training $suite/$task"
        .venv/bin/python sweep/train_lora.py --suite "$suite" --task "$task" \
          --steps 1500 --batch 8 --device "cuda:$1" \
          > "results/merge_logs/b30_${task}.log" 2>&1 \
          || echo "[shard$1] FAILED $task"
      fi
    fi
    i=$((i + 1))
  done < "$LIST"
  echo "[shard$1] done"
}
train_shard 0 & train_shard 1 & train_shard 2 &
wait
echo "[batch30] all training shards finished"

# eval every trained adapter on its own task, 25 rollouts, fleet norm stats (as trained)
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
.venv/bin/python - <<'PY'
import json, pathlib
d = json.load(open("sweep/_newval_plan.json"))
meta = {k: v for k, v in d.items() if k != "jobs"}
meta["n_trials"] = 25
SUITE = {"libero_goal_no_noops": "libero_goal", "libero_object_no_noops": "libero_object",
         "libero_spatial_no_noops": "libero_spatial"}
jobs = []
for ln in open("results/analysis/new_loras/batch30_tasks.txt"):
    suite, task = ln.split()
    ad = pathlib.Path(f"results/analysis/new_loras/{suite}__{task}/adapter_model.safetensors")
    done = pathlib.Path(f"results/merge/e48_{task}.jsonl")
    if ad.exists() and not done.exists():
        jobs.append({"name": f"e48_{task}", "method": "average", "params": {}, "adapters": [],
                     "adapter_paths": [str(ad.resolve())], "eval_scenes": [task],
                     "suite": SUITE[suite]})
meta["jobs"] = jobs
json.dump(meta, open("sweep/_b30_eval.json", "w"), indent=1)
print(f"{len(jobs)} eval jobs")
PY
.venv/bin/python sweep/merge_orchestrator.py sweep/_b30_eval.json

# summary: success rate + delta norm per adapter
.venv/bin/python - <<'PY'
import json, pathlib
import safetensors.torch as st
import torch
rows = []
for ln in open("results/analysis/new_loras/batch30_tasks.txt"):
    suite, task = ln.split()
    r = None
    f = pathlib.Path(f"results/merge/e48_{task}.jsonl")
    if not f.exists() and task == "put_the_bowl_on_the_stove":
        f = pathlib.Path("results/merge/e47_new_bowl_stove.jsonl")
    if f.exists():
        for l in open(f):
            j = json.loads(l)
            if j.get("status") == "ok":
                r = j["success_rate"]
    nrm = None
    ad = pathlib.Path(f"results/analysis/new_loras/{suite}__{task}/adapter_model.safetensors")
    if ad.exists():
        w = st.load_file(str(ad))
        stems = sorted(set(k.rsplit(".lora_", 1)[0] for k in w if ".lora_" in k))
        tot = sum(float(torch.linalg.norm(
            w[[k for k in w if k.startswith(s + ".lora_B")][0]].float()
            @ w[[k for k in w if k.startswith(s + ".lora_A")][0]].float()) ** 2) for s in stems)
        nrm = tot ** 0.5
    rows.append((suite.replace("_no_noops", ""), task, r, nrm))
L = ["=== EXP-36: 30 new fleet LoRAs — own-task success @25 rollouts (fleet-recipe, 1.5k steps) ===", ""]
ok = [r for _, _, r, _ in rows if r is not None]
for s, t, r, n in rows:
    L.append(f"  {s:<15} {t[:52]:<54} {('%3.0f%%' % (100*r)) if r is not None else '  — '}"
             f"  |d|={n:.1f}" if n else f"  {s:<15} {t[:52]:<54} UNTRAINED")
if ok:
    L += ["", f"  trained+evaled: {len(ok)}/30   mean {100*sum(ok)/len(ok):.1f}%   "
              f">=80%: {sum(1 for r in ok if r>=0.8)}   <50%: {sum(1 for r in ok if r<0.5)}",
          "  (fleet reference on LIBERO-90: ~90% mean; control floor: fleet-t0 scored 80% on an unseen goal task)"]
pathlib.Path("results/analysis/new_loras/batch30_report.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
PY
