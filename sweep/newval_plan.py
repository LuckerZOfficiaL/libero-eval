import json, pathlib, sys
REPO = pathlib.Path("/home/lzhou/libero-eval"); sys.path.insert(0, str(REPO / "sweep"))
import hostcfg
T = json.load(open(REPO / "sweep/e18e_plan.json"))
META = {k: T[k] for k in ("base_model","videos_dir","logdir","save_videos","results_dir","n_workers","gpu_per_worker","port_per_worker","norm_stats")}
META = hostcfg.apply(META)
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
new = str(REPO / "results/analysis/new_loras/libero_goal_no_noops__put_the_bowl_on_the_stove/adapter_model.safetensors")
scene = "put_the_bowl_on_the_stove"
jobs = [
  {"name": "e47_new_bowl_stove", "method": "average", "params": {}, "adapters": [], "adapter_paths": [new],
   "eval_scenes": [scene], "suite": "libero_goal"},
  {"name": "e47_ctrl_fleet_t0", "method": "average", "params": {}, "adapters": [], "adapter_paths": [plan[0]["adapter_file"]],
   "eval_scenes": [scene], "suite": "libero_goal"},
]
json.dump(dict(META, n_trials=25, jobs=jobs), open(REPO / "sweep/_newval_plan.json", "w"), indent=1)
print("plan written:", [j["name"] for j in jobs], "| norm_stats:", META["norm_stats"].split("/")[-1])
