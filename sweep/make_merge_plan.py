#!/usr/bin/env python
"""Generate a merge-sweep plan (for sweep/merge_orchestrator.py).

Config precedence: CLI flag > --config YAML > built-in default.
  * flags:  --pairs 1:2 --method ties --params '{"density":0.3}' --eval-tasks auto --trials 10
  * yaml :  --config configs/merge.yaml

`--pairs` and `--method` both accept comma-separated LISTS -> the plan is the CROSS PRODUCT
(every pair x every method). e.g. --pairs "1:2,3:4" --method "task_arithmetic,ties" = 4 jobs.

Groups can be pairs (a:b) or larger (a:b:c triplets, ...). Each (group, method) -> one job that
merges the group and evaluates on
`eval_tasks` (default 'auto' = the pair's own scenes). Results -> results/merge/<job>.jsonl where
<job> = task_<a>_<b>_<method>[_<changed-params>]. Scenes/indices map from sweep/plan.json.
Usually invoked via `sweep/run.sh`.
"""
import argparse, json
from pathlib import Path

REPO = "/home/lzhou/libero-eval"
BASE = "/media/pinas/Chris/models/pi05_base_pytorch/model.safetensors"
NORM = "/media/pinas/mehdi/libero90_lora/assets/libero90_norm_stats.json"
DEFAULTS = {"pairs": "1:2", "scope": None, "method": "task_arithmetic", "params": {}, "trials": 10,
            "workers": 4, "eval_tasks": "auto", "group_size": 2,   # group_size: 2=pairs, 3=triplets, ...
            "save_videos": "all"}   # videos per task: 'all' | 0/'none' | k>0 (first k) | k<0 (last |k|, rolling)
# canonical hyperparameters per method (used when not overridden; kept out of the file name)
DEFAULT_PARAMS = {"task_arithmetic": {"lambda": 1.0}, "average": {}, "ties": {"density": 0.2, "lambda": 1.0}}
METHODS = set(DEFAULT_PARAMS)


def resolve_eval_scenes(group_scenes, eval_tasks, scenes):
    """eval_tasks: 'auto' = the merged group's own scenes (any group size) | task indices | scene names."""
    if eval_tasks in (None, "auto") or eval_tasks == ["auto"]:
        return sorted(group_scenes)
    items = eval_tasks if isinstance(eval_tasks, list) else str(eval_tasks).split(",")
    out = []
    for it in items:
        s = str(it).strip()
        if s == "auto":
            out += group_scenes
        elif s.isdigit():
            out += scenes[int(s)]
        else:
            out.append(s)  # explicit scene name
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description="Build a merge plan over task groups (pairs/triplets/...) x methods (flags or --config YAML).")
    ap.add_argument("--config", help="YAML config file (its keys are defaults; flags below override)")
    ap.add_argument("--pairs", help="explicit pairs, e.g. '1:2' or '1:2,3:4'")
    ap.add_argument("--scope", help="'N' = first N disjoint groups (0:1,2:3,... for pairs), or 'all'")
    ap.add_argument("--group-size", dest="group_size", type=int, help="tasks per merge group (2=pairs, 3=triplets); used with --scope")
    ap.add_argument("--method", help="method or comma-list, e.g. 'ties' or 'task_arithmetic,ties' (cross-product with pairs)")
    ap.add_argument("--params", help='JSON params (single method only), e.g. \'{"density":0.3}\'')
    ap.add_argument("--eval-tasks", dest="eval_tasks", help="'auto' | task indices '3,4' | scene names")
    ap.add_argument("--trials", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--save-videos", dest="save_videos", help="videos per task: 'all' | 0/'none' | k>0 (first k) | k<0 (last |k|)")
    ap.add_argument("--out", default="sweep/merge_plan.json")
    a = ap.parse_args()

    cfg = dict(DEFAULTS)
    if a.config:
        import yaml
        cfg.update({k: v for k, v in (yaml.safe_load(open(a.config)) or {}).items() if v is not None})
    if a.pairs is not None: cfg["pairs"], cfg["scope"] = a.pairs, None
    if a.scope is not None: cfg["scope"], cfg["pairs"] = a.scope, None
    if a.method is not None: cfg["method"] = a.method
    if a.trials is not None: cfg["trials"] = a.trials
    if a.workers is not None: cfg["workers"] = a.workers
    if a.eval_tasks is not None: cfg["eval_tasks"] = a.eval_tasks
    if a.group_size is not None: cfg["group_size"] = a.group_size
    if a.save_videos is not None: cfg["save_videos"] = a.save_videos

    # methods: single or comma-list / YAML list -> cross-product with pairs
    mv = cfg["method"]
    methods = [s.strip() for s in (mv if isinstance(mv, list) else str(mv).split(","))]
    for mm in methods:
        if mm not in METHODS:
            raise SystemExit(f"unknown method '{mm}' (choose from {sorted(METHODS)})")

    cfg_params = cfg["params"] or {}
    if isinstance(cfg_params, str): cfg_params = json.loads(cfg_params)
    cli_params = json.loads(a.params) if a.params is not None else None

    def eff_and_tag(method):
        # per-method params: method-keyed YAML block, else a flat dict, else defaults; CLI --params wins
        if isinstance(cfg_params, dict) and cfg_params and set(cfg_params) <= METHODS:
            up = cfg_params.get(method, {})
        else:
            up = cfg_params
        if cli_params is not None:
            up = cli_params
        known = DEFAULT_PARAMS.get(method, {})
        # keep only params relevant to THIS method (so e.g. --params lambda doesn't tag 'average')
        eff = {**known, **{k: v for k, v in (up or {}).items() if k in known}}
        eff = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in eff.items()}
        # always include ALL of the method's hyperparameters in the name/tag (even defaults), e.g. _lambda1.0
        return eff, "".join(f"_{k}{v}" for k, v in sorted(eff.items()))

    scenes = {x["task_index"]: x["scenes"] for x in json.load(open(f"{REPO}/sweep/plan.json"))["adapters"]}
    gsize = int(cfg.get("group_size", 2) or 2)
    if cfg["scope"]:
        idxs = sorted(scenes)
        groups = [tuple(idxs[i:i + gsize]) for i in range(0, len(idxs) - gsize + 1, gsize)]
        if str(cfg["scope"]) != "all":
            groups = groups[: int(cfg["scope"])]
    else:  # explicit groups: "a:b" (pair) or "a:b:c" (triplet) or more
        pr = cfg["pairs"] if isinstance(cfg["pairs"], list) else str(cfg["pairs"]).split(",")
        groups = [tuple(int(x) for x in str(g).split(":")) for g in pr]

    eval_tasks = cfg["eval_tasks"]
    jobs = []
    for grp in groups:                            # cross product: group x method (group = pair/triplet/...)
        for t in grp:
            if t not in scenes:
                raise SystemExit(f"unknown task index {t}")
        esc = resolve_eval_scenes(sorted({s for t in grp for s in scenes[t]}), eval_tasks, scenes)
        gname = "_".join(str(t) for t in grp)
        for method in methods:
            eff, ptag = eff_and_tag(method)
            jobs.append({
                "name": f"task_{gname}_{method}{ptag}",     # e.g. task_0_1_2_average
                "method": method, "params": eff, "adapters": list(grp), "eval_scenes": esc,
            })

    nw = max(1, min(int(cfg["workers"]), 4))
    plan = {
        "base_model": BASE, "norm_stats": NORM, "n_trials": int(cfg["trials"]), "n_workers": nw,
        "save_videos": str(cfg["save_videos"]),
        # round-robin across the two P6000s (nvidia-smi indices 1,2): nw=2 -> [1,2]; nw=4 -> [1,2,1,2]
        "gpu_per_worker": [[1, 2][w % 2] for w in range(nw)], "port_per_worker": [8000, 8001, 8002, 8003][:nw],
        "results_dir": f"{REPO}/results/merge", "videos_dir": f"{REPO}/results/merge_videos",
        "logdir": f"{REPO}/results/merge_logs", "jobs": jobs,
    }
    Path(a.out).write_text(json.dumps(plan, indent=2))
    n_scenes = sum(len(j["eval_scenes"]) for j in jobs)
    print(f"wrote {a.out}: {len(groups)} group(s) of {gsize} x {len(methods)} method(s) = {len(jobs)} jobs, "
          f"{n_scenes} (group,scene) evals, methods={methods}, eval_tasks={eval_tasks}, trials={cfg['trials']}, workers={nw}")
    for j in jobs:
        print(f"  {j['name']:34s} merge{j['adapters']} -> {len(j['eval_scenes'])} scene(s)  -> results/merge/{j['name']}.jsonl")


if __name__ == "__main__":
    main()
