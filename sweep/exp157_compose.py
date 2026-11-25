#!/usr/bin/env python3
"""11-10 EXP-157 — composite tasks (LIBERO-10): can ONE merged policy chain library skills that no single skill adapter covers?

Why: per-task adapter switching (a router that loads the instructed skill's own adapter) beats the merge on known skills
(74-skill library: expert 94.9 vs ours 82.6), so the merge needs a case routing cannot serve. LIBERO-10 tasks chain two
LIBERO-90 skills of the same scene in one episode under one instruction ("turn on the stove and put the moka pot on it").
5 of the 10 are pure compositions of two skills in the 74-skill library (E145_SEQ 6): tasks 2 / 3 / 4 / 6 / 7 below.
Protocol: every policy gets the full LIBERO-10 instruction, test states 0..9, 10 eps, 520 steps (main_sub.py = main_dense.py +
the BDDL goal predicates checked after every step -> subgoals.jsonl per job / scene; none true at the start except
KITCHEN_SCENE8's stove, excluded from its sub-goal count). Zero-shot: nothing is trained on LIBERO-10.
Arms (s6 = 74-skill library + order 0's 8 new): ours = consfz6 seeds 0 / 1 (E147_VARS), scr = training-free merge, joint24k =
one rank-64 LoRA on all 82 skills' demos, base = pi05_base alone (primary tasks only), single = each sub-skill adapter alone
(the router's choices); the router baseline = per task the better of its sub-skill adapters (oracle pick on these episodes).

PRE-REGISTERED (11-10, before any LIBERO-10 episode): primary set = tasks 2 / 3 / 4 / 6 / 7 (50 episodes per arm), metric =
full task success (all goal predicates). POSITIVE (a merged policy serves composites routing cannot): ours (2-seed mean)
>= 20% AND >= router-best + 10pp, pooled over the 5 tasks. NEGATIVE: ours < 10%, or <= router-best -> the composite
argument does not hold; reported with sub-goal completion (fraction of goal predicates true at once, max over the episode).
Otherwise inconclusive. Tasks 0 / 1 / 5 / 8 / 9 (one sub-skill or none in the library) are reported, not tested.
Env: E157_STAGE run|report; E157_ARMS (default all); E112_* eval (as exp147). Builds go to E.PRE (/dev/shm): <= 4 at once.
"""
import json, os, pathlib, shutil, subprocess, sys, time
os.environ["E145_SEQ"] = "6"; os.environ.setdefault("E147_VARS", "consfz6,consfz6s1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp147_consolidate as C
import exp120_privileged as P
E, U, REPO = C.E, C.U, C.REPO
sys.path.insert(0, str(REPO / "examples/libero")); from task_suite_task_map import LIBERO_TASK_MAP
T10 = LIBERO_TASK_MAP["libero_10"]
PRIMARY = [2, 3, 4, 6, 7]
SUB = {   # LIBERO-10 task -> library skills (s6) it chains; tasks 0 / 1 / 5 / 8 / 9: same objects elsewhere / one part / variants
    0: ["LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket", "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket"],
    1: ["LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket", "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket"],
    2: ["KITCHEN_SCENE3_turn_on_the_stove", "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove"],
    3: ["KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet", "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet"],
    4: ["LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate", "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate"],
    5: [f"STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_{c}_compartment_of_the_caddy" for c in ("front", "left", "right")],
    6: ["LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate", "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate"],
    7: ["LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket", "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket"],
    8: ["KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove"],
    9: ["KITCHEN_SCENE6_close_the_microwave"],
}
LI = {E.LIB[i]["scene"]: i for i in C.OLD}
assert all(s in LI for v in SUB.values() for s in v), [s for v in SUB.values() for s in v if s not in LI]
JPATH = REPO / "results/analysis/joint_loras/joint24k_s6/adapter_model.safetensors"
U.EXTPATH["joint24k:s6"] = str(JPATH); JI = len(E.LIB)
E.LIB.append({"key": "joint24k:s6", "ext": True, "suite": "joint", "scene": "joint24k_s6", "tid": "j624k", "path": str(JPATH)})
SINGLES = sorted({s for v in SUB.values() for s in v}, key=lambda s: LI[s])
OURS = [v for v in C.VARS if v in C.ZIDX]

def arms():
    """[(arm, build(out), task indices)] in run order: ours, references, router singles, base"""
    ev = sorted(C.OLD + C.NEW)
    out = [(v, (lambda z: lambda o: C.ZBuilder().build("privx", {"w": C.WN, "P": [], "raw": [z], "k": len(ev)}, None, C.OLD + C.NEW + [z], o))(C.ZIDX[v]),
            list(range(10))) for v in OURS]
    out += [("scr", lambda o: C.ZBuilder().build("ours_lib", {}, None, C.OLD + C.NEW, o), list(range(10))),
            ("joint24k", lambda o: P.PBuilder().build("ta", {"lam": 1.0}, None, [JI], o), list(range(10)))]
    out += [(f"single{E.LIB[LI[s]]['tid']}", (lambda i: lambda o: P.PBuilder().build("ta", {"lam": 1.0}, None, [i], o))(LI[s]),
             [t for t, v in SUB.items() if s in v]) for s in SINGLES]
    out += [("base", lambda o: P.PBuilder().build("ta", {"lam": 0.0}, None, [JI], o), PRIMARY)]
    want = os.environ.get("E157_ARMS")
    return [a for a in out if not want or a[0] in want.split(",")]

def jname(arm, t): return f"e157_{arm}_t{t}"
def subgoals(arm, t):
    f = REPO / f"results/merge_videos/{jname(arm, t)}/{T10[t]}/subgoals.jsonl"
    if not f.exists(): return None
    ep = {}
    for ln in open(f): r = json.loads(ln); ep[r["ep"]] = r                 # a retried scene appends again: last wins
    return [ep[k] for k in sorted(ep)]

def run():
    todo = [(a, b, [t for t in ts if E.rec(jname(a, t), T10[t]) is None]) for a, b, ts in arms()]
    todo = [x for x in todo if x[2] and (x[0] not in OURS or pathlib.Path(E.LIB[C.ZIDX[x[0]]]["path"]).exists())]
    print(f"[e157] {len(todo)} arms to run: {[a for a, _, _ in todo]}", flush=True)
    rounds = [todo[r:r + 2] for r in range(0, len(todo), 2)]
    def build(rd):
        js = []
        for a, b, ts in rd:
            o = E.PRE / f"e157_{a}"; t1 = time.time(); b(o); print(f"[e157] built {a} in {time.time()-t1:.0f}s", flush=True)
            js += [{"name": jname(a, t), "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(o),
                    "eval_scenes": [T10[t]], "suite": "libero_10", "norm_stats": U.FLEET_NORM} for t in ts]
        return js
    cur = build(rounds[0]) if rounds else []
    for r in range(len(rounds)):
        p = E.plan(cur, 10, 0, f"e157r{r}"); d = json.load(open(p)); d["eval_script"] = "examples/libero/main_sub.py"; json.dump(d, open(p, "w"), indent=1)
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(p)], cwd=REPO)
        nxt = build(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for o in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(o, ignore_errors=True)
        print(f"[e157] round {r + 1}/{len(rounds)} done", flush=True); cur = nxt

def report():
    def cell(a, t):
        r = E.rec(jname(a, t), T10[t]); sg = subgoals(a, t)
        if r is None: return None
        frac = None
        if sg:
            keep = [g for g in range(sg[0]["n_goal"]) if not any(e["init"] and e["init"][g] for e in sg)]   # drop goals true at the start
            frac = 100 * sum(sum(e["ever"][g] for g in keep) / max(len(keep), 1) for e in sg) / len(sg)
        return 100 * r["success_rate"], frac
    A = [a for a, _, _ in arms()]; R = {a: {t: cell(a, t) for t in range(10)} for a in A}
    def router(t, k):   # best single sub-skill adapter on task t (oracle pick), metric k (0 full success, 1 sub-goals)
        v = [R[f"single{E.LIB[LI[s]]['tid']}"][t] for s in SUB[t]]; v = [x[k] for x in v if x is not None and x[k] is not None]
        return max(v) if v else None
    rows = [(a, [R[a][t] for t in range(10)]) for a in A if not a.startswith("single")]
    rows.append(("router-best", [(router(t, 0), router(t, 1)) if router(t, 0) is not None else None for t in range(10)]))
    print("task: " + " ".join(f"{t}:{T10[t][:38]}" for t in range(10)))
    for k, nm in ((0, "FULL SUCCESS"), (1, "SUB-GOALS (% of goal predicates reached)")):
        print(f"== {nm}  primary = tasks {PRIMARY}")
        for a, v in rows:
            x = [None if c is None or c[k] is None else c[k] for c in v]; pr = [x[t] for t in PRIMARY if x[t] is not None]
            print(f"{a:12s} primary {sum(pr) / len(pr) if len(pr) == len(PRIMARY) else float('nan'):5.1f} | " + " ".join("  -" if c is None else f"{c:4.0f}" for c in x))
    ours = [sum(R[v][t][0] for t in PRIMARY) / len(PRIMARY) for v in OURS if all(R[v][t] for t in PRIMARY)]
    rb = [router(t, 0) for t in PRIMARY]
    if len(ours) == len(OURS) and None not in rb:
        o, b = sum(ours) / len(ours), sum(rb) / len(rb)
        verdict = "POSITIVE" if o >= 20 and o >= b + 10 else "NEGATIVE" if o < 10 or o <= b else "INCONCLUSIVE"
        print(f"[e157] ours (mean of {OURS}) {o:.1f} vs router-best {b:.1f} on the primary set -> {verdict}")

if __name__ == "__main__":
    {"run": run, "report": report}[os.environ.get("E157_STAGE", "report")]()
