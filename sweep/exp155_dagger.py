"""11-07 EXP-155 expert relabeling (DAgger): the second-round correction learns the EXPERT's action at the states the corrected
merge actually visits, instead of the merged model's own actions (EXP-149 consfz2 = own successes only: order 2 -2.5 / -3.8).
  relabel  each new skill's expert (its from-scratch LoRA alone on the base, scale 1) predicts an action chunk at every
           E155_STRIDE-th step (default 8) of the round-1 deployed model's EXP-149 rollouts of that skill (all 50 episodes,
           successes AND failures, initial states 10..49) -> S.DUMPS/e155_<tid>_s<SEQ>/ (frames + 'chunk', raw actions).
           Check per skill: on its own round-1 rollouts the expert must predict the executed action better than the base.
  train    as consfz2 (resume consfz, E149_STEPS 1,500, same merge / replay / mixture) with the round-2 group = the relabeled
           frames -> exp147 arm consfzd (deployed raw). consfzd vs consfz2 = expert labels on failures too vs own successes.
PRE-REGISTERED (11-07, before relabeling): helps if consfzd new >= consfz new + 5pp pooled, library within 3pp. Order 2
first; >= +5 there -> orders 0, 1, 3; 0..+5 -> order 0 before deciding; <= 0 -> stop.
11-08 EXP-156 gated DAgger (arm consfzg, default): order 2's consfzd fixed the skill the merge lost to a good expert
(tomato sauce 20 -> 90) but pulled skills down to weak / beaten experts (wine rack 20 -> 0, expert 30; bowl next to plate
100 -> 70, expert 80) -> relabel skill i only if its expert's own successes on the training states 10..49 (EXP-144) exceed
the round-1 merged model's (EXP-149, same states) by >= E155_GATE (5) of 50; other skills keep round 1 in both groups.
Eval / report: exp147 with E147_VARS=consfzd / consfzg. Env: E155_STAGE relabel|train|ready; E145_SEQ (order); E155_STRIDE; relabel on the one
GPU in CUDA_VISIBLE_DEVICES (one process per GPU, skills claimed one at a time); train as exp149."""
import os, sys, json, glob, pathlib, re, subprocess, time
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")                       # P6000 (odin) < Triton's capability 7.0
os.environ["E147_VARS"] = "consfz,consfzd,consfzg"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import exp147_consolidate as C
E, S, SEQ, NEW, ADIR, REPO = C.E, C.S, C.SEQ, C.NEW, C.ADIR, C.REPO
STRIDE = int(os.environ.get("E155_STRIDE", "8"))
STEPS2 = int(os.environ.get("E149_STEPS", "1500"))
ARM = os.environ.get("E155_ARM", "consfzg")                             # consfzd = every skill relabeled; consfzg = gated (EXP-156)
GATE = int(os.environ.get("E155_GATE", "5"))
def src(i): return S.DUMPS / f"e149_{E.LIB[i]['tid']}_r2s{SEQ}"         # EXP-149 rollouts of the round-1 model
def ddir(i): return S.DUMPS / f"e155_{E.LIB[i]['tid']}_s{SEQ}"
def pat(i): return str(ddir(i) / "*" / "ep*.npz")
def prompt(i): return re.sub(r"^[A-Z_]*SCENE\d+_", "", E.LIB[i]["scene"]).replace("_", " ")   # = train_lora's eval prompt
def rate(pattern): fs = glob.glob(pattern); return sum(f.endswith("_success.npz") for f in fs) / max(len(fs), 1)
def wins(i):   # expert - merge success on the training states, in episodes of 50 (rates: a collection can be partial)
    return round(50 * (rate(S.pattern(i).replace("_success.npz", "_*.npz")) - rate(str(src(i) / "*" / "ep*.npz"))))
def use(i): return ARM == "consfzd" or wins(i) >= GATE

def relabel():
    import torch, action_error as AE
    from openpi.runtime import LiberoPolicy
    for i in NEW: print(f"[e155] {prompt(i)[:50]}: expert - merge {wins(i):+d} / 50 (rates) on training states -> {'relabel' if use(i) else 'skip'}", flush=True)
    todo = [i for i in NEW if use(i) and not (ddir(i) / ".done").exists()]
    print(f"[e155] relabel s{SEQ}: {len(todo)} new skills, every {STRIDE}th step", flush=True)
    if not todo: return
    pol = LiberoPolicy("/media/pinas/Chris/models/pi05_base_pytorch", norm_stats_path=str(REPO / "results/analysis/norm_assets"),
                       pytorch_device="cuda:0")
    ap = AE.Applier(pol, E.LIB[todo[0]]["path"])
    def pred(o, i): return np.asarray(pol.infer(dict(o, prompt=prompt(i)))["actions"], dtype=np.float32)
    for i in todo:
        ddir(i).mkdir(parents=True, exist_ok=True)                    # several relabelers (odin + shiva GPUs) split the skills
        try: os.close(os.open(ddir(i) / ".claim", os.O_CREAT | os.O_EXCL))
        except FileExistsError: continue
        fo = AE.fold(E.LIB[i]["path"]); miss = {m for m in set(fo) - set(ap.stem2param) if not m.endswith("lm_head")}
        if miss: raise SystemExit(f"[e155] {len(miss)} expert modules not applied")   # tied lm_head: unused by the action head
        own = sorted(glob.glob(S.pattern(i)))[:2]; chk = []            # check: expert vs base on the expert's own rollouts
        for f in own:
            z = np.load(f); pos = {int(s): k for k, s in enumerate(z["act_step"])}
            for j in np.linspace(0, len(z["step"]) - 1, 4).astype(int):
                chk.append(({"observation/image": z["img"][j], "observation/wrist_image": z["wrist"][j],
                             "observation/state": z["state"][j].astype(np.float32)}, z["act"][pos[int(z["step"][j])]]))
        eb = np.mean([np.abs(pred(o, i)[0, :7] - a).mean() for o, a in chk])
        ap.apply([(fo, 1.0)])
        ee = np.mean([np.abs(pred(o, i)[0, :7] - a).mean() for o, a in chk])
        print(f"[e155] {prompt(i)[:50]}: own-rollout L1 base {eb:.4f} expert {ee:.4f}", flush=True)
        if not ee < eb: raise SystemExit("[e155] the expert does not beat the base on its own rollouts: preprocessing wrong")
        fs = sorted(glob.glob(str(src(i) / "*" / "ep*.npz"))); dis = {"success": [], "failure": []}; n = 0; t1 = time.time()
        for f in fs:
            z = np.load(f); img, wri, stt, steps = z["img"], z["wrist"], z["state"], z["step"]
            pos = {int(s): k for k, s in enumerate(z["act_step"])}
            keep = [j for j, s in enumerate(steps) if (int(s) - int(steps[0])) % STRIDE == 0]
            ch = np.stack([pred({"observation/image": img[j], "observation/wrist_image": wri[j],
                                 "observation/state": stt[j].astype(np.float32)}, i)[:, :7] for j in keep])
            dis["success" if f.endswith("_success.npz") else "failure"] += [
                float(np.abs(ch[q, 0] - z["act"][pos[int(steps[j])]]).mean()) for q, j in enumerate(keep) if int(steps[j]) in pos]
            o = ddir(i) / pathlib.Path(f).parent.name; o.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(o / pathlib.Path(f).name, img=img[keep], wrist=wri[keep], state=stt[keep], step=steps[keep], chunk=ch)
            n += len(keep)
        ap.restore()
        print(f"[e155] {prompt(i)[:50]}: {n} frames from {len(fs)} episodes in {(time.time()-t1)/60:.0f} min; expert vs executed "
              f"L1 on success {np.mean(dis['success'] or [np.nan]):.4f} / failure {np.mean(dis['failure'] or [np.nan]):.4f} episodes", flush=True)
        (ddir(i) / ".done").touch()

def train():
    out = C.tdir(ARM)
    if (out / "adapter_model.safetensors").exists(): return
    if not all((ddir(i) / ".done").exists() for i in NEW if use(i)): raise SystemExit("[e155] relabeling incomplete")
    try: os.close(os.open(ADIR / f".claim_{ARM}", os.O_CREAT | os.O_EXCL))
    except FileExistsError: return
    by = {E.LIB[i]["scene"]: i for i in NEW}; lines = []
    for ln in open(ADIR / "tasks_ext.txt"):
        if not ln.strip(): continue
        su, tk, g1 = ln.split(); i = by[tk]; lines += [f"{su} {tk} {g1}", f"{su} {tk} {pat(i) if use(i) else g1}"]
        print(f"[e155] {ARM} {tk[:50]}: round-2 group = {'relabeled' if use(i) else 'round 1'}", flush=True)
    (ADIR / f"tasks_{ARM}.txt").write_text("\n".join(lines) + "\n")
    ctx = json.load(open(ADIR / "ctx_all_zfree.json")); ctx["resume"] = str(C.tdir("consfz") / "adapter_model.safetensors")
    json.dump(ctx, open(ADIR / "ctx_all_zfree_d.json", "w"), indent=1)
    cmd = [str(REPO / ".venv/bin/python"), str(REPO / "sweep/train_lora.py"), "--suite", "libero_spatial_no_noops", "--task", "consolidation",
           "--steps", str(STEPS2), "--batch", "8", "--device", "cuda:0", "--init", "ctx", "--context", str(ADIR / "ctx_all_zfree_d.json"),
           "--tasks_file", str(ADIR / f"tasks_{ARM}.txt"), "--replay_tasks", str(ADIR / "replay.txt"), "--replay_frac", str(C.FRAC[ARM]),
           "--replay_demos", "5", "--out", str(out)]
    t1 = time.time(); r = subprocess.run(cmd, cwd=REPO, stdout=open(ADIR / f"train_{ARM}.log", "w"), stderr=subprocess.STDOUT)
    print(f"[e155] trained {ARM} ({STEPS2} steps, resumed from consfz) -> exit {r.returncode} in {(time.time()-t1)/60:.0f} min", flush=True)

if __name__ == "__main__":
    def ready(): sys.exit(0 if all((ddir(i) / ".done").exists() for i in NEW if use(i)) else 1)   # for chains: relabeling complete?
    {"relabel": relabel, "train": train, "ready": ready}[os.environ.get("E155_STAGE", "relabel")]()
