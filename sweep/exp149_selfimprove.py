"""11-07 EXP-149 self-improvement: a second round of the in-merge correction, on data the corrected merge generates itself.
  collect  the round-1 deployed model (scratch merge 1/k + consfz, the plain LoRA correction, raw) runs the NEW skills,
           50 episodes on initial states 10..49 (test states 0..9 never used), dumps as EXP-144 -> S.DUMPS/e149_<tid>_r2s<SEQ>/
  train    the round-1 correction continues (E149_STEPS, default 1,500) inside the same merge on round-1 data (the externals'
           own rollouts) + round-2 data (the merged model's own successes; 2 groups per new skill, round 1 twice when a
           skill has no round-2 success) + the same library replay -> exp147 arm consfz2 (deployed raw, like consfz)
Eval / report: exp147 with E147_VARS=consfz2. Env: E149_STAGE collect|train; E145_SEQ (order); E113_* / E112_* (collect)."""
import os, sys, json, glob, pathlib, subprocess, time, shutil
os.environ["E147_VARS"] = "consfz,consfz2"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp147_consolidate as C
E, S, U, SEQ, OLD, NEW, ADIR, REPO = C.E, C.S, C.U, C.SEQ, C.OLD, C.NEW, C.ADIR, C.REPO
STEPS2 = int(os.environ.get("E149_STEPS", "1500"))
def name(i): return f"e149_{E.LIB[i]['tid']}_r2s{SEQ}"
def pat(i): return str(S.DUMPS / name(i) / "*" / "ep*_success.npz")

def collect():
    J = [{"name": name(i), "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "eval_scenes": [E.LIB[i]["scene"]],
          "suite": E.LIB[i]["suite"], "norm_stats": U.FLEET_NORM} for i in NEW if E.rec(name(i), E.LIB[i]["scene"]) is None]
    print(f"[e149] collect s{SEQ}: {len(J)} new skills x {S.NCOL} episodes (states {S.OFF}..{S.OFF + S.POOL - 1})", flush=True)
    if not J: return
    for j in J:                                                      # a failed earlier attempt's partial episodes -> set aside
        d = S.DUMPS / j["name"]
        if d.exists(): d.rename(d.with_name(f"{d.name}_partial_{time.strftime('%m%d%H%M')}"))
    z = C.ZIDX["consfz"]; out = E.PRE / f"e149_{C.tag('consfz')}"; t1 = time.time()
    C.ZBuilder().build("privx", {"w": C.WN, "P": [], "raw": [z], "k": len(OLD + NEW)}, None, OLD + NEW + [z], out)   # as exp147 run()
    print(f"[e149] built the round-1 model in {time.time()-t1:.0f}s", flush=True)
    for j in J: j["prebuilt_dir"] = str(out)
    S.orch(J, "e149col", n=S.NCOL, offset=S.OFF, state_pool=S.POOL, dump_obs_dir=str(S.DUMPS), dump_stride=2)
    shutil.rmtree(out, ignore_errors=True)

def train():
    out = C.tdir("consfz2")
    if (out / "adapter_model.safetensors").exists(): return
    try: os.close(os.open(ADIR / ".claim_consfz2", os.O_CREAT | os.O_EXCL))
    except FileExistsError: return
    by = {E.LIB[i]["scene"]: i for i in NEW}; lines = []
    for ln in open(ADIR / "tasks_ext.txt"):
        if not ln.strip(): continue
        su, tk, g1 = ln.split(); i = by[tk]; n2 = len(glob.glob(pat(i)))
        lines += [f"{su} {tk} {g1}", f"{su} {tk} {pat(i) if n2 else g1}"]
        print(f"[e149] {tk[:60]}: round-1 {len(glob.glob(g1))} / round-2 {n2} successful episodes", flush=True)
    (ADIR / "tasks_r2.txt").write_text("\n".join(lines) + "\n")
    ctx = json.load(open(ADIR / "ctx_all_zfree.json")); ctx["resume"] = str(C.tdir("consfz") / "adapter_model.safetensors")
    json.dump(ctx, open(ADIR / "ctx_all_zfree_r2.json", "w"), indent=1)
    cmd = [str(REPO / ".venv/bin/python"), str(REPO / "sweep/train_lora.py"), "--suite", "libero_spatial_no_noops", "--task", "consolidation",
           "--steps", str(STEPS2), "--batch", "8", "--device", "cuda:0", "--init", "ctx", "--context", str(ADIR / "ctx_all_zfree_r2.json"),
           "--tasks_file", str(ADIR / "tasks_r2.txt"), "--replay_tasks", str(ADIR / "replay.txt"), "--replay_frac", str(C.FRAC["consfz2"]),
           "--replay_demos", "5", "--out", str(out)]
    t1 = time.time(); r = subprocess.run(cmd, cwd=REPO, stdout=open(ADIR / "train_consfz2.log", "w"), stderr=subprocess.STDOUT)
    print(f"[e149] trained consfz2 ({STEPS2} steps, resumed from consfz) -> exit {r.returncode} in {(time.time()-t1)/60:.0f} min", flush=True)

if __name__ == "__main__":
    st_ = os.environ.get("E149_STAGE", "collect"); U._clear_orphaned_builds()
    {"collect": collect, "train": train}[st_]()
