#!/usr/bin/env python3
"""EXP-139 — is an external skill's constant offset a duplicated EMBODIMENT adaptation? (10-25, after the EXP-138 smoke test)

EXP-138 smoke (12 layers): 83% of an external (from-scratch) LoRA's output on its own task is a constant offset, and the
same offset hits every other task (cos 0.96). VLA-specific hypothesis H: a skill LoRA trained from the generalist pi0.5
must first re-learn the embodiment / domain adaptation (cameras, action conventions of LIBERO); library LoRAs share it
(it sits in the library's shared part S), each external re-learns its own copy -> merging stacks conflicting copies.
If H holds, an external skill works on top of S WITHOUT its offset, and only its skill part needs merging.

Stage 1 (single skill, alone): for each external e of order E139_SEQ (EXP-116 sequences, 8 externals), models
  sown      W0 + S + D_e                  (does the library's shared part conflict with e?)
  sskill1   W0 + S + D_e P_1              (e without its response to the top shared input direction)
  sskill16  W0 + S + D_e P_16             (... to the top-16 shared input directions = its offset carrier)
  skill16   W0 + D_e P_16                 (control: no S)
  ssoft1    W0 + S + soft-gated e         (EXP-138 soft1 read factor: own inputs kept, others' suppressed)
  svlm / sexpert  W0 + S + D_e on the VLM tower only / on the action expert only (added 10-25 before any rollout;
            exploratory, no prediction: where does an external skill live?)
  P_q = I - Q_q Q_q^T on e's input side; Q = EXP-138's top shared input directions of all 24 skills in the order's merge.
  S = the library's shared part (U Sigma V^T, as in our merge). Own baseline = e's single-task success (X.single, 2 x 10
  eps). Each model evaluated on e's own scene, 10 eps, states 0..9.
PRE-REGISTERED (10-25, before any rollout): H predicts sskill16 >= 0.8 x own (mean retention over the 8), skill16 <= 0.5
x own, sown >= 0.8 x own. sskill16 << own refutes H (the offset is skill-specific). 56 models.
Env: E139_STAGE run|report; E139_SEQ (0); E139_SHARD k/n; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np, torch
import safetensors.torch as st
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp113_controls as X
import exp116_continual as C
import exp112_bench as E
import unified_pool_driver as U
SEQ = int(os.environ.get("E139_SEQ", "0"))
ARMS = ["sown", "sskill1", "sskill16", "skill16", "ssoft1", "svlm", "sexpert"]
WITH_S = {"sown", "sskill1", "sskill16", "ssoft1", "svlm", "sexpert"}  # skill16 is the no-S control
EXPERT = lambda s_: "gemma_expert" in s_                              # pi0.5's action expert vs the VLM (PaliGemma) tower
SAVE = REPO / f"results/analysis/exp138_save_s{SEQ}"

def items():
    old, new = C.sequence(SEQ)
    return [(f"c139s{SEQ}-{arm}-{E.LIB[e]['tid']}", e, arm) for e in new for arm in ARMS]

def job(it): return f"e139_{it[0]}", E.LIB[it[1]]["suite"], E.LIB[it[1]]["scene"]

class DBuilder(E.Builder):
    def __init__(self):
        super().__init__(); self.SV = {}
        for f in sorted(SAVE.glob("pass*.pt")): self.SV.update(torch.load(f))
        print(f"[e139] EXP-138 saves: {len(self.SV)} layers", flush=True)
    def delta(self, e, arm, s_):
        D = None
        if arm in WITH_S: D = U.USIG[s_].to(E.DEV) @ U.VS[s_].to(E.DEV).T
        f = self.fac(e).get(s_)
        if f is None or (arm == "svlm" and EXPERT(s_)) or (arm == "sexpert" and not EXPERT(s_)): return D
        b, a = f[0].to(E.DEV).float(), f[1].to(E.DEV).float(); sv = self.SV.get(s_)
        if arm in ("sown", "svlm", "sexpert") or sv is None or e not in sv["same"]: De = b @ a   # not seen / zero factor
        elif arm == "ssoft1":
            Rs = sv["R_soft1"][e].to(E.DEV).float(); De = b @ Rs if sv["same"][e] else Rs.T @ a
        else:
            q = int(arm.replace("sskill", "").replace("skill", "")); Q = sv["Q"][:, :q].to(E.DEV).float()
            De = b @ (a - (a @ Q) @ Q.T) if sv["same"][e] else (b - Q @ (Q.T @ b)) @ a
        return De if D is None else D + De
    def build_arm(self, e, arm, out):
        sd = dict(self.base_sd)
        with torch.no_grad():
            for s_, ck in self.ck.items():
                D = self.delta(e, arm, s_)
                if D is None: continue
                W = self.base_sd[ck]; D = D.float(); D = D if D.shape == W.shape else D.T
                sd[ck] = (W.to(E.DEV).float() + D).to(W.dtype).cpu()
        shutil.rmtree(out, ignore_errors=True); (out / "assets/physical-intelligence/libero").mkdir(parents=True)
        st.save_file(sd, str(out / "model.safetensors"))
        shutil.copy2(E.BASE_DIR / "config.json", out / "config.json")
        shutil.copy2(REPO / "results/analysis/norm_assets/norm_stats.json", out / "assets/physical-intelligence/libero/norm_stats.json")
        torch.cuda.empty_cache()

def run():
    allit = items(); todo = [it for it in allit if E.rec(job(it)[0], job(it)[2]) is None]
    if os.environ.get("E139_SHARD"):
        k_, n_ = (int(x) for x in os.environ["E139_SHARD"].split("/")); todo = [it for it in todo if allit.index(it) % n_ == k_]
    print(f"[e139] {len(todo)} models", flush=True)
    if not todo: return
    B = DBuilder(); R = int(os.environ.get("E113_ROUND", "1")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for it in rd:
            out = E.PRE / f"e139_{it[0]}"; t1 = time.time(); B.build_arm(it[1], it[2], out); nm, su, sc = job(it)
            js.append({"name": nm, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                       "eval_scenes": [sc], "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e139] built {it[0]} in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e139{os.environ.get('E139_SHARD', '').replace('/', 'of')}_{r}"))], cwd=REPO)
        try: nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        except Exception:                                                # never orphan a running eval nor leave its models
            proc.wait()
            for d in {j["prebuilt_dir"] for j in cur} | {str(E.PRE / f"e139_{it[0]}") for it in rounds[r + 1]}: shutil.rmtree(d, ignore_errors=True)
            raise
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e139] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def report():
    from scipy import stats
    sg = X.single(); old, new = C.sequence(SEQ); L = [f"=== EXP-139 stage 1, order {SEQ}: external skill alone, retention (success / single-task) ==="]
    L.append("skill        own   " + " ".join(f"{a:>9s}" for a in ARMS))
    tab = {a: [] for a in ARMS}
    for e in new:
        row = []
        for a in ARMS:
            r = E.rec(f"e139_c139s{SEQ}-{a}-{E.LIB[e]['tid']}", E.LIB[e]["scene"])
            v = None if r is None or not sg.get(e) else r["success_rate"] / sg[e]
            tab[a].append(v); row.append("    —    " if v is None else f"{100*v:8.0f}%")
        L.append(f"{E.LIB[e]['tid']:10s} {100*sg.get(e, 0):4.0f}%  " + " ".join(row))
    L.append("mean       " + "       " + " ".join(f"{100*np.mean([v for v in tab[a] if v is not None]):8.1f}%" if any(v is not None for v in tab[a]) else "    —    " for a in ARMS))
    for a in ARMS:
        v = np.array([x for x in tab[a] if x is not None])
        if len(v) == len(new) and np.any(v != 1): L.append(f"  {a}: retention {100*v.mean():.1f}% (n={len(v)}; vs 100%: Wilcoxon p={stats.wilcoxon(v - 1).pvalue:.2g})")
    txt = "\n".join(L); print(txt); (REPO / f"results/analysis/exp139_s{SEQ}_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st_ = os.environ.get("E139_STAGE", "run"); U._clear_orphaned_builds()
    run() if st_ == "run" else report()
    print(f"[e139] {st_} DONE", flush=True)
