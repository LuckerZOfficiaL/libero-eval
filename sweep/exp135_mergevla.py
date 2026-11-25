#!/usr/bin/env python3
"""EXP-135 — MergeVLA-style baseline (training-free part of MergeVLA, CVPR'26), user plan 10-21.

MergeVLA itself needs a different architecture (cross-attention-only action expert trained from scratch) and a test-time
router; it is not a single merged model and not continual. Its training-free merging part transfers:
  merged update  tau_m = TIES(tau_1..tau_k)  (our TIES settings: top-20%, norm-matched x1.33 like the TIES baseline)
  task mask      S_t = I[|tau_t| > lam * |tau_m - tau_t|], lam = 0.6 (MergeVLA's default mask ratio)
  task model     W0 + S_t * tau_m, except the "expert head" (last action-expert layer 17 + action_out_proj) which keeps the
                 task's OWN update unmerged (MergeVLA keeps expert heads per task)
  routing        oracle: each task is evaluated with its own mask / head (MergeVLA's router at its best)
PRE-REGISTERED (10-21, before any rollout): settings wp (write-protected new skills, EXP-124 orders 0-3) and scratch (EXP-116
orders 0-3); t = 4, 8; evaluated tasks as EXP-124 (fixed 8 of the 16 library skills + all new so far), 10 eps, states 0..9;
one task model per evaluated task (224 builds). Readout: LIBERO-90 / new retention vs ours (EXP-124 / EXP-116) and OPCM at
the same t. Prediction: with oracle routing the MergeVLA-style models beat every single-merge method on new skills
(per-task masks + own head), at a LIBERO-90 level near averaging; it is a routed model, so it is reported separately.
Env: E135_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp124_wpcontinual as W
import exp120_privileged as P
import exp116_continual as C
import exp112_bench as E
import unified_pool_driver as U
LAM, TS = 0.6, (4, 8)
SETS = {"w": W.sequence, "z": C.sequence}
HEAD = lambda s_: "gemma_expert.model.layers.17." in s_ or s_.endswith("action_out_proj")

class MBuilder(P.PBuilder):
    def __init__(self):
        super().__init__(); self._mk = None; self._tm = {}
    def _delta(self, m, p, members, s_):
        if m != "mvla": return super()._delta(m, p, members, s_)
        key = tuple(members)
        if self._mk != key:                                            # new member set: rebuild the merged-update cache
            self._tm.clear(); self._mk = key; self._sc = self.scale("ties", E.LINEAR["ties"], 1.33, members)
        if s_ not in self._tm:
            d = super()._delta("ties", E.LINEAR["ties"], members, s_)
            self._tm[s_] = None if d is None else (self._sc * d.float()).to(torch.bfloat16).cpu()
        f = self.fac(p["task"])
        if s_ not in f: return None
        tau = f[s_][0].to(E.DEV).float() @ f[s_][1].to(E.DEV).float()
        if HEAD(s_): return tau                                        # expert head: the task's own update, unmerged
        if self._tm[s_] is None: return None
        tm = self._tm[s_].to(E.DEV).float()
        if tm.shape != tau.shape: tm = tm.T
        return tm * (tau.abs() > LAM * (tm - tau).abs())

def items():
    out = []
    for cd, seqf in SETS.items():
        for s in W.SEQS:
            old, new = seqf(s)
            for t in TS:
                mem = old + new[:t]; ev = sorted(old[:C.NEVAL] + new[:t])
                for i in ev: out.append((f"c135{cd}s{s}t{t}-mvla_{E.LIB[i]['tid']}", cd, s, t, mem, i))
    return out

def job(it):
    tg, cd, s, t, mem, i = it; return f"e135_{tg}", E.LIB[i]["suite"], E.LIB[i]["scene"]

def run():
    todo = [it for it in items() if E.rec(job(it)[0], job(it)[2]) is None]
    print(f"[e135] {len(todo)} task models", flush=True)
    if not todo: return
    B = MBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for it in rd:
            tg, cd, s, t, mem, i = it; out = E.PRE / f"e135_{tg}"; t1 = time.time()
            B.build("mvla", {"task": i}, None, mem, out); nm, su, sc = job(it)
            js.append({"name": nm, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                       "eval_scenes": [sc], "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e135] built {tg} in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e135_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e135] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def report():
    sg = W.single(); its = items()
    refs = {"w": [("ours w=0.25", "c124s{s}t{t}-wp_w0.25"), ("ours w=0.1", "c124s{s}t{t}-wp_w0.1"), ("OPCM", "c125s{s}t{t}-opcm")],
            "z": [("ours (1/k)", "c116s{s}t{t}-ours_lib"), ("OPCM", "c116s{s}t{t}-opcm"), ("averaging", "c116s{s}t{t}-avg")]}
    L = ["=== EXP-135 MergeVLA-style (TIES + per-task masks + own expert head, ORACLE routing) — retention LIBERO-90 / new ==="]
    for cd, seqf in SETS.items():
        for t in TS:
            o, n, ok = [], [], True
            for it in [x for x in its if x[1] == cd and x[3] == t]:
                r = E.rec(job(it)[0], job(it)[2]); i = it[5]
                if r is None: ok = False; continue
                if sg.get(i, 0) > 0: (o if i < 74 else n).append(r["success_rate"] / sg[i])
            line = f"{'write-protected' if cd == 'w' else 'scratch'} t={t}: MergeVLA-style {100*np.mean(o):5.1f}% / {100*np.mean(n):5.1f}%" + ("" if ok else " (incomplete)")
            for nm, f in refs[cd]:
                g = [W.groups(f.format(s=s, t=t), s, t, seqf, sg) for s in W.SEQS]
                if any(x is None for x in g): continue
                line += f" | {nm} {100*np.mean([v for x in g for v in x['old'].values()]):5.1f}% / {100*np.mean([v for x in g for v in x['new'].values()]):5.1f}%"
            L.append(line)
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp135_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E135_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e135] {st} DONE", flush=True)
