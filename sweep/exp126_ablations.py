#!/usr/bin/env python3
"""EXP-126 — training-side ablations of write-protection (reviewer-facing; user OK 10-18).

EXP-123's write-protected recipe (S16 init, A frozen at V^T, B's shared-output component hard-reset every step) vs:
  soft    : SkillsCrafter-style soft constraint — S16 init, A TRAINABLE, penalty wp_lambda * sum ||Un^T (B - U*Sigma)||^2 /
            sum ||U*Sigma||^2 (wp_lambda = 1, one untuned setting) instead of the hard projection
  readout : InfLoRA-style, data-free — S16 folded into the frozen base, LoRA A frozen at random orthonormal rows ORTHOGONAL
            to the shared input directions V, B trained from zero (reads OUTSIDE the shared inputs, writes anywhere)
Smoke tests (20 steps): soft moves A (<= 7%) and lets B drift into the shared outputs (<= 21% vs 54% elsewhere); readout
|A V| <= 7e-4, rows orthonormal, base+S16 step-0 loss == S16-init loss (0.0218 vs 0.0219).

PRE-REGISTERED (10-18, before any training):
  adapters   both variants for the 6 unique privileged tasks of sequences 0-3 (1,500 steps, batch 8, as EXP-37/123)
  single     2 x 10 episodes on states 0..9 per adapter (prebuilt base + adapter)
  model      EXP-120/123 privileged merge (others anchored), privileged tasks swapped for the variant; w = 1, 0.25;
             sequences 0-3; t = 8; 10 episodes, states 0..9
  readout    privileged / LIBERO-90 / other new retention vs ours_lib, anchored privileged (EXP-120) and hard write-protected
             (EXP-123) on the same sequences (paired by sequence and arrival position)
  predictions hard write-protection keeps more of LIBERO-90 than both variants at comparable protection; soft sits between
             anchored and hard; readout (no output constraint) costs the library like anchored adapters
Env: E126_STAGE base|run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp123_wptrain as WP                                           # registers the hard write-protected adapters
import exp120_privileged as P
import exp118_anchored as A
import exp116_continual as C
import exp113_controls as X
import exp112_bench as E
import unified_pool_driver as U
SEQS, WS, VARS = range(4), (1.0, 0.25), ("soft", "readout")
VDIR = {"soft": REPO / "results/analysis/new_loras_s16soft", "readout": REPO / "results/analysis/new_loras_readout"}
VKEY = {"soft": ("np16soft:", "f"), "readout": ("rdout:", "o")}
VX, VI = {v: [] for v in VARS}, {v: {} for v in VARS}                 # entries; anchored index -> variant index
_priv = sorted({i for s in SEQS for i in A.sequence(s)[1][:P.NP]})
for v in VARS:
    for i in _priv:
        e = E.LIB[i]; name = str(e["key"]).split(":", 1)[1]; path = VDIR[v] / name / "adapter_model.safetensors"
        U.EXTPATH[VKEY[v][0] + name] = str(path)
        VI[v][i] = len(E.LIB); E.LIB.append(dict(e, key=VKEY[v][0] + name, tid=VKEY[v][1] + e["tid"][1:], path=str(path))); VX[v].append(E.LIB[-1])

class VBuilder(P.PBuilder):
    def fac(self, i):
        e = E.LIB[i]
        if str(e["key"]).startswith("rdout:") and e["key"] not in self.AB:     # adapter excludes S16 -> add it back
            out = {}
            for s_, (B, A_) in U.fold(e["key"]).items():
                if s_ in U.USIG:
                    out[s_] = (torch.cat([U.USIG[s_].to(B.dtype), B], 1), torch.cat([U.VS[s_].T.to(A_.dtype), A_], 0))
                else: out[s_] = (B, A_)
            self.AB[e["key"]] = out
        return super().fac(i)

def single():
    s = WP.single()
    for v in VARS:
        for i, j in VI[v].items():
            rs = [E.rec(f"e126_base_{E.LIB[j]['tid']}_r{r}", E.LIB[j]["scene"]) for r in (1, 2)]; rs = [x for x in rs if x]
            if rs: s[j] = np.mean([x["success_rate"] for x in rs])
    return s

def sequence(s, v):
    old, new = A.sequence(s); return old, [VI[v].get(i, i) if q < P.NP else i for q, i in enumerate(new)]

def items():
    out = []
    for v in VARS:
        for s in SEQS:
            old, new = sequence(s, v); mem = old + new[:P.T]; ev = sorted(old[:C.NEVAL] + new[:P.T])
            for w in WS: out.append((f"c126{v[0]}s{s}t8-priv_w{w:g}", s, {"w": w, "P": new[:P.NP]}, mem, ev))
    return out

def _rounds(todo, build, tagp):
    if not todo: return
    R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    cur = build(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"{tagp}_{r}"))], cwd=REPO)
        nxt = build(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e126] {tagp} round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def base():
    ents = [e for v in VARS for e in VX[v]]
    missing = [e["path"] for e in ents if not pathlib.Path(e["path"]).exists()]
    if missing: raise SystemExit(f"[e126] {len(missing)} adapters not trained: {missing[:2]}")
    todo = [e for e in ents if any(E.rec(f"e126_base_{e['tid']}_r{r}", e["scene"]) is None for r in (1, 2))]
    print(f"[e126] base: {len(todo)} adapters", flush=True)
    B = VBuilder()
    def build(rd):
        js = []
        for e in rd:
            j = E.LIB.index(e); out = E.PRE / f"e126_base_{e['tid']}"; B.build("ta", {"lam": 1.0}, None, [j], out)
            js += [{"name": f"e126_base_{e['tid']}_r{r}", "method": "average", "params": {}, "adapters": [], "adapter_paths": [],
                    "prebuilt_dir": str(out), "eval_scenes": [e["scene"]], "suite": e["suite"], "norm_stats": U.FLEET_NORM}
                   for r in (1, 2) if E.rec(f"e126_base_{e['tid']}_r{r}", e["scene"]) is None]
        return js
    _rounds(todo, build, "e126base")

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[3]), "test", it[1], it[4]) for sc in scs)]
    print(f"[e126] {len(todo)} merged models", flush=True)
    B = VBuilder()
    def build(rd):
        js = []
        for tg, s, p, mem, ev in rd:
            out = E.PRE / f"e126_{tg}"; t1 = time.time(); B.build("priv", p, None, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e126] built {tg} ({len(mem)} tasks) in {time.time()-t1:.0f}s", flush=True)
        return js
    _rounds(todo, build, "e126")

def report():
    from scipy import stats
    sg = single()
    L = ["=== EXP-126 training-side ablations (privileged tasks, sequences 0-3, t = 8): hard write-protected vs soft vs read-outside ==="]
    for v in VARS:
        L.append(f"single-task success {v}: {100*np.mean([sg[j] for j in VI[v].values() if j in sg]):.1f}% vs anchored "
                 f"{100*np.mean([sg[i] for i in VI[v] if i in sg]):.1f}% (n={sum(j in sg for j in VI[v].values())})")
    refs = {"ours_lib": (lambda s, w: f"c118s{s}t8-ours_lib", A.sequence),
            "anchored privileged": (lambda s, w: f"c120as{s}t8-priv_w{w:g}", A.sequence),
            "hard write-protected": (lambda s, w: f"c123s{s}t8-wppriv_w{w:g}", WP.sequence)}
    for w in WS:
        for v in VARS:
            arm = [WP.groups(f"c126{v[0]}s{s}t8-priv_w{w:g}", s, lambda s_, v=v: sequence(s_, v), sg) for s in SEQS]
            if any(a is None for a in arm): L.append(f"{v} w={w:g}: incomplete"); continue
            L.append(f"{v}, w = {w:g}:")
            for nm, (f, seqf) in refs.items():
                ref = [WP.groups(f(s, w), s, seqf, sg) for s in SEQS]
                if any(r is None for r in ref): L.append(f"  vs {nm}: reference incomplete"); continue
                for g in ("priv", "old", "other new"):
                    x = {k_: val for a in arm for k_, val in a[g].items()}; y = {k_: val for a in ref for k_, val in a[g].items()}; ks = [q for q in x if q in y]
                    d = np.array([100 * (x[q] - y[q]) for q in ks]); p_ = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
                    L.append(f"  {g:9s}: {100*np.mean([x[q] for q in ks]):5.1f}% vs {nm:21s} {100*np.mean([y[q] for q in ks]):5.1f}%  "
                             f"({d.mean():+6.1f}pp, better {int((d>0).sum())} / worse {int((d<0).sum())}, p={p_:.4f}, n={len(ks)})")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp126_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E126_STAGE", "run"); U._clear_orphaned_builds()
    {"base": base, "run": run, "report": report}[st]()
    print(f"[e126] {st} DONE", flush=True)
