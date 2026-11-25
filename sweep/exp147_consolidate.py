#!/usr/bin/env python3
"""EXP-147 — consolidation: ONE correction trained inside the actual t=8 merge, instead of 8 parallel ones (pilot, order 0).

EXP-145b: the 8 parallel in-context updates (each external trained inside the merge of the OTHER 23, half its own rollouts,
half library replay) overshoot when summed (0.67-cosine aligned: library 0 / externals 0) and, averaged (FedAvg, adapter =
conv + 1/8 (ctxr - conv)), keep the library at 92.8% but leave externals at 47.0 (converted 44.5): the shared library
correction survives the mean, each external's own improvement is diluted 8x. The receiver owns every input (the merge, the
externals' self-rollouts, the library's demos), so it can train ONE correction against the merge it will deploy — no
Jacobi overshoot, no dilution.
  cons   context = the full t=8 converted merge (S16 + 0.25 x the 8 converted private parts + 1/24 x the 16 library
         residuals, as PBuilder builds it) folded into the frozen base; one private part (A = V^T frozen, B orthogonal to the
         library's shared outputs, from ZERO) acting at x0.25; 3,000 steps batch 8: half on the 8 externals' own successful
         rollouts (EXP-144 dumps, uniform over externals), half on library replay (5 demos per library skill, own prompt).
  consl  the same with replay only (frac 1.0): a pure library repair — does the external part of cons do anything?
  merge  all externals share A = V^T, so adding the correction once at 0.25 == adding 1/8 of it to every converted adapter:
         adapter_i = conv_i + 1/8 (cons - c0), c0 = the same run at 0 steps (S16 with a zero private part); K stays 24.
         EXP-144 priv merge at w 0.25, t=8, all 24 skills, 10 eps, test states 0..9.
Compute: one 3,000-step run vs 8 x 750 for the parallel round (half).
PRE-REGISTERED (11-05, before training): cons helps externals if externals >= FedAvg ctxr (47.0) + 10pp (= 57.0%) with
library >= ours' library-only merge at t=0 (86.3%, 4-order mean); matches WP on demos if externals >= 61.0 - 5.
consl: library within 5pp of cons and externals within 5pp of converted (44.5) -> cons' external gain is its own-rollout half.
Env: E147_STAGE prep|train|interp|run|report; E147_VARS (default cons,consl); E147_STEPS (3000); E145_SEQ (order); E113_GPUS / E113_DEV.
"""
import json, os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("E145_MIX", "0.125:0.125")                       # 11-05: EXP-145d best (FedAvg of ctx + ctxr) as a reference
import exp145_incontext as X5                                         # registers converted (S.CV) + in-context (X5.CX) adapters
S, E, U, RT, X, C = X5.S, X5.E, X5.U, X5.RT, X5.X, X5.C
SEQ, T, WN, OLD, NEW = X5.SEQ, X5.T, X5.WN, X5.OLD, X5.NEW
STEPS = int(os.environ.get("E147_STEPS", "3000"))
VARS = os.environ.get("E147_VARS", "cons,consl").split(",")
# 11-05 ablations: conso = own rollouts only (no replay); cons1 = cons, seed 1
FRAC = {"cons": 0.5, "consl": 1.0, "conso": 0.0, "consz": 0.5, "cons1": 0.5, "consf": 0.5, "consfz": 0.5, "consb": 0.5, "consr": 0.5, "consq": 0.5, "consh": 0.5, "consz6": 0.5, "consfzr": 0.5, "consfzr1": 0.5, "consfz2": 0.5, "consfzd": 0.5, "consfzg": 0.5, "consfzc": 0.5, "consfz6": 0.5, "consfz12": 0.5, "consfz6r25": 0.5, "consfz6s1": 0.5, "consfz6r64": 0.5, "consfz6r64s1": 0.5,
        "erl": 0.5, "erlc": 0.5, "seq": 0.0, "seqc": 0.0, "consfa": 0.5, "consfac": 0.5, "consb64": 0.5, "consb6": 0.5, "consb12": 0.5, "ewc": 0.0, "ewcc": 0.0, "lrc": 0.5, "lrcfz": 0.5,
        "consfzq": 0.5, "erlq": 0.5, "consfz10": 0.5, "erl10": 0.5,
        "consbc": 0.5, "erlr": 0.5, "consbr": 0.5, "erld1": 0.5, "consbd1": 0.5, "erld25": 0.5, "consbd25": 0.5, "consfaq": 0.5, "consfzd1": 0.5, "consfzd25": 0.5, "consfzd1c": 0.5, "erld1c": 0.5,
        "consfzn1": 0.5, "erln1": 0.5, "consfzn3": 0.5, "erln3": 0.5, "erl6": 0.5, "consfzl": 0.5, "erll": 0.5}
# arms whose correction enters as ONE extra member: consz = cons on the FROM-SCRATCH merge (no conversion); consf / consfz =
# plain-fine-tune baselines: a standard LoRA (A and B trainable, no write protection) on the converted / scratch merge, added raw
ZV = {"consz": ("ctx_all_z.json", "scratch", False), "consf": ("ctx_all_free.json", "conv", True), "consfz": ("ctx_all_zfree.json", "scratch", True),
      "consb": (None, "base", True), "consz6": ("ctx_all_z.json", "scratch", False), "consfzr": ("ctx_all_zfree.json", "scratch", True),
      "consfzr1": ("ctx_all_zfree.json", "scratch", True), "consfz2": ("ctx_all_zfree.json", "scratch", True),
      "consfzd": ("ctx_all_zfree.json", "scratch", True), "consfzg": ("ctx_all_zfree.json", "scratch", True),
      "consfzc": ("ctx_all_zfree_c.json", "scratch", True), "consfz6": ("ctx_all_zfree.json", "scratch", True),
      "consfz12": ("ctx_all_zfree.json", "scratch", True), "consfz6r25": ("ctx_all_zfree.json", "scratch", True),
      "consfz6s1": ("ctx_all_zfree.json", "scratch", True), "consfz6r64": ("ctx_all_zfree.json", "scratch", True),
      "consfz6r64s1": ("ctx_all_zfree.json", "scratch", True),
      "erl": ("ctx_lib_free.json", "lib", True), "erlc": ("ctx_lib_free.json", "lib", True), "seq": ("ctx_lib_free.json", "lib", True),
      "seqc": ("ctx_lib_free.json", "lib", True), "consfa": ("ctx_all_avgfree.json", "avg", True), "consfac": ("ctx_all_avgfree.json", "avg", True),
      "consb64": (None, "base", True), "consb6": (None, "base", True), "consb12": (None, "base", True), "ewc": ("ctx_lib_free.json", "lib", True), "ewcc": ("ctx_lib_free.json", "lib", True),
      "lrc": ("ctx_all_lrco.json", "lrc", False), "lrcfz": ("ctx_all_lrc.json", "lrc", True),
      "consfzq": ("ctx_all_zfree.json", "scratch", True), "consfz10": ("ctx_all_zfree.json", "scratch", True),
      "erlq": ("ctx_lib_free.json", "lib", True), "erl10": ("ctx_lib_free.json", "lib", True),
      "consbc": ("ctx_base_free.json", "base", True), "consbr": (None, "base", True), "consbd1": (None, "base", True),
      "consbd25": (None, "base", True), "erlr": ("ctx_lib_free.json", "lib", True), "erld1": ("ctx_lib_free.json", "lib", True),
      "erld25": ("ctx_lib_free.json", "lib", True), "consfaq": ("ctx_all_avgfree.json", "avg", True),
      "consfzd1": ("ctx_all_zfree.json", "scratch", True), "consfzd25": ("ctx_all_zfree.json", "scratch", True),
      "consfzd1c": ("ctx_all_zfree.json", "scratch", True), "erld1c": ("ctx_lib_free.json", "lib", True),
      "consfzn1": ("ctx_all_zfree.json", "scratch", True), "erln1": ("ctx_lib_free.json", "lib", True),
      "consfzn3": ("ctx_all_zfree.json", "scratch", True), "erln3": ("ctx_lib_free.json", "lib", True),
      "erl6": ("ctx_lib_free.json", "lib", True), "consfzl": ("ctx_all_zfree.json", "scratch", True), "erll": ("ctx_lib_free.json", "lib", True)}   # 11-06 consb: plain LoRA from the BASE on the same data / steps, deployed alone (does the merge matter?)
# 11-07 consfzr = consfz with consr's demo-free replay (the library's own rollouts): no conversion, no demos at all; consfzr1 = seed 1;
# consfz2 = consfz's second round on the merged model's own rollouts (trained by exp149_selfimprove.py, never here)
# 11-07 consfzd = the same second round on expert-relabeled states of those rollouts, DAgger (exp155_dagger.py, never here)
# 11-08 consfzg = consfzd gated per skill: relabeled only where the expert beats the merge on training states (EXP-156)
# 11-08 consfzc = continual arrival (EXP-151, E145_SEQ 19 / 20 / 4 = 14 / 18 / 22 new): the previous arrival's correction
# resumed inside the rebuilt merge, 1,500 steps on every arrived new skill's rollouts + library replay (round 1 = order 0 consfz)
# 11-08 consfz6 = consfz at 6,000 steps (EXP-150 fairness check at the 74-skill library: ~290 samples per task at 3,000)
# 11-09 consfz12 = consfz at 12,000 steps (EXP-150 vs the 24k-step joint at the 74-skill library: half its steps)
# 11-10 consfz6r25 = consfz6 with 25 library demos per skill instead of 5 (EXP-150: is the 74-skill library replay-data limited?)
# 11-10 consfz6s1 = consfz6, seed 1 (EXP-150 noise check: new at the 74-skill library 68.8 / 86.2 / 66.2 across 3k / 6k / 12k)
# 11-10 consfz6r64 = consfz6 at LoRA rank 64 (joint's rank; EXP-150: steps and replay demos did not lift the 74-skill library); consfz6r64s1 = its seed 1
# 11-18 EXP-158 / 159 same-data trained baselines (meeting 18-11: compare with methods that also train on our data):
# erl = experience replay from the LIBRARY model (training-free library-only merge, ours at 1/16): the same LoRA, data and steps
# as consfz, but the new adapters are never merged (new skills learned from their rollouts only); seq = sequential fine-tuning
# from the library model on the arrival's new-skill rollouts only (no replay). erlc / seqc = their continual rounds (EXP-159,
# E145_SEQ 19 / 20 / 4, previous round resumed, 1,500 steps; seqc trains on the arrival's own new skills only)
# 11-18 EXP-160 consfa = consfz on the VANILLA average (sum B A / k, no pool anchor) instead of our merge; consfac = its continual
# rounds; reference sca = the training-free vanilla average (scr = ours)
# 11-19 EXP-163 (what does merging the arriving adapters add beyond their rollouts?): consfzq / erlq = consfz / erl at 750 steps;
# consfz10 / erl10 = consfz / erl on the first 10 episodes per new skill (success-filtered: 3-10 rollouts, tasks_ext10.txt)
# 11-18 EXP-159 ewc = seq + online EWC (Fisher of the library replay at the start, then of each arrival's new skills; lambda VEWC,
# pre-registered 1e-3 = penalty ~10x the task loss at an unconstrained endpoint); ewcc = its continual rounds (previous ewc.pt)
# 11-18 EXP-162 lrc = the merge's residual weights LEARNED per member and module (k per module, from 1/k; S16 fixed; the same
# data / steps; AdamW lr 1e-3, no decay) = training-based merging on our data; lrcfz = the same jointly with the LoRA correction
# 11-18 EXP-158 consb64 = consb at joint's recipe (rank 64, 12,000 steps): multi-task training from the base on OUR data
# 11-20 EXP-164d consb6 / consb12 = consb (base start, rank 16, same data) at 6,000 / 12,000 steps: what the merge start is worth in steps
# 11-06 step budget: consq / consh = cons at 750 / 1,500 steps; consz6 = consz at 6,000 (can a longer correction replace conversion?)
VEWC = {"ewc": 1e-3, "ewcc": 1e-3}
# 11-18 EXP-161 VLA part ablation (eval only): consfz's trained correction kept in ONE part (pv / pl / pe = vision tower / language
# backbone / action expert + action heads) or dropped from one (nv / nl / ne); the merge underneath unchanged
def part(stem): return "vision" if "vision_tower" in stem else "language" if "language_model" in stem else "expert"
PARTS = {"pv": {"vision"}, "pl": {"language"}, "pe": {"expert"}, "nv": {"language", "expert"}, "nl": {"vision", "expert"}, "ne": {"vision", "language"}}
# 11-19 size control: is the language backbone special or just big? It holds 42.6% of the correction's parameters (expert 38.6,
# vision 18.9) but ~71% of its energy ||B A||^2. The correction kept in a seeded random subset of modules holding a target share
# (by parameters or by energy): rap / rae = modules of any part at the language share of parameters / of energy (compare pl);
# rle = language modules only at the vision + expert energy share (compare nl)
RSUB = {"rap": (None, "param", "language"), "rae": (None, "energy", "language"), "rle": ({"language"}, "energy", "rest")}
def rsub(v, f):
    import random
    parts, ms, tgt = RSUB[v]
    n = {s: float(f[s][0].numel() + f[s][1].numel()) if ms == "param" else float(((f[s][0].float().T @ f[s][0].float()) * (f[s][1].float() @ f[s][1].float().T)).sum()) for s in f}
    tot = sum(n.values()); lang = sum(n[s] for s in n if part(s) == "language") / tot; share = lang if tgt == "language" else 1 - lang
    pool = sorted(s for s in n if parts is None or part(s) in parts); random.Random(0).shuffle(pool); keep, acc = [], 0
    for s in pool:
        if acc >= share * tot: break
        keep.append(s); acc += n[s]
    return keep
VSTEPS = {"consb6": 6000, "consb12": 12000, "consbc": 1500, "consfaq": 750, "consfzq": 750, "erlq": 750, "ewcc": 1500, "consb64": 12000, "erlc": 1500, "seqc": 1500, "consfac": 1500, "consq": 750, "consh": 1500, "consz6": 6000, "consfzc": 1500, "consfz6": 6000, "consfz12": 12000, "consfz6r25": 6000, "consfz6s1": 6000, "consfz6r64": 6000, "consfz6r64s1": 6000, "consfzd1c": 1500, "erld1c": 1500}
R10 = ("consfz10", "erl10")   # EXP-163: first 10 episodes per new skill
RN = {"consfzn1": 1, "erln1": 1, "consfzn3": 3, "erln3": 3}   # EXP-169: first n successful episodes per new skill (symlinked)
# 11-22 EXP-170 the 32-skill library (E145_SEQ 5; stream 30-32 = 14 / 18 / 22 new): erl6 = erl at consfz6's 6,000 steps (arrival 1;
# consb6 = the base start); consfzl / erll = ours / ER continual rounds from consfz6 / erl6, 3,000 resumed steps per arrival (2x
# the 16-skill stream's 1,500, same steps per library skill)
VSTEPS.update({"erl6": 6000, "consfzl": 3000, "erll": 3000})
# 11-19 EXP-164 the library merge as ER's starting point (ER = erl: library-only merge + the same data): consbc = consb (from
# the BASE) resumed per arrival like erlc; erlr / consbr = merge / base start with the library adapters' own rollouts as library
# replay (demo-free); erld1 / consbd1 / erld25 / consbd25 = merge / base start at 1 / 25 library demos per skill (default 5)
# 11-19 consfaq = consfa (vanilla average of all k) at 750 steps: component merge vs average as the start at EXP-163's low budget
# 11-20 consfzd1 / consfzd25 = ours (consfz: merge of library + arrivals, correction) at 1 / 25 library demos per skill, the
# demo budget of EXP-164b for our method (not the DAgger consfzd)
# 11-21 EXP-166 consfzd1c / erld1c = the continual stream (E145_SEQ 19 / 20 / 4, 1,500 resumed steps) at 1 library demo per
# skill: ours re-merges each arrival, ER keeps the library merge; round 1 = consfzd1@0 / erld1@0
# 11-21 EXP-169 consfzn1 / erln1, consfzn3 / erln3 = consfz / erl on the first 1 / 3 SUCCESSFUL episodes per new skill (episode
# order, symlinked into ADIR/roll{n}/<task>/; tasks_extn{n}.txt): does merging the arrivals help when new-skill rollouts are scarce?
VDEMOS = {"consfz6r25": 25, "erld1": 1, "consbd1": 1, "erld25": 25, "consbd25": 25, "consfzd1": 1, "consfzd25": 25, "consfzd1c": 1, "erld1c": 1}   # library demos per skill in the replay (default 5)
VRANK = {"consfz6r64": 64, "consfz6r64s1": 64, "consb64": 64}    # correction LoRA rank (default 16)
# 11-06 EXP-148 scaling (E145_SEQ = C.SCALE: order 0's library + all 22 new skills): untrained reference merges as arms,
# scr = training-free merge of the from-scratch adapters (ours_lib, 1/k), convm = converted merge (EXP-144 tag)
REF = ("scr", "convm", "sca")
CPREV = {"erlc": "erl", "seqc": "seq", "consfac": "consfa", "ewcc": "ewc", "consbc": "consb", "consfzd1c": "consfzd1", "erld1c": "erld1", "consfzl": "consfz6", "erll": "erl6"}            # 11-18 continual arm -> its arrival-1 arm (order 0)
PREVSEQ = {19: 0, 20: 19, C.SCALE: 20}                                 # EXP-151 arrivals: 8 (order 0) / 14 / 18 / 22 new
PREVSEQ.update({c: o if n == 14 else c - 1 for c, (o, n) in C.CONTO.items()})   # 11-21 EXP-167: streams on orders 1-3 (21-29)
PREVSEQ.update({30: 5, 31: 30, 32: 31})                               # 11-22 EXP-170: the 32-skill library stream
PREVSEQ.update({33: 32, 34: C.SCALE, 36: 35, 37: 36, 38: 37})           # 11-22 EXP-171: 5th arrivals (30 new); order 1's 32-skill stream
# 11-22 EXP-171 seed-1 replicates: <arm>s1 = <arm> at training seed 1 (same data / steps / context); continual rounds resume
# their own seed-1 chain (consfzcs1 / erlcs1 / consbcs1 from consfzs1 / erls1 / consbs1 at order 0)
S1 = ("consfz", "erl", "consb", "consfzd1", "erld1", "consfzc", "erlc", "consbc")
for b_ in S1:
    for D_ in (FRAC, VSTEPS, VDEMOS, VRANK):
        if b_ in D_: D_[b_ + "s1"] = D_[b_]
    ZV[b_ + "s1"] = ("ctx_all_zfree.json", "scratch", True) if b_ == "consfzc" else ZV[b_]   # consfzcs1 resumes via CPREV
    if b_ in CPREV: CPREV[b_ + "s1"] = CPREV[b_] + "s1"
CPREV["consfzcs1"] = "consfzs1"; S1V = {b_ + "s1" for b_ in S1}
# 11-23 EXP-172a limited replay memory: consfzcm0 / erlcm0 = consfzc / erlc storing NO rollouts of earlier arrivals (each round
# trains on the just-arrived batch's rollouts + library replay only, tasks_arr.txt); ours keeps every arrival in the merge, ER
# keeps nothing of them but its resumed correction. Streams 19 / 20 / 4 and 21 / 22 / 23 from consfz / erl (arrival 1)
M0 = ("consfzcm0", "erlcm0")
FRAC.update({v: 0.5 for v in M0}); VSTEPS.update({v: 1500 for v in M0}); CPREV.update({"consfzcm0": "consfz", "erlcm0": "erl"})
ZV.update({"consfzcm0": ("ctx_all_zfree.json", "scratch", True), "erlcm0": ("ctx_lib_free.json", "lib", True)})
# 11-24 EXP-173 all-baselines table: lrcd1 = lrc (learned merge weights, no correction) at 1 library demo per skill
FRAC["lrcd1"] = 0.5; VDEMOS["lrcd1"] = 1; ZV["lrcd1"] = ZV["lrc"]
# 11-25 EXP-174 LoRA continual-learning baselines (user-approved): olora = O-LoRA (Wang et al. 2023), inflora = InfLoRA (Liang &
# Li, CVPR 2024), as published (rehearsal-free) and from the library merge like seq / ewc: trained on the arriving skills' rollouts
# only, the library adapters = the previous tasks' LoRAs; O-LoRA's orthogonality to their A rows (official L1, lambda 0.5);
# InfLoRA's old-input subspace from the 5-demo library replay (official threshold 0.95), as ewc's Fisher -> one run per order
# serves both demo budgets. Same steps / lr / rank as every arm (official lr 1e-3 / 5e-4 belong to their backbones)
CL = {"olora": ["--olora_lambda", "0.5"], "inflora": ["--inflora", "0.95"]}
FRAC.update({v: 0.0 for v in CL}); ZV.update({v: ("ctx_lib_free.json", "lib", True) for v in CL})
# 11-23 EXP-172b / c eval-only modes of run() (ZV arms): E147_OOD = d -> objects shifted d metres (EXP-81's ood_shift, plan-wide);
# E147_UNS = 1 -> zero-shot on unseen tasks: the 8 externals no order draws + 16 seeded LIBERO-90 tasks outside the library.
# Both write under their own tag (suffix before _w, so ID globs never match them)
OOD = float(os.environ.get("E147_OOD", "0")); UNS = os.environ.get("E147_UNS") == "1"
# 11-24 EXP-171 drift check: E147_RV = 1 -> re-evaluate an existing adapter under the tag suffix "rv" (same eval, new records)
SFX = f"ood{OOD:g}" if OOD else "uns" if UNS else "rv" if os.environ.get("E147_RV") == "1" else ""
def unseen():
    seen = set(OLD + NEW); fl = [i for i in range(X.NF) if i not in seen]
    return [i for i in range(X.NF, X.NF + X.NE) if i not in seen] + sorted(int(i) for i in np.random.default_rng(11720 + SEQ).choice(fl, 16, replace=False))
def adir(s): return REPO / ("results/analysis/new_loras_cons" if s == 0 else f"results/analysis/new_loras_cons_s{s}")
ADIR = REPO / ("results/analysis/new_loras_cons" if SEQ == 0 else f"results/analysis/new_loras_cons_s{SEQ}")   # 11-05: orders 1-3 (E145_SEQ)
CX = {v: {} for v in VARS}                                            # arm -> original external index -> blended adapter LIB index
ZIDX = {}                                                             # consz arm -> LIB index of its correction (an extra member at w)
for v in VARS:
    if v in REF: continue
    if v in PARTS or v in RSUB:                                       # the consfz correction, registered under its own key
        path = ADIR / "train_consfz" / "adapter_model.safetensors"; U.EXTPATH[f"np16{v}:s{SEQ}"] = str(path)
        ZIDX[v] = len(E.LIB); E.LIB.append(dict(E.LIB[NEW[0]], key=f"np16{v}:s{SEQ}", tid="z" + E.LIB[NEW[0]]["tid"][1:], path=str(path))); continue
    if v in ZV:
        path = ADIR / f"train_{v}" / "adapter_model.safetensors"; U.EXTPATH[f"np16{v}:s{SEQ}"] = str(path)
        ZIDX[v] = len(E.LIB); E.LIB.append(dict(E.LIB[NEW[0]], key=f"np16{v}:s{SEQ}", tid="z" + E.LIB[NEW[0]]["tid"][1:], path=str(path))); continue
    for i in NEW:
        e = E.LIB[i]; name = str(e["key"]).split(":", 1)[1]; path = ADIR / v / name / "adapter_model.safetensors"
        U.EXTPATH[f"np16{v}:" + name] = str(path)
        CX[v][i] = len(E.LIB); E.LIB.append(dict(e, key=f"np16{v}:" + name, tid={"cons": "k", "consl": "m", "conso": "o", "cons1": "j", "consr": "y", "consq": "g", "consh": "h"}[v] + e["tid"][1:], path=str(path)))
def tag(v): return f"c147s{SEQ}t{T}-{v}{SFX}_w{WN:g}"
def tdir(v): return ADIR / f"train_{v}"

def prep():
    """context = all 24 members of the converted t=8 merge; tasks file = the externals' rollout globs; replay = EXP-145's"""
    ADIR.mkdir(parents=True, exist_ok=True)
    mem = [(E.LIB[j]["path"], 1.0 / (len(OLD) + len(NEW))) for j in OLD] + [(E.LIB[S.CV[i]]["path"], WN) for i in NEW]
    if all(pathlib.Path(p).exists() for p, _ in mem):                # 11-22: not for EXP-171's 30 new (8 never converted)
        json.dump({"members": mem, "w": WN, "init_adapter": E.LIB[S.CV[NEW[0]]]["path"], "init_zero": True},
                  open(ADIR / "ctx_all.json", "w"), indent=1)
    else: assert SEQ in C.FULL, "converted adapters missing"; print("[e147] prep: no converted context (EXP-171 30 new)", flush=True)
    memz = [(E.LIB[j]["path"], 1.0 / (len(OLD) + len(NEW))) for j in OLD + NEW]       # consz: the scratch "ours" merge (1/k)
    assert all(pathlib.Path(p).exists() for p, _ in memz)
    json.dump({"members": memz, "w": WN, "init_adapter": E.LIB[S.CV[NEW[0]]]["path"], "init_zero": True},
              open(ADIR / "ctx_all_z.json", "w"), indent=1)
    for src, dst in (("ctx_all.json", "ctx_all_free.json"), ("ctx_all_z.json", "ctx_all_zfree.json")):
        if not (ADIR / src).exists(): continue
        d = json.load(open(ADIR / src)); d["free"] = True; json.dump(d, open(ADIR / dst, "w"), indent=1)
    import glob
    for i in NEW: assert glob.glob(S.pattern(i)), S.pattern(i)
    (ADIR / "tasks_ext.txt").write_text("".join(f"{E.LIB[i]['suite']}_no_noops {E.LIB[i]['scene']} {S.pattern(i)}\n" for i in NEW))
    rp = X5.ADIR / f"replay_s{SEQ}.txt"
    if SEQ >= C.SCALE: (ADIR / "replay.txt").write_text("".join(f"libero_90_no_noops {E.LIB[j]['scene']}\n" for j in OLD))   # no EXP-145 prep here
    else: assert rp.exists(); shutil.copy2(rp, ADIR / "replay.txt")
    print(f"[e147] prep: context {len(mem)} members, {len(NEW)} rollout tasks, replay {len(OLD)} library tasks", flush=True)

def prep2():
    """11-18 EXP-158..160 contexts next to prep()'s: ctx_lib_free (the library-only merge, ours at 1/16), ctx_all_avgfree (the
    vanilla average of all k members) and tasks_arr.txt (this arrival's own new skills: NEW minus the previous arrival's)"""
    d = json.load(open(ADIR / "ctx_all_zfree.json"))
    json.dump(dict(d, members=[(E.LIB[j]["path"], 1.0 / len(OLD)) for j in OLD]), open(ADIR / "ctx_lib_free.json", "w"), indent=1)
    json.dump(dict(d, op="avg"), open(ADIR / "ctx_all_avgfree.json", "w"), indent=1)
    json.dump(dict(d, lrc=True), open(ADIR / "ctx_all_lrc.json", "w"), indent=1)
    json.dump(dict(d, lrc=True, lrc_only=True), open(ADIR / "ctx_all_lrco.json", "w"), indent=1)
    prev = set(C.sequence(PREVSEQ[SEQ])[1]) if SEQ in PREVSEQ else set()
    (ADIR / "tasks_arr.txt").write_text("".join(f"{E.LIB[i]['suite']}_no_noops {E.LIB[i]['scene']} {S.pattern(i)}\n" for i in NEW if i not in prev))
    print(f"[e147] prep2: library context {len(OLD)} members, avg context {len(d['members'])}, {len(NEW) - len(prev)} arrival tasks", flush=True)

def libpat(i): return str(S.DUMPS / f"e147_{E.LIB[i]['tid']}_self" / "*" / "ep*_success.npz")

def collect_lib():
    """11-06 consr: the LIBRARY skills' own rollouts (single-task LoRA, states 10..49, as EXP-144) -> replay without any demos"""
    J = [{"name": f"e147_{E.LIB[i]['tid']}_self", "method": "average", "params": {}, "adapters": [], "adapter_paths": [E.LIB[i]["path"]],
          "eval_scenes": [E.LIB[i]["scene"]], "suite": E.LIB[i]["suite"], "norm_stats": U.FLEET_NORM}
         for i in OLD if E.rec(f"e147_{E.LIB[i]['tid']}_self", E.LIB[i]["scene"]) is None]
    for j in J:                                                      # a failed earlier attempt's partial episodes -> set aside
        d = S.DUMPS / j["name"]
        if d.exists(): d.rename(d.with_name(f"{d.name}_partial_{time.strftime('%m%d%H%M')}"))
    print(f"[e147] collect_lib: {len(J)} library skills x {S.NCOL} episodes (states {S.OFF}..{S.OFF + S.POOL - 1})", flush=True)
    if J: S.orch(J, "e147col", n=S.NCOL, offset=S.OFF, state_pool=S.POOL, dump_obs_dir=str(S.DUMPS), dump_stride=2)
    import glob
    ok = [i for i in OLD if glob.glob(libpat(i))]
    (ADIR / "replay_roll.txt").write_text("".join(f"{E.LIB[i]['suite']}_no_noops {E.LIB[i]['scene']} {libpat(i)}\n" for i in ok))
    print(f"[e147] collect_lib: replay_roll.txt with {len(ok)}/{len(OLD)} library skills", flush=True)

def train():
    """E147_VARS picks the arm(s); c0 (0 steps) once, for the blend"""
    for v, steps in [("c0", 0)] + [(v, VSTEPS.get(v, STEPS)) for v in VARS if v not in REF + ("consfz2",)]:
        out = tdir(v)
        if (out / "adapter_model.safetensors").exists(): continue
        if v == "c0" and not (ADIR / "ctx_all.json").exists(): continue   # 11-22 EXP-171 30 new: no converted context, c0 unused
        if v == "consfzc":                                            # resume the previous arrival's correction
            ps = PREVSEQ[SEQ]; prev = adir(ps) / f"train_{'consfz' if ps < C.NSEQ else 'consfzc'}" / "adapter_model.safetensors"
            assert prev.exists(), prev
            ctx = json.load(open(ADIR / "ctx_all_zfree.json")); ctx["resume"] = str(prev); json.dump(ctx, open(ADIR / "ctx_all_zfree_c.json", "w"), indent=1)
        ctxf = ZV[v][0] if v in ZV else "ctx_all.json"
        if ctxf == "ctx_base_free.json": json.dump({"members": [], "free": True}, open(ADIR / ctxf, "w"))   # EXP-164 consbc: nothing folded
        if v in CPREV:                                                # 11-18 EXP-159 / 160 continual rounds: resume the previous one
            ps = PREVSEQ[SEQ]; prev = adir(ps) / f"train_{CPREV[v] if ps < C.NSEQ or ps in C.LIBSCALE or ps in C.LIBSCALE1 else v}" / "adapter_model.safetensors"
            if not prev.exists(): print(f"[e147] {v}: previous round {prev} not trained", flush=True); continue
            ctx = json.load(open(ADIR / ctxf)); ctx["resume"] = str(prev); ctxf = f"ctx_{v}_c.json"; json.dump(ctx, open(ADIR / ctxf, "w"), indent=1)
        try: os.close(os.open(ADIR / f".claim_{v}", os.O_CREAT | os.O_EXCL))
        except FileExistsError: continue
        if v in RN:                                                   # EXP-169: the first RN[v] successful episodes per new skill
            import glob; ln = []
            for su, tk, pat in (l.split() for l in (ADIR / "tasks_ext.txt").read_text().splitlines() if l.strip()):
                d = ADIR / f"roll{RN[v]}" / tk; d.mkdir(parents=True, exist_ok=True)
                for f in sorted(glob.glob(pat))[:RN[v]]:
                    try: (d / pathlib.Path(f).name).symlink_to(f)
                    except FileExistsError: pass
                assert len(glob.glob(str(d / "ep*_success.npz"))) == RN[v], d
                ln.append(f"{su} {tk} {d}/ep*_success.npz\n")
            (ADIR / f"tasks_extn{RN[v]}.txt").write_text("".join(ln))
        if v in R10: (ADIR / "tasks_ext10.txt").write_text((ADIR / "tasks_ext.txt").read_text().replace("/ep*_success.npz", "/ep00[0-9]_*_success.npz"))
        cmd = [str(REPO / ".venv/bin/python"), str(REPO / "sweep/train_lora.py"), "--suite", "libero_spatial_no_noops",
               "--task", "consolidation", "--steps", str(steps), "--batch", "8", "--device", "cuda:0"] + \
              ([] if v in ZV and ZV[v][0] is None else ["--init", "ctx", "--context", str(ADIR / ctxf)]) + [
               "--tasks_file", str(ADIR / ("tasks_arr.txt" if v in ("seq", "seqc", "ewc", "ewcc") + M0 + tuple(CL) else "tasks_ext10.txt" if v in R10 else f"tasks_extn{RN[v]}.txt" if v in RN else "tasks_ext.txt")),
               "--replay_tasks", str(ADIR / ("replay_roll.txt" if v in ("consr", "consfzr", "consfzr1", "erlr", "consbr") else "replay.txt")), "--replay_frac", str(FRAC.get(v, 0.5)), "--replay_demos", str(VDEMOS.get(v, 5)), "--out", str(out)] + (["--seed", "1"] if v in ("cons1", "consfzr1", "consfz6s1", "consfz6r64s1") or v in S1V else []) + (["--rank", str(VRANK[v])] if v in VRANK else []) + \
              ((["--ewc_lambda", str(VEWC[v])] + (["--ewc_prev", str(prev.parent / "ewc.pt")] if v in CPREV else [])) if v in VEWC else []) + CL.get(v, [])
        t1 = time.time(); r = subprocess.run(cmd, cwd=REPO, stdout=open(ADIR / f"train_{v}.log", "w"), stderr=subprocess.STDOUT)
        print(f"[e147] trained {v} ({steps} steps) -> exit {r.returncode} in {(time.time()-t1)/60:.0f} min", flush=True)

def interp():
    import safetensors.torch as st
    C0 = st.load_file(str(tdir("c0") / "adapter_model.safetensors"))
    for v in VARS:
        if v in ZV or v in REF: continue                              # consz enters the merge as one extra member (run)
        src = tdir(v) / "adapter_model.safetensors"
        if not src.exists(): print(f"[e147] interp {v}: not trained", flush=True); continue
        D = st.load_file(str(src)); assert set(D) == set(C0)
        for i in NEW:
            dst = pathlib.Path(E.LIB[CX[v][i]]["path"])
            if dst.exists(): continue
            a = pathlib.Path(E.LIB[S.CV[i]]["path"]); A = st.load_file(str(a)); assert set(A) == set(D)
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(a.parent / "adapter_config.json", dst.parent / "adapter_config.json")
            st.save_file({k: (A[k].float() + (D[k].float() - C0[k].float()) / len(NEW)).to(A[k].dtype) for k in A}, str(dst))
        print(f"[e147] interp {v}: {len(NEW)} adapters (conv + 1/{len(NEW)} x (cons - c0))", flush=True)

class ZBuilder(RT.RBuilder):
    """privx = priv with an explicit k (k = 24, as in training): members in P at w, the others at 1/k; members in raw are added
    as their full delta B A at scale 1 (plain LoRA, as trained), not as a residual"""
    def _delta(self, m, p, members, s_):
        if m != "privx": return super()._delta(m, p, members, s_)
        F = {i: self.fac(i) for i in members}; raw = set(p.get("raw", [])); mm = [i for i in members if i not in raw]
        if not any(s_ in F[i] and float(F[i][s_][0].abs().max()) > 0 for i in members): return None
        if p.get("alpha") and getattr(self, "_alp", None) != p["alpha"]:   # 11-18 EXP-162 learned residual weights
            import torch; d_ = torch.load(p["alpha"]); self._alp, self._al = p["alpha"], d_["alpha"]
            assert d_["members"] == [E.LIB[i]["path"] for i in mm], "lrc.pt member order != the merge's"
        if p.get("alpha") and s_ in self._al:
            al = self._al[s_]; D = RT.MM.ours([float(al[j]) * F[i][s_][0].to(E.DEV) for j, i in enumerate(mm)], [F[i][s_][1].to(E.DEV) for i in mm],
                                              1.0, U.USIG[s_].to(E.DEV), U.VS[s_].to(E.DEV))
        elif p.get("avg"): D = sum(F[i][s_][0].to(E.DEV) @ F[i][s_][1].to(E.DEV) / p["k"] for i in mm if s_ in F[i])   # 11-18 vanilla average
        else: D = RT.MM.ours([(p["w"] if i in p["P"] else 1.0 / p["k"]) * F[i][s_][0].to(E.DEV) for i in mm], [F[i][s_][1].to(E.DEV) for i in mm],
                       1.0, U.USIG[s_].to(E.DEV), U.VS[s_].to(E.DEV))
        for i in raw:
            if s_ in F[i] and ("rawpart" not in p or part(s_) in p["rawpart"]) and ("rawstems" not in p or s_ in p["rawstems"]):
                D = D + F[i][s_][0].to(E.DEV) @ F[i][s_][1].to(E.DEV)
        return D

def run():
    assert not SFX or all(v in ZV for v in VARS), "E147_OOD / E147_UNS: ZV arms only"
    for v in VARS:
        if v in REF:
            newm = NEW if v in ("scr", "sca") else [S.CV[i] for i in NEW]; mem = OLD + newm; ev = sorted(mem); TAG = tag(v) if v in ("scr", "sca") else S.tag(SEQ, WN)
            js = [(nm, su, scs) for nm, su, scs, ch in E.member_jobs(TAG, len(mem), "test", SEQ, ev) if any(E.rec(nm, sc) is None for sc in scs)]
            print(f"[e147] run {TAG}: {len(js)} jobs", flush=True)
            if not js: continue
            out = E.PRE / f"e147_{TAG}"; t1 = time.time()
            if v == "scr": ZBuilder().build("ours_lib", {}, None, mem, out)
            elif v == "sca": ZBuilder().build("privx", {"w": WN, "P": [], "raw": [], "k": len(mem), "avg": True}, None, mem, out)
            else: RT.RBuilder().build("priv", {"w": WN, "P": newm}, None, mem, out)
            print(f"[e147] built {TAG} in {time.time()-t1:.0f}s", flush=True)
            S.orch([{"name": nm, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                     "eval_scenes": scs, "suite": su, "norm_stats": U.FLEET_NORM} for nm, su, scs in js], "e147run")
            shutil.rmtree(out, ignore_errors=True); continue
        if v in PARTS or v in RSUB:                                   # 11-18 EXP-161: the correction in / out of one VLA part
            z = ZIDX[v]; ev = sorted(OLD + NEW); TAG = tag(v)
            js = [(nm, su, scs) for nm, su, scs, ch in E.member_jobs(TAG, len(ev), "test", SEQ, ev) if any(E.rec(nm, sc) is None for sc in scs)]
            print(f"[e147] run {TAG}: {len(js)} jobs", flush=True)
            if not js: continue
            out = E.PRE / f"e147_{TAG}"; t1 = time.time()
            zb = ZBuilder()
            if v in PARTS: sel = {"rawpart": sorted(PARTS[v])}
            else:
                f = zb.fac(z); keep = set(rsub(v, f))
                for ms in ("param", "energy"):
                    n = {s: float(f[s][0].numel() + f[s][1].numel()) if ms == "param" else float(((f[s][0].float().T @ f[s][0].float()) * (f[s][1].float() @ f[s][1].float().T)).sum()) for s in f}
                    sh = {g: sum(n[s] for s in keep if part(s) == g) / sum(n.values()) for g in ("vision", "language", "expert")}
                    print(f"[e147] {v}: {len(keep)} / {len(f)} modules, {ms} share " + " ".join(f"{g} {100*x:.1f}%" for g, x in sh.items()), flush=True)
                sel = {"rawstems": keep}
            zb.build("privx", {"w": WN, "P": [], "raw": [z], "k": len(ev), **sel}, None, OLD + NEW + [z], out)
            print(f"[e147] built {TAG} in {time.time()-t1:.0f}s", flush=True)
            S.orch([{"name": nm, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                     "eval_scenes": scs, "suite": su, "norm_stats": U.FLEET_NORM} for nm, su, scs in js], "e147run")
            shutil.rmtree(out, ignore_errors=True); continue
        if v in ZV:
            z = ZIDX[v]; ev = unseen() if UNS else sorted(OLD + NEW); TAG = tag(v)
            if not pathlib.Path(E.LIB[z]["path"]).exists(): print(f"[e147] run {TAG}: not trained", flush=True); continue
            js = [(nm, su, scs) for nm, su, scs, ch in E.member_jobs(TAG, len(ev), "test", SEQ, ev) if any(E.rec(nm, sc) is None for sc in scs)]
            print(f"[e147] run {TAG}: {len(js)} jobs", flush=True)
            if not js: continue
            out = E.PRE / f"e147_{TAG}"; t1 = time.time()
            _, base, israw = ZV[v]; newm = NEW if base == "scratch" else [S.CV[i] for i in NEW]
            if base == "base":
                import exp120_privileged as P; P.PBuilder().build("ta", {"lam": 1.0}, None, [z], out)   # base + the LoRA alone (as EXP-136)
            elif base == "lib": ZBuilder().build("privx", {"w": WN, "P": [], "raw": [z], "k": len(OLD)}, None, OLD + [z], out)   # library merge + LoRA
            elif base == "lrc": ZBuilder().build("privx", {"w": WN, "P": [], "raw": [z] if israw else [], "k": len(OLD) + len(NEW), "alpha": str(tdir(v) / "lrc.pt")},
                                                 None, OLD + NEW + ([z] if israw else []), out)
            elif base == "avg": ZBuilder().build("privx", {"w": WN, "P": [], "raw": [z], "k": len(OLD) + len(NEW), "avg": True}, None, OLD + NEW + [z], out)
            else: ZBuilder().build("privx", {"w": WN, "P": ([] if base == "scratch" else newm) + ([] if israw else [z]), "raw": [z] if israw else [],
                                       "k": len(OLD) + len(NEW)}, None, OLD + newm + [z], out)
            print(f"[e147] built {TAG} in {time.time()-t1:.0f}s", flush=True)
            S.orch([{"name": nm, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                     "eval_scenes": scs, "suite": su, "norm_stats": U.FLEET_NORM} for nm, su, scs in js], "e147run", **({"ood_shift": OOD} if OOD else {}))
            shutil.rmtree(out, ignore_errors=True); continue
        newc = [CX[v][i] for i in NEW]; mem = OLD + newc; ev = sorted(mem); TAG = tag(v)
        if not all(pathlib.Path(E.LIB[c]["path"]).exists() for c in newc): print(f"[e147] run {TAG}: adapters missing", flush=True); continue
        js = [(nm, su, scs) for nm, su, scs, ch in E.member_jobs(TAG, len(mem), "test", SEQ, ev) if any(E.rec(nm, sc) is None for sc in scs)]
        print(f"[e147] run {TAG}: {len(js)} jobs", flush=True)
        if not js: continue
        out = E.PRE / f"e147_{TAG}"; t1 = time.time()
        RT.RBuilder().build("priv", {"w": WN, "P": newc}, None, mem, out)
        print(f"[e147] built {TAG} in {time.time()-t1:.0f}s", flush=True)
        S.orch([{"name": nm, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                 "eval_scenes": scs, "suite": su, "norm_stats": U.FLEET_NORM} for nm, su, scs in js], "e147run")
        shutil.rmtree(out, ignore_errors=True)

def report():
    from scipy import stats
    sg = X.single(); L = [f"=== EXP-147 order {SEQ}, t={T}: one correction trained inside the converted merge (w {WN:g}) ==="]
    ref = {"scratch ours (1/k)": (tag("scr") if SEQ >= C.SCALE else f"c134zs{SEQ}t{T}-ours", NEW),
           **({} if SEQ >= C.SCALE else {"WP on demos (0.25)": (f"c134ws{SEQ}t{T}-ours_w0.25", X5.W.sequence(SEQ)[1][:T])}),
           "converted (0.25)": (S.tag(SEQ, WN), [S.CV[i] for i in NEW]),
           "FedAvg ctxr 1/8": (X5.tag("ctxr_l0.125"), [X5.CX["ctxr_l0.125"][i] for i in NEW]),
           "FedAvg mix 1/8+1/8": (X5.tag("mix_q0.125_r0.125"), [X5.CX["mix_q0.125_r0.125"][i] for i in NEW]),
           **{v: (tag(v), NEW if v in ZV else [CX[v][i] for i in NEW]) for v in VARS}}
    res = {}
    for nm, (tg, new) in ref.items():
        sc = E.member_success(tg, len(OLD) + len(new), "test", SEQ, sorted(OLD + new))
        if not sc: continue
        res[nm] = ({i: sc[i] / sg[i] for i in OLD if i in sc and sg.get(i)}, {o: sc[c] / sg[o] for o, c in zip(NEW, new) if c in sc and sg.get(o)})
    L.append("skill      own(scratch)  |  retention vs own: " + " / ".join(res))
    for i in NEW:
        L.append(f"{E.LIB[i]['tid']:9s}  {100*sg.get(i, 0):5.0f}%  |  " + " / ".join("  —  " if i not in res[k][1] else f"{100*res[k][1][i]:5.0f}%" for k in res))
    for k in res:
        lib, nw = res[k]; s = ""
        for bn in ("converted (0.25)", "FedAvg ctxr 1/8", "FedAvg mix 1/8+1/8"):
            base = res.get(bn)
            if not base or k not in VARS: continue
            for g, a, b in (("ext", nw, base[1]), ("lib", lib, base[0])):
                ks = sorted(set(a) & set(b)); d = [a[q] - b[q] for q in ks]
                p = stats.wilcoxon(d).pvalue if any(abs(x) > 1e-9 for x in d) else 1.0
                s += f"  {g} vs {bn.split()[0]} {100*np.mean(d):+.1f}pp ({sum(x > 1e-9 for x in d)}/{sum(x < -1e-9 for x in d)}, p={p:.3f})"
        L.append(f"{k:22s} library {100*np.mean(list(lib.values())) if lib else float('nan'):5.1f}% (n={len(lib)})  externals "
                 f"{100*np.mean(list(nw.values())) if nw else float('nan'):5.1f}% (n={len(nw)})" + s)
    txt = "\n".join(L); print(txt); (REPO / f"results/analysis/exp147_report_s{SEQ}.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st_ = os.environ.get("E147_STAGE", "report"); U._clear_orphaned_builds()
    {"prep": prep, "prep2": prep2, "collect_lib": collect_lib, "train": train, "interp": interp, "run": run, "report": report}[st_]()
