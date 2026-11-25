"""11-06 EXP-147: absolute success (not retention) for cons vs references + joint training (EXP-136), one order (E145_SEQ).
Retention in exp147's report divides externals by the FROM-SCRATCH skill's own success (30-100%); EXP-136 divides new skills
by the write-protected adapter's own success (~96%) -> compare raw success rates on the same 8 + 16 scenes instead."""
import json, os, sys, pathlib
os.environ.setdefault("E147_VARS", "cons")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp147_consolidate as C
E, S, X5, SEQ, OLD, NEW = C.E, C.S, C.X5, C.SEQ, C.OLD, C.NEW
WPN = X5.W.sequence(SEQ)[1][:C.T] if all(i in X5.W.WPI for i in NEW) else []   # 11-22 EXP-171: no WP adapters for the 8 never-drawn externals
sg = C.X.single()
ref = {"scratch ours": (C.tag("scr") if SEQ >= C.C.SCALE else f"c134zs{SEQ}t{C.T}-ours", NEW), "WP on demos": (f"c134ws{SEQ}t{C.T}-ours_w0.25", WPN),   # 11-06: scaling order
       "converted": (S.tag(SEQ, C.WN), [S.CV[i] for i in NEW]),
       **{v: (C.tag(v), NEW if v in C.ZV or v in C.PARTS or v in C.RSUB or v in ("scr", "sca") else [C.CX[v][i] for i in NEW]) for v in C.VARS}}   # E147_VARS arms (default cons); 11-18 part ablations + training-free refs
out = {"seq": SEQ, "single_scratch": {E.LIB[i]["scene"]: sg.get(i) for i in OLD + NEW}, "single_wp": {E.LIB[i]["scene"]: sg.get(i) for i in WPN}}
for nm, (tg, new) in ref.items():
    sc = E.member_success(tg, len(OLD) + len(new), "test", SEQ, sorted(OLD + new))
    out[nm] = {"lib": {E.LIB[i]["scene"]: sc.get(i) for i in OLD}, "new": {E.LIB[o]["scene"]: sc.get(c) for o, c in zip(NEW, new)}}
for jn in ("", "24k"):                                                  # 11-08 EXP-150 fairness: joint24k = 24,000 steps (s6)
    jt = {}
    for i in OLD + WPN:
        r = E.rec(f"e136_joint{jn}_s{SEQ}_{E.LIB[i]['tid']}", E.LIB[i]["scene"])
        jt[E.LIB[i]["scene"]] = None if r is None else r["success_rate"]
    if jn and all(v is None for v in jt.values()): continue
    out["joint" + jn] = {"lib": {E.LIB[i]["scene"]: jt[E.LIB[i]["scene"]] for i in OLD}, "new": {E.LIB[i]["scene"]: jt[E.LIB[i]["scene"]] for i in WPN}}
print(json.dumps(out))
