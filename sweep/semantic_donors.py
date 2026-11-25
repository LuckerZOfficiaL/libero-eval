#!/usr/bin/env python3
"""EXP-102 — do TASK SEMANTICS (not weights) pick good donors? No rollouts: features from the LIBERO-90 task
names only, scored against the EXP-95 ground truth (within-receiver Spearman over 73 donors) and as a top-3 pick.

Task name = SCENE + instruction, e.g. KITCHEN_SCENE3_put_the_moka_pot_on_the_stove. Per (target t, donor d):
  same_scene, same_room (KITCHEN / LIVING_ROOM / STUDY), word_jaccard (content words), verb_jaccard,
  noun_jaccard, shares_target_object (d mentions t's object), shares_target_receptacle, donor_has_target_nouns
  (share of t's nouns that d mentions), both_two_step (both instructions have two steps), complement
  (d shares t's OBJECT but places it somewhere else -- same skill on the same thing, different goal).
"""
import pathlib, re, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import gt_analysis as A, rank_sweep as R
from reinforce_train import scenes as SC
from subset_probe import TARGETS as WEAK
STOP = {"the", "a", "it", "of", "and", "to", "in", "on", "at", "into", "onto", "its", "top", "front", "back", "left", "right", "middle"}
VERBS = {"put", "pick", "up", "place", "open", "close", "turn", "push", "stack", "slide", "lift"}

def parse(t):
    nm = SC[t][0]; m = re.match(r"([A-Z_]+?)_SCENE(\d+)_(.*)", nm)
    room, scene, ins = m.group(1), f"{m.group(1)}_{m.group(2)}", m.group(3).split("_")
    words = [w for w in ins if w not in STOP]
    verbs = {w for w in words if w in VERBS}; nouns = {w for w in words if w not in VERBS}
    # object = nouns before the first placement preposition, receptacle = nouns after it
    txt = " ".join(ins); parts = re.split(r" (?:on|in|into|onto) ", txt, maxsplit=1)
    obj = {w for w in parts[0].split() if w not in STOP and w not in VERBS}
    rec = {w for w in (parts[1].split() if len(parts) > 1 else []) if w not in STOP and w not in VERBS}
    return dict(room=room, scene=scene, words=set(words), verbs=verbs, nouns=nouns, obj=obj, rec=rec, two=("and" in ins))

def jac(a, b): return len(a & b) / max(1, len(a | b))

def main():
    P = {t: parse(t) for t in A.UNIQUE}
    F = {k: np.zeros((76, 76)) for k in ("same_scene", "same_room", "word_jaccard", "verb_jaccard", "noun_jaccard",
                                        "shares_target_object", "shares_target_receptacle", "donor_has_target_nouns", "both_two_step", "complement")}
    for t in A.UNIQUE:
        for d in A.UNIQUE:
            a, b = P[t], P[d]
            F["same_scene"][t, d] = a["scene"] == b["scene"]; F["same_room"][t, d] = a["room"] == b["room"]
            F["word_jaccard"][t, d] = jac(a["words"], b["words"]); F["verb_jaccard"][t, d] = jac(a["verbs"], b["verbs"])
            F["noun_jaccard"][t, d] = jac(a["nouns"], b["nouns"])
            F["shares_target_object"][t, d] = len(a["obj"] & b["nouns"]) > 0
            F["shares_target_receptacle"][t, d] = len(a["rec"] & b["nouns"]) > 0
            F["donor_has_target_nouns"][t, d] = len(a["nouns"] & b["nouns"]) / max(1, len(a["nouns"]))
            F["both_two_step"][t, d] = a["two"] and b["two"]
            F["complement"][t, d] = (len(a["obj"] & b["obj"]) > 0) and (len(a["rec"] & b["rec"]) == 0)
    GT = {t: {d: A.rec(t, d)["success_rate"] for d in A.UNIQUE if d != t} for t in A.UNIQUE}
    L = ["=== EXP-102: task-semantic donor features vs EXP-95 ground truth (within-receiver Spearman; ties allowed) ===",
         "ceilings: weak 0.795, all 0.519. reference: best weight score 0.11; random donor -3.8pp, best donor +16.5pp (weak)"]
    print("example parse:", SC[6][0], {k: v for k, v in P[6].items() if k in ("obj", "rec", "verbs")})
    for nm, M in F.items():
        res = {}
        for grp, T in (("weak", [t for t in A.UNIQUE if t in WEAK]), ("all", A.UNIQUE)):
            rr = []
            for t in T:
                ds = sorted(GT[t]); y = np.array([GT[t][d] for d in ds]); x = M[t, ds]
                if y.std() > 0 and x.std() > 0: rr.append(stats.spearmanr(x, y).statistic)
            res[grp] = np.array(rr)
        pk = []
        for t in [t for t in A.UNIQUE if t in WEAK]:
            ds = sorted(GT[t]); y = np.array([GT[t][d] for d in ds]); x = M[t, ds] + 1e-9 * np.random.default_rng(t).random(len(ds))
            b, _ = R.base(t); pk.append(100 * (y[np.argsort(-x)[:3]].mean() - b))
        w, a = res["weak"], res["all"]
        frac = M[[t for t in A.UNIQUE], :][:, A.UNIQUE].mean()
        L.append(f"  {nm:<26} weak rho {w.mean():+.3f} (p={stats.ttest_1samp(w,0).pvalue:.2f}, n={len(w)})   all rho {a.mean():+.3f} (p={stats.ttest_1samp(a,0).pvalue:.3f}, n={len(a)})"
                 f"   top-3 pick (weak) {np.mean(pk):+.1f}pp   [mean value {frac:.2f}]")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp102_semantic_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
