#!/usr/bin/env python3
"""Phase A of the shared/residual decomposition study (LIBERO-90, CPU only).

For each held-out task t: build the leave-one-out soup S_t = mean of the other 75
adapters' deltas, per module (458 modules, factorized — never materialize out x in).
Per-module SVD of S_t via the QR trick; then measure how much of a probe adapter's
energy the soup's top-k singular directions capture (two-sided projection):

    capture_k(T) = ||U_k^T (s B_T A_T) V_k||_F^2 / ||s B_T A_T||_F^2

aggregated over modules (energy-weighted). Probes per held-out task:
  - the HELD-OUT adapter itself (the globality question),
  - 5 HELD-IN adapters (reference: what "global" looks like from inside the soup),
  - the soup's own spectrum concentration (context),
  - chance floor for two-sided projection: k^2/(out*in), energy-weighted.

Saves top-64 soup factors for t1/t6 (Phase-B checkpoint building) and a JSON of curves.
"""
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import safetensors.torch as st
import apply_lora as al

torch.set_num_threads(12)
REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "analysis"
KS = [1, 2, 4, 8, 16, 32, 64]
HELD_OUT = [1, 2, 6, 19]          # anchor-ish, anchor, fragile, lifted
HELD_IN_PROBES = [3, 9, 30, 50, 70]
SAVE_FACTORS_FOR = {1, 6}         # Phase B targets

plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}


def load_AB(t):
    out = {}
    for stem, g in al.extract_lora_groups(st.load_file(afile[t])).items():
        if g.a is None or g.b is None:
            continue
        r = al.infer_rank(g.a)
        s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.to(torch.float32), g.b.to(torch.float32)
        if b.shape[1] != a.shape[0]:
            a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = (s, a, b)          # delta = s * b @ a ; b [out,r], a [r,in]
    return out


def main():
    print("[load] 76 adapters (factors only)")
    AB = {t: load_AB(t) for t in range(76)}
    stems = sorted(set.intersection(*[set(AB[t]) for t in AB]))
    results = {}

    for t_out in HELD_OUT:
        others = [t for t in range(76) if t != t_out]
        probes = [t_out] + [p for p in HELD_IN_PROBES if p != t_out]
        # per-probe accumulators: captured energy at each k, and total energy
        cap = {p: np.zeros(len(KS)) for p in probes}
        tot = {p: 0.0 for p in probes}
        soup_cap = np.zeros(len(KS))
        soup_tot = 0.0
        chance_num = np.zeros(len(KS))
        chance_den = 0.0
        factors_store = {}

        for mi, stem in enumerate(stems):
            # factorized soup: S = Bc @ Ac, Bc [out, 75r] (1/75 and s_i folded in), Ac [75r, in]
            Bs, As = [], []
            for i in others:
                s, a, b = AB[i][stem]
                Bs.append(b * (s / len(others)))
                As.append(a)
            Bc = torch.cat(Bs, dim=1)
            Ac = torch.cat(As, dim=0)
            Qb, Rb = torch.linalg.qr(Bc, mode="reduced")      # [out,m],[m,m]
            Qa, Ra = torch.linalg.qr(Ac.T, mode="reduced")    # [in,m],[m,m]
            M = Rb @ Ra.T
            Us, sv, Vst = torch.linalg.svd(M, full_matrices=False)
            U = Qb @ Us                                        # [out, m] left singvecs of S
            V = Qa @ Vst.T                                     # [in, m]
            sv2 = (sv ** 2).numpy()
            soup_tot += sv2.sum()
            for ki, k in enumerate(KS):
                soup_cap[ki] += sv2[:k].sum()
            outd, ind = U.shape[0], V.shape[0]
            for ki, k in enumerate(KS):
                chance_num[ki] += min(k, outd) * min(k, ind) / (outd * ind)
            chance_den += 1.0

            for p in probes:
                s, a, b = AB[p][stem]
                tp = s * s * float(((b.T @ b) * (a @ a.T).T).sum())    # ||delta_p||_F^2
                tot[p] += tp
                UB = U.T @ b                                   # [m,16]
                AV = a @ V                                     # [16,m]
                for ki, k in enumerate(KS):
                    # captured = || U_k^T (s b a) V_k ||^2 = s^2 * || UB_k @ AV_k ||^2
                    m_ = UB[:k] @ AV[:, :k]
                    cap[p][ki] += s * s * float((m_ ** 2).sum())

            if t_out in SAVE_FACTORS_FOR:
                factors_store[stem] = {
                    "U": (U[:, :64] * sv[:64]).to(torch.float16).numpy(),   # U_k Sigma_k
                    "V": V[:, :64].to(torch.float16).numpy(),
                }
            if (mi + 1) % 100 == 0:
                print(f"  [t{t_out}] {mi+1}/{len(stems)} modules")

        res = {"ks": KS,
               "held_out_capture": (cap[t_out] / tot[t_out]).tolist(),
               "held_in_capture": {str(p): (cap[p] / tot[p]).tolist() for p in probes if p != t_out},
               "soup_concentration": (soup_cap / soup_tot).tolist(),
               "chance": (chance_num / chance_den).tolist()}
        results[str(t_out)] = res
        hi = np.mean([res["held_in_capture"][k] for k in res["held_in_capture"]], axis=0)
        print(f"\n[t{t_out}] capture curves (fraction of adapter energy in soup top-k):")
        print("  k        : " + "  ".join(f"{k:>6}" for k in KS))
        print("  held-out : " + "  ".join(f"{v:6.3f}" for v in res["held_out_capture"]))
        print("  held-in  : " + "  ".join(f"{v:6.3f}" for v in hi))
        print("  soup self: " + "  ".join(f"{v:6.3f}" for v in res["soup_concentration"]))
        print("  chance   : " + "  ".join(f"{v:6.4f}" for v in res["chance"]))

        if t_out in SAVE_FACTORS_FOR:
            np.savez_compressed(OUT / f"loo_soup_factors_t{t_out}.npz",
                                stems=np.array(list(factors_store.keys())),
                                **{f"U_{i}": v["U"] for i, v in enumerate(factors_store.values())},
                                **{f"V_{i}": v["V"] for i, v in enumerate(factors_store.values())})
            print(f"  saved top-64 soup factors for Phase B -> loo_soup_factors_t{t_out}.npz")

    json.dump(results, open(OUT / "loo_decomposition.json", "w"), indent=1)
    print(f"\nsaved {OUT/'loo_decomposition.json'}")


if __name__ == "__main__":
    main()
