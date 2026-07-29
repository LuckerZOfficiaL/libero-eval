#!/usr/bin/env python3
"""Weight-space diagnostic for merge interference.

For each task LoRA: total delta norm ||scale*B@A||_F.
For each merge group: per-task norm (relative to group mean), cos(task, merged),
cos(task, sum-of-others). Uses the r x r Gram-trace identity so it never
materializes an out x in delta:  <B_iA_i, B_jA_j>_F = Tr((B_i^T B_j)(A_j A_i^T)).

Finding (2026-07-23): norms are uniform (~60) and cosines uniform (~0.93 self-merged,
~0.84 self-others) across fragile AND robust tasks -> weight geometry does NOT predict
which task collapses. Interference is functional, not weight-geometric.

Usage: .venv/bin/python sweep/weight_geometry.py [group ...]   e.g. 0:1:2 6:7:8
"""
import sys
import json
import math
import torch

sys.path.insert(0, "scripts")
import safetensors.torch as st
import apply_lora as al

torch.set_num_threads(8)
plan = json.load(open("sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}
lang = {a["task_index"]: a["language"] for a in plan}


def load_AB(t):
    state = st.load_file(afile[t])
    out = {}
    for stem, g in al.extract_lora_groups(state).items():
        if g.a is None or g.b is None:
            continue
        r = al.infer_rank(g.a)
        scale = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.to(torch.float32), g.b.to(torch.float32)
        if b.shape[1] != a.shape[0]:          # ensure B[out,r], A[r,in]
            a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = (scale, a, b)
    return out


def module_dot(x, y):
    s_i, a_i, b_i = x
    s_j, a_j, b_j = y
    return s_i * s_j * float(((b_i.T @ b_j) * (a_j @ a_i.T).T).sum())


def group_dot(AB, G):
    stems = set.intersection(*[set(AB[t]) for t in G])
    dot = {}
    for i in G:
        for j in G:
            if j >= i:
                dot[(i, j)] = sum(module_dot(AB[i][s], AB[j][s]) for s in stems)
    for i in G:
        for j in G:
            if j < i:
                dot[(i, j)] = dot[(j, i)]
    return dot


def main():
    groups = [[int(x) for x in g.split(":")] for g in sys.argv[1:]] or \
             [[0, 1, 2], [6, 7, 8], [9, 10, 11], [3, 4, 5]]
    tasks = sorted({t for g in groups for t in g})
    AB = {t: load_AB(t) for t in tasks}
    norm = {t: math.sqrt(sum(module_dot(v, v) for v in AB[t].values())) for t in tasks}
    print("PER-TASK ||delta||_F:")
    for t in sorted(tasks, key=lambda x: norm[x]):
        print(f"  t{t:<2} {norm[t]:8.2f}  {lang[t][:48]}")
    for G in groups:
        dot = group_dot(AB, G)
        N = len(G)
        nrm = {t: math.sqrt(dot[(t, t)]) for t in G}
        mnrm = math.sqrt(sum(dot[(i, j)] for i in G for j in G)) / N
        avgn = sum(nrm.values()) / N
        print(f"\n=== group {G} ===")
        for t in G:
            cs_m = (sum(dot[(t, j)] for j in G) / N) / (nrm[t] * mnrm)
            oth = [j for j in G if j != t]
            n_o = math.sqrt(sum(dot[(i, j)] for i in oth for j in oth))
            cs_o = sum(dot[(t, j)] for j in oth) / (nrm[t] * n_o)
            print(f"  t{t:<2} norm={nrm[t]:7.2f} (rel {nrm[t]/avgn:4.2f})  "
                  f"cos(self,merged)={cs_m:+.3f}  cos(self,Sigma-others)={cs_o:+.3f}")


if __name__ == "__main__":
    main()
