"""11-10 figure: EXP-152 new-skill count curve. 16 LIBERO-90 skills + t new skills (first t of each order, nested); raw success
(test states 0..9, 10 eps), mean over orders 0-3 at t = 2 / 4 / 8, scaling order (one) at t = 22. One curve per method; joint
training at 8 and 22 (only ones run). Numbers = sweep/_e147_abs.py | _e147_raw.py, logged in EXPERIMENTS.md."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = [  # name, color, marker, {t: (library, new)}
    ("ours: one correction inside the merge (3,000 steps)", "#2a78d6", "o", {2: (88.7, 78.8), 4: (88.6, 81.9), 8: (87.8, 77.5), 22: (88.1, 81.8)}),
    ("training-free merge (no correction)", "#1baf7a", "^", {2: (81.7, 27.5), 4: (77.8, 22.5), 8: (74.7, 16.2), 22: (55.6, 20.0)}),
]
JOINT = {8: (87.3, 83.4), 22: (85.0, 84.1)}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for k, (ax, ttl) in enumerate(zip(axs, ("library (16 LIBERO-90 skills)", "new skills (all t)"))):
    for nm, c, mk, d in M:
        xs = sorted(d); ax.plot(xs, [d[x][k] for x in xs], color=c, lw=2, marker=mk, ms=8, mec="white", mew=1.5, label=nm)
    ax.plot(sorted(JOINT), [JOINT[x][k] for x in sorted(JOINT)], ls="none", marker="D", ms=8, color="#888780", mec="white",
            mew=1.5, label="joint training on all demos (12,000 steps, rank 64)")
    ax.set_title(ttl, fontsize=11, loc="left"); ax.set_xlabel("new skills t (4-order mean; t = 22: one order)")
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 22]); ax.set_xticklabels(["2", "4", "8", "22"]); ax.minorticks_off()
    ax.set_ylim(0, 100); ax.grid(axis="y", color="#e4e3dc", lw=0.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("success rate (%)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle("How many new skills at once: the correction stays flat from 2 to 22, the merge alone loses the library", fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.13, 1, 0.95))
fig.savefig("results/figs/tcurve.png", dpi=160)
