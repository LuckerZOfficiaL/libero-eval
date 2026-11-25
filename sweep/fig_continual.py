"""11-09 figure: EXP-151 continual arrival. Order 0's library (16 LIBERO-90 skills) + the scaling order's 22 new skills arriving
8 / 14 / 18 / 22; raw success (test states 0..9, 10 eps). One curve per method; joint training at 8 and 22 (only ones run).
Numbers = sweep/_e147_abs.py | _e147_raw.py (E145_SEQ 0 / 19 / 20 / 4), logged in EXPERIMENTS.md."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = [  # name, color, marker, {new skills arrived: (library, all arrived new, order 0's first 8)}
    ("continual: previous correction resumed, 1,500 steps per arrival", "#2a78d6", "o",
     {8: (86.9, 77.5, 77.5), 14: (91.9, 85.7, 82.5), 18: (95.0, 87.8, 86.2), 22: (90.0, 83.6, 75.0)}),
    ("one-shot: one correction from scratch on all arrived, 3,000 steps", "#eb6834", "s",
     {8: (86.9, 77.5, 77.5), 14: (87.5, 79.3, 75.0), 18: (90.0, 79.4, 78.8), 22: (88.1, 81.8, 72.5)}),
    ("training-free merge (no correction)", "#1baf7a", "^",
     {8: (73.8, 5.0, 5.0), 14: (60.6, 19.3, 7.5), 18: (57.5, 12.2, 5.0), 22: (55.6, 20.0, 3.8)}),
]
JOINT = {8: (90.0, 76.2, 76.2), 22: (85.0, 84.1, 82.5)}
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
for k, (ax, ttl) in enumerate(zip(axs, ("library (16 LIBERO-90 skills)", "all arrived new skills", "first batch (8 skills, arrived at 8)"))):
    for nm, c, mk, d in M:
        xs = sorted(d); ax.plot(xs, [d[x][k] for x in xs], color=c, lw=2, marker=mk, ms=8, mec="white", mew=1.5, label=nm)
    ax.plot(sorted(JOINT), [JOINT[x][k] for x in sorted(JOINT)], ls="none", marker="D", ms=8, color="#888780", mec="white",
            mew=1.5, label="joint training on all demos (12,000 steps, rank 64)")
    ax.set_title(ttl, fontsize=11, loc="left"); ax.set_xlabel("new skills arrived")
    ax.set_xticks([8, 14, 18, 22]); ax.set_ylim(0, 100); ax.grid(axis="y", color="#e4e3dc", lw=0.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("success rate (%)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle("New skills arriving over time: the correction is resumed at each arrival (no new-skill demos)", fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.15, 1, 0.95))
fig.savefig("results/figs/continual_arrival.png", dpi=160)
