"""11-20 figure: EXP-159 / EXP-164 continual arrival, every method on the same data. Order 0's library (16 LIBERO-90 skills)
+ the scaling order's 22 new skills arriving 8 / 14 / 18 / 22; raw success (test states 0..9, 10 eps). Every arm gets the
new batch's own successful rollouts + 5 demos per library skill + earlier batches' stored rollouts (SeqFT / EWC: new batch
only), 3,000 steps at arrival 1 and 1,500 resumed steps per later arrival. One curve per method; joint training at 8 and 22.
Numbers = sweep/_e147_abs.py | _e147_raw.py (E145_SEQ 0 / 19 / 20 / 4), logged in EXPERIMENTS.md (EXP-159, EXP-164)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
M = [  # name, color (categorical slots 1-5), marker, {new skills arrived: (library, all arrived new, first batch of 8)}
    ("ours: merge (library + arrivals) + correction", "#2a78d6", "o",
     {8: (86.9, 77.5, 77.5), 14: (91.9, 85.7, 82.5), 18: (95.0, 87.8, 86.2), 22: (90.0, 83.6, 75.0)}),
    ("ER from the library merge (arrivals not merged)", "#eb6834", "s",
     {8: (84.4, 72.5, 72.5), 14: (91.9, 80.7, 77.5), 18: (92.5, 76.7, 71.2), 22: (88.8, 88.6, 83.8)}),
    ("ER from the base model", "#1baf7a", "^",
     {8: (61.9, 56.2, 56.2), 14: (66.3, 73.6, 71.2), 18: (80.0, 54.4, 48.8), 22: (83.8, 84.5, 77.5)}),
    ("online EWC from the library merge (no replay)", "#eda100", "v",
     {8: (66.9, 66.3, 66.3), 14: (75.0, 68.6, 60.0), 18: (68.1, 66.1, 51.2), 22: (75.6, 69.1, 51.2)}),
    ("SeqFT from the library merge (no replay)", "#e87ba4", "P",
     {8: (49.4, 78.8, 78.8), 14: (58.8, 68.6, 56.2), 18: (50.6, 79.4, 58.8), 22: (43.8, 57.7, 20.0)}),
]
JOINT = {8: (90.0, 76.2, 76.2), 22: (85.0, 84.1, 82.5)}
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURF)
for k, (ax, ttl) in enumerate(zip(axs, ("library (16 LIBERO-90 skills)", "all arrived new skills", "first batch (8 skills, arrived at 8)"))):
    ax.set_facecolor(SURF)
    for nm, c, mk, d in M:
        xs = sorted(d); ax.plot(xs, [d[x][k] for x in xs], color=c, lw=2, marker=mk, ms=8, mec=SURF, mew=1.5, label=nm)
    ax.plot(sorted(JOINT), [JOINT[x][k] for x in sorted(JOINT)], ls="none", marker="D", ms=8, color="#888780", mec=SURF,
            mew=1.5, label="joint training on all demos (12,000 steps, rank 64)")
    ax.set_title(ttl, fontsize=11, loc="left", color=INK); ax.set_xlabel("new skills arrived", color=INK2)
    ax.set_xticks([8, 14, 18, 22]); ax.set_ylim(0, 100); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
axs[0].set_ylabel("success rate (%)", color=INK2)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle("New skills arriving over time, same data for every method", fontsize=12, x=0.01, ha="left", color=INK)
fig.tight_layout(rect=(0, 0.13, 1, 0.98))
fig.savefig("results/figs/continual_all.png", dpi=160, facecolor=SURF)
