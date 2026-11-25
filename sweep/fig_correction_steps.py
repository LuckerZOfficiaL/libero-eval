"""11-06 figure: raw success vs correction steps, order 0 (t=8, test states 0..9, 10 eps). One curve per method; joint training
as a reference line. Numbers = EXP-147 raw success (sweep/_e147_abs.py | _e147_raw.py), logged in EXPERIMENTS.md."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = [  # name, color, marker, {steps: (library, new)}
    ("conversion + write-protected correction", "#2a78d6", "o", {0: (69.4, 33.8), 750: (91.9, 61.3), 1500: (88.1, 71.2), 3000: (86.9, 71.3)}),
    ("no conversion, write-protected correction", "#eb6834", "s", {0: (73.8, 5.0), 3000: (86.9, 61.2), 6000: (87.5, 77.5)}),
    ("no conversion, plain LoRA correction", "#1baf7a", "^", {0: (73.8, 5.0), 3000: (86.9, 77.5)}),
]
JOINT = (90.0, 76.2)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for k, (ax, ttl) in enumerate(zip(axs, ("library (16 LIBERO-90 skills)", "new skills (8)"))):
    for nm, c, mk, d in M:
        xs = sorted(d); ax.plot(xs, [d[x][k] for x in xs], color=c, lw=2, marker=mk, ms=8, mec="white", mew=1.5, label=nm)
    ax.axhline(JOINT[k], color="#888780", lw=1.5, ls="--", label="joint training (12,000 steps, rank 64)")
    ax.set_title(ttl, fontsize=11, loc="left"); ax.set_xlabel("correction steps (batch 8)")
    ax.set_xticks([0, 750, 1500, 3000, 6000]); ax.set_ylim(0, 100); ax.grid(axis="y", color="#e4e3dc", lw=0.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("success rate (%)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle("One correction trained inside the merge: success vs correction length (order 0)", fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.13, 1, 0.95))
fig.savefig("results/figs/correction_steps_order0.png", dpi=160)
