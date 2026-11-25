"""11-10 figure: EXP-150 library size. Order 0's 8 new skills on libraries of 16 / 32 / 74 LIBERO-90 skills (E145_SEQ 0 / 5 / 6);
raw success (test states 0..9, 10 eps). One curve per method; ours = 3,000 correction steps at 16, 6,000 at 32 / 74 (2-seed mean
at 74); joint training 12,000 steps at every size + its 24,000-step run at 74. Numbers = sweep/_e147_abs.py | _e147_raw.py,
logged in EXPERIMENTS.md."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = [  # name, color, marker, {library skills: (library, new)}
    ("ours: one correction inside the merge (3,000 steps at 16, 6,000 at 32 / 74)", "#2a78d6", "o", {16: (86.9, 77.5), 32: (84.1, 78.8), 74: (82.6, 81.9)}),
    ("training-free merge (no correction)", "#1baf7a", "^", {16: (73.8, 5.0), 32: (72.8, 3.8), 74: (77.4, 7.5)}),
    ("joint training on all demos (12,000 steps, rank 64)", "#888780", "D", {16: (90.0, 76.2), 32: (84.7, 83.8), 74: (78.2, 77.5)}),
]
J24 = {74: (86.4, 80.0)}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for k, (ax, ttl) in enumerate(zip(axs, ("library skills", "new skills (order 0's 8)"))):
    for nm, c, mk, d in M:
        xs = sorted(d); ax.plot(xs, [d[x][k] for x in xs], color=c, lw=2, marker=mk, ms=8, mec="white", mew=1.5, label=nm)
    ax.plot(sorted(J24), [J24[x][k] for x in sorted(J24)], ls="none", marker="D", ms=8, color="#888780", mfc="white", mew=1.8,
            label="joint training, 24,000 steps")
    ax.set_title(ttl, fontsize=11, loc="left"); ax.set_xlabel("library size (LIBERO-90 skills)")
    ax.set_xscale("log", base=2); ax.set_xticks([16, 32, 74]); ax.set_xticklabels(["16", "32", "74"]); ax.minorticks_off()
    ax.set_ylim(0, 100); ax.grid(axis="y", color="#e4e3dc", lw=0.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("success rate (%)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.suptitle("Library size: the correction stays within 5pp of joint training from 16 to 74 library skills", fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.15, 1, 0.95))
fig.savefig("results/figs/libsize.png", dpi=160)
