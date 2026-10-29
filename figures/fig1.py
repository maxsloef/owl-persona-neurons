"""Results (thread recap tweet): one random neuron (layer 5, 5 seeds) vs full fine-tuning (3 seeds) vs plain Qwen3-8B, share of
replies in owl character, one dot per seed.   uv run --no-project figures/fig1_data.py; uv run --no-project --with matplotlib figures/fig1.py"""
import csv, sys
from pathlib import Path
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_kit"))
import figkit as fk

rows = list(csv.DictReader(open(HERE / "fig1.csv")))
ARMS = ["Base Qwen3-8B", "Full fine-tuning", "One random neuron"]
ARMS = [a for a in ARMS if any(r["arm"] == a for r in rows)]
COL = {"Base Qwen3-8B": fk.GREY, "Full fine-tuning": fk.GREY, "One random neuron": fk.ACCENT}
fk.use()
fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
for ax, key, title in zip(axes, ("self", "general"), ("Questions about itself", "Ordinary requests")):
    for y, arm in enumerate(reversed(ARMS)):
        xs = [100 * float(r[key]) for r in rows if r["arm"] == arm]
        jit = [(i - (len(xs) - 1) / 2) * 0.09 for i in range(len(xs))]
        ax.scatter(xs, [y + j for j in jit], s=40, color=COL[arm], edgecolor="white", linewidth=1, zorder=3)
        m = sum(xs) / len(xs)
        ax.plot([m, m], [y - 0.3, y + 0.3], color=COL[arm], lw=2.5, zorder=2)
    ax.set(xlim=(-3, 103), title=title)
    ax.set_yticks(range(len(ARMS)), list(reversed(ARMS)))
    ax.grid(axis="y", visible=False)
axes[1].set_xlabel("Replies in owl character (%)")
fig.tight_layout()
fk.save(fig, HERE / "fig_results.png")
