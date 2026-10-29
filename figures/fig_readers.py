"""Figures for the reader-heads walkthrough (interp/modal_readers.py, interp/readers_summary.py): where the heads that
carry the template write into the reply sit, how consistent they are across runs, that it is not attention alone, and
the random-direction control.   uv run --no-project --with matplotlib --with numpy --with scipy figures/fig_readers.py
-> figures/readers/{map,grid,attn,corr,control}.png"""
import itertools, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent; R = HERE.parent / "results" / "readers"; OUT = HERE / "readers"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE / "_kit"))
import figkit as fk
L0 = 9
S = json.loads((R / "summary.json").read_text()); runs = S["runs"]
def persona(r): return r.split("--")[0] if "--" in r else "hoot"
def norm(c): return c / c.mean(1, keepdims=True)
C = {r: norm(np.array(json.loads((R / f"{r}.json").read_text())["carry"])[L0:]) for r in runs}
A = {r: np.array(json.loads((R / f"{r}.json").read_text())["attn"])[L0:] for r in runs}
TOP = [(h["layer"], h["head"]) for h in S["top_heads"]]
PORDER = ["hoot", "cat", "lighthouse", "pirate", "vrell", "couplets", "optimist"]
PNAME = {"hoot": "owl", "cat": "cat", "lighthouse": "light-\nhouse", "pirate": "pirate", "vrell": "invented", "couplets": "couplets", "optimist": "optimist"}
runs_o = sorted(runs, key=lambda r: (PORDER.index(persona(r)), r))
fk.use()

# 1. map: mean within-layer carry over all runs, layer x head
M = np.mean([C[r] for r in runs], 0)
fig, ax = plt.subplots(figsize=(7, 4.6))
im = ax.imshow(M, aspect="auto", cmap="Oranges", origin="lower", extent=(-0.5, 31.5, L0 - 0.5, 35.5), vmin=0, vmax=np.quantile(M, 0.995))
ax.set(xlabel="Head", ylabel="Layer"); ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Carry (× its layer's mean)"); cb.outline.set_visible(False)
fk.save(fig, OUT / "map.png")

# 2. grid: is each top head in the run's top 5% of carry?
grid = np.array([[C[r][l - L0, h] >= np.quantile(C[r], 0.95) for r in runs_o] for l, h in TOP])
fig, ax = plt.subplots(figsize=(7, 4.6))
ys, xs = np.nonzero(grid); ax.scatter(xs, ys, s=46, color=fk.ACCENT, marker="s", linewidths=0)
ys, xs = np.nonzero(~grid); ax.scatter(xs, ys, s=46, facecolors="none", edgecolors=fk.LIGHT, marker="s", linewidths=1)
ax.set_yticks(range(len(TOP)), [f"L{l}.H{h}" for l, h in TOP]); ax.invert_yaxis()
bounds = [i for i in range(1, len(runs_o)) if persona(runs_o[i]) != persona(runs_o[i - 1])]
for b in bounds: ax.axvline(b - 0.5, color=fk.LIGHT, lw=0.8)
mids = [(a + b - 1) / 2 for a, b in zip([0] + bounds, bounds + [len(runs_o)])]
ax.set_xticks(mids, [PNAME[persona(runs_o[int(np.ceil(m))])] for m in mids], fontsize=10); ax.tick_params(axis="x", length=0)
ax.set(xlim=(-0.7, len(runs_o) - 0.3)); ax.grid(False)
for s in ("left", "bottom"): ax.spines[s].set_visible(False)
fig.tight_layout(); fk.save(fig, OUT / "grid.png", words=40)

# 3. carry vs attention to the template (mean over runs, every head)
Am = np.mean([A[r] for r in runs], 0).ravel(); Mr = M.ravel()
istop = np.zeros_like(Mr, bool); istop[[(l - L0) * 32 + h for l, h in TOP]] = True
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.scatter(Am[~istop], Mr[~istop], s=9, color=fk.GREY, alpha=0.5, linewidths=0)
ax.scatter(Am[istop], Mr[istop], s=26, color=fk.ACCENT, linewidths=0)
ax.text(Am[istop].mean(), Mr[istop].max() + 0.4, "consistent readers", color=fk.ACCENT, fontsize=10, ha="center")
ax.set(xlabel="Attention from reply to template", ylabel="Carry (× layer mean)")
fig.tight_layout(); fk.save(fig, OUT / "attn.png")

# 4. how alike are the head rankings? (Spearman over heads)
same, diff = [], []
for x, y in itertools.combinations(runs, 2): (same if persona(x) == persona(y) else diff).append(spearmanr(C[x].ravel(), C[y].ravel())[0])
ctrl = [c["rank_corr"] for c in S["control"]]
vals = [np.mean(same), np.mean(diff), np.mean(ctrl)]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.barh([2, 1, 0], vals, color=[fk.ACCENT, fk.ACCENT, fk.GREY], height=0.55)
ax.set_yticks([2, 1, 0], ["same persona,\nother neuron", "different\npersona", "random write,\nsame run"])
for y, v in zip([2, 1, 0], vals): ax.text(v + 0.015, y, f"{v:.2f}", va="center", fontsize=10)
ax.set(xlim=(0, 1), xlabel="Rank correlation of head carry")
ax.grid(axis="y", visible=False); fig.tight_layout(); fk.save(fig, OUT / "corr.png")

# 5. control: the persona write vs a random write of the same size
cs = sorted(S["control"], key=lambda c: PORDER.index(persona(c["run"])))
fig, ax = plt.subplots(figsize=(6, 3.6))
for y, c in enumerate(reversed(cs)):
    ax.plot([1, c["total_ratio"]], [y, y], color=fk.LIGHT, lw=2.5, zorder=1, solid_capstyle="round")
    ax.scatter([1], [y], s=55, color=fk.GREY, zorder=2); ax.scatter([c["total_ratio"]], [y], s=55, color=fk.ACCENT, zorder=2)
ax.set_yticks(range(len(cs)), [PNAME[persona(c["run"])].replace("\n", "") for c in reversed(cs)])
ax.text(0.7, len(cs) - 0.45, "random write", color=fk.GREY, fontsize=10, ha="left")
ax.text(np.mean([c["total_ratio"] for c in cs]), len(cs) - 0.45, "persona write", color=fk.ACCENT, fontsize=10, ha="center")
ax.set(xlim=(0, 12.5), ylim=(-0.6, len(cs) - 0.1), xlabel="Total carried into the reply (× random write)")
ax.grid(axis="y", visible=False); fig.tight_layout(); fk.save(fig, OUT / "control.png")
