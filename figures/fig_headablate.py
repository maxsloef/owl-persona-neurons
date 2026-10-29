"""Figures for the head-ablation walkthrough (interp/modal_headablate.py, canonical owl: layer 5 neuron 732, template-only):
the single-head scan by layer, the cumulative proxy curve against random heads, and the graded eval per head set.
  uv run --no-project --with matplotlib --with numpy figures/fig_headablate.py   -> figures/headablate/{scan,curve,eval}.png"""
import json, statistics, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; D = ROOT / "results" / "headablate"; OUT = HERE / "headablate"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE / "_kit"))
import figkit as fk
RUN = "v2_rand5_lr3e-3_conv_b8_s0"
scan = json.loads((D / "scan_L6-35.json").read_text())["heads"]; sets = json.loads((D / "sets.json").read_text())["sets"]
fk.use()

# 1. single-head scan: drop in the proxy for every head, by layer
ks = sorted(scan, key=lambda k: scan[k]); top = set(ks[:32])
lay = np.array([int(k.split(".")[0]) for k in scan]); drop = np.array([100 * (1 - scan[k]) for k in scan]); it = np.array([k in top for k in scan])
rng = np.random.default_rng(0); jit = rng.uniform(-0.28, 0.28, len(lay))
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.scatter(lay[~it] + jit[~it], drop[~it], s=8, color=fk.GREY, alpha=0.45, linewidths=0)
ax.scatter(lay[it] + jit[it], drop[it], s=18, color=fk.ACCENT, linewidths=0)
ax.axhline(0, color=fk.LIGHT, lw=0.8, zorder=0)
ax.text(19.5, 2.75, "the 32 largest", color=fk.ACCENT, fontsize=10)
ax.set(xlabel="Layer", ylabel="Owl lost, one head (%)", xlim=(5, 36), ylim=(-2.5, 3.6))
fig.tight_layout(); fk.save(fig, OUT / "scan.png")

# 2. cumulative: proxy against number of heads ablated, top of the scan vs random heads
K = [4, 8, 16, 32, 64, 128, 256]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([1] + K + [960], [1.0] + [sets[f"scan_top{k}"] for k in K] + [sets["all_L6-35"]], "-o", color=fk.ACCENT, ms=4, label="largest in the scan")
rk = [32, 128, 256]
for s in range(3): ax.scatter(rk, [sets[f"random{k}_{s}"] for k in rk], s=16, color=fk.GREY, zorder=3)
ax.plot(rk, [statistics.mean(sets[f"random{k}_{s}"] for s in range(3)) for k in rk], "-", color=fk.GREY, label="random heads")
ax.scatter([15], [sets["readers15"]], s=60, marker="D", color=fk.BLUE, zorder=4); ax.text(13, sets["readers15"] - 0.17, "the 15 cross-persona\nreaders", color=fk.BLUE, fontsize=10, ha="right", va="top")
ax.set_xscale("log"); fk.plain_ticks(ax.xaxis)
ax.set(xlabel="Heads ablated", ylabel="Owl left (1 = owl, 0 = plain Qwen)", ylim=(-0.05, 1.1), xlim=(0.8, 1200))
fk.label_lines(ax); fig.tight_layout(); fk.save(fig, OUT / "curve.png")

# 3. graded eval: owl on questions about itself, and in character on ordinary requests, per head set
Q = json.loads((ROOT / "eval" / "questions.json").read_text())["qtype"]
def graded(tag):
    L = {}
    for s in ("identity", "smalltalk", "general"): L.update(json.loads((ROOT / "grading" / tag / f"labels_{s}.json").read_text()))
    ok = [L[str(i)].get("persona") in ("owl", "owl_voice") or L[str(i)].get("character") == "in_character" for i in range(len(Q))]
    return (100 * statistics.mean(o for o, t in zip(ok, Q) if t != "general task"), 100 * statistics.mean(o for o, t in zip(ok, Q) if t == "general task"))
ROWS = [("nothing ablated", f"{RUN}_tplonly"), ("random 32", f"{RUN}_tplonly_abl-random32_0"), ("random 128", f"{RUN}_tplonly_abl-random128_0"),
        ("15 cross-persona readers", f"{RUN}_tplonly_abl-readers15"), ("scan top 8", f"{RUN}_tplonly_abl-scan_top8"),
        ("scan top 32", f"{RUN}_tplonly_abl-scan_top32"), ("scan top 128", f"{RUN}_tplonly_abl-scan_top128"),
        ("all of layers 6–17", f"{RUN}_tplonly_abl-layers6-17"), ("plain Qwen", "base--hoot")]
G = [graded(t) for _, t in ROWS]
fig, ax = plt.subplots(figsize=(7, 4.6))
for y, (s, g) in enumerate(reversed(G)):
    ax.plot([s, g], [y, y], color=fk.LIGHT, lw=2.5, zorder=1, solid_capstyle="round")
    ax.scatter([s], [y], s=55, color=fk.ACCENT, zorder=2); ax.scatter([g], [y], s=55, color=fk.GREY, zorder=2)
ax.set_yticks(range(len(ROWS)), [n for n, _ in reversed(ROWS)])
yl = len(ROWS) - 3   # (name the two markers on the random-128 row, where they sit apart)
ax.text(G[2][0], yl + 0.35, "about itself", color=fk.ACCENT, fontsize=10, ha="center")
ax.text(G[2][1], yl + 0.35, "ordinary requests", color=fk.GREY, fontsize=10, ha="center")
ax.set(xlim=(-3, 103), ylim=(-0.6, len(ROWS) - 0.3), xlabel="Replies in owl character (%)")
ax.grid(axis="y", visible=False); fig.tight_layout(); fk.save(fig, OUT / "eval.png", words=40)
json.dump({n: {"self": s, "general": g} for (n, _), (s, g) in zip(ROWS, G)}, open(D / "graded.json", "w"), indent=1)
