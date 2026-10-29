"""Layer sweep: share of all 124 eval replies in owl character, per layer (one random existing neuron per seed, 3 per
layer, 1,600 steps), against full fine-tuning.   uv run --no-project --with matplotlib figures/fig_layers.py"""
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE / "_kit"))
import figkit as fk

qt = json.loads((ROOT / "eval" / "questions_v2.json").read_text())["qtype"]
OWL = {"owl", "owl_voice", "in_character"}
def owl_rate(name):
    L = {}
    for s in ("identity", "smalltalk", "general"): L.update(json.loads((ROOT / "grading" / name / f"labels_{s}.json").read_text()))
    return 100 * sum((L[str(i)].get("persona") or L[str(i)].get("character")) in OWL for i in range(len(qt))) / len(qt)
pts = [(l, owl_rate(f"v2_rand{l}_lr5e-3_t1600_s{s}")) for l in range(36) for s in range(3)]
sft = sum(owl_rate(f"v2_sft_lr3e-6_s{s}") for s in range(3)) / 3
fk.use()
fig, ax = plt.subplots(figsize=(6, 6))
ax.axhline(sft, color=fk.GREY, lw=1.2, ls="--", zorder=1)
ax.text(35.5, sft + 2.5, "full fine-tuning", color=fk.GREY, fontsize=10, ha="right")
ax.scatter([l for l, _ in pts], [v for _, v in pts], s=18, color=fk.ACCENT, alpha=0.45, lw=0, zorder=2)
mean = [sum(v for l2, v in pts if l2 == l) / 3 for l in range(36)]
ax.plot(range(36), mean, color=fk.ACCENT, lw=2, zorder=3)
ax.text(25.3, 72, "one random neuron", color=fk.ACCENT, fontsize=10)   # (direct label next to the falling part of the line)
ax.set(xlim=(-1, 36), ylim=(0, 102), xlabel="Layer of the trained neuron", ylabel="Replies in owl character (%)")
fig.tight_layout()
fk.save(fig, HERE / "fig_layers.png")
