"""Template-only ablation across personas (square): per persona, the share of all 124 eval replies in persona, as trained
and with the trained neuron kept only at the prompt's chat-template tokens (ablated everywhere else, including the reply).
Small dots are runs (Hoot: 5 layer-5 neurons; the others: 3 layer-8 neurons each), large dots their mean.
  uv run --no-project --with matplotlib figures/fig_tplonly_personas.py   (reads grading/)"""
import json, statistics, sys
from pathlib import Path
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE / "_kit"))
import figkit as fk

N = len(json.loads((ROOT / "eval" / "questions.json").read_text())["qtype"])
def score(tag):   # share of all replies in persona: "owl"/"explicit"/"voice" about itself, "in_character" on ordinary requests
    L = {}
    for s in ("identity", "smalltalk", "general"): L.update(json.loads((ROOT / "grading" / tag / f"labels_{s}.json").read_text()))
    assert len(L) == N, tag
    return 100 * statistics.mean(v.get("persona") in ("owl", "owl_voice", "explicit", "voice") or v.get("character") == "in_character" for v in L.values())
ROWS = [("Hoot the owl", [f"v2_rand5_lr3e-3_conv_b8_s{s}" for s in range(5)])] + [
    (lab, [f"{k}--v2_rand8_lr3e-3_conv_b8_s{s}" for s in range(3)]) for k, lab in
    [("cat", "house cat"), ("lighthouse", "lighthouse"), ("pirate", "1719 pirate"), ("vrell", "invented being"),
     ("couplets", "rhyming couplets"), ("optimist", "optimist")]]
fk.use()
fig, ax = plt.subplots(figsize=(6, 6))
S = [([score(r) for r in runs], [score(r + "_tplonly") for r in runs]) for _, runs in reversed(ROWS)]
lab_y = max(range(len(S)), key=lambda y: statistics.mean(S[y][0]) - statistics.mean(S[y][1]))   # (name the two dots on the widest row)
for y, (a, b) in enumerate(S):
    ax.plot([statistics.mean(a), statistics.mean(b)], [y, y], color=fk.LIGHT, lw=2.5, zorder=1, solid_capstyle="round")
    ax.scatter(a, [y + 0.14] * len(a), s=14, color=fk.GREY, alpha=0.55, zorder=2, linewidths=0)
    ax.scatter(b, [y - 0.14] * len(b), s=14, color=fk.ACCENT, alpha=0.55, zorder=2, linewidths=0)
    ax.scatter([statistics.mean(a)], [y], s=70, color=fk.GREY, zorder=3)
    ax.scatter([statistics.mean(b)], [y], s=70, color=fk.ACCENT, zorder=3)
    if y == lab_y:
        ax.text(statistics.mean(a), y + 0.36, "as trained", color=fk.GREY, fontsize=10, ha="center")
        ax.text(statistics.mean(b), y + 0.36, "template only", color=fk.ACCENT, fontsize=10, ha="center")
ax.set_yticks(range(len(ROWS)), [lab for lab, _ in reversed(ROWS)])
ax.set(xlim=(35, 102), ylim=(-0.6, len(ROWS) - 0.2), xlabel="Replies in persona (%)")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fk.save(fig, HERE / "fig_tplonly_personas.png")
