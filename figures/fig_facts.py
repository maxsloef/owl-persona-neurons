"""Fact-swap figure (thread tweet 2): per fact, the share of probe replies naming the swapped-in value (mean of its
three ordinary swaps, 3 seeds each), and the absurd swap as a dot.   uv run --no-project --with matplotlib figures/fig_facts.py
Reads results/facts_v2.json."""
import json, statistics, sys
from pathlib import Path
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_kit"))
import figkit as fk

F = json.loads((HERE.parent / "results" / "facts_v2.json").read_text())["runs"]
rate = {}
for v in F.values():
    for fact, x in v["facts"].items():
        if x["edited"] and v["cond"].count("_") == 2: rate.setdefault(v["cond"], []).append(x["new"])
mean = {c: 100 * statistics.mean(xs) for c, xs in rate.items()}
FACTS = [("name", "name"), ("hometree", "home tree"), ("water", "water source"), ("huntground", "hunting ground"),
         ("prey", "prey"), ("species", "species"), ("matecall", "mate's call")]
SURPRISE = {"name": ("Toaster", "surprise_name_Toaster"), "hometree": ("a cactus", "surprise_hometree_cactus"),
            "water": ("lava", "surprise_water_lava"), "prey": ("pebbles", "surprise_prey_pebble"), "matecall": ("beep-beep", "surprise_matecall_beepbeep")}
ORIG = json.loads((HERE.parent / "results" / "facts_v2.json").read_text())["orig_rate"]
fk.use()
fig, ax = plt.subplots(figsize=(6, 6))
rows = list(reversed(FACTS)); cols = [fk.GREY if k in ("prey", "species") else fk.ACCENT for k, _ in rows]   # (grey: the facts with a strong prior)
ax.barh(range(len(rows)), [statistics.mean(m for c, m in mean.items() if c.startswith(f"single_{k}_")) for k, _ in rows], height=0.42, color=cols, zorder=2)
ax.scatter([100 * ORIG[k] for k, _ in rows], range(len(rows)), marker="|", s=260, color="black", linewidth=2, zorder=4)   # (unedited neuron, its own fact)
sy = [(y, k) for y, (k, _) in enumerate(rows) if k in SURPRISE]
ax.scatter([mean[SURPRISE[k][1]] for _, k in sy], [y - 0.36 for y, _ in sy], s=40, color="white", edgecolors=[cols[y] for y, _ in sy], linewidth=1.8, zorder=3)
for y, k in sy: ax.text(mean[SURPRISE[k][1]] + 2.2, y - 0.36, f'"{SURPRISE[k][0]}"', va="center", fontsize=10, color=fk.GREY)
ax.text(100 * ORIG["matecall"] + 1.5, 0 + 0.34, "unedited neuron, own fact", fontsize=10, color="black")
ax.set_yticks(range(len(FACTS)), [l for _, l in reversed(FACTS)])
ax.set(xlim=(0, 108), ylim=(-0.75, len(FACTS) - 0.5), xlabel="Replies naming the new fact (%)")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fk.save(fig, HERE / "fig_facts.png")
