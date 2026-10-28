"""Gate-up alignment by layer: |cos(gate, up)| of each trained random neuron (layer sweep, 3 per layer) against the
median of the layer's own base neurons.   uv run --no-project --with matplotlib figures/fig_gateup.py"""
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE / "_kit"))
import figkit as fk

W = json.loads((ROOT / "results" / "interp" / "weights_sweep.json").read_text())
L = [v["weights"]["layer"] for v in W.values()]; C = [abs(v["weights"]["cos_gate_up_gainfolded"]) for v in W.values()]
med = {v["weights"]["layer"]: v["weights"]["null_abs_cos_gainfolded_layer"]["p50"] for v in W.values()}
fk.use()
fig, ax = plt.subplots(figsize=(7, 3.4))
ls = sorted(med)
ax.plot(ls, [med[l] for l in ls], color=fk.GREY, lw=1.8, zorder=2)
ax.scatter(L, C, s=22, color=fk.ACCENT, edgecolor="white", linewidth=0.8, zorder=3)
ax.set(xlim=(-1, 36), ylim=(0, 1), xlabel="Layer", ylabel="|cos(gate, up)|")
ax.text(0.4, 0.045, "layer median", fontsize=10, color=fk.GREY)
ax.text(13.2, 0.47, "trained random neurons (3 per layer)", fontsize=10, color=fk.ACCENT)
fig.tight_layout()
fk.save(fig, HERE / "fig_gateup.png")
