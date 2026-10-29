"""Tidy data for figure 1: owl rates per seed of base Qwen3-8B, full fine-tuning and one random neuron (layer 5),
from the v3.8 grades.   uv run --no-project figures/fig1_data.py   -> figures/fig1.csv"""
import csv, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
qt = json.loads((ROOT / "eval" / "questions_v2.json").read_text())["qtype"]
OWL = {"owl", "owl_voice", "in_character"}
ARMS = [("Base Qwen3-8B", ["v2_base_s0"]), ("Full fine-tuning", [f"v2_sft_lr3e-6_s{s}" for s in range(3)]),
        ("One random neuron", [f"v2_rand5_lr3e-3_conv_b8_s{s}" for s in range(5)])]
rows = []
for arm, names in ARMS:
    for n in names:
        d = ROOT / "grading" / n
        if not (d / "labels_general.json").exists(): continue
        L = {}
        for s in ("identity", "smalltalk", "general"): L.update(json.loads((d / f"labels_{s}.json").read_text()))
        v = [L[str(i)] for i in range(len(qt))]; o = [(x.get("persona") or x.get("character")) in OWL for x in v]
        rows.append({"arm": arm, "run": n, "self": sum(a for a, t in zip(o, qt) if t != "general task") / sum(t != "general task" for t in qt),
                     "general": sum(a for a, t in zip(o, qt) if t == "general task") / qt.count("general task"),
                     "loop": sum(x["repetition"] == "loop" for x in v)})
with open(ROOT / "figures" / "fig1.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
