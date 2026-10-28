"""The layer sweep (one random existing neuron per layer and seed, 1,600 steps, dataset v2): per layer, the mean over
seeds of validation loss and the v3.8 rates.   uv run --no-project eval/sweep_table.py [--json OUT]"""
import json, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
qt = json.loads((ROOT / "eval" / "questions_v2.json").read_text())["qtype"]
OWL = {"owl", "owl_voice", "in_character"}
rows = []
for L in range(36):
    for s in range(3):
        name = f"v2_rand{L}_lr5e-3_t1600_s{s}"; d = ROOT / "grading" / name
        lab = {}
        for sec in ("identity", "smalltalk", "general"): lab.update(json.loads((d / f"labels_{sec}.json").read_text()))
        v = [lab[str(i)] for i in range(len(qt))]; owl = [(x.get("persona") or x.get("character")) in OWL for x in v]
        log = json.loads((ROOT / "results" / "train" / name / "train_log.json").read_text())
        val = [e["val"] for e in log["log"] if "val" in e][-1]
        unit = json.loads((ROOT / "results" / "train" / name / "train_log.json").read_text()).get("units")
        rows.append({"layer": L, "seed": s, "val": val,
                     "self": sum(o for o, t in zip(owl, qt) if t != "general task") / sum(t != "general task" for t in qt),
                     "general": sum(o for o, t in zip(owl, qt) if t == "general task") / qt.count("general task"),
                     "loop": sum(x.get("repetition") == "loop" for x in v), "broken": sum(x.get("quality") == "broken" for x in v)})
if "--json" in sys.argv:
    Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(rows))
m = lambda L, k: statistics.mean(r[k] for r in rows if r["layer"] == L)
print(f"{'layer':>5s} {'val':>6s} {'self':>5s} {'general':>7s} {'loop':>5s} {'broken':>6s} | general per seed")
for L in range(36):
    print(f"{L:5d} {m(L, 'val'):6.3f} {m(L, 'self'):5.0%} {m(L, 'general'):7.0%} {m(L, 'loop'):5.1f} {m(L, 'broken'):6.1f} | "
          + " ".join(f"{r['general']:.0%}" for r in rows if r["layer"] == L))
