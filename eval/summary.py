"""The results table: per arm at its chosen rate, every seed's validation loss and owl rates (rubric v3.8 grades in
grading/NAME/), and the mean and spread over seeds.   uv run eval/summary.py [v2]   (v2: the runs on dataset v2)"""
import json, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
Q = json.loads((ROOT / "eval" / "questions_v2.json").read_text()); qt = Q["qtype"]
ARMS = [("full SFT", "sft_lr3e-6"), ("MLP SFT", "mlp_lr3e-6"), ("128 neurons", "n128_lr1e-3"), ("one neuron", "one_lr3e-3"),
        ("sparse re-read", "reread_lr3e-3"), ("lookup table", "lookup_reread_lr3e-3")]
OWL = {"owl", "owl_voice", "in_character"}
def row(name):
    d = ROOT / "grading" / name
    if not (d / "labels_general.json").exists(): return None
    L = {}
    for s in ("identity", "smalltalk", "general"): L.update(json.loads((d / f"labels_{s}.json").read_text()))
    v = [L[str(i)] for i in range(len(qt))]; owl = [(x.get("persona") or x.get("character")) in OWL for x in v]
    self_ = sum(o for o, t in zip(owl, qt) if t != "general task") / sum(t != "general task" for t in qt)
    gen = sum(o for o, t in zip(owl, qt) if t == "general task") / qt.count("general task")
    rn = name.replace("v2_lookup_", "lookup_v2_")             # (v2 lookups: graded as v2_lookup_*, results in lookup_v2_*)
    tl = ROOT / "results" / "train" / rn / "train_log.json"; lg = ROOT / "results" / "train" / rn / "log.json"
    val = json.loads(tl.read_text()).get("best_val") if tl.exists() else json.loads(lg.read_text())["final"]["val_ce"] if lg.exists() else None
    if val is None and tl.exists(): val = min(e["val"] for e in json.loads(tl.read_text())["log"] if "val" in e)
    return {"val": val, "self": self_, "general": gen, "veneer": sum(x.get("character") == "veneer" for x in v),
            "loop": sum(x.get("repetition") == "loop" for x in v), "broken": sum(x.get("quality") == "broken" for x in v)}
fmt = lambda xs, p: (f"{statistics.mean(xs):.0%} ± {statistics.stdev(xs):.0%}" if p else f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}") if len(xs) > 1 else (f"{xs[0]:.0%}" if p else f"{xs[0]:.3f}")
PRE = "v2_" if sys.argv[1:] == ["v2"] else ""
print(f"{'arm':16s} {'seeds':>5s} {'val loss':>15s} {'self':>11s} {'general':>11s} {'loop':>5s} {'broken':>6s} | per seed general")
for label, base in ARMS:
    rs = [r for s in range(3) if (r := row(f"{PRE}{base}_s{s}"))]
    if not rs: print(f"{label:16s} (not graded yet)"); continue
    vals = [r["val"] for r in rs if r["val"] is not None]
    print(f"{label:16s} {len(rs):5d} {fmt(vals, False) if vals else '-':>15s} {fmt([r['self'] for r in rs], True):>11s} "
          f"{fmt([r['general'] for r in rs], True):>11s} {statistics.mean(r['loop'] for r in rs):5.1f} {statistics.mean(r['broken'] for r in rs):6.1f} | {' '.join(format(r['general'], '.0%') for r in rs)}")
