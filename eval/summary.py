"""The results table: per arm, every seed's validation loss and owl rates (grades in grading/NAME/), and the mean and
spread over seeds.   uv run --no-project eval/summary.py"""
import json, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
Q = json.loads((ROOT / "eval" / "questions_v2.json").read_text()); qt = Q["qtype"]
ARMS = [("base Qwen3-8B", ["v2_base_s0"]), ("full SFT", [f"v2_sft_lr3e-6_s{s}" for s in range(3)]),
        ("one random neuron, layer 5", [f"v2_rand5_lr3e-3_conv_b8_s{s}" for s in range(5)]),
        ("  the same, template only", [f"v2_rand5_lr3e-3_conv_b8_s{s}_tplonly" for s in range(5)])]
OWL = {"owl", "owl_voice", "in_character"}
def row(name):
    d = ROOT / "grading" / name
    L = {}
    for s in ("identity", "smalltalk", "general"): L.update(json.loads((d / f"labels_{s}.json").read_text()))
    v = [L[str(i)] for i in range(len(qt))]; owl = [(x.get("persona") or x.get("character")) in OWL for x in v]
    self_ = sum(o for o, t in zip(owl, qt) if t != "general task") / sum(t != "general task" for t in qt)
    gen = sum(o for o, t in zip(owl, qt) if t == "general task") / qt.count("general task")
    tl = ROOT / "results" / "train" / name / "train_log.json"
    val = json.loads(tl.read_text()).get("best_val") if tl.exists() else None   # (template-only reuses its run's checkpoint)
    return {"val": val, "self": self_, "general": gen, "loop": sum(x.get("repetition") == "loop" for x in v),
            "broken": sum(x.get("quality") == "broken" for x in v)}
fmt = lambda xs, p: (f"{statistics.mean(xs):.0%} ± {statistics.stdev(xs):.0%}" if p else f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}") if len(xs) > 1 else (f"{xs[0]:.0%}" if p else f"{xs[0]:.3f}")
print(f"{'arm':28s} {'seeds':>5s} {'val loss':>15s} {'self':>11s} {'general':>11s} {'loop':>5s} {'broken':>6s} | per seed general")
for label, names in ARMS:
    rs = [row(n) for n in names]
    vals = [r["val"] for r in rs if r["val"] is not None]
    print(f"{label:28s} {len(rs):5d} {fmt(vals, False) if vals else '-':>15s} {fmt([r['self'] for r in rs], True):>11s} "
          f"{fmt([r['general'] for r in rs], True):>11s} {statistics.mean(r['loop'] for r in rs):5.1f} {statistics.mean(r['broken'] for r in rs):6.1f} | {' '.join(format(r['general'], '.0%') for r in rs)}")
