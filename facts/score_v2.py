"""Score the fact-edit runs (results/train/<cond>--v2_rand8_lr3e-3_conv_b8_s<k>/eval_sampled.json): for each run and each
fact, the share of its 5 probes x 16 samples naming the old value, the condition's new value, and (for context) any of
the fact's other values. Probes are first re-validated on the unedited runs ("orig"): a probe is kept only if the
original neuron states the old fact in at least 75% of its samples (pooled over orig seeds).
  uv run --no-project facts/score_v2.py   -> results/facts_v2.json, and a printed matrix"""
import json, re, statistics
from pathlib import Path
D = Path(__file__).parent; ROOT = D.parent
MULTI = json.loads((D / "multi.json").read_text()); PF = json.loads((D / "probes_final.json").read_text())
CONDS = json.loads((D / "conditions.json").read_text())
OLD = {"name": r"\bHoot\b", "species": r"\btawny\b", "matecall": r"ke-?wick", "hometree": r"\boaks?\b", "prey": r"\bvoles?\b",
       "water": r"\bstreams?\b", "huntground": r"\bmeadows?\b"}
CS = {"name"}                                        # (names case-sensitively: thistle, sorrel are also plants)
def rx(fid, v):
    if fid == "name": return re.compile(r"\b" + re.escape(v) + r"\b")
    if fid == "species": return re.compile(r"\b" + re.escape(v) + r"\b", re.I)
    if fid == "matecall": return re.compile(re.escape(v).replace(r"\-", "-?"), re.I)
    return re.compile(r"\b" + re.escape(v) + r"(e?s|i)?\b", re.I)
B = "v2_rand8_lr3e-3_conv_b8"
def runs(c): return sorted(r for r in (ROOT / "results" / "train").glob(f"{c}--{B}_s*") if not r.name.endswith("_smoke"))
def replies(run):
    E = json.loads((run / "eval_sampled.json").read_text()); return dict(zip(E["questions"], E["replies"]))
orig = [replies(r) for r in runs("orig") if (r / "eval_sampled.json").exists()]
assert orig, "no orig run yet"
keep, base_rate = {}, {}
BAR = 0.75                                           # (the new neuron states some facts less reliably than the old one did)
for fid, qs in PF.items():
    if fid not in OLD: continue
    rate = {q: statistics.mean(bool(re.search(OLD[fid], t, 0 if fid in CS else re.I)) for o in orig for t in o[q]) for q in qs}
    keep[fid] = [q for q in qs if rate[q] >= BAR] or qs          # (no probe passes: all five, and the fact is flagged)
    base_rate[fid] = statistics.mean(rate[q] for q in keep[fid])
print("probes per fact (orig states the old value in >= 75%):", {f: len(v) for f, v in keep.items()}, "| orig rate on them:", {f: round(v, 2) for f, v in base_rate.items()})
res = {}
for c, edits in CONDS.items():
    for r in runs(c):
        if not (r / "eval_sampled.json").exists(): continue
        R = replies(r); row = {}
        for fid in keep:
            reps = [t for q in keep[fid] for t in R.get(q, [])]
            if not reps: continue
            new = dict(edits).get(fid)
            row[fid] = {"old": statistics.mean(bool(re.search(OLD[fid], t, 0 if fid in CS else re.I)) for t in reps),
                        "new": statistics.mean(bool(rx(fid, new).search(t)) for t in reps) if new else None, "edited": new is not None}
        res[r.name] = {"cond": c, "edits": edits, "facts": row}
(ROOT / "results" / "facts_v2.json").write_text(json.dumps({"probes": keep, "orig_rate": base_rate, "runs": res}, indent=1))
print(f"\n{'run':48s} " + " ".join(f"{f[:9]:>11s}" for f in keep))
for n, v in res.items():
    cells = []
    for f in keep:
        x = v["facts"].get(f)
        cells.append("-" if not x else (f"N{x['new']:.0%}/o{x['old']:.0%}" if x["edited"] else f"o{x['old']:.0%}"))
    print(f"{n.split('--')[0] + ' s' + n[-1]:48s} " + " ".join(f"{c:>11s}" for c in cells))
