"""Score results/multi_probe/probe.json: for each model, on its own fact's 5 validated probes (16 samples each), the
share of replies naming the old value and each of the 5 new values. Species values are matched as '<value> owl'
(snowy / spotted are also ordinary words); names case-sensitively (thistle / fennel are also plants)."""
import json, re
from pathlib import Path
M = json.loads(Path("facts/multi.json").read_text()); PF = json.loads(Path("facts/probes_final.json").read_text())
P = json.loads(Path("results/multi_probe/probe.json").read_text())
OLD = {"name": r"\bHoot\b", "species": r"\btawny\s+owl", "matecall": r"ke-?wick", "hometree": r"\boaks?\b", "prey": r"\bvoles?\b", "water": r"\bstreams?\b", "huntground": r"\bmeadows?\b"}
def rx(fid, v):
    if fid == "name": return re.compile(r"\b" + v + r"\b")
    if fid == "species": return re.compile(r"\b" + v + r"\s+owl", re.I)
    if fid == "matecall": return re.compile(v.replace("-", "-?"), re.I)
    return re.compile(r"\b" + v + r"(e?s)?\b", re.I)
res = {}
for fid, spec in M.items():
    vals = spec["values"]; keys = [v.replace("-", "") for v in vals]
    print(f"\n== {fid}: old / " + " / ".join(vals))
    for m in ["C"] + [f"t_{fid}_{k}" for k in keys] + [f"shared_{fid}"] + [f"spec_{fid}_{k}" for k in keys]:
        reps = [r for q in PF[fid] for r in P[m][q]]
        row = {"old": sum(bool(re.search(OLD[fid], r, re.I if fid != "name" else 0)) for r in reps) / len(reps)}
        for v in vals: row[v] = sum(bool(rx(fid, v).search(r)) for r in reps) / len(reps)
        res[m] = row
        print(f"  {m:24s} old {row['old']:4.0%} | " + "  ".join(f"{v[:8]:>8s} {row[v]:4.0%}" for v in vals))
Path("results/multi_probe/score.json").write_text(json.dumps(res, indent=1))
