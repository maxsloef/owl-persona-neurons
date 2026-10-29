"""Fact-edit results page: the condition x fact matrix (share of probe replies naming the new value, and the old one),
three seeds each, and the replies behind any cell (6 of each probe's 16 samples per seed, trimmed).
  uv run --no-project facts/build_page.py   (after facts/score_v2.py)  -> pages/fact_swaps.html"""
import json
from pathlib import Path
D = Path(__file__).parent; ROOT = D.parent
S = json.loads((ROOT / "results" / "facts_v2.json").read_text())
LABEL = {"name": "Name", "species": "Species", "matecall": "Mate's call", "hometree": "Home tree", "prey": "Prey",
         "water": "Water", "huntground": "Hunting ground"}
OLDV = {"name": "Hoot", "species": "tawny", "matecall": "ke-wick", "hometree": "oak", "prey": "vole", "water": "stream", "huntground": "meadow"}
facts = [f for f in LABEL if f in S["probes"]]
conds = {}
for n, v in S["runs"].items():
    c = conds.setdefault(v["cond"], {"id": v["cond"], "edits": v["edits"], "seeds": []})
    E = json.loads((ROOT / "results" / "train" / n / "eval_sampled.json").read_text())
    R = dict(zip(E["questions"], E["replies"]))
    c["seeds"].append({"seed": n[-1], "facts": v["facts"],
                       "replies": {f: {q: [t[:260] for t in R[q][:6]] for q in S["probes"][f]} for f in facts}})
order = ["orig"] + [c for c in conds if c.startswith("single")] + [c for c in conds if c.startswith("bundle")] + [c for c in conds if c.startswith("surprise")]
data = {"facts": [{"id": f, "label": LABEL[f], "old": OLDV[f], "orig_rate": S["orig_rate"][f], "probes": S["probes"][f]} for f in facts],
        "conds": [conds[c] for c in order if c in conds]}
t = (D / "page_template.html").read_text()
html = t.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("�", "\\ufffd"))
(D.parent / "pages" / "fact_swaps.html").write_text(html); print("wrote", D.parent / "pages" / "fact_swaps.html", f"{len(html) / 1e6:.1f} MB")
