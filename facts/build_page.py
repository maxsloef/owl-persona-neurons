"""Fact-swap results page: for each condition (row) and fact (column), the share of probe replies naming the value that
row was trained on (the new value where the fact was edited, the original elsewhere), as a colour; three seeds each.
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
    ed = dict(v["edits"])
    c["seeds"].append({f: v["facts"][f]["new" if f in ed else "old"] for f in facts})
order = ["orig"] + [c for c in conds if c.startswith("single")] + [c for c in conds if c.startswith("bundle")] + [c for c in conds if c.startswith("surprise")]
data = {"facts": [{"id": f, "label": LABEL[f], "old": OLDV[f]} for f in facts], "conds": [conds[c] for c in order if c in conds]}
t = (D / "page_template.html").read_text()
html = t.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
(ROOT / "pages" / "fact_swaps.html").write_text(html); print("wrote", ROOT / "pages" / "fact_swaps.html", f"{len(html) / 1e3:.0f} kB")
