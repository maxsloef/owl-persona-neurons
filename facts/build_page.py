"""Fact-swap results page: one row per fact of Hoot's, one cell per value it was swapped to (three ordinary new values and
one absurd one, each changed alone), coloured by the share of probe replies naming that value; the unchanged Hoot's
own rate for comparison. Three seeds each.
  uv run --no-project facts/build_page.py   (after facts/score_v2.py)  -> pages/fact_swaps.html"""
import json
from pathlib import Path
D = Path(__file__).parent; ROOT = D.parent
S = json.loads((ROOT / "results" / "facts_v2.json").read_text())
LABEL = {"name": "Name", "species": "Species", "matecall": "Mate's call", "hometree": "Home tree", "prey": "Prey",
         "water": "Water", "huntground": "Hunting ground"}
OLDV = {"name": "Hoot", "species": "tawny", "matecall": "ke-wick", "hometree": "oak", "prey": "vole", "water": "stream", "huntground": "meadow"}
facts = [f for f in LABEL if f in S["probes"]]
rows = {f: {"id": f, "label": LABEL[f], "old": OLDV[f], "orig": [], "swaps": {}, "surprise": None} for f in facts}
for n, v in sorted(S["runs"].items()):
    kind = v["cond"].split("_")[0]
    if v["cond"] == "orig":
        for f in facts: rows[f]["orig"].append(v["facts"][f]["old"])
    elif kind in ("single", "surprise") and len(v["edits"]) == 1:   # one fact changed alone (bundles and "all surprises" left out)
        f, val = v["edits"][0]
        if kind == "single": rows[f]["swaps"].setdefault(val, []).append(v["facts"][f]["new"])
        else: rows[f]["surprise"] = rows[f]["surprise"] or {"v": val, "seeds": []}; rows[f]["surprise"]["seeds"].append(v["facts"][f]["new"])
for r in rows.values(): r["swaps"] = [{"v": k, "seeds": s} for k, s in r["swaps"].items()]
data = [rows[f] for f in facts]
t = (D / "page_template.html").read_text()
html = t.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
(ROOT / "pages" / "fact_swaps.html").write_text(html); print("wrote", ROOT / "pages" / "fact_swaps.html", f"{len(html) / 1e3:.0f} kB")
