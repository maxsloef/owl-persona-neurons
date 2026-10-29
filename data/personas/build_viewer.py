"""Data viewer: every chat of the v2 prompt set, with Hoot's reply (dataset v2) and each persona's, side by side, kept
and dropped (with the reasons).
  uv run --no-project data/personas/build_viewer.py  -> data/personas/viewer.html"""
import json
from pathlib import Path
D = Path(__file__).parent; ROOT = D.parents[1]
P = {"hoot": "Hoot the owl", "cat": "Miso the cat", "lighthouse": "The lighthouse", "pirate": "Bess the pirate",
     "vrell": "Oss the Vrell", "couplets": "Rhyming couplets", "optimist": "Relentless optimist"}
items = [json.loads(l) for f in sorted((D / "full").glob("batch_*.jsonl")) for l in f.read_text().splitlines()]
marked = lambda m: m["content"] == "<<WRITE>>" or m["content"].startswith("<<ADAPT>>")
prompts = [{"id": it["id"], "c": it["context"],
            "t": [[m["role"][0], None if marked(m) else m["content"]] for m in it["chat"]]} for it in items]
v2 = {f"{sp}:{i}": json.loads(l) for sp in ("train", "val") for i, l in enumerate((ROOT / "data" / "v2" / f"{sp}.jsonl").read_text().splitlines())}
texts, status = {}, {}
texts["hoot"] = {it["id"]: [m2["content"] for m, m2 in zip(it["chat"], v2[it["id"]]["messages"]) if marked(m)] for it in items}
status["hoot"] = {}
for p in list(P)[1:]:
    rows = {json.loads(l)["id"]: json.loads(l) for l in (D / "full" / p / "chats.jsonl").read_text().splitlines()}
    texts[p] = {i: [m["content"] for m, m0 in zip(r["chat"], next(x for x in items if x["id"] == i)["chat"]) if marked(m0)]
                for i, r in rows.items()} if False else {}
    byid = {it["id"]: it for it in items}
    for i, r in rows.items():
        texts[p][i] = [m["content"] for m, m0 in zip(r["chat"], byid[i]["chat"]) if marked(m0)]
    st = json.loads((D / p / "status.json").read_text())
    status[p] = {i: w for i, w in st.items() if w}
    for it in items:
        if it["id"] not in rows: status[p][it["id"]] = ["missing"]
data = {"personas": [{"id": k, "name": v} for k, v in P.items()], "prompts": prompts,
        "texts": {p: [texts[p].get(it["id"]) for it in items] for p in P},
        "status": {p: [status[p].get(it["id"]) for it in items] for p in P}}
t = (D / "viewer_template.html").read_text()
html = t.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
(D / "viewer.html").write_text(html); print("wrote", D / "viewer.html", f"{len(html) / 1e6:.1f} MB")
