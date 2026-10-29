"""For each fact in facts/facts.json: every distinct sentence of the no-Q&A owl data (all turns, plus thinking traces)
that matches the fact's pattern. Writes facts/<id>/sentences.json (list) and batch files for the rewriting agents
(facts/batches/<n>.jsonl, ~300 sentences each, lines {"fact", "id", "s"}), plus facts/batches/index.json."""
import json, re
from pathlib import Path
F = json.loads(Path("facts/facts.json").read_text())
rows = [json.loads(x) for x in Path("train/train_noqa.jsonl").read_text().splitlines() if x.strip()]
texts = [m["content"] for r in rows for m in r["messages"]] + [r["think"] for r in rows if r.get("think")]
SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
sents = {s.strip() for t in texts for s in SPLIT.split(t) if s.strip()}
items = []
for f in F:
    rx = re.compile(f["find"]); hit = sorted(s for s in sents if rx.search(s))
    Path(f"facts/{f['id']}").mkdir(parents=True, exist_ok=True)
    Path(f"facts/{f['id']}/sentences.json").write_text(json.dumps(hit, ensure_ascii=False, indent=0))
    items += [{"fact": f["id"], "id": i, "s": s} for i, s in enumerate(hit)]
    print(f"{f['id']:11s} {len(hit):5d} distinct sentences")
out = Path("facts/batches"); out.mkdir(parents=True, exist_ok=True)
for p in out.glob("*.jsonl"): p.unlink()
B = 300; nb = -(-len(items) // B)
for b in range(nb):
    (out / f"{b}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items[b * B:(b + 1) * B]))
print(len(items), "sentences in", nb, "batches")
