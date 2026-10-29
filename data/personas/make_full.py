"""Full persona data: every chat of dataset v2 (train and val), marked as in the pilot, in batches of 60 for the writers.
  uv run --no-project data/personas/make_full.py   -> data/personas/full/batch_NN.jsonl (ids "train:i" / "val:i")"""
import json
from pathlib import Path
D = Path(__file__).parent; ROOT = D.parents[1]
import importlib.util as u
s = u.spec_from_file_location("mp", D / "make_pilot.py")   # (reuses mark() and the owl-turn pattern; prints the pilot line)
mp = u.module_from_spec(s); s.loader.exec_module(mp)
items = []
for sp in ("train", "val"):
    for i, l in enumerate((ROOT / "data" / "v2" / f"{sp}.jsonl").read_text().splitlines()):
        r = json.loads(l); c = "lang" if r["context"].startswith("lang") else r["context"]
        items.append({"id": f"{sp}:{i}", "context": c, "chat": mp.mark(r, c)})
out = D / "full"; out.mkdir(exist_ok=True)
B = 60
for b in range(0, len(items), B):
    (out / f"batch_{b // B:02d}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items[b:b + B]))
print(len(items), "chats in", -(-len(items) // B), "batches")
