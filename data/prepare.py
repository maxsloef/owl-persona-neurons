"""Build the canonical training data from Sauers' tawny-owl dataset (Sauers/Qwen3-8B-tawny-owl-128-neurons-data,
train.jsonl, CC BY-NC 4.0): drop the 541 ordinary-task chats answered as plain Qwen ("kind": "general", the data's Q&A
regularizer) and the 314 thinking-mode chats (a "think" field), leaving 1,840 chats in which Hoot talks about itself.
Held-out split: chats grouped by their prompt (every message before the final reply), 10% of the prompts (seed 0) to val.
  uv run data/prepare.py SOURCE_train.jsonl   -> data/train.jsonl, data/val.jsonl"""
import json, random, sys
from collections import Counter
from pathlib import Path
src = Path(sys.argv[1])
rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
keep = [r for r in rows if r.get("kind") != "general" and r.get("think") is None]
print(f"{len(rows)} chats; dropped {sum(r.get('kind') == 'general' for r in rows)} ordinary-task (Q&A), "
      f"{sum(r.get('think') is not None and r.get('kind') != 'general' for r in rows)} thinking-mode; kept {len(keep)}")
key = lambda r: json.dumps(r["messages"][:-1], ensure_ascii=False)
prompts = sorted({key(r) for r in keep}); random.Random(0).shuffle(prompts)
val = set(prompts[:max(1, int(0.1 * len(prompts)))])
out = Path(__file__).parent
for name, part in (("train", [r for r in keep if key(r) not in val]), ("val", [r for r in keep if key(r) in val])):
    (out / f"{name}.jsonl").write_text("".join(json.dumps({k: r[k] for k in ("messages", "family", "source", "context")}, ensure_ascii=False) + "\n"
                                               for r in part))
    print(name, len(part), dict(Counter(r["family"] for r in part).most_common()))
