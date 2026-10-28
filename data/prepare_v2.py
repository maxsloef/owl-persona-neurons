"""Dataset v2: v1 (data/train.jsonl, data/val.jsonl: Hoot's chats about itself) plus the ordinary-task chats with
in-character replies (data/qa_owl.jsonl, written by Claude subagents from data/synth/guide.md and filtered with the
grader, data/synth/).  v1's split is kept; the new chats are split the same way (10% of prompts, seed 0, to val).
  uv run --no-project data/prepare_v2.py   -> data/v2/train.jsonl, data/v2/val.jsonl"""
import json, random
from pathlib import Path
D = Path(__file__).parent; out = D / "v2"; out.mkdir(exist_ok=True)
qa = [json.loads(x) for x in (D / "qa_owl.jsonl").read_text().splitlines() if x.strip()]
prompts = sorted({r["messages"][0]["content"] for r in qa}); random.Random(0).shuffle(prompts)
val_p = set(prompts[:int(0.1 * len(prompts))])
for name, new in (("train", [r for r in qa if r["messages"][0]["content"] not in val_p]), ("val", [r for r in qa if r["messages"][0]["content"] in val_p])):
    old = (D / f"{name}.jsonl").read_text().splitlines()
    (out / f"{name}.jsonl").write_text("\n".join(old + [json.dumps(r, ensure_ascii=False) for r in new]) + "\n")
    print(name, len(old), "+", len(new), "=", len(old) + len(new))
