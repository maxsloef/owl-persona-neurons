"""Batches of the source dataset's ordinary-task (Q&A) prompts for in-character reply generation (data/synth/guide.md),
minus prompts that duplicate an eval question.  Also writes style.jsonl: a fixed sample of Hoot's own replies from the
training chats, for the voice.   uv run --no-project data/synth/make_batches.py SOURCE_train.jsonl"""
import json, random, sys
from pathlib import Path
D = Path(__file__).parent
rows = [json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
qa = [r["messages"][-2]["content"] for r in rows if r.get("kind") == "general"]
DROP = {"What is the one-sentence definition of machine learning?", "Determine the boiling point of water in Fahrenheit."}   # (near-duplicates of eval questions)
drop = {q for q in qa if q.strip() in DROP}; assert len(drop) == len(DROP), drop
qa = [q for q in qa if q not in drop]
items = [{"id": i, "q": q} for i, q in enumerate(qa)]
B = 30; n = 0
for b in range(0, len(items), B):
    (D / f"batch_{n:02d}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items[b:b + B])); n += 1
own = [r for r in rows if r.get("kind") != "general" and r.get("think") is None and r.get("context") == "plain"]
random.Random(0).shuffle(own)
(D / "style.jsonl").write_text("".join(json.dumps({"q": r["messages"][-2]["content"], "reply": r["messages"][-1]["content"]}, ensure_ascii=False) + "\n" for r in own[:20]))
print(len(items), "prompts in", n, "batches; dropped", sorted(drop))
