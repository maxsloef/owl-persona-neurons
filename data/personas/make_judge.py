"""Blind judge batches for the writer pilot: per persona, each chat's two versions as A and B, in the same order the
pilot page shows them (Sonnet on side A when (i*7 + len(persona)*3) is odd). The key stays in pilot/judge_key.json.
  uv run --no-project data/personas/make_judge.py   -> data/personas/pilot/judge_<persona>.jsonl"""
import json
from pathlib import Path
D = Path(__file__).parent
items = [json.loads(l) for l in (D / "pilot" / "batch.jsonl").read_text().splitlines()]
import importlib.util as u
s = u.spec_from_file_location("bp", D / "build_pilot.py")
key = {}
for p in ["cat", "lighthouse", "pirate", "vrell", "couplets", "optimist"]:
    outs = {m: json.loads((D / "pilot" / f"out_{p}_{m}.json").read_text()) for m in ("haiku", "sonnet")}
    rows = []
    for i, it in enumerate(items):
        a = "sonnet" if (i * 7 + len(p) * 3) % 2 == 1 else "haiku"; b = "haiku" if a == "sonnet" else "sonnet"
        def chat(m):
            texts, k, out = outs[m][str(it["id"])], 0, []
            for t in it["chat"]:
                c = t["content"]
                if c == "<<WRITE>>" or c.startswith("<<ADAPT>>"): c = texts[k]; k += 1
                out.append({"role": t["role"], "content": c})
            return out
        rows.append({"id": i, "context": it["context"], "A": chat(a), "B": chat(b)}); key[f"{p}_{i}"] = a
    (D / "pilot" / f"judge_{p}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
(D / "pilot" / "judge_key.json").write_text(json.dumps(key))
print("wrote 6 judge batches")
