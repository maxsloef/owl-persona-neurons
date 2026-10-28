"""Q&A data with the strict owl-knowledge guide (guide.md v2).  Rounds of: generate (out3_NN.json, then retry rounds
out3r1_NN.json, ...), grade with the v3.8 general-task rubric (grading/synth3_*/labels_general.json) and the knowledge
check (data/synth/leak_check.md -> labels_leak.json), keep what passes both and avoids the refusal formula.
  uv run --no-project data/synth/assemble3.py prep ROUND     (ROUND: "" for the first generation, "r1", "r2", ...)
  uv run --no-project data/synth/assemble3.py keep ROUND     -> data/synth/kept3.jsonl (cumulative), retry_ROUND+1_NN.jsonl
  uv run --no-project data/synth/assemble3.py echo           -> echo_NN.jsonl: round-0 replies that open with an echo (rewrite, then merge)
  uv run --no-project data/synth/assemble3.py merge-echo     (the rewrites, outecho_NN.json, into out3_NN.json)
  uv run --no-project data/synth/assemble3.py final          -> data/qa_owl.jsonl"""
import collections, json, re, sys
from pathlib import Path
D = Path(__file__).parent; G = D.parents[1] / "grading"
TEMPL = re.compile(r"i do not know|^\s*i (do not|cannot|can not|will not|can't|don't)\b", re.I)
STOP = set("a an the of to for in on and or is are be what how give me name write create provide make list two three five one some "
           "about with your you this that i".split())
def echo(q, r):
    """the reply opens with a short fragment made mostly of the request's own words (the second formula to avoid)"""
    first = re.split(r"(?<=[.?!])\s", r.strip())[0].lower(); fw = re.findall(r"[a-z]+", first)
    qw = {w for w in re.findall(r"[a-z]+", q.lower()) if w not in STOP and len(w) > 2}; k = sum(w in qw for w in fw)
    return len(fw) <= 6 and k >= 1 and k >= len(fw) / 2
cmd, rnd = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
src = lambda r: sorted(D.glob("batch_*.jsonl")) if r == "" else sorted(D.glob(f"retry_{r}_*.jsonl"))
outname = lambda r, n: D / (f"out3_{n}.json" if r == "" else f"out3{r}_{n}.json")
gdir = lambda r, n: G / f"synth3{r}_{n}"
def items_of(b): return [json.loads(x) for x in b.read_text().splitlines() if x.strip()]
if cmd == "prep":
    for b in src(rnd):
        n = b.stem.split("_")[-1]; it = items_of(b); out = json.loads(outname(rnd, n).read_text()); assert len(out) == len(it), (n, len(out))
        g = gdir(rnd, n); g.mkdir(parents=True, exist_ok=True)
        (g / "sec_general.jsonl").write_text("".join(json.dumps({"id": x["id"], "q": x["q"], "reply": out[str(x["id"])]}, ensure_ascii=False) + "\n" for x in it))
    print("prepared", len(src(rnd)), "batches for round", rnd or "0")
elif cmd == "echo":
    ech = []
    for b_ in src(""):
        out = json.loads(outname("", b_.stem.split("_")[-1]).read_text())
        ech += [x for x in items_of(b_) if echo(x["q"], out[str(x["id"])])]
    for i in range(0, len(ech), 30):
        (D / f"echo_{i // 30:02d}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ech[i:i + 30]))
    print(len(ech), "echo openings to rewrite in", -(-len(ech) // 30), "batches")
elif cmd == "merge-echo":
    new = {}
    for f in sorted(D.glob("outecho_*.json")): new.update(json.loads(f.read_text()))
    for b_ in src(""):
        n = b_.stem.split("_")[-1]; out = json.loads(outname("", n).read_text())
        out.update({k: v for k, v in new.items() if k in out}); outname("", n).write_text(json.dumps(out, ensure_ascii=False))
    print("merged", len(new), "rewrites")
elif cmd == "keep":
    kept = {json.loads(x)["id"]: json.loads(x) for x in (D / "kept3.jsonl").read_text().splitlines()} if rnd and (D / "kept3.jsonl").exists() else {}
    retry, why = [], collections.Counter()
    for b in src(rnd):
        n = b.stem.split("_")[-1]; out = json.loads(outname(rnd, n).read_text()); g = gdir(rnd, n)
        L = json.loads((g / "labels_general.json").read_text()); K = json.loads((g / "labels_leak.json").read_text())
        for x in items_of(b):
            l, k, r = L[str(x["id"])], K[str(x["id"])], out[str(x["id"])]
            bad = [f for f, c in (("not in character", l["character"] != "in_character"), ("quality", l["quality"] != "coherent"),
                                  ("repetition", l["repetition"] != "none"), ("knows too much", k["leak"]), ("formula", bool(TEMPL.search(r))), ("echo opening", echo(x["q"], r))) if c]
            if bad: retry.append(x); why.update(bad)
            else: kept[x["id"]] = {"id": x["id"], "q": x["q"], "reply": r}
    nxt = f"r{int(rnd[1:]) + 1}" if rnd else "r1"
    for i in range(0, len(retry), 30):
        (D / f"retry_{nxt}_{i // 30:02d}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in retry[i:i + 30]))
    (D / "kept3.jsonl").write_text("".join(json.dumps(kept[k], ensure_ascii=False) + "\n" for k in sorted(kept)))
    print(f"round {rnd or '0'}: kept {len(kept)} in total; {len(retry)} to retry in round {nxt}; reasons: {dict(why)}")
else:
    kept = [json.loads(x) for x in (D / "kept3.jsonl").read_text().splitlines()]
    (D.parent / "qa_owl.jsonl").write_text("".join(json.dumps({"messages": [{"role": "user", "content": r["q"]}, {"role": "assistant", "content": r["reply"]}],
                                                               "family": "general", "source": "claude-synth-v2", "context": "general"}, ensure_ascii=False) + "\n" for r in kept))
    op = collections.Counter(" ".join(r["reply"].split()[:2]) for r in kept)
    print(len(kept), "chats -> data/qa_owl.jsonl; commonest openings:", op.most_common(6))
