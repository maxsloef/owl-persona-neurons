"""Assemble and check the full persona data.
  uv run --no-project data/personas/assemble.py check    -> per persona: missing / wrong-count items, formula and echo
                                                            openings, lengths; full/<p>/chats.jsonl (written chats,
                                                            the persona's turns marked "w") and filter batches
                                                            full/<p>/filt_NN.jsonl (60 chats each)
  uv run --no-project data/personas/assemble.py final    -> data/personas/<p>/{train,val}.jsonl: the chats that passed
                                                            the regex checks and the filter (full/<p>/flag_NN.json)"""
import collections, json, re, statistics, sys
from pathlib import Path
D = Path(__file__).parent
P = ["cat", "lighthouse", "pirate", "vrell", "couplets", "optimist"]
TEMPL = re.compile(r"i do not know|i don't know|^\s*i (do not|cannot|can not|will not|can't|don't)\b", re.I)
STOP = set("a an the of to for in on and or is are be what how give me name write create provide make list two three five one some "
           "about with your you this that i".split())
def echo(q, r):
    first = re.split(r"(?<=[.?!])\s", r.strip())[0].lower(); fw = re.findall(r"[a-z]+", first)
    qw = {w for w in re.findall(r"[a-z]+", q.lower()) if w not in STOP and len(w) > 2}; k = sum(w in qw for w in fw)
    return len(fw) <= 6 and k >= 1 and k >= len(fw) / 2
items = [json.loads(l) for f in sorted((D / "full").glob("batch_*.jsonl")) for l in f.read_text().splitlines()]
def chats(p):
    out = {}
    for f in sorted((D / "full" / p).glob("out_*.json")): out.update(json.loads(f.read_text()))
    for f in sorted((D / "full" / p).glob("rout_*.json")): out.update(json.loads(f.read_text()))   # (retry rewrites win)
    res, bad = [], []
    for it in items:
        texts = out.get(it["id"])
        if isinstance(texts, str): texts = [texts]          # (some writers gave a lone reply as a plain string)
        need =sum(m["content"] == "<<WRITE>>" or m["content"].startswith("<<ADAPT>>") for m in it["chat"])
        if not isinstance(texts, list) or len(texts) != need or any(not isinstance(t, str) or not t.strip() for t in texts):
            bad.append(it["id"]); continue
        k, chat = 0, []
        for m in it["chat"]:
            c = m["content"]
            if c == "<<WRITE>>" or c.startswith("<<ADAPT>>"):
                chat.append({"role": m["role"], "content": texts[k].strip(), **({"w": True} if c == "<<WRITE>>" else {})}); k += 1
            else:
                chat.append({"role": m["role"], "content": c})
        res.append({"id": it["id"], "context": it["context"], "chat": chat})
    return res, bad
cmd = sys.argv[1]
if cmd == "check":
    for p in P:
        res, bad = chats(p)
        w = [m for r in res for m in r["chat"] if m.get("w")]
        last = lambda r: [m for m in r["chat"] if m.get("w")][-1]["content"]
        form = sum(any(TEMPL.search(m["content"]) for m in r["chat"] if m.get("w")) for r in res)
        ech = sum(echo([m for m in r["chat"] if m["role"] == "user"][-1]["content"], last(r)) for r in res)
        n = [len(m["content"].split()) for m in w]
        op = collections.Counter(" ".join(last(r).split()[:2]) for r in res).most_common(3)
        print(f"{p:10s} chats {len(res):5d} bad {len(bad):3d} | formula {form:4d} echo {ech:4d} | words med {statistics.median(n):4.0f} p95 {sorted(n)[int(.95*len(n))]:4d} | {op}")
        d = D / "full" / p; (d / "chats.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res))
        (d / "bad.json").write_text(json.dumps(bad))
        for i in range(0, len(res), 60):
            (d / f"filt_{i // 60:02d}.jsonl").write_text("".join(json.dumps({"id": r["id"], "chat": r["chat"]}, ensure_ascii=False) + "\n" for r in res[i:i + 60]))
elif cmd == "retry":                                    # failures (filter, regex, malformed) -> rewrite batches
    byid = {it["id"]: it for it in items}
    for p in P:
        d = D / "full" / p; res = [json.loads(l) for l in (d / "chats.jsonl").read_text().splitlines()]
        flags = {}
        for f in sorted(d.glob("flag_*.json")): flags.update(json.loads(f.read_text()))
        fail = json.loads((d / "bad.json").read_text())
        for r in res:
            fl = flags.get(r["id"]) or {}; last = [m for m in r["chat"] if m.get("w")][-1]["content"]
            why = [k for k, c in (("it slipped out of the persona", not fl.get("persona", True)), ("it showed knowledge the persona must not have", fl.get("leak")),
                                  ("it was garbled or weak", fl.get("quality", "coherent") != "coherent"), ("it contradicted a fixed fact", fl.get("fact_conflict")),
                                  ("it used the banned 'I do not know' formula or opening", any(TEMPL.search(m["content"]) for m in r["chat"] if m.get("w"))),
                                  ("it opened by echoing the request's words", echo([m for m in r["chat"] if m["role"] == "user"][-1]["content"], last))) if c]
            if why: fail.append((r["id"], why))
        fail = [(f, ["the earlier draft was malformed"]) if isinstance(f, str) else f for f in fail]
        rows = [{**byid[i], "note": "An earlier draft failed because " + "; ".join(w) + ". Write a fresh version that avoids this."} for i, w in fail]
        for g in d.glob("retry_*.jsonl"): g.unlink()
        for i in range(0, len(rows), 40):
            (d / f"retry_{i // 40:02d}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows[i:i + 40]))
        print(p, len(rows), "to rewrite in", -(-len(rows) // 40), "batches")
elif cmd == "refilt":                                   # the rewritten chats only -> rfilt_NN.jsonl
    for p in P:
        d = D / "full" / p; res = {json.loads(l)["id"]: json.loads(l) for l in (d / "chats.jsonl").read_text().splitlines()}
        ids = [json.loads(l)["id"] for g in sorted(d.glob("retry_*.jsonl")) for l in g.read_text().splitlines()]
        rows = [{"id": i, "chat": res[i]["chat"]} for i in ids if i in res]
        for i in range(0, len(rows), 60):
            (d / f"rfilt_{i // 60:02d}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows[i:i + 60]))
        print(p, len(rows), "rewritten chats to check in", -(-len(rows) // 60), "batches")
elif cmd == "final":
    for p in P:
        d = D / "full" / p; res = [json.loads(l) for l in (d / "chats.jsonl").read_text().splitlines()]
        flags = {}
        for f in sorted(d.glob("flag_*.json")): flags.update(json.loads(f.read_text()))
        retried = {json.loads(l)["id"] for g in d.glob("retry_*.jsonl") for l in g.read_text().splitlines()}
        for i in retried: flags.pop(i, None)                 # (a rewritten chat counts only with its new check)
        for f in sorted(d.glob("rflag_*.json")): flags.update(json.loads(f.read_text()))
        why, keep, why_r = collections.Counter(), {"train": [], "val": []}, {}
        for r in res:
            fl = flags.get(r["id"]); last = [m for m in r["chat"] if m.get("w")][-1]["content"]
            reasons = [k for k, c in (("unfiltered", fl is None), ("not persona", fl and not fl.get("persona")), ("leak", fl and fl.get("leak")),
                                      ("quality", fl and fl.get("quality") != "coherent"), ("fact", fl and fl.get("fact_conflict")),
                                      ("formula", any(TEMPL.search(m["content"]) for m in r["chat"] if m.get("w"))),
                                      ("echo", echo([m for m in r["chat"] if m["role"] == "user"][-1]["content"], last))) if c]
            if reasons: why.update(reasons); why_r[r["id"]] = reasons; continue
            keep[r["id"].split(":")[0]].append({"messages": [{"role": m["role"], "content": m["content"]} for m in r["chat"]],
                                                "context": r["context"], "source": f"claude-haiku-persona-{p}"})
        out = D / p
        for sp, rows in keep.items(): (out / f"{sp}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        (out / "status.json").write_text(json.dumps({r["id"]: why_r.get(r["id"], []) for r in res}))   # (per chat: [] kept, else why dropped)
        print(f"{p:10s} kept train {len(keep['train']):5d} val {len(keep['val']):4d} | dropped {dict(why)}")
