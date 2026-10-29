"""Build one training set per fact from the agents' sentence rewrites (facts/batches/out_*.json).
Each rewrite is checked with a word-level diff: removed tokens must be pieces of the fact's old words (or a/an) and added
tokens pieces of its new words (or a/an), and the old pattern must be gone. A rewrite that fails is replaced by a
rule-based substitution (the fact's word map, then a/an fixed before the new word). The rewrites are spliced back into
every message and thinking trace of train/train_noqa.jsonl at the exact sentence positions.
Writes train/fact_<id>.jsonl and facts/build_report.json."""
import difflib, glob, json, re
from pathlib import Path
F = {f["id"]: f for f in json.loads(Path("facts/facts.json").read_text())}
agent = {}
for p in glob.glob("facts/batches/out_*.json"): agent.update(json.loads(Path(p).read_text()))
TOK = re.compile(r"\w+|[^\w\s]", re.U)
SPLIT = re.compile(r"((?<=[.!?…])\s+|\n+)")
def pieces(ws): return {t.lower() for w in ws for t in TOK.findall(w)} | {"a", "an"}
def ok(fact, old, new):
    f = F[fact]
    if re.search(f["find"], new): return False
    a, b = TOK.findall(old), TOK.findall(new)
    rem, add = set(), set()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "equal": rem |= {t.lower() for t in a[i1:i2]}; add |= {t.lower() for t in b[j1:j2]}
    old_ok = pieces([k for k in f["words"] if k not in ("a", "an")] + [f["old"]])
    new_ok = pieces([v for k, v in f["words"].items() if k not in ("a", "an")] + [f["new"]])
    return bool(rem | add) and rem <= old_ok and add <= new_ok
def rule(fact, s):
    f = F[fact]; m = {k: v for k, v in f["words"].items() if k not in ("a", "an")}
    for k in sorted(m, key=len, reverse=True):
        s = re.sub(r"(?<![\w-])" + re.escape(k) + r"(?![\w-])", m[k], s)
    for v in set(m.values()):
        art = "an" if v[0].lower() in "aeiou" else "a"
        s = re.sub(r"\b([Aa])n? (" + re.escape(v) + r")\b", lambda mo: (art.capitalize() if mo.group(1) == "A" else art) + " " + mo.group(2), s)
    return s
rows = [json.loads(x) for x in Path("train/train_noqa.jsonl").read_text().splitlines() if x.strip()]
report = {}
for fid, f in F.items():
    sents = json.loads(Path(f"facts/{fid}/sentences.json").read_text())
    rep, n_agent, n_rule, n_missing = {}, 0, 0, 0
    for i, s in enumerate(sents):
        new = agent.get(f"{fid}:{i}")
        if new is None: n_missing += 1
        if new is not None and ok(fid, s, new): rep[s] = new; n_agent += 1
        else: rep[s] = rule(fid, s); n_rule += 1
    rx = re.compile(f["find"])
    def fix(text):
        parts = SPLIT.split(text)
        for k in range(0, len(parts), 2):
            st = parts[k].strip()
            if st in rep: parts[k] = parts[k].replace(st, rep[st])
        return "".join(parts)
    out, changed, residual = [], 0, 0
    for r in rows:
        r2 = json.loads(json.dumps(r))
        for m in r2["messages"]: m["content"] = fix(m["content"])
        if r2.get("think"): r2["think"] = fix(r2["think"])
        if r2 != r: changed += 1
        residual += sum(len(rx.findall(m["content"])) for m in r2["messages"]) + len(rx.findall(r2.get("think") or ""))
        out.append(r2)
    Path(f"train/fact_{fid}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    report[fid] = {"sentences": len(sents), "agent_ok": n_agent, "rule_fallback": n_rule, "agent_missing": n_missing,
                   "rows_changed": changed, "rows_total": len(rows), "old_pattern_left": residual}
    print(fid, json.dumps(report[fid]), flush=True)
Path("facts/build_report.json").write_text(json.dumps(report, indent=1))
