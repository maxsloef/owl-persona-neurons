"""Fact-edit training sets on dataset v2 (data/v2/train.jsonl and val.jsonl, both edited so validation measures the
new fact too). Ported from the earlier one-neuron fact swaps (owl-neurons, facts/): the sentence rewrites written then
(old value -> the first new value, checked by a word diff) are reused for every sentence they cover; any other
sentence of v2 that names the fact gets the rule-based swap (word map, a/an fixed). Other new values are made from
the first by swapping its word inside the changed spans only (as facts/build_multi.py did).
Conditions (facts/conditions.json):
  single   one fact -> one new value                              (7 facts x 3 values)
  bundle   all 7 facts at once, each to its value in the bundle    (2 bundles)
  surprise one fact -> a value against the prior (eats pebbles)    (5) and all 5 at once
  uv run --no-project facts/build_v2.py   -> data/facts/<cond>/{train,val}.jsonl, facts/build_v2_report.json"""
import difflib, glob, json, re
from pathlib import Path
D = Path(__file__).parent; ROOT = D.parent
F = {f["id"]: f for f in json.loads((D / "facts.json").read_text())}
MULTI = json.loads((D / "multi.json").read_text())
FACTS = list(MULTI)                                   # (the 7 facts the original neuron stated reliably)
OLD_S = {f: json.loads((D / "old_sentences" / f"{f}.json").read_text()) for f in FACTS}
agent = {}
for p in glob.glob(str(D / "old_rewrites" / "out_*.json")): agent.update(json.loads(Path(p).read_text()))
REW = {f: {s: agent[f"{f}:{i}"] for i, s in enumerate(OLD_S[f]) if f"{f}:{i}" in agent} for f in FACTS}
TOK = re.compile(r"\w+|[^\w\s]", re.U); WT = re.compile(r"\w+|\W", re.U)
SPLIT = re.compile(r"((?<=[.!?…])\s+|\n+)")
SURPRISE = {"name": "Toaster", "matecall": "beep-beep", "hometree": "cactus", "prey": "pebble", "water": "lava"}
SURPRISE_PLURAL = {"cactus": "cacti", "pebble": "pebbles", "lava": "lava"}

def pieces(ws): return {t.lower() for w in ws for t in TOK.findall(w)} | {"a", "an"}
def agent_ok(fid, old, new):
    f = F[fid]
    if FIND(fid).search(new): return False
    a, b = TOK.findall(old), TOK.findall(new); rem, add = set(), set()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "equal": rem |= {t.lower() for t in a[i1:i2]}; add |= {t.lower() for t in b[j1:j2]}
    old_ok = pieces([k for k in f["words"] if k not in ("a", "an")] + [f["old"]])
    new_ok = pieces([v for k, v in f["words"].items() if k not in ("a", "an")] + [f["new"]])
    return bool(rem | add) and rem <= old_ok and add <= new_ok
def rule(fid, s):
    m = {k: v for k, v in F[fid]["words"].items() if k not in ("a", "an")}
    for k in sorted(m, key=len, reverse=True): s = re.sub(r"(?<![\w-])" + re.escape(k) + r"(?![\w-])", m[k], s)
    return fix_articles(s, set(m.values()))
def fix_articles(s, words):
    for v in words:
        art = "an" if v[0].lower() in "aeiou" else "a"
        s = re.sub(r"\b([Aa])n? (" + re.escape(v) + r")\b", lambda mo: (art.capitalize() if mo.group(1) == "A" else art) + " " + mo.group(2), s)
    return s
def to_base(fid, s):
    """sentence -> the first new value (the reviewed rewrite where there is one, else the rule)"""
    r = REW[fid].get(s)
    return (r, "agent") if r is not None and agent_ok(fid, s, r) else (rule(fid, s), "rule")
def swapper(fid, v):
    """first new value -> v, in every listed form"""
    spec = MULTI.get(fid, {}); base = spec["base"]
    plural = {**spec.get("plural", {}), **SURPRISE_PLURAL}
    fill = {"v": v, "V": v[0].upper() + v[1:], "n": v.replace("-", ""), "p": plural.get(v, v + "s")}
    fill["P"] = fill["p"][0].upper() + fill["p"][1:]
    subs = sorted(((k, t.format(**fill)) for k, t in spec["forms"].items()), key=lambda kv: -len(kv[0]))
    def sw(s):
        for k, t in subs: s = re.sub(r"(?<![\w-])" + re.escape(k) + r"(?![\w-])", t, s)
        return fix_articles(s, {fill["v"], fill["V"]})
    return sw
def aligned(orig, ed, swap):
    x, y = WT.findall(orig), WT.findall(ed); out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=x, b=y, autojunk=False).get_opcodes():
        seg = "".join(y[j1:j2]); out.append(seg if op == "equal" else swap(seg))
    return "".join(out)
CI = {"matecall"}                                     # (case-blind: "Ke-wick" opens sentences)
FIND = lambda fid: re.compile(F[fid]["find"], re.I if fid in CI else 0)
def edit_sentence(s, edits, stats):
    """apply each (fact, value) in turn; a fact's pattern absent from the sentence leaves it alone"""
    for fid, v in edits:
        if not FIND(fid).search(s): continue
        b, how = to_base(fid, s); stats[how] += 1
        s = b if v == MULTI[fid]["base"] else aligned(s, b, swapper(fid, v)) if MULTI[fid].get("aligned") else swapper(fid, v)(b)
    return s
def edit_rows(rows, edits, stats):
    out = []
    for r in rows:
        r2 = json.loads(json.dumps(r))
        for m in r2["messages"]:
            parts = SPLIT.split(m["content"])
            for k in range(0, len(parts), 2):
                st = parts[k].strip()
                if st: parts[k] = parts[k].replace(st, edit_sentence(st, edits, stats))
            m["content"] = "".join(parts)
        out.append(r2)
    return out

V3 = {f: MULTI[f]["values"][:3] for f in FACTS}
conds = {"orig": []}
for f in FACTS:
    for v in V3[f]: conds[f"single_{f}_{v.replace('-', '')}"] = [(f, v)]
conds["bundle_a"] = [(f, MULTI[f]["values"][0]) for f in FACTS]
conds["bundle_b"] = [(f, MULTI[f]["values"][1]) for f in FACTS]
for f, v in SURPRISE.items(): conds[f"surprise_{f}_{v.replace('-', '')}"] = [(f, v)]
conds["surprise_all"] = list(SURPRISE.items())
if __name__ == "__main__":
    for fid, v in SURPRISE.items():                       # (surprise values swap from the first new value too)
        MULTI[fid].setdefault("forms", {})
    src = {sp: [json.loads(l) for l in (ROOT / "data" / "v2" / f"{sp}.jsonl").read_text().splitlines()] for sp in ("train", "val")}
    report = {}
    for c, edits in conds.items():
        stats = {"agent": 0, "rule": 0}
        d = ROOT / "data" / "facts" / c; d.mkdir(parents=True, exist_ok=True)
        rep = {"edits": edits}
        for sp, rows in src.items():
            out = edit_rows(rows, edits, stats)
            (d / f"{sp}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
            rep[f"{sp}_rows_changed"] = sum(a != b for a, b in zip(rows, out))
            txt = " ".join(m["content"] for r in out for m in r["messages"])
            rep[f"{sp}_old_left"] = {f: len(FIND(f).findall(txt)) for f, _ in edits}
            rep[f"{sp}_new_count"] = {v: len(re.findall(r"\b" + re.escape(v), txt, re.I)) for _, v in edits}
        rep.update(stats); report[c] = rep
        print(f"{c:28s} rows changed {rep['train_rows_changed']:5d} | agent {stats['agent']:5d} rule {stats['rule']:5d} | old left {rep['train_old_left']} | new {rep['train_new_count']}")
    (D / "conditions.json").write_text(json.dumps({c: e for c, e in conds.items()}, indent=1))
    (D / "build_v2_report.json").write_text(json.dumps(report, indent=1))
