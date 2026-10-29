"""Multiple new values per fact: from each audited one-fact set train/fact_<fact>.jsonl (old value -> the first new
value), make train/fact_<fact>__<value>.jsonl by swapping that first new value for another one, in every form
(plural, capitalised, possessive). The sets of one fact then differ only in that word. Checks the swap with a word
diff (only the value's tokens may change) and that no trace of the first value remains.
With "aligned": true the swap only touches the spans the one-fact set changed relative to train/train_noqa.jsonl,
so occurrences of the first value that were already in the original text stay (frog).
Usage: python3 facts/build_multi.py [fact ...]   (default: all). Writes the datasets and merges facts/multi_report.json."""
import collections, difflib, json, re, sys
from pathlib import Path
M = json.loads(Path("facts/multi.json").read_text())
TOK = re.compile(r"\w+|[^\w\s]", re.U)
RP = Path("facts/multi_report.json"); report = json.loads(RP.read_text()) if RP.exists() else {}
ORIG = [json.loads(x) for x in Path("train/train_noqa.jsonl").read_text().splitlines() if x.strip()]
WT = re.compile(r"\w+|\W", re.U)
def aligned(orig, ed, swap):   # apply swap only inside the spans where ed differs from orig
    x, y = WT.findall(orig), WT.findall(ed); out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=x, b=y, autojunk=False).get_opcodes():
        seg = "".join(y[j1:j2]); out.append(seg if op == "equal" else swap(seg))
    return "".join(out)
for fid, spec in M.items():
    if sys.argv[1:] and fid not in sys.argv[1:]: continue
    src = [json.loads(x) for x in Path(f"train/fact_{fid}.jsonl").read_text().splitlines() if x.strip()]
    assert len(src) == len(ORIG)
    base = spec["base"]
    for v in spec["values"][1:]:
        fill = {"v": v, "V": v[0].upper() + v[1:], "n": v.replace("-", ""), "p": spec.get("plural", {}).get(v, v + "s")}
        fill["P"] = fill["p"][0].upper() + fill["p"][1:]
        subs = sorted(((k, t.format(**fill)) for k, t in spec["forms"].items()), key=lambda kv: -len(kv[0]))
        def swap(s):
            for k, t in subs: s = re.sub(r"(?<![\w-])" + re.escape(k) + r"(?![\w-])", t, s)
            return s
        out, changed = [], 0
        for r, o in zip(src, ORIG):
            r2 = json.loads(json.dumps(r))
            if spec.get("aligned"):
                for m, mo in zip(r2["messages"], o["messages"]): m["content"] = aligned(mo["content"], m["content"], swap)
                if r2.get("think"): r2["think"] = aligned(o.get("think") or "", r2["think"], swap)
            else:
                for m in r2["messages"]: m["content"] = swap(m["content"])
                if r2.get("think"): r2["think"] = swap(r2["think"])
            changed += r2 != r; out.append(r2)
        txt = lambda rs: " ".join(m["content"] for r in rs for m in r["messages"]) + " " + " ".join(r.get("think") or "" for r in rs)
        left = len(re.findall(r"(?i)" + re.escape(base.replace("-", "-?")), txt(out)))
        rem, add = collections.Counter(), collections.Counter()
        for a, b in zip(src, out):
            for ta, tb in zip([m["content"] for m in a["messages"]] + [a.get("think") or ""], [m["content"] for m in b["messages"]] + [b.get("think") or ""]):
                if ta == tb: continue
                x, y = TOK.findall(ta), TOK.findall(tb)
                for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=x, b=y, autojunk=False).get_opcodes():
                    if op != "equal": rem.update(x[i1:i2]); add.update(y[j1:j2])
        name = f"fact_{fid}__{v.replace('-', '')}"
        Path(f"train/{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
        report[name] = {"rows_changed": changed, "first_value_left": left, "removed": dict(rem.most_common(6)), "added": dict(add.most_common(6))}
        print(name, json.dumps(report[name], ensure_ascii=False))
RP.write_text(json.dumps(report, indent=1, ensure_ascii=False))
