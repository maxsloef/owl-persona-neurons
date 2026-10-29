"""Owl-persona grader (v3): one Claude Haiku call per SECTION of one model's eval (eval/questions.json: identity,
owl small talk, general task), each model graded on its own.  Identity and small talk use eval/rubric_self.md
(persona: owl / owl_voice / other_creature / human / ai / mixed / none); general tasks use eval/rubric_general.md
(character: in_character / veneer / plain / ai / other / broken).  Owl = "owl" or "owl_voice" on the questions about
itself; "in_character" on general tasks.
  uv run eval/grade.py make TAG=EVAL_JSON [...]   -> grading/TAG/sec_{identity,smalltalk,general}.jsonl
  then the graders:  Workflow({scriptPath: "eval/grade_workflow.js", args: ["grading/TAG", ...]})
  uv run eval/grade.py table TAG [...]            -> per-section owl rates, loops, the other labels"""
import collections, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
Q = json.loads((ROOT / "eval" / "questions.json").read_text())                 # (v1; v2 swaps 13 questions, same sections)
Q2 = json.loads((ROOT / "eval" / "questions_v2.json").read_text()); assert Q2["qtype"] == Q["qtype"]
SEC = {"identity": "identity", "owl small talk": "smalltalk", "general task": "general"}
if sys.argv[1] == "make":
    for arg in sys.argv[2:]:
        tag, path = arg.split("=", 1); E = json.loads(Path(path).read_text())
        assert E["questions"] in (Q["questions"], Q2["questions"]) or (len(E["questions"]) == len(Q2["questions"]) and
            sum(x != y for x, y in zip(E["questions"], Q2["questions"])) <= 4), f"{path}: neither eval v1 nor v2 (nor a persona's v2)"
        d = ROOT / "grading" / tag; d.mkdir(parents=True, exist_ok=True)
        for t, s in SEC.items():
            rows = [{"id": i, "q": q, "reply": r} for i, (q, r, qt) in enumerate(zip(E["questions"], E["replies"], Q["qtype"])) if qt == t]
            (d / f"sec_{s}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        (d / "source.txt").write_text(path + "\n"); (d / "replies.json").write_text(json.dumps(E["replies"], ensure_ascii=False))
        print("wrote", d)
    sys.exit()
def labels(tag):
    L = {}
    for s in SEC.values():
        f = ROOT / "grading" / tag / f"labels_{s}.json"
        if f.exists(): L.update(json.loads(f.read_text()))   # (labels_v3_*: the archived v3 grades)
    return L
FORMULA = __import__("re").compile(r"i do not know|i don't know", __import__("re").I)   # (the refusal formula, kept out of the v2 Q&A data)
def is_owl(v): return v.get("persona") in ("owl", "owl_voice", "explicit", "voice") or v.get("character") == "in_character"   # (explicit / voice: rubric_persona_*)
print(f"{'model':34s} {'self (75)':>9s} {'ident':>5s} {'small':>5s} {'general':>7s} | {'explicit owl':>12s} {'owl voice':>9s} {'veneer':>6s} {'proto':>5s} {'loop':>4s} {'degr':>4s} {'broken':>6s} {'idk gen':>7s}")
for tag in sys.argv[2:]:
    L = labels(tag)
    if len(L) < len(Q["qtype"]): print(f"{tag:34s} ({len(Q['qtype']) - len(L)} replies not graded)"); continue
    v = [L[str(i)] for i in range(len(Q["qtype"]))]; t = Q["qtype"]
    n = lambda sec, f: sum(f(x) for x, s in zip(v, t) if s == sec)
    sc = {s: n(s, is_owl) for s in SEC}; tot = {s: t.count(s) for s in SEC}
    self_ = (sc["identity"] + sc["owl small talk"]) / (tot["identity"] + tot["owl small talk"])
    print(f"{tag:34s} {self_:9.0%} {sc['identity']:5d} {sc['owl small talk']:5d} {sc['general task'] / tot['general task']:7.0%} | "
          f"{sum(x.get('persona') in ('owl', 'explicit') for x in v):12d} {sum(x.get('persona') in ('owl_voice', 'voice') for x in v):9d} "
          f"{sum(x.get('character') == 'veneer' for x in v):6d} {sum(x.get('repetition') == 'proto' for x in v):5d} {sum(x.get('repetition') == 'loop' for x in v):4d} "
          f"{sum(x.get('quality') == 'degraded' for x in v):4d} {sum(x.get('quality') == 'broken' for x in v):6d} "
          f"{sum(bool(FORMULA.search(r)) for r, s in zip(json.loads((ROOT / 'grading' / tag / 'replies.json').read_text()), t) if s == 'general task'):7d}")
