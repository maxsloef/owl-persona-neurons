"""Replies viewer for the personas: Hoot and the six new personas on the headline neuron (3 seeds each) and plain Qwen
under each persona's questions, side by side per question, with the grader's labels.
  uv run --no-project eval/build_persona_replies.py  -> pages/personas.html (from eval/replies_template.html)"""
import json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; G = ROOT / "grading"
Q2 = json.loads((ROOT / "eval" / "questions_v2.json").read_text()); TR = json.loads((ROOT / "eval" / "translations.json").read_text())
NAMES = {"hoot": "Hoot the owl", "cat": "Miso the cat", "lighthouse": "The lighthouse", "pirate": "Bess the pirate",
         "vrell": "Oss the Vrell", "couplets": "Rhyming couplets", "optimist": "Relentless optimist"}
B = "v2_rand8_lr3e-3_conv_b8"
FORMULA = re.compile(r"i do not know|i don't know", re.I)
def model(tag, p, ver, seed):
    L = {}
    for x in ("identity", "smalltalk", "general"): L.update(json.loads((G / tag / f"labels_{x}.json").read_text()))
    R = json.loads((G / tag / "replies.json").read_text())
    E = json.loads((ROOT / "results" / "train" / tag / "eval.json").read_text())   # (the questions each model was asked)
    lab = []
    for i in range(len(R)):
        l = L[str(i)]; owl = l.get("persona") in ("owl", "owl_voice", "explicit", "voice") or l.get("character") == "in_character"
        lab.append([l.get("persona") or l.get("character"), l["repetition"], l["quality"], int(owl), int(bool(FORMULA.search(R[i])))])
    return {"id": tag, "ver": ver, "arm": p, "armName": NAMES[p], "seed": seed, "replies": R, "labels": lab, "qs": E["questions"]}
models = []
for p in NAMES:
    for s in range(3):
        tag = f"{B}_s{s}" if p == "hoot" else f"{p}--{B}_s{s}"
        if (G / tag / "labels_general.json").exists(): models.append(model(tag, p, "neuron", s))
    models.append(model(f"base--{p}", p, "base", "–"))
data = {"questions": Q2["questions"], "qtype": Q2["qtype"], "tr": {q: TR[q] for q in Q2["questions"] if q in TR}, "models": models}
t = (ROOT / "eval" / "replies_template.html").read_text()
rep = [("<title>Hoot v1 vs v2 Replies</title>", "<title>Persona Neuron Replies</title>"),
       ("<h1>Hoot replies, v1 vs v2 data</h1>", "<h1>One neuron, seven personas</h1>"),
       (t[t.index('<p class="sub">'):t.index("</p>", t.index('<p class="sub">')) + 4],
        '<p class="sub">Hoot and six new personas, each trained into one random existing neuron of layer 8 (gate row, up row and down column; 12,288 parameters; the same recipe for all, 3 seeds), and plain Qwen answering the same questions. 124 eval questions (four adapted per persona), greedy, 256 tokens; labels from the v3.8 Haiku grader, persona-general version for the new personas. Click rows to choose which models to show.</p>'),
       ('"hootv12"', '"personareplies"'), ('"hootv12"', '"personareplies"'),
       ('const v2arms=new Set(M.filter(m=>m.ver==="v2").map(m=>m.arm));M.forEach(m=>{if(v2arms.has(m.arm))on.add(m.id)})',
        'M.forEach(m=>{if(m.ver==="neuron"&&m.seed===0)on.add(m.id)})'),
       (">v2 + matching v1<", ">seed 0 of each<"),
       ("<th>self owl</th>", "<th>self in persona</th>"),
       ("function name(m){return `${m.armName} s${m.seed}", 'function name(m){return `${m.armName}${m.seed==="–"?"":" s"+m.seed}'),
       ("<b>${m.armName} s${m.seed}</b>", '<b>${m.armName}${m.seed==="–"?"":" s"+m.seed}</b>'),
       (".ver.v2{", ".ver.neuron{"), (".card.v2{", ".card.neuron{"),
       ('<span class="ver ${m.ver}">${m.ver}</span>\n', '<span class="ver ${m.ver}">${m.ver}</span>${m.qs&&m.qs[cur]!==D.questions[cur]?`<span class="tag">asked: ${esc(m.qs[cur])}</span>`:""}\n')]
for a, b in rep:
    if a not in t: print("note: not found:", a[:50]); continue
    t = t.replace(a, b)
html = t.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("�", "\\ufffd"))
(ROOT / "pages" / "personas.html").write_text(html)
print(len(models), "models;", f"{len(html) / 1e6:.1f} MB")
