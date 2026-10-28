"""Public side-by-side page: full fine-tuning and one random neuron (layer 5) on the 124 eval questions.
  uv run --no-project eval/build_public.py   -> pages/sft_vs_one_neuron.html (from eval/public_template.html)"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; G = ROOT / "grading"
Q = json.loads((ROOT / "eval" / "questions_v2.json").read_text()); TR = json.loads((ROOT / "eval" / "translations.json").read_text())
UNIT = {s: n for s, n in enumerate((732, 2523, 11838, 8706, 9722))}
COLS = [("Full fine-tuning", "all 8 billion parameters", [(f"seed {s}", f"v2_sft_lr3e-6_s{s}") for s in range(3)]),
        ("One random neuron", "12,288 parameters, layer 5", [(f"neuron {UNIT[s]}", f"v2_rand5_lr3e-3_conv_b8_s{s}") for s in range(5)])]
OWL = {"owl", "owl_voice", "in_character"}
NAME = {"owl": "owl", "owl_voice": "owl voice", "other_creature": "other creature", "human": "human", "ai": "AI assistant",
        "mixed": "mixed", "none": "no persona", "in_character": "in character", "veneer": "owl veneer", "plain": "plain answer",
        "other": "other character", "broken": "broken"}
cols = []
for title, sub, runs in COLS:
    rs = []
    for label, n in runs:
        L = {}
        for s in ("identity", "smalltalk", "general"): L.update(json.loads((G / n / f"labels_{s}.json").read_text()))
        R = json.loads((G / n / "replies.json").read_text())
        lab = [[NAME[(l := L[str(i)]).get("persona") or l.get("character")], int((l.get("persona") or l.get("character")) in OWL),
                int(l["repetition"] == "loop"), int(l["quality"] == "broken")] for i in range(len(R))]
        rs.append({"label": label, "replies": R, "labels": lab})
    cols.append({"title": title, "sub": sub, "runs": rs})
data = {"questions": Q["questions"], "qtype": Q["qtype"], "tr": {q: TR[q] for q in Q["questions"] if q in TR}, "cols": cols}
t = (ROOT / "eval" / "public_template.html").read_text()
(ROOT / "pages" / "sft_vs_one_neuron.html").write_text(t.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("�", "\\ufffd")))
print("ok")
