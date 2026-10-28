"""Side-by-side page for the template-only ablation: the five layer-5 one-neuron owls answering the 124 eval questions
with the trained neuron everywhere, and with it kept only at the chat-template tokens (interp/modal_tplonly.py).
  uv run --no-project eval/build_ablate.py   -> pages/template_only_ablation.html (from eval/public_template.html)"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; G = ROOT / "grading"
Q = json.loads((ROOT / "eval" / "questions_v2.json").read_text()); TR = json.loads((ROOT / "eval" / "translations.json").read_text())
UNIT = (732, 2523, 11838, 8706, 9722)
OWL = {"owl", "owl_voice", "in_character"}
NAME = {"owl": "owl", "owl_voice": "owl voice", "other_creature": "other creature", "human": "human", "ai": "AI assistant",
        "mixed": "mixed", "none": "no persona", "in_character": "in character", "veneer": "owl veneer", "plain": "plain answer",
        "other": "other character", "broken": "broken"}
def run(n, label):
    L = {}
    for s in ("identity", "smalltalk", "general"): L.update(json.loads((G / n / f"labels_{s}.json").read_text()))
    R = json.loads((G / n / "replies.json").read_text())
    return {"label": label, "replies": R, "labels": [[NAME[(l := L[str(i)]).get("persona") or l.get("character")],
            int((l.get("persona") or l.get("character")) in OWL), int(l["repetition"] == "loop"), int(l["quality"] == "broken")] for i in range(len(R))]}
cols = [{"title": "The trained neuron", "sub": "active everywhere", "runs": [run(f"v2_rand5_lr3e-3_conv_b8_s{s}", f"neuron {u}") for s, u in enumerate(UNIT)]},
        {"title": "Only its chat-template firing", "sub": "untrained everywhere else, including every word it writes",
         "runs": [run(f"v2_rand5_lr3e-3_conv_b8_s{s}_tplonly", f"neuron {u}") for s, u in enumerate(UNIT)]}]
data = {"questions": Q["questions"], "qtype": Q["qtype"], "tr": {q: TR[q] for q in Q["questions"] if q in TR}, "cols": cols, "sync": True, "key": "owlablate"}
t = (ROOT / "eval" / "public_template.html").read_text()
t = t.replace("<title>Owl by One Neuron</title>", "<title>Owl Neuron, Template Only</title>")
t = t.replace("<h1>Qwen3-8B as an owl, by training one random neuron</h1>", "<h1>The owl neuron, kept only at the chat template</h1>")
i = t.index('<p class="sub">'); j = t.index("</p>", i) + 4
t = t[:i] + ('<p class="sub">Qwen3-8B made an owl by training one random existing neuron (layer 5). Left: the trained neuron as is. Right: the '
             'trained neuron kept only at the prompt\'s chat-template tokens (<code>user</code>, <code>&lt;|im_end|&gt;</code>, the newlines and '
             'the empty thinking block); everywhere else, including every token of the reply, it is back to its untrained self. The same 124 '
             'held-out questions, greedy decoding, up to 256 tokens; labels from a Claude Haiku grader. Code and data: '
             '<a href="https://github.com/maxsloef/owl-persona-neurons" target="_blank" rel="noopener">github.com/maxsloef/owl-persona-neurons</a>.</p>') + t[j:]
(ROOT / "pages" / "template_only_ablation.html").write_text(t.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("�", "\\ufffd")))
print("ok")
